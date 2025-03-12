import torch
import argparse
import yaml
from DataLoader import DataLoaderFactory
from tqdm import tqdm
from copy import deepcopy
from transformers import Mask2FormerForUniversalSegmentation
import os
import numpy as np
import cv2
from utils import Plotter, ColorMap
from pathlib import Path
from accelerate import Accelerator
import gc
from PIL import Image
from matplotlib import pyplot as plt


class Inference:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        cmap = ColorMap()
        self.cmap = cmap.get_cmap(config.get('color_map'))

        # Data Loading
        dlf = DataLoaderFactory()
        seg_task = self.config['seg_task']
        self.accelerator = Accelerator()
        is_main_process = self.accelerator.is_main_process
        self.test_loader = dlf(deepcopy(self.config), action='pseudo_label', seg_task=seg_task, is_main_process=is_main_process)

        self.post_process = self.test_loader.dataset.datasets[0].image_processor.post_process_semantic_segmentation

        pretrained_path = self.config.get('pretrained')

        self.accelerator.print(f"Loading model: {pretrained_path}")
        if os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path)
            self.model = Mask2FormerForUniversalSegmentation(config=ckpt['model_config'])
            self.model.load_state_dict(state_dict=ckpt['model_state_dict'])
        else:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(pretrained_path)

        # Get index to label and id to label mappings. id2label has no void class by huggingface convention.
        self.id2label = self.test_loader.dataset.datasets[0].label_mapper.master_id2master_label
        self.idx2label = self.id2label.copy()
        void_label = self.idx2label[255]
        del self.idx2label[255]
        self.idx2label[len(self.idx2label)] = void_label
        del self.id2label[255]

        self.plotter = Plotter(idx2label=self.idx2label, colormap=self.cmap)
        if config['show_extra']:
            idx2label_extra = {k: k for k in range(255)} # increase if necessary
            config_cmap = config.get('color_map_extra')
            if config_cmap is not None:
                cmap_extra = ColorMap()
                cmap_extra = cmap_extra.get_cmap(config_cmap)
            else:
                cmap_extra = None
            self.plotter_extra = Plotter(idx2label=idx2label_extra, colormap=cmap_extra)

        self.save_softmax = config.get('save_softmax', False)

        self.model.eval()
        self.fac = 1

        self.model, self.test_loader = self.accelerator.prepare(self.model, self.test_loader)

    @torch.no_grad()
    def predict(self, visualize=False, save_results=False, show_extra=False):
        # Do a prediction w/o ground truth priors

        if save_results:
            save_dir = Path('datasets', 'generated_labels', str(Path(self.config['pretrained']).parts[1])) / 'no_priors'

        self.accelerator.print("Total number of images to label:", len(self.test_loader))

        disable_tqdm = not self.accelerator.is_main_process
        for i, data in enumerate(tqdm(self.test_loader, disable=disable_tqdm)):

            if save_results:
                if not (mask_path := data.get('mask_path')):
                    image_path = data['image_path'][0]
                    image_name = os.path.join(*Path(image_path).parts[-2:]).replace('.jpg', '.png').replace('.png', '_prediction.png')
                else:
                    image_name = os.path.join(*Path(mask_path[0]).parts[1:])
                new_path = save_dir / image_name
                if new_path.exists():
                    print("Deprecated. You shouldn't be seeing this message.")
                    print(f"Skipping {new_path}")
                    continue

            image = data['image_orig'][0]
            prediction = self.model(data['pixel_values'])
            target_sizes = [list(pv.shape[1:]) for pv in data["pixel_values"]]

            if tta := self.config.get('test_time_augmentation') and (augmentations := data.get('augmentations')):
                augmented_predictions = []
                if 'flip' in tta:
                    flipped_prediction = augmentations['flip']
                    flipped_prediction = self.model(flipped_prediction)
                    flipped_prediction.masks_queries_logits = torch.flip(flipped_prediction.masks_queries_logits, dims=(3,))
                    augmented_predictions.append(flipped_prediction)
                    del flipped_prediction
                    gc.collect()
                    torch.cuda.empty_cache()
                if 'scales' in tta:
                    for scale, scaled_image in augmentations['rescaled'].items():
                        scaled_prediction = None  # Initialize to None
                        try:
                            scaled_prediction = self.model(scaled_image)
                            augmented_predictions.append(scaled_prediction)
                        except RuntimeError as e:
                            if 'out of memory' in str(e):
                                print(f"Out of memory error at scale {scale}. Skipping this scale.")
                                # Free up GPU memory
                                del scaled_image
                                if scaled_prediction is not None:
                                    del scaled_prediction
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e  # Re-raise exception if it's not an OOM error
                        finally:
                            # Ensure tensors are deleted to free up memory
                            if scaled_prediction is not None:
                                del scaled_prediction
                            gc.collect()
                            torch.cuda.empty_cache()

                augmented_predictions.append(prediction)
                prediction_post = [self.post_process(prediction, target_sizes=target_sizes, return_softmax=True) for prediction in augmented_predictions]
                prediction_post = torch.stack(prediction_post).mean(dim=0)
                prediction_post = torch.argmax(prediction_post, dim=1)
                prediction_post = prediction_post[0].cpu().numpy().astype(np.uint8)

                if show_extra:
                    # single prediction
                    result = self.plotter.draw_semantic_segmentation(prediction_post, rgb_image=image, alpha=1.0, figsize=(10, 10))
                    cv2.imshow("Normal Prediction", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

                    # tta prediction
                    tta_prediction_cv = prediction_post[0].cpu().numpy().argmax(axis=0).astype(np.uint8)
                    result_tta = self.plotter.draw_semantic_segmentation(tta_prediction_cv, rgb_image=image, alpha=1.0, figsize=(10, 10))
                    cv2.imshow("TTA Prediction", cv2.cvtColor(result_tta, cv2.COLOR_RGB2BGR))

            else:
                prediction_post = self.post_process(prediction, target_sizes=target_sizes, return_softmax=False, remove_null_class=True)
                prediction_post = prediction_post[0].cpu().numpy().astype(np.uint8)

            if save_results:
                new_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(new_path), prediction_post)

            if visualize:

                result = self.plotter.draw_semantic_segmentation(prediction_post, rgb_image=image, alpha=0.6, figsize=(13, 13))
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                h, w = image.shape[:2]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                img_x = np.argwhere(gray_result[0] == 255).squeeze()[0] - 1
                self.fac = w / img_x

                cv2.namedWindow("Image")
                cv2.setMouseCallback("Image", self.mouse_callback, prediction_post)

                w_view = 1080
                h_view = int(w_view * h / w)
                image_resized = cv2.resize(image, (w_view, h_view))
                cv2.imshow("Original", image_resized)
                cv2.imshow("Image", result)

                key = cv2.waitKey(0)
                if key == 27:  # esc
                    break

            if tta:
                del augmented_predictions
            del prediction_post
            del prediction
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def predict_with_priors(self, visualize=False, save_results=False, show_extra=False):
        # Do a prediction with ground truth priors

        if save_results:
            save_dir = Path('datasets', 'generated_labels', str(Path(self.config['pretrained']).parts[1]))

        self.accelerator.print("Total number of images to label:", len(self.test_loader))

        disable_tqdm = not self.accelerator.is_main_process
        for i, data in enumerate(tqdm(self.test_loader, disable=disable_tqdm)):

            print(data['image_path'][0])

            if save_results:
                if not (mask_path := data.get('mask_path')):
                    image_path = data['image_path'][0]
                    image_name = os.path.join(*Path(image_path).parts[-2:]).replace('.jpg', '.png').replace('.png', '_prediction.png')
                elif data['dataset'] == ['cityscapesextra']:
                    # use image name for path
                    image_path = Path(data['image_path'][0])
                    image_name = os.path.join(*Path(image_path).parts[1:])
                else:
                    image_name = os.path.join(*Path(mask_path[0]).parts[1:])
                new_path = save_dir / image_name
                if new_path.exists():
                    print("Deprecated. You shouldn't be reading this message.")
                    print(f"Skipping {new_path}")
                    continue

            image = data['image_orig'][0]
            prediction = self.model(data['pixel_values'])
            # print(i, data['image_path'])
            target_sizes = [list(pv.shape[1:]) for pv in data["pixel_values"]]

            if (tta := self.config.get('test_time_augmentation', False)) and (augmentations := data.get('augmentations', False)):
                augmented_predictions = []
                if 'flip' in tta:
                    flipped_prediction = augmentations['flip']
                    flipped_prediction = self.model(flipped_prediction)
                    flipped_prediction.masks_queries_logits = torch.flip(flipped_prediction.masks_queries_logits, dims=(3,))
                    augmented_predictions.append(flipped_prediction)
                    del flipped_prediction
                    gc.collect()
                    torch.cuda.empty_cache()
                if 'scales' in tta:
                    for scale, scaled_image in augmentations['rescaled'].items():
                        scaled_prediction = None  # Initialize to None
                        try:
                            scaled_prediction = self.model(scaled_image)
                            augmented_predictions.append(scaled_prediction)
                        except RuntimeError as e:
                            if 'out of memory' in str(e):
                                print(f"Out of memory error at scale {scale}. Skipping this scale.")
                                # Free up GPU memory
                                del scaled_image
                                if scaled_prediction is not None:
                                    del scaled_prediction
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e  # Re-raise exception if it's not an OOM error
                        finally:
                            # Ensure tensors are deleted to free up memory
                            if scaled_prediction is not None:
                                del scaled_prediction
                            gc.collect()
                            torch.cuda.empty_cache()

                augmented_predictions.append(prediction)
                predictions_post = [self.post_process(prediction, target_sizes=target_sizes, return_softmax=True) for prediction in augmented_predictions]
                prediction_post = torch.stack(predictions_post).mean(dim=0)

                if self.save_softmax:
                    img_name = Path(data['image_path'][0]).stem
                    self.save_softmax_as_images(prediction_post, folder_name=f"junk/{img_name}")

                if show_extra:
                    # single prediction
                    normal_prediction_cv = predictions_post[0][0].cpu().numpy().argmax(axis=0).astype(np.uint8)
                    result = self.plotter.draw_semantic_segmentation(normal_prediction_cv, rgb_image=image, alpha=1.0, figsize=(15, 15))
                    cv2.imshow("Normal Prediction", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

                    # tta prediction
                    tta_prediction_cv = prediction_post[0].cpu().numpy().argmax(axis=0).astype(np.uint8)
                    result_tta = self.plotter.draw_semantic_segmentation(tta_prediction_cv, rgb_image=image, alpha=1.0, figsize=(15, 15))
                    cv2.imshow("TTA Prediction", cv2.cvtColor(result_tta, cv2.COLOR_RGB2BGR))

                    # ground truth
                    gt = cv2.imread(data['mask_path'][0], cv2.IMREAD_GRAYSCALE)
                    result_gt = self.plotter_extra.draw_semantic_segmentation(gt, rgb_image=image, alpha=1.0, figsize=(15, 15))
                    cv2.imshow("Ground Truth", cv2.cvtColor(result_gt, cv2.COLOR_RGB2BGR))

            else:
                prediction_post = self.post_process(prediction, target_sizes=target_sizes, return_softmax=True)

            correction_mask = data.get('correction_mask')
            if correction_mask is not None:
                prediction_post_corrected = prediction_post * correction_mask
                prediction_post_corrected = prediction_post_corrected / prediction_post_corrected.sum(dim=1, keepdim=True)
                prediction_post_corrected = prediction_post_corrected.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                prediction_post_numpy = prediction_post_corrected

            else:
                prediction_post_numpy = prediction_post[0].cpu().numpy().argmax(axis=0).astype(np.uint8)

            if save_results:
                new_path.parent.mkdir(exist_ok=True, parents=True)
                #print(str(new_path))
                cv2.imwrite(str(new_path), prediction_post_numpy)

            if visualize:

                result = self.plotter.draw_semantic_segmentation(prediction_post_numpy, rgb_image=image, alpha=1.0, figsize=(13, 13))
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                h, w = image.shape[:2]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                img_x = np.argwhere(gray_result[0] == 255).squeeze()[0] - 1
                self.fac = w / img_x

                cv2.namedWindow("Image")
                cv2.setMouseCallback("Image", self.mouse_callback, prediction_post_numpy)

                w_view = 1080
                h_view = int(w_view * h / w)
                image_resized = cv2.resize(image, (w_view, h_view))
                cv2.imshow("Original", image_resized)
                cv2.imshow("Image", result)

                key = cv2.waitKey(0)
                if key == 27:  # esc
                    break

            del augmented_predictions
            del predictions_post
            del prediction_post
            del prediction
            del prediction_post_corrected
            gc.collect()
            torch.cuda.empty_cache()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Extract pixel value
            x = int(x * self.fac)
            y = int(y * self.fac)
            if x < param.shape[1] and y < param.shape[0]:
                pixel_value = param[y, x]
                # Update window title with pixel value
                label = self.idx2label[pixel_value]
                window_title = f"Label at ({x}, {y}): {label}"
            else:
                window_title = f"Label at ({x}, {y}): Out of bounds"
            cv2.setWindowTitle("Image", window_title)

    @staticmethod
    def save_softmax_as_images(tensor, folder_name="junk"):
        """
        Saves each layer of a tensor as an 8-bit image in a folder.

        Args:
            tensor (torch.Tensor): A tensor of shape [1, N, H, W] (softmax output).
            folder_name (str): Directory name where images will be saved.
        """
        # Ensure tensor shape is [1, N, H, W]
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("Input tensor must have shape [1, N, H, W]")

        # Create the output directory if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)

        # Remove the batch dimension
        tensor = tensor.squeeze(0)  # Shape becomes [N, H, W]

        # Loop through each layer and save as an image
        for i in range(tensor.shape[0]):
            layer = tensor[i]  # Shape [H, W]

            # Normalize layer to range [0, 1]
            layer = (layer - layer.min()) / (layer.max() - layer.min() + 1e-8)  # Normalize to [0, 1]

            # Apply viridis colormap
            colormap = plt.cm.viridis
            layer_colored = colormap(layer.cpu().numpy())[:, :, :3]  # Extract RGB channels

            # Convert to 8-bit image
            layer_colored = (layer_colored * 255).astype(np.uint8)

            # Convert to PIL image
            image = Image.fromarray(layer_colored)

            # Downscale the image to 360 pixels wide while preserving aspect ratio
            width = 360
            aspect_ratio = image.height / image.width
            height = int(width * aspect_ratio)
            image = image.resize((width, height), Image.ANTIALIAS)

            # Save the image
            image.save(os.path.join(folder_name, f"layer_{i:03d}.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, default='config_inference.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    inference = Inference(config, args)
    save_results = config['save_results']
    visualize = config['visualize']
    assert save_results or visualize, "Either save_results or visualize must be True."
    assert not save_results and visualize or save_results and not visualize, "Only one of save_results or visualize can be True."
    print(config['dataset'])
    if config['use_priors']:
        inference.predict_with_priors(save_results=save_results, visualize=visualize, show_extra=config.get('show_extra', False))
    else:
        inference.predict(save_results=save_results, visualize=visualize)