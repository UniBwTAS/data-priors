import numpy as np
import os
import random
import cv2
from glob import glob

from responses import target
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug import SegmentationMapsOnImage
from abc import ABC, abstractmethod
from utils import LabelMapper, SoftLabelImageProcessor, PadCustom
from transformers.image_transforms import rgb_to_id
from transformers import Mask2FormerImageProcessor
import torch
from pathlib import Path
from copy import deepcopy
import sqlite3
import json
import warnings
from tqdm import tqdm


class RNG:
    def __init__(self, deterministic):
        self.deterministic = deterministic
        seed = 42 if deterministic else None
        self.random_rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)


class BaseDataset(Dataset, ABC):

    def __init__(self, dataset_dir, action='train', seg_task='semantic', config=None, is_main_process=True, transform=None, debug=False):
        super(BaseDataset, self).__init__()

        self.seg_task = seg_task

        if seg_task == 'semantic':
            self._getitem = self._getitem_semantic
        elif seg_task == 'instance':
            self._getitem = self._getitem_instance
        elif seg_task == 'panoptic':
            self._getitem = self._getitem_panoptic

        self.action = action
        self.mean = [80.85,  77.914, 76.967]
        self.std = [61.877, 61.501, 64.752]
        self.debug = debug
        self.slice_len = 2

        # SQLite database for storing paths of images and masks that contain (mostly) void pixels
        self.sqlite_file = config.get('excluded_void_images')
        if self.sqlite_file and not os.path.exists(self.sqlite_file):
            warnings.warn(f"SQLite file {self.sqlite_file} does not exist.")
            conn = sqlite3.connect(self.sqlite_file)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS path_lists (id TEXT PRIMARY KEY, data TEXT)")
            conn.commit()
            conn.close()

        self.name = self.__class__.__name__.lower()
        train_with_val = self.name in config.get('train_with_val', [])
        if self.action == 'pseudo_label' or (self.action == 'train' and train_with_val):
            actions = ['train', 'val']
            self.tta = config.get('test_time_augmentation')
        elif self.action == 'val' and train_with_val:
            actions = []
        else:
            actions = [self.action]
        self.data_paths = {'image_paths': [], f'{seg_task}_mask_paths': []}
        for action in actions:
            data_paths = self._collect_paths(dataset_dir, seg_task=seg_task, action=action)
            for key, value in data_paths.items():
                if key not in self.data_paths:
                    self.data_paths[key] = []
                self.data_paths[key] += value

        if self.action == 'pseudo_label' and config['save_results']:
            # filter out image and label paths if pseudo labels already exist
            keep_image_paths = []
            keep_label_paths = []
            for image_path, label_path in zip(self.data_paths['image_paths'], self.data_paths['semantic_mask_paths']):
                if config['use_priors']:
                    save_dir = Path('datasets', 'generated_labels', str(Path(config['pretrained']).parts[1]))
                else:
                    save_dir = Path('datasets', 'generated_labels', str(Path(config['pretrained']).parts[1]), 'no_priors')
                if not config['dataset'][0] == 'cityscapesextra':
                    pseudo_label_path = os.path.join(*Path(label_path).parts[1:]).replace('.jpg', '.png')
                    pseudo_label_path = os.path.join(save_dir, pseudo_label_path)
                    if not os.path.exists(pseudo_label_path):
                        keep_image_paths.append(image_path)
                        keep_label_paths.append(label_path)
                else:
                    image_name = os.path.join(*Path(image_path).parts[1:])
                    pseudo_label_path = save_dir / image_name

                    if not os.path.exists(pseudo_label_path):
                        keep_image_paths.append(image_path)
                        keep_label_paths.append(label_path)

            num_removed = len(self.data_paths['image_paths']) - len(keep_image_paths)
            if num_removed > 0 and is_main_process:
                print(f"{self.name}: Removed {num_removed} images and labels that already have pseudo labels."
                      f"\n{len(keep_image_paths)} more to label.")
            self.data_paths['image_paths'] = keep_image_paths
            self.data_paths['semantic_mask_paths'] = keep_label_paths

        self.hard_labels = True # because training with soft labels is deprecated
        self.validate_hard = self.action == 'val' and self.name in config['validation_datasets']
        self.source_dataset = self.name == config.get('source_dataset')
        if self.hard_labels:
            if not (self.source_dataset or (self.action == 'val' and self.name in config['validation_datasets'])):
                for data_type, path_list in self.data_paths.items():
                    if 'mask' in data_type and 'pseudo_labels_root' in config:
                        new_root = config['pseudo_labels_root']
                        dataset_root = os.path.join(*Path(dataset_dir).parts[:-1])
                        self.data_paths[data_type] = [path.replace(dataset_root, new_root) for path in path_list]
                        for path in self.data_paths[data_type]:
                            assert os.path.exists(path), f"Path {path} does not exist."

        # print(f"before {self.name}", len(self.data_paths['image_paths']))
        if self.sqlite_file and len(self.data_paths['image_paths']) > 0:
            self._filter_void_paths(seg_task)
        # print(f"after {self.name}", len(self.data_paths['image_paths']))

        self.len = len(self.data_paths['image_paths'])
        self.transform = transform
        self.label_mapper = LabelMapper(lists_dir=os.path.join(dataset_dir, 'lists'),
                                        seg_task=seg_task,
                                        master_labels=config['master_labels'],
                                        verbose=is_main_process,
                                        dataset_name=self.name)

        self.rng = RNG(deterministic=('val' in action or debug))

        self.image_processor = SoftLabelImageProcessor(do_resize=False,
                                                       do_rescale=False,
                                                       do_normalize=False)
        self.m2f_processor = Mask2FormerImageProcessor(ignore_index=255,
                                                        do_resize=False,
                                                       do_rescale=False,
                                                       do_normalize=False)

        self.tb_images = config.get('tensorboard_imgs', [])
        void_id_list = self.label_mapper.master_id2id[255]
        self.void_id = max(void_id_list)    # get any void id for mask padding background
        assert self.void_id >= 0

        self.invalid_ids = set(void_id_list)

        self.indices = list(range(len(self.data_paths['image_paths'])))

        if aug_config := config.get('augmentation'):
            aug_config = deepcopy(aug_config)
            aug_base_config = aug_config['base']
            class_config = aug_config.get(self.name, {})
            aug_base_config.update(class_config)
            self.aug_config = aug_base_config
        else:
            self.aug_config = {}
        if 'image_dim' not in self.aug_config and self.action != 'pseudo_label':
            assert config['training_params']['batch_size'] == 1, "Only batch size 1 supported for variable image dimensions."

    def __len__(self):
        return len(self.data_paths['image_paths'])

    def __getitem__(self, idx):
        image_path = self.data_paths['image_paths'][idx]

        img_name = Path(image_path).name
        full_img = (self.action == 'val' and img_name in self.tb_images) or self.action == 'pseudo_label' #or True
        out_dict = self._getitem(idx, image_path, full_img)

        out_dict.update({"dataset": self.name, "image_path": image_path, "full_img": full_img})

        return out_dict

    def _augment(self, image, mask, pixel_mask=None, mask_path=None):

        ac = self.aug_config

        # Randomly crop images and masks
        longest_side = max(image.shape[:2])
        max_crop_size = min(int(longest_side*ac['max_crop_scale']), ac['max_crop_size'])
        min_crop_size = max(int(ac['min_crop_scale'] * max_crop_size), ac['min_crop_size'])
        # in case image is smaller than min_crop_size
        max_crop_size = max(max_crop_size, min_crop_size)

        tries = 5
        for try_ in range(tries):

            if try_ > tries-1:
                print(f"Warning: Augmentation already on try {try_}!")

            # center-pad the image to according to the random crop so that the crop box is always on image and not padded area
            aspect_ratio = ac['image_dim']['height'] / ac['image_dim']['width']
            crop_size = self.rng.random_rng.randint(min_crop_size, max_crop_size)
            cropw = crop_size
            croph = int(crop_size * aspect_ratio)
            padx = max(0, (cropw - image.shape[1]))
            pady = max(0, (croph - image.shape[0]))

            pad = PadCustom(padx, pady, color_pad_value=127, grayscale_pad_value=self.void_id)

            # same crop, flip and affine for image and masks --> to_deterministic()

            crop = iaa.CropToFixedSize(width=cropw, height=croph, position="uniform", seed=self.rng.np_rng).to_deterministic()
            flip = iaa.Fliplr(ac['flip'], seed=self.rng.np_rng).to_deterministic()
            affine = iaa.Affine(**ac['affine'], cval=127).to_deterministic()
            affine._cval_segmentation_maps = self.void_id
            mask_transform = [affine, crop, flip, iaa.Resize(ac['image_dim'])]

            mask = pad(mask)
            mask_object = SegmentationMapsOnImage(mask, shape=mask.shape)

            # Augment the image and masks
            seq_mask = iaa.Sequential(mask_transform)
            mask_aug = seq_mask(segmentation_maps=mask_object)
            mask_aug = np.squeeze(mask_aug.arr).astype(np.uint8)

            # Check that non-void pixels in mask
            mask_aug_set = set(mask_aug.flatten())
            if bool(mask_aug_set - self.invalid_ids):
                # checking if mask has only void pixels
                break
            if try_ == tries - 1: # failed on final try
                return None, None, None, None

        # transform rgb image
        seq_image_list = [
            # blur only sometimes
            iaa.Sometimes(ac['blur_prob'], iaa.GaussianBlur(sigma=(0.0, 3.0))),
            iaa.Sometimes(ac['noise_prob'], iaa.AdditiveGaussianNoise(scale=(0, ac['gaussian_noise']*255))),
            iaa.WithColorspace(
                to_colorspace='HSV',
                from_colorspace='RGB',
                children=iaa.Sequential([
                    iaa.WithChannels(0, iaa.Add(ac['hue_shift'])),
                    iaa.WithChannels(1, iaa.Multiply(ac['saturation'])),
                ])
            ),
            iaa.LinearContrast(ac['contrast']),
            iaa.Multiply(ac['brightness'])]

        seq_color = iaa.Sequential([*seq_image_list])
        image_aug_full = seq_color(image=image)
        image_aug_full = pad(image_aug_full)
        image_aug_full = seq_mask(image=image_aug_full)

        image_aug_affine = pad(image)
        image_aug_affine = seq_mask(image=image_aug_affine)

        # transform pixel mask
        if pixel_mask is not None:
            pad.grayscale_pad_value = 0
            seq_mask[0]._cval_segmentation_maps = 0
            pixel_mask = pad(pixel_mask)
            pixel_mask_object = SegmentationMapsOnImage(pixel_mask, shape=pixel_mask.shape)
            pixel_mask = np.squeeze(seq_mask(segmentation_maps=pixel_mask_object).arr)

        return image_aug_full, image_aug_affine, mask_aug, pixel_mask

    def _resize_and_pad(self, image, mask, pixel_mask=None):
        ac = self.aug_config

        # resize and preserve aspect ratio
        if 'image_dim' in ac:
            target_height = ac['image_dim']['height']
            target_width = ac['image_dim']['width']
        else:
            target_height, target_width = image.shape[:2]

        h, w = image.shape[:2]
        input_aspect_ratio = h / w
        target_aspect_ratio = target_height / target_width
        if input_aspect_ratio > target_aspect_ratio:
            new_h = target_height
            new_w = int(new_h / input_aspect_ratio)
        elif input_aspect_ratio < target_aspect_ratio:
            new_w = target_width
            new_h = int(new_w * input_aspect_ratio)
        else:
            # square image --> resize to smaller target dimension
            new_h, new_w = target_height, target_width

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Calculate padding needed to center the image
        pad_top = (target_height - new_h) // 2
        pad_bottom = target_height - new_h - pad_top
        pad_left = (target_width - new_w) // 2
        pad_right = target_width - new_w - pad_left

        # Apply constant value padding
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                   value=(127, 127, 127))
        mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                  value=self.void_id)

        if pixel_mask is not None:
            pixel_mask = cv2.resize(pixel_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            pixel_mask = cv2.copyMakeBorder(pixel_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        pad_tuple = (pad_top, pad_bottom, pad_left, pad_right)

        return image, mask, pixel_mask, pad_tuple

    def _test_time_augment(self, image):
        augmentations = {}

        # flip
        if self.tta['flip']:
            image_flip = cv2.flip(image, 1)
            image_flip = self._process_image(image_flip)
            augmentations['flip'] = image_flip

        # scale
        rescaled_images = {}
        for scale in self.tta['scales']:
            if scale != 1.0:
                rescaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                rescaled_image = self._process_image(rescaled_image)
                rescaled_images[scale] = rescaled_image
        augmentations['rescaled'] = rescaled_images

        return augmentations

    def _process_image(self, image):
        image = ((image - self.mean) / self.std)
        image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32)

    def _exclude_file_path(self, image_path, mask_path):
        # Connect to SQLite database
        conn = sqlite3.connect(self.sqlite_file)
        cursor = conn.cursor()

        # Try to retrieve the tuple of lists associated with the given key
        cursor.execute("SELECT data FROM path_lists WHERE id = ?", (self.name,))
        result = cursor.fetchone()

        if result is not None:
            # Key exists, load the tuple of lists and add the new paths
            image_paths, mask_paths = json.loads(result[0])
            image_paths.append(image_path)
            mask_paths.append(mask_path)
            # Update the existing tuple in the database
            updated_data = json.dumps((image_paths, mask_paths))
            cursor.execute("UPDATE path_lists SET data = ? WHERE id = ?", (updated_data, self.name))
        else:
            # Key does not exist, create a new tuple of lists with the provided paths
            data = json.dumps(([image_path], [mask_path]))
            cursor.execute("INSERT INTO path_lists (id, data) VALUES (?, ?)", (self.name, data))

        print(f"Added {image_path} and {mask_path} to void images for dataset '{self.name}'.")

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    def _filter_void_paths(self, seg_task):
        # Connect to SQLite database
        conn = sqlite3.connect(self.sqlite_file)
        cursor = conn.cursor()

        # Retrieve the stored void paths
        cursor.execute("SELECT data FROM path_lists WHERE id = ?", (self.name,))
        result = cursor.fetchone()

        if result is not None:
            # Unpack the stored image and mask paths
            void_image_paths, void_mask_paths = json.loads(result[0])

            # Extract folder and filename for each void path
            void_image_set = {(Path(path).parent.name, Path(path).name) for path in void_image_paths}
            void_mask_set = {(Path(path).parent.name, Path(path).name) for path in void_mask_paths}

            # Filter image and mask paths in tandem based on folder and filename match
            filtered_image_paths, filtered_mask_paths = [], []

            for img, msk in zip(self.data_paths['image_paths'], self.data_paths[f"{seg_task}_mask_paths"]):
                img_key = (Path(img).parent.name, Path(img).name)
                msk_key = (Path(msk).parent.name, Path(msk).name)

                # Only add pairs where both folder+filename keys are not in the void sets
                if img_key not in void_image_set and msk_key not in void_mask_set:
                    filtered_image_paths.append(img)
                    filtered_mask_paths.append(msk)

            self.data_paths['image_paths'] = filtered_image_paths
            self.data_paths[f"{seg_task}_mask_paths"] = filtered_mask_paths

        # Close the connection
        conn.close()

    @abstractmethod
    def _getitem_panoptic(self, idx, image, full_img):
        pass

    @abstractmethod
    def _getitem_instance(self, idx, image, full_img):
        pass

    def _getitem_semantic(self, idx, image_path, full_img):
        tries = 3
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for try_ in range(tries+1):
            mask_path = self.data_paths['semantic_mask_paths'][idx]
            semantic_mask = cv2.imread(mask_path, -1)
            if semantic_mask is None:
                raise ValueError(f"Failed to read mask at {mask_path}")
            if semantic_mask.dtype == np.uint16:
                semantic_mask = (semantic_mask // 1000).astype(np.uint8)
            assert semantic_mask.shape == image.shape[:2], \
                f"Image and mask dimensions do not match. image_path = {image_path} with dimension {image.shape[:2]}, mask_path = {mask_path} with dimension {semantic_mask.shape}"

            pixel_mask = np.ones(semantic_mask.shape, dtype=np.uint8)
            out_dict = {"padding": None}
            if self.action == 'train':
                image_aug, image_affine, semantic_mask, pixel_mask = self._augment(image, semantic_mask, pixel_mask, mask_path)
                if image_affine is not None and not self.hard_labels:
                    out_dict["image_affine"] = self._process_image(image_affine)
            elif self.action == 'val':
                image_aug, semantic_mask, pixel_mask, padding = self._resize_and_pad(image, semantic_mask, pixel_mask)
                if full_img:
                    out_dict["padding"] = padding
            elif self.action == 'pseudo_label':
                image_aug = image
                if self.tta:
                    augmentations = self._test_time_augment(image)
                    out_dict["augmentations"] = augmentations

            if semantic_mask is not None:
                # successful augmentation
                break
            else:
                print(f"Warning: Augmentation failed on {mask_path}!")
                if try_ < tries:
                    if self.sqlite_file:
                        self._exclude_file_path(image_path, mask_path)
                    self.indices.remove(idx)
                    idx = self.rng.random_rng.choice(self.indices)
                    image_path = self.data_paths['image_paths'][idx]
                    image = cv2.imread(image_path, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Augmentation failed permanently on {self.name}.")

        out_dict["pixel_values"] = self._process_image(image_aug)
        out_dict["mask_path"] = mask_path

        if not self.hard_labels:
            prediction_correction_mask_3D, class_labels, class_ambiguity = self.label_mapper.map_classes_soft(semantic_mask, pixel_mask)
            ambiguity_map_full = prediction_correction_mask_3D / np.sum(prediction_correction_mask_3D, axis=0)  # 3D
            # check for nan
            if np.isnan(ambiguity_map_full).any():
                raise ValueError(f"NaN values in ambiguity map for {mask_path}")
            # negative labels also have no ambiguity

            if self.action == 'val':
                ambiguity_map_2d = np.max(ambiguity_map_full, axis=0)
                ambiguity_map_2d_bool = ambiguity_map_2d == 1.
                out_dict.update({"ambiguity_map_bool": ambiguity_map_2d_bool.astype(float)})

            # TODO: since we are only using hard labels, we may as well get rid of soft label computation and all other unnecessary computations
            prediction_correction_mask_3D = torch.from_numpy(prediction_correction_mask_3D)
            out_dict.update({"hard_labels": None})
        else:
            prediction_correction_mask_3D = torch.tensor([])    # don't need this anymore
            if self.source_dataset or self.validate_hard:
                semantic_mask, _ = self.label_mapper.map_classes_hard(semantic_mask, pixel_mask)
            else:
                semantic_mask[semantic_mask == self.void_id] = 255

            inputs = self.m2f_processor(out_dict["pixel_values"], segmentation_maps=semantic_mask, return_tensors='pt')

            class_labels = inputs["class_labels"][0]
            mask_labels = inputs["mask_labels"][0]
            out_dict.update({"hard_labels": mask_labels})
            if self.action == 'val':
                out_dict.update({"hard_labels_2d": semantic_mask})

        out_dict.update({"correction_mask": prediction_correction_mask_3D,
                    "class_labels": class_labels,
                    "pixel_mask": pixel_mask,})

        out_dict["image_orig"] = image_aug if full_img else None

        return out_dict

    @abstractmethod
    def _collect_paths(self, dataset_dir, seg_task, action):
        pass


class Cityscapes(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(
            glob(os.path.join(dataset_dir, 'leftImg8bit', action, '**', '*_leftImg8bit.png'), recursive=True))
        paths = {"image_paths": image_paths}

        if seg_task == 'panoptic':
            panoptic_mask_paths = sorted(
                glob(os.path.join(dataset_dir, 'gtFine', 'cityscapes_panoptic_' + action, '**', '*_panoptic.png'),
                     recursive=True))

            assert len(image_paths) == len(panoptic_mask_paths)
            paths["panoptic_mask_paths"] = panoptic_mask_paths

        elif seg_task == 'semantic':
            semantic_mask_paths = sorted(
                glob(os.path.join(dataset_dir, 'gtFine', action, '**', '*_labelIds.png'), recursive=True))

            assert len(image_paths) == len(semantic_mask_paths)
            paths["semantic_mask_paths"] = semantic_mask_paths

        if self.debug:
            paths = {key: paths[key][:self.slice_len] for key in paths}

        return paths

    # deprecated
    def _getitem_panoptic(self, idx, image, action):
        # deprecated
        panoptic_mask = cv2.imread(self.data_paths['panoptic_mask_paths'][idx], 1)
        panoptic_mask = cv2.cvtColor(panoptic_mask, cv2.COLOR_BGR2RGB)
        panoptic_mask = rgb_to_id(panoptic_mask)

        image, panoptic_mask = self._augment(image, panoptic_mask)

        panoptic_mask = self.label_mapper.map_seg_classes(panoptic_mask)

        out_dict = {}
        if action == 'val':
            semantic_mask = panoptic_mask.copy()
            semantic_mask[semantic_mask >= 1000] = semantic_mask[semantic_mask >= 1000] // 1000
            out_dict["semantic_mask"] = semantic_mask

        inst_classes = np.unique(panoptic_mask)
        seg_classes = inst_classes.copy()
        seg_classes[seg_classes >= 1000] = seg_classes[seg_classes >= 1000] // 1000
        inst2class = {inst: seg for inst, seg in zip(inst_classes, seg_classes)}

        image_norm = ((image - self.mean) / self.std)

        inputs = self.image_processor(
            image_norm,
            segmentation_maps=panoptic_mask,
            return_tensors='pt',
            instance_id_to_semantic_id=inst2class
        )

        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        out_dict.update(inputs)

        return out_dict

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class CityscapesExtra(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = []
        mask_paths = []

        dataset_dir = os.path.join(dataset_dir, 'extra')

        if action == 'train':

            # video
            for video_dir in sorted(
                    glob(os.path.join(dataset_dir, 'leftImg8bit_sequence_trainvaltest', 'leftImg8bit_sequence', '*', '*'))):
                image_paths_city = sorted(glob(os.path.join(video_dir, '*_leftImg8bit.png')))
                if self.action == 'pseudo_label':
                    image_paths_city_len = len(image_paths_city)
                    city = video_dir.split('/')[-1]  # too lazy to make windows compatible
                    mask_paths_city = [os.path.join(dataset_dir, 'city_ego_gt', f'{city}.png')] * image_paths_city_len
                else:
                    mask_paths_city = image_paths_city
                assert len(mask_paths_city) > 0

                image_paths += image_paths_city
                mask_paths += mask_paths_city

            # coarse data
            for coarse_dir in sorted(
                    glob(os.path.join(dataset_dir, 'leftImg8bit_trainextra', 'leftImg8bit', 'train_extra', '*'))):
                image_paths_coarse_city = sorted(glob(os.path.join(coarse_dir, '*_leftImg8bit.png')))
                if self.action == 'pseudo_label':
                    image_paths_coarse_city_len = len(image_paths_coarse_city)
                    city = coarse_dir.split('/')[-1]  # too lazy to make windows compatible
                    mask_paths_coarse_city = [os.path.join(dataset_dir, 'city_ego_gt', f'{city}.png')] * image_paths_coarse_city_len
                else:
                    mask_paths_coarse_city = image_paths_coarse_city
                if len(mask_paths_coarse_city) == 0:
                    mask_paths_coarse_city = [os.path.join(dataset_dir, 'city_ego_gt', f'{city}.png')]
                assert len(mask_paths_coarse_city) > 0

                image_paths += image_paths_coarse_city
                mask_paths += mask_paths_coarse_city

            assert len(image_paths) == len(mask_paths) > 0

        paths = {"image_paths": image_paths, "semantic_mask_paths": mask_paths}

        if self.debug:
            paths = {key: paths[key][:self.slice_len] for key in paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class Goose(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # default
    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, 'images', action, '**', '*.png'), recursive=True))

        if seg_task == 'semantic':
            mask_paths = sorted(
                glob(os.path.join(dataset_dir, 'labels', action, '**', '*_labelids.png'), recursive=True))
        elif seg_task == 'instance':
            mask_paths = sorted(
                glob(os.path.join(dataset_dir, 'labels', action, '**', '*_instanceids.png'), recursive=True))
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class GooseUrban(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # urban subset
    def _collect_paths(self, dataset_dir, seg_task, action):
        txt_file = os.path.join(dataset_dir, 'lists', f'{action}_urban_subset.txt')
        names_txt = open(txt_file, 'r')
        image_paths = [os.path.join(dataset_dir, 'images', action, name.strip()) for name in names_txt]

        if seg_task == 'semantic':
            mask_paths = [image_path.replace('images', 'labels').replace('windshield_vis', 'labelids') for image_path in image_paths]
        elif seg_task == 'instance':
            mask_paths = sorted(
                glob(os.path.join(dataset_dir, 'labels', action, '**', '*_instanceids.png'), recursive=True))
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class TAS500(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, action, '*.png')))

        if seg_task == 'semantic':
            mask_paths = sorted(
                glob(os.path.join(dataset_dir, f'{action}_labels_ids', '*.png')))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class Yamaha(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, action, '**', 'rgb.jpg')))

        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, action, '**', 'label_ids.png')))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class Lanes(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, action, 'images', '*.jpg')))

        if seg_task == 'semantic':
            mask_paths = [path.replace('.jpg', '_id.png').replace('/images/', '/masks/') for path in image_paths]
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class RUGD(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, 'RUGD_frames-with-annotations', action, '**', '*.png'), recursive=True))

        if seg_task == 'semantic':
            mask_paths = sorted(
                glob(os.path.join(dataset_dir, 'RUGD_annotations', action, '**', '*_bw_*.png'), recursive=True))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class Rellis3D(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = []
        mask_paths = []

        list_path = os.path.join(dataset_dir, 'Rellis_3D_image_split', action + '.lst')
        img_prefix = os.path.join(dataset_dir, "Rellis_3D_pylon_camera_node", "Rellis-3D")
        label_prefix = os.path.join(dataset_dir, "Rellis_3D_pylon_camera_node_label_id", "Rellis-3D")
        with open(list_path, 'r') as file:
            for line in file:
                image_path, mask_path = line.strip().split()
                image_paths.append(os.path.join(img_prefix, image_path))
                mask_paths.append(os.path.join(label_prefix, mask_path))

        assert len(image_paths) == len(mask_paths) > 0
        return {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class WildScenes(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):

        base_path = os.path.join(dataset_dir, "61541v003", "data" , "WildScenes", "WildScenes2d")
        img_paths = sorted(glob(os.path.join(base_path, "**", "image", "*.png"), recursive=True))
        mask_paths = sorted(glob(os.path.join(base_path, "**", "indexLabel", "*.png"), recursive=True))

        assert len(img_paths) == len(mask_paths) > 0
        return {"image_paths": img_paths, f"{seg_task}_mask_paths": mask_paths}

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class GreatOutdoors(BaseDataset):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def _collect_paths(self, dataset_dir, seg_task, action):
            img_paths = sorted(glob(os.path.join(dataset_dir, "pylon_camera_node", "**", '*.png'), recursive=True))
            mask_paths = sorted(glob(os.path.join(dataset_dir, "pylon_camera_node_label_id", "**", '*.png'), recursive=True))

            assert len(img_paths) == len(mask_paths) > 0
            return {"image_paths": img_paths, f"{seg_task}_mask_paths": mask_paths}

        def _getitem_panoptic(self, idx, image):
            raise NotImplementedError

        def _getitem_instance(self, idx, image):
            raise NotImplementedError

class COCO(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        txt_file = os.path.join(dataset_dir, 'lists', f'{action}_urban_subset.txt')
        names_txt = open(txt_file, 'r')
        image_paths = [os.path.join(dataset_dir, f'{action}2017', name.strip() + '.jpg') for name in names_txt]

        if seg_task == 'semantic':
            mask_paths = [path.replace(f'{action}2017', f'panoptic_semseg_{action}2017').replace('.jpg', '.png') for path in image_paths]
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class FreiburgForest(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, 'download', 'freiburg_forest_annotated', action, 'rgb', '*.jpg'), recursive=True))

        if seg_task == 'semantic':
            mask_paths = [path.replace('rgb', 'GT_id').replace('Clipped.jpg', 'mask.png') for path in image_paths]
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        if self.debug:
            image_paths = image_paths[:self.slice_len]
            mask_paths = mask_paths[:self.slice_len]

        return {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class CamVid(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, action,'*.png')))
        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, f'{action}_labels', '*_id.png')))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class ApolloScape(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):

        if action == 'val':
            # not going to use this dataset for validation
            image_paths = []
            mask_paths = []
        else:
            image_paths = sorted(glob(os.path.join(dataset_dir, 'road*', 'ColorImage', '*', '*', '*.jpg')))
            if seg_task == 'semantic':
                mask_paths = []
                for cam_id in [5, 6]:
                    for task in ['seg', 'ins']:
                        mask_paths += glob(os.path.join(dataset_dir, f'road*_{task}', 'Label', '*', '*', f'*Camera_{cam_id}_bin.png'))
                        mask_paths += glob(os.path.join(dataset_dir, f'road*_{task}', 'Label', '*', '*', f'*Camera_{cam_id}.png'))
                mask_paths = sorted(mask_paths)
            elif seg_task == 'instance':
                raise NotImplementedError
            elif seg_task == 'panoptic':
                raise NotImplementedError

            assert len(image_paths) > 0
            assert len(image_paths) == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class MapillaryVistas(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):

        image_paths = sorted(glob(os.path.join(dataset_dir, action, "images", '*.jpg')))
        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, f'{action}', 'labels_id', '*.png')))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths), f"{len_image_paths} != {len(mask_paths)}"

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class IndiaDrivingDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, "idd-20k-II", "idd20kII", "leftImg8bit", action, "**", "*.jpg")) +
                             glob(os.path.join(dataset_dir, "idd-segmentation", "IDD_Segmentation", "leftImg8bit", action, "**", "*.png")))
        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, "idd-20k-II", "idd20kII", "gtFine", "labels", action, "**", "*.png")) +
                             glob(os.path.join(dataset_dir, "idd-segmentation", "IDD_Segmentation", "gtFine", "labels", action, "**", "*.png")))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError


class BerkeleyDeepDrive(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, "bdd100k_seg", "bdd100k", "seg", "images", action, "*.jpg")))
        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, "bdd100k_seg", "bdd100k", "seg", "labels", action, "*.png")))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class KITTI(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        image_paths = sorted(glob(os.path.join(dataset_dir, action, "image_2", "*.png")))
        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, action, "semantic", "*.png")))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class A2D2(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        if action == "val":
            image_paths = []
            mask_paths = []
        else:
            image_paths = sorted(glob(os.path.join(dataset_dir, "camera_lidar_semantic", "*", "camera", "*", "*.png")))
            if seg_task == 'semantic':
                mask_paths = sorted(glob(os.path.join(dataset_dir, "camera_lidar_semantic", "*", "label_id", "*", "*.png")))
            elif seg_task == 'instance':
                raise NotImplementedError
            elif seg_task == 'panoptic':
                raise NotImplementedError

            len_image_paths = len(image_paths)
            assert len_image_paths > 0
            assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class NuImages(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):

        image_paths = sorted(glob(os.path.join(dataset_dir, action, "img", "*.jpg")))
        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, action, "masks", "*.png")))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class WildDash(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):
        if action == "train":
            image_paths = []
            mask_paths = []
        else:
            image_paths = sorted(glob(os.path.join(dataset_dir, "wd_public_v2p0", "images", "*.jpg")))
            if seg_task == 'semantic':
                mask_paths = sorted(glob(os.path.join(dataset_dir, "wd_public_v2p0", "semantic", "*.png")))
            elif seg_task == 'instance':
                raise NotImplementedError
            elif seg_task == 'panoptic':
                raise NotImplementedError

            len_image_paths = len(image_paths)
            assert len_image_paths > 0
            assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class Waymo(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _collect_paths(self, dataset_dir, seg_task, action):

        image_paths = sorted(glob(os.path.join(dataset_dir, action, 'images', '*.png')))

        if seg_task == 'semantic':
            mask_paths = sorted(glob(os.path.join(dataset_dir, action, 'segmentation', '*.png')))
        elif seg_task == 'instance':
            raise NotImplementedError
        elif seg_task == 'panoptic':
            raise NotImplementedError

        len_image_paths = len(image_paths)
        assert len_image_paths > 0
        assert len_image_paths == len(mask_paths)

        paths = {"image_paths": image_paths, f"{seg_task}_mask_paths": mask_paths}

        return paths

    def _getitem_panoptic(self, idx, image):
        raise NotImplementedError

    def _getitem_instance(self, idx, image):
        raise NotImplementedError

class TestDataset(Dataset):
    def __init__(self, dataset_dir, config):
        assert os.path.isdir(dataset_dir)

        super(Dataset, self).__init__()
        # self.mean = np.array([80.85, 77.914, 76.967], dtype=np.float32)
        # self.std = np.array([61.877, 61.501, 64.752], dtype=np.float32)

        self.image_paths = (sorted(glob(os.path.join(dataset_dir, '**', '*.jpg'))) +
                            sorted(glob(os.path.join(dataset_dir, '**', '*.png'))) +
                            sorted(glob(os.path.join(dataset_dir, '*.jpeg'))) +
                            sorted(glob(os.path.join(dataset_dir, '*.png'))))
        self.len = len(self.image_paths)
        assert self.len > 0

        self.image_processor = SoftLabelImageProcessor(do_resize=False,
                                                       do_rescale=False,
                                                       do_normalize=False)
        self.label_mapper = LabelMapper(lists_dir=config['list_dir'], seg_task=config['seg_task'],
                                        master_labels=config['master_labels'])

        self.tta = config.get('test_time_augmentation')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, 1)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixel_values = self._process_image(image)
        augmentations = self._test_time_augment(image) if self.tta else None

        return {"pixel_values": pixel_values, "image_path": image_path, "class_labels": None,
                "padding": None, "image_orig": image, "hard_labels": None, "augmentations": augmentations}

    def _process_image(self, image):
        image = ((image - self.mean) / self.std)
        image = np.transpose(image, (2, 0, 1))
        return image

    def _test_time_augment(self, image):
        augmentations = {}

        # flip
        if self.tta['flip']:
            image_flip = cv2.flip(image, 1)
            image_flip = self._process_image(image_flip)
            augmentations['flip'] = image_flip

        # scale
        rescaled_images = {}
        for scale in self.tta['scales']:
            if scale != 1.0:
                rescaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                rescaled_image = self._process_image(rescaled_image)
                rescaled_images[scale] = rescaled_image
        augmentations['rescaled'] = rescaled_images

        return augmentations

if __name__ == "__main__":
    from DataLoader import DataLoaderFactory
    from tqdm import tqdm
    import yaml

    dlf = DataLoaderFactory()
    seg_task = 'semantic'

    # load config
    with open('config_test.yaml') as f:
        config = yaml.safe_load(f)
    config['dataset'] = ["cityscapes"]
    config['training_params']['batch_size'] = 1
    config['training_params']['num_workers'] = 0

    loader = dlf(config, action='train', debug=False, seg_task=seg_task)

    id2label = loader.dataset.datasets[0].label_mapper.master_id2master_label
    idx2label = id2label.copy()
    void_label = idx2label[255]
    del idx2label[255]
    idx2label[len(idx2label)] = void_label
    del id2label[255]

    print(len(loader))
    for sample in tqdm(loader):

        # class_labels = sample['class_labels'][0].numpy()
        # class_labels = class_labels.tolist()
        # print(class_labels)

        # image = sample['image_affine'][0]
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mask = sample['correction_mask'][0].numpy()
        unique_ids = [i for i in range(len(mask)) if np.any(mask[i])]
        print(unique_ids)

        # cv2.imshow('image', image)
        # # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(0)
        # has_variable_values = False
        # for i in range(len(idx2label)):
        #     layer = mask[i]
        #     unique_values = np.unique(layer)
        #     if len(unique_values) > 2:
        #         print(f"Class: {idx2label}")
        #         print(f"Unique values: {unique_values}")
        #         print(idx2label[i])
        #         has_variable_values = True
        # if has_variable_values:
        #     print(sample['image_path'])
            # window_name = f"{idx2label[i]}"
            # if i in class_labels:
            #     window_name += " (featured)"
            # cv2.imshow(window_name, layer.astype(np.uint8) * 255)
            # cv2.waitKey(0)
            # cv2.destroyWindow(window_name)
