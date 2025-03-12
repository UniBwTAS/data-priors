import torch
import argparse
import yaml
from DataLoader import DataLoaderFactory
from tqdm import tqdm
from accelerate import Accelerator
from copy import deepcopy
import os
from utils import profile, ENABLE_PROFILING, plot_confusion_matrix, Plotter, ColorMap, ModelEma, get_input_labels, format_duration
from glob import glob
from evaluation import SegmentationEvaluator
from tracker import DistributedTensorBoardSummaryWriter, CSVLogger
from pathlib import Path
from transformers import AutoConfig, Mask2FormerForUniversalSegmentation
from datetime import datetime
from time import time


class TrainAgent:

    def __init__(self, config, project_dir, exper_name):

        self.config = config
        self.project_dir = project_dir

        # Set gradient accumulation steps
        nbs = self.config['training_params'].get('nominal_batch_size', -1)
        bs = self.config['training_params']['batch_size']
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        gas = max(round(nbs / (bs * num_devices)), 1)

        self.accelerator = Accelerator(gradient_accumulation_steps=gas, )
        self.accelerator = Accelerator()
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.state.num_processes

        # Init tracking
        logdir = os.path.join(self.project_dir, 'logs')
        self.writer = DistributedTensorBoardSummaryWriter(self.accelerator, logdir)

        self.accelerator.print(f'Detected {num_devices} GPUs.')
        if gas > 1:
            self.accelerator.print(
                f'Doing gradient accumulation with {gas} steps for a nominal batch size of {nbs}.')
        else:
            self.accelerator.print(f'Using batch size {bs} on each device. '
                                   f'This equates to a full batch size of {bs * num_devices}.\n')

        self.device = self.accelerator.device
        self.epoch = 1
        self.start = 1
        self.step = 1
        self.valid_epoch = 1
        self.fitness = 0
        self.no_improvement_epochs = 0
        self.patience = self.config.get('patience', 10)
        self.best = False
        self.save_best = self.config["training_params"]['save_best']
        self.first_save = self.config["training_params"].get('first_epoch_save', 1)
        self.val_stats = {'val/loss': torch.inf, 'val/iou': 0}
        self.hard_labels = True # because training with soft labels is deprecated
        self.start_time = 0
        self.gradclip = self.config.get('gradclip', False)
        if self.gradclip:
            self.accelerator.print(f"Gradient clipping with norm {self.gradclip}")

        # Data Loading
        dlf = DataLoaderFactory()
        seg_task = self.config['seg_task']
        self.train_loader = dlf(deepcopy(self.config), action='train', debug=args.debug, seg_task=seg_task, is_main_process=self.accelerator.is_main_process)
        self.val_loader = dlf(deepcopy(self.config), action='val', debug=args.debug, seg_task=seg_task, is_main_process=self.accelerator.is_main_process)

        # Get index to label and id to label mappings. id2label has no void class by huggingface convention.
        self.id2label = self.train_loader.dataset.datasets[0].label_mapper.master_id2master_label
        self.idx2label = self.id2label.copy()
        void_label = self.idx2label[255]
        del self.idx2label[255]
        self.idx2label[len(self.idx2label)] = void_label
        del self.id2label[255]

        cmap = ColorMap()
        self.cmap = cmap.get_cmap(config.get('color_map'))
        self.plotter = Plotter(idx2label=self.idx2label, colormap=self.cmap)

        self.evaluator = SegmentationEvaluator(num_classes=len(self.id2label),
                                               device=self.accelerator.device,
                                               label_map=self.idx2label)
        if source_dataset := config.get("source_dataset"):
            self.accelerator.print(f"Target dataset: {source_dataset}")
            self.target_evaluator = SegmentationEvaluator(num_classes=len(self.id2label),
                                                   device=self.accelerator.device,
                                                   label_map=self.idx2label)
        else:
            self.target_evaluator = None

        self.post_process = self.train_loader.dataset.datasets[0].image_processor.post_process_semantic_segmentation

        model = Mask2FormerForUniversalSegmentation
        model_config = self.config.get('model_config')

        if pretrained_path := self.config.get('pretrained'):
            if not os.path.exists(pretrained_path):
                # is not local path
                self.accelerator.print(
                    f"{pretrained_path} is not a local path. Loading pretrained model from Huggingface Hub.")
                hf_model_config = AutoConfig.from_pretrained(self.config['pretrained'])
                hf_model_config.update(model_config | {'id2label': self.id2label})
                # labels = self.idx2label if self.config['model'] == 'segformer' else self.id2label
                self.model = model.from_pretrained(
                    pretrained_path,
                    # id2label=labels, # id2label must have no void class!
                    ignore_mismatched_sizes=True,
                    config=hf_model_config)
                if self.config.get('resume'):
                    self.accelerator.print("Setting resume to False because Huggingface Hub checkpoint is used.")
                    self.config['resume'] = False

            else:
                # is local path
                self.accelerator.print(f"Loading model: {pretrained_path}")
                ckpt = torch.load(pretrained_path)
                model_config = self.config.get('model_config') or ckpt['config']['model_config']
                model_config.update({'id2label': self.id2label})

                # not the prettiest solution getting the size from the config
                hf_model_config = AutoConfig.from_pretrained(f"facebook/mask2former-swin-{self.config['size']}-ade-semantic")
                hf_model_config.update(model_config | {'id2label': self.id2label})

                self.model = model(hf_model_config)
                try:
                    self.model.load_state_dict(state_dict=ckpt['model_state_dict'], strict=False)
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        self.load_state_dict_ignore_mismatch(ckpt['model_state_dict'])

        if self.config.get('resume') and pretrained_path:
            self.best = True
            self.start = ckpt['epoch'] + 1
            num_epochs = self.config['training_params']['epochs']
            pretrained0 = ckpt['config'].get('pretrained0') or ckpt['config']['pretrained']
            self.config['pretrained0'] = pretrained0
            if self.start >= num_epochs:
                self.accelerator.print(f"Number of epochs specified in config file {num_epochs} must be larger than start epoch {self.start}."
                                       "Shutting down.")
                exit(0)
            self.step = ckpt['global_step']
            self.first_save = 0
            self.val_stats = {k: v for k, v in ckpt.items() if k.startswith('val')}
            self.fitness = ckpt['fitness'].item()
            self.accelerator.print(f'Resuming training from {self.start} epoch.')
            self.id2label = self.model.config.id2label
            self.load_from_checkpoint('optimizer', ckpt)
            self.load_from_checkpoint('lr_scheduler', ckpt)

        self.get_loss = self.model.get_loss

        self.accelerator.print(f'Training from epoch {self.start}.')

        if self.config['optimizer'].get('freeze_backbone'):
            self.freeze_backbone()

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_lr_scheduler()

        if self.config.get('ema'):
            # init EMA (teacher) model (deprecated)
            self.ema_model = ModelEma(self.model, decay=0.999, device=self.device)

        # save copy of config
        config_name = 'config'
        if self.config.get('resume'):
            config_name += '_resume'
        with open(os.path.join(self.project_dir, f'{config_name}.yaml'), 'w') as f:
            yaml.dump(self.config, f, sort_keys=False, default_flow_style=False)

        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = (
            self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader))

        self.csv_logger = CSVLogger(exper_name=exper_name, accelerator=self.accelerator, csv_path="experiments/summaries.csv")
        self.csv_logger.update_config(self.config)

        self.accelerator.print(f"Training with {len(self.train_loader)*bs*self.world_size} training "
                               f"and {len(self.val_loader)*bs*self.world_size} validation samples.")

    def train(self):

        disable_tqdm = not self.accelerator.is_main_process
        self.accelerator.print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting training.")
        self.start_time = time()
        try:
            for self.epoch in tqdm(range(self.start, self.config['training_params']['epochs'] + 1), desc="epoch",
                                   position=0, disable=disable_tqdm):
                self._train_one_epoch()
            # end training -----------------------------------------------------------------------------------------

        except KeyboardInterrupt:
            self.accelerator.print("Keyboard interrupt detected.")
            if self.epoch > 1:
                self._save_model()

        self._end_training()

    @profile(ENABLE_PROFILING)
    def _train_one_epoch(self):

        self.model.train()

        disable_tqdm = not self.accelerator.is_main_process
        for batch_idx, sample in enumerate(
                tqdm(self.train_loader, desc="batch", position=1, disable=disable_tqdm)):
            self.optimizer.zero_grad()

            pixel_values = sample["pixel_values"]

            correction_mask = sample["correction_mask"]
            pixel_mask = sample["pixel_mask"]

            # Forward pass
            student_prediction = self.model(pixel_values, pixel_mask=pixel_mask, output_auxiliary_logits=True)
            # Forward pass EMA model
            if not self.hard_labels:
                if self.config['ema']:
                    with torch.inference_mode():
                        teacher_prediction = self.ema_model(sample['image_affine'], pixel_mask=pixel_mask, output_auxiliary_logits=False)
                    pre_labels = teacher_prediction
                else:
                    pre_labels = student_prediction

                target_sizes = [list(pv.shape[1:]) for pv in pixel_values]
                with torch.no_grad():
                    prediction_post = self.post_process(pre_labels, target_sizes=target_sizes, return_softmax=True)
                    corrected_predictions = prediction_post * correction_mask
                    # get unique classes
                    input_labels, class_labels = get_input_labels(corrected_predictions)
            else:
                input_labels = sample["hard_labels"]
                class_labels = sample["class_labels"]

            if self.config['model'] in ['mask2former', 'soft_mask2former', 'mask2former_dinov2']:

                # will have to go through each dataset and do a test training
                loss = self.get_loss(
                    masks_queries_logits=student_prediction.masks_queries_logits,
                    class_queries_logits=student_prediction.class_queries_logits,
                    mask_labels=input_labels,
                    class_labels=class_labels,
                    auxiliary_predictions=student_prediction.auxiliary_logits,
                )
            elif self.config['model'] == 'segformer':
                loss = self.get_loss(student_prediction.logits, corrected_predictions)

            self.accelerator.backward(loss)

            if self.step % 10 == 0:
                # don't slow down training with gather
                self.writer.add_scalar('train/loss', loss, self.step)

            if self.gradclip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradclip)

            self.optimizer.step()
            self.scheduler.step()

            if self.config['ema']:
                # Update the EMA model parameters
                self.ema_model.update(self.model)

            self.step += 1
            # end batch ------------------------------------------------------------------------------------

        if len(self.optimizer.param_groups) == 2:
            lrs = {'encoder': self.optimizer.param_groups[0]['lr'],
                   'decoder': self.optimizer.param_groups[1]['lr']}
            self.writer.add_scalars('train/lr', lrs, self.step)
        else:
            lrs = {'decoder': self.optimizer.param_groups[0]['lr']}
            self.writer.add_scalar('train/lr', lrs['decoder'], self.step)

        # self.scheduler.step()
        # end epoch ----------------------------------------------------------------------------------------

        if len(self.val_loader):
            self._validate()

        if self.epoch >= self.first_save:
            if self.best and self.save_best:
                # save only if best model
                self._save_model(delete_old=True)

            elif self.epoch % self.config["training_params"]["save_interval"] == 0 or self.epoch == self.config['training_params']['epochs']:
                # save model
                self._save_model()

    @torch.inference_mode()
    @profile(ENABLE_PROFILING)
    def _validate(self):
        self.valid_epoch = self.epoch
        self.model.eval()
        losses = []

        disable_tqdm = not self.accelerator.is_main_process

        for batch_idx, sample in enumerate(tqdm(self.val_loader, desc="val", position=2, disable=disable_tqdm)):

            pixel_values = sample["pixel_values"]
            pixel_mask = sample["pixel_mask"]
            correction_mask = sample["correction_mask"]
            # class_labels = sample["class_labels"]

            # Forward pass
            prediction = self.model(pixel_values, pixel_mask=pixel_mask, output_auxiliary_logits=True)
            target_sizes = [list(pv.shape[1:]) for pv in pixel_values]

            prediction_post = self.post_process(prediction, target_sizes=target_sizes, return_softmax=True)

            if not self.hard_labels:
                corrected_predictions = prediction_post * correction_mask
                input_labels, class_labels = get_input_labels(corrected_predictions)
                # semi_soft_labels = get_semi_soft_labels(corrected_predictions, class_labels)
            else:
                input_labels = sample["hard_labels"]
                class_labels = sample["class_labels"]

            if self.config['model'] in ['mask2former', 'soft_mask2former', 'mask2former_dinov2']:
                loss = self.get_loss(
                    masks_queries_logits=prediction.masks_queries_logits,
                    class_queries_logits=prediction.class_queries_logits,
                    mask_labels=input_labels,
                    class_labels=class_labels,
                    auxiliary_predictions=prediction.auxiliary_logits,
                )

            elif self.config['model'] == 'segformer':
                loss = self.get_loss(prediction.logits, corrected_predictions.contiguous())

            losses.append(loss)

            prediction_post = prediction_post / prediction_post.sum(dim=1, keepdim=True)
            hard_predictions = prediction_post.argmax(dim=1)
            if not self.hard_labels:
                corrected_predictions = corrected_predictions / corrected_predictions.sum(dim=1, keepdim=True)
                hard_labels = corrected_predictions.argmax(dim=1)
            else:
                hard_labels = sample["hard_labels_2d"]

            if not self.hard_labels:
                valid_mask = pixel_mask * sample["ambiguity_map_bool"]
            else:
                valid_mask = pixel_mask
            # what if all pixels are ambiguous?

            self.evaluator.add_batch(ground_truth=hard_labels, predictions=hard_predictions, pixel_mask=valid_mask)

            if source_dataset := self.config.get("source_dataset"):
                target_indices = torch.tensor([s.lower() == source_dataset for s in sample["dataset"]])
                if len(target_indices) > 0:
                    hard_labels_target = hard_labels[target_indices]
                    hard_predictions_target = hard_predictions[target_indices]
                    pixel_mask_target = pixel_mask[target_indices]
                    self.target_evaluator.add_batch(ground_truth=hard_labels_target, predictions=hard_predictions_target, pixel_mask=pixel_mask_target)

            tensorboard_indices = torch.nonzero(sample["full_img"], as_tuple=False).flatten().tolist()
            for idx in tensorboard_indices:

                # add sample prediction to tensorboard

                semantic_pred_mask = hard_predictions[idx].cpu().numpy()
                rgb_image = sample["image_orig"][idx]
                # remove padding
                padding = sample["padding"][idx]
                H, W = semantic_pred_mask.shape
                rgb_image = rgb_image[padding[0]:H-padding[1], padding[2]:W-padding[3], :]
                semantic_pred_mask = semantic_pred_mask[padding[0]:H-padding[1], padding[2]:W-padding[3]]

                annotated_img = self.plotter.draw_semantic_segmentation(semantic_pred_mask, rgb_image, alpha=0.67)
                img_name = sample["dataset"][idx] + "_" + Path(sample["image_path"][idx]).stem

                self.writer.add_image(f"{img_name}", annotated_img, self.step, dataformats='HWC')

        self.evaluator.compute_iou()
        self.evaluator.compute_accuracy()

        if self.accelerator.state.num_processes > 1:
            iou = {'mean_iou': self.accelerator.gather(self.evaluator.mIoU).mean(),
                   'per_category_iou': self.accelerator.gather(self.evaluator.IoU.unsqueeze(0)).mean(0), }
            accuracy = {'mean_accuracy': self.accelerator.gather(self.evaluator.mPA).mean(),
                        'per_category_accuracy': self.accelerator.gather(self.evaluator.mCA.unsqueeze(0)).mean(0), }
            confusion_matrix = self.accelerator.gather(self.evaluator.confusion_matrix.unsqueeze(0)).float().mean(0)
        else:
            iou = {'mean_iou': self.evaluator.mIoU,
                   'per_category_iou': self.evaluator.IoU, }
            accuracy = {'mean_accuracy': self.evaluator.mPA,
                        'per_category_accuracy': self.evaluator.mCA, }
            confusion_matrix = self.evaluator.confusion_matrix

        if self.accelerator.is_main_process:
            confusion_matrix_img = plot_confusion_matrix(confusion_matrix, self.idx2label)
            self.writer.add_image("val/confusion_matrix", confusion_matrix_img, self.step, dataformats='HWC')

        self.evaluator.reset()

        if self.config.get("source_dataset"):
            # compute metrics for target dataset

            self.target_evaluator.compute_iou()
            self.target_evaluator.compute_accuracy()

            iou_target = {'mean_iou': self.accelerator.gather(self.target_evaluator.mIoU).mean(),
                          'per_category_iou': self.accelerator.gather(self.target_evaluator.IoU.unsqueeze(0)).mean(0)}
            accuracy_target = {'mean_accuracy': self.accelerator.gather(self.target_evaluator.mPA).mean(),
                               'per_category_accuracy': self.accelerator.gather(self.target_evaluator.mCA.unsqueeze(0)).mean(0)}
            confusion_matrix_target = self.accelerator.gather(self.target_evaluator.confusion_matrix.unsqueeze(0)).float().mean(0)

            if self.accelerator.is_main_process:
                confusion_matrix_img_target = plot_confusion_matrix(confusion_matrix_target, self.idx2label)
                self.writer.add_image("val/confusion_matrix_target", confusion_matrix_img_target, self.step,
                                      dataformats='HWC')

            fitness_target = iou_target['mean_iou'] * 0.8 + accuracy_target['mean_accuracy'] * 0.2

            per_category_iou_target = iou_target.pop('per_category_iou')
            per_category_iou_target = {f'{self.idx2label[idx]}': val for idx, val in enumerate(per_category_iou_target)}
            per_category_accuracy_target = accuracy_target.pop('per_category_accuracy')
            per_category_accuracy_target = {f'{self.idx2label[idx]}': val for idx, val in enumerate(per_category_accuracy_target)}

            self.writer.gather_add_scalar("val/fitness_target", fitness_target, self.step)
            self.writer.gather_add_scalars("val/class_iou_target", per_category_iou_target, self.step)
            self.writer.gather_add_scalars("val/class_accuracy_target", per_category_accuracy_target, self.step)
            self.writer.gather_add_scalar("val/iou_target", iou_target['mean_iou'], self.step)
            self.writer.gather_add_scalar("val/accuracy_target", accuracy_target['mean_accuracy'], self.step)

            self.target_evaluator.reset()

        losses = torch.stack(losses)

        fitness = iou['mean_iou'] * 0.8 + accuracy['mean_accuracy'] * 0.2

        diff = fitness - self.fitness

        per_category_iou = iou.pop('per_category_iou')
        per_category_iou = {f'{self.idx2label[idx]}': val for idx, val in enumerate(per_category_iou)}
        per_category_accuracy = accuracy.pop('per_category_accuracy')
        per_category_accuracy = {f'{self.idx2label[idx]}': val for idx, val in enumerate(per_category_accuracy)}

        val_stats_short = {'fitness': fitness, **iou, **accuracy, 'loss': losses.mean()}
        self.val_stats = val_stats_short.copy()
        self.val_stats.update({"per_category_iou": per_category_iou})
        self.val_stats.update({"per_category_accuracy": per_category_accuracy})

        if diff > 0:
            self.fitness = fitness
            self.best = True
            self.no_improvement_epochs = 0
        else:
            self.best = False
            self.no_improvement_epochs += 1

        nc = int(max(4 - len(str(self.epoch)), 0))
        report = '\n' + f"{self.epoch}: " + nc * ' ' + ' | '.join(
            f'{k}: {v.item():.4f}' for k, v in val_stats_short.items())
        report += f" | stop: {self.no_improvement_epochs}/{self.patience}"
        if self.best:
            report += f' | improvement: {diff:.4f}'
            self.csv_logger.update_metrics({"fitness": round(fitness.item(), 3),
                                            "iou": round(iou['mean_iou'].item(), 3),
                                            "accuracy": round(accuracy['mean_accuracy'].item(), 3),
                                            "loss": round(losses.mean().item(), 3),
                                            "epoch": self.epoch})

        self.accelerator.print(report)

        self.writer.gather_add_scalar("val/loss", losses.mean(), self.step)
        self.writer.gather_add_scalar("val/fitness", fitness, self.step)
        self.writer.gather_add_scalar("val/iou", iou['mean_iou'], self.step)
        self.writer.gather_add_scalar("val/accuracy", accuracy['mean_accuracy'], self.step)
        self.writer.gather_add_scalars("val/class_iou", per_category_iou, self.step)
        self.writer.gather_add_scalars("val/class_accuracy", per_category_accuracy, self.step)

        if self.no_improvement_epochs >= self.patience:
            self.accelerator.print(f"Early stopping after {self.epoch} epochs.")
            self._end_training()

    def _save_model(self, delete_old=False):
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if delete_old:
                name = f'{self.epoch}_best'
            elif self.epoch == self.config['training_params']['epochs']:
                name = f'{self.epoch}_last'
            else:
                name = str(self.epoch)

            save_dir = os.path.join(self.project_dir, 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            save_name = f'{name}.pth'
            save_path = os.path.join(save_dir, save_name)

            self.accelerator.print(f"Saving model: {save_path}.")

            self.accelerator.save({'model_state_dict': unwrapped_model.state_dict(),
                                   'optimizer_state_dict': self.optimizer.state_dict(),
                                   'scheduler_state_dict': self.scheduler.state_dict(),
                                   'epoch': self.epoch,
                                   'global_step': self.step,
                                   'config': self.config,
                                   'model_config': unwrapped_model.config,
                                   'fitness': self.fitness,
                                   **self.val_stats},
                                  save_path)

            old_best_model_path = glob(os.path.join(save_dir, '*_best.pth'))
            if delete_old and len(old_best_model_path) > 1:
               # delete old best model
                old_best_model_path.remove(save_path)
                self.accelerator.print("\nOverwriting former best model.")
                for path in old_best_model_path:
                    if os.path.exists(path):
                        # self.accelerator.print(f"\nDeleting old best model {path}.")
                        os.remove(path)

    def get_optimizer(self):
        """
        This function is used to initialize the optimizer with the parameters
        specified in the configuration file
        """
        optimizer_config = self.config['optimizer']
        freeze_backbone = optimizer_config.get('freeze_backbone', False)
        optim_name = optimizer_config['name']

        self.accelerator.print("Optimizer:", optim_name)

        if 'encoder_lr' in optimizer_config[optim_name].keys() and not freeze_backbone:
            encoder_params = []
            decoder_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name.startswith("model.pixel_level_module.encoder"):
                        encoder_params.append(param)
                    else:
                        decoder_params.append(param)

            encoder_lr = optimizer_config[optim_name].pop('encoder_lr')
            decoder_lr = optimizer_config[optim_name].pop('lr')

            self.accelerator.print("Encoder LR:", encoder_lr)
            self.accelerator.print("Decoder LR:", decoder_lr)

            optimizer_params = [
                {'params': encoder_params, 'lr': encoder_lr},  # Adjust the LR for encoder
                {'params': decoder_params, 'lr': decoder_lr}  # Other parameters with default LR
            ]
            optimizer = getattr(torch.optim, optim_name)(optimizer_params, **optimizer_config[optim_name])
            optimizer_config[optim_name]['encoder_lr'] = encoder_lr
            optimizer_config[optim_name]['lr'] = decoder_lr
        else:
            if freeze_backbone:
                self.accelerator.print("Skipping encoder LR setting because backbone is frozen.")
            optimizer_params = self.model.parameters()
            if 'encoder_lr' in optimizer_config[optim_name].keys():
                optimizer_config[optim_name].pop('encoder_lr')
            optimizer = getattr(torch.optim, optim_name)(optimizer_params, **optimizer_config[optim_name])

        return optimizer

    def get_lr_scheduler(self):
        """
        This function is used to initialize the learning rate scheduler with the parameters
        specified in the configuration file.
        """
        scheduler_config = self.config.get('lr_scheduler')
        iterations_per_epoch = len(self.train_loader)

        # No scheduler
        if scheduler_config is None or scheduler_config['name'] == 'None':
            main_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda iteration: 1.0)
        else:
            name = scheduler_config['name']
            self.accelerator.print("Scheduler:", name)

            if name == 'MultiStepLR':
                milestones = [int(fac * self.config['training_params']['epochs'] * iterations_per_epoch) for fac in
                              scheduler_config[name]['milestones']]
                scheduler_config[name]['milestones'] = milestones
            if name == 'CosineAnnealingLR':
                scheduler_config[name]['T_max'] = self.config['training_params']['epochs'] * iterations_per_epoch

            main_scheduler = getattr(torch.optim.lr_scheduler, scheduler_config['name'])(self.optimizer,
                                                                                         **scheduler_config[name])

        # Warm-up scheduler setup
        if 'warmup_epochs' in scheduler_config:
            warmup_epochs = scheduler_config['warmup_epochs']
            self.accelerator.print(f"Warm-up epochs: {warmup_epochs}")
            warmup_iterations = warmup_epochs * iterations_per_epoch

            # Warm-up scheduler: linear increase from 0 to 1 over warmup_iterations
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                 lr_lambda=lambda
                                                                     iteration: (iteration+1) / (warmup_iterations+1))

            # Combine warm-up and main scheduler using SequentialLR
            scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer,
                                                              schedulers=[warmup_scheduler, main_scheduler],
                                                              milestones=[warmup_iterations])
        else:
            scheduler = main_scheduler

        return scheduler

    def load_from_checkpoint(self, obj, ckpt):
        assert obj in ['optimizer', 'lr_scheduler']

        current_obj_name = self.config[obj]['name']
        ckpt_obj_name = ckpt['config'][obj]['name']
        current_obj_config = self.config[obj].get(current_obj_name)
        ckpt_obj_config = ckpt['config'][obj].get(ckpt_obj_name)
        if current_obj_config != ckpt_obj_config or current_obj_name != ckpt_obj_name:
            self.accelerator.print(
                f"Warning: Detected scheduler mismatch between current config: {current_obj_name} "
                f"and checkpoint config: {ckpt_obj_name}. Using checkpoint config.")
        self.config[obj][current_obj_name] = ckpt_obj_config
        self.config[obj]['name'] = ckpt_obj_name
        if obj == 'optimizer':
            self.optimizer = self.get_optimizer()
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        elif obj == 'lr_scheduler':
            self.scheduler = self.get_lr_scheduler()
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    def freeze_backbone(self):
        self.accelerator.print("Freezing backbone.")
        for name, param in self.model.named_parameters():
            if name.startswith("model.pixel_level_module.encoder"):
                # don't freeze adapter part!
                if "spm" not in name and "interactions" not in name and "spm" not in name and "up" not in name and "model.pixel_level_module.encoder.norm" not in name:
                    param.requires_grad = False

    def load_state_dict_ignore_mismatch(self, state_dict):
        """
        Load a state_dict into 'model', ignoring parameters
        whose shapes don't match the model's current parameters.
        """
        model_dict = self.model.state_dict()

        # Make a copy of the input state_dict so we can safely modify it
        filtered_state_dict = dict(state_dict)

        for param_name in list(filtered_state_dict.keys()):
            if param_name in model_dict:
                # Compare shapes
                if filtered_state_dict[param_name].shape != model_dict[param_name].shape:
                    print(
                        f"[WARNING] Skipping parameter '{param_name}': "
                        f"checkpoint shape {filtered_state_dict[param_name].shape}, "
                        f"model shape {model_dict[param_name].shape}"
                    )
                    del filtered_state_dict[param_name]
            else:
                # Key does not exist in the model
                print(f"[WARNING] Skipping non-existent parameter '{param_name}' in the model.")
                del filtered_state_dict[param_name]

        self.model.load_state_dict(filtered_state_dict, strict=False)

    def init_head(self):
        # hacky solution to re-init mask2former head after downloading pretrained weights
        self.accelerator.print("Initializing head.")
        modules_decoder = self.model.model.pixel_level_module.decoder.modules()
        transformer_module = self.model.model.transformer_module.modules()
        for module in modules_decoder:
            self.model._init_weights(module)
        for module in transformer_module:
            self.model._init_weights(module)

    def init_full(self):
        # hacky solution to re-init mask2former head after downloading pretrained weights
        self.accelerator.print("Initializing model.")
        modules = self.model.modules()
        for module in modules:
            self.model._init_weights(module)

    def _shrink_perturb(self, lamda=0.5, sigma=0.01):
        # use this when fine-tuning on new data
        self.accelerator.print("Shrinking and perturbing pretrained weights")
        for (name, param) in self.model.named_parameters():
            if 'weight' in name:  # just weights
                mu = torch.zeros_like(param.data)
                param.data = param.data * lamda + torch.normal(mean=mu, std=sigma)

    def _end_training(self):
        self.writer.close()
        duration = format_duration(time() - self.start_time)
        self.accelerator.print(
            f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. With a duration of {duration}.")
        exit(0)


def main(args):

    config_path = args.config if not args.debug else 'config_debug.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.batch_size:
        config['training_params']['batch_size'] = args.batch_size
    if args.workers:
        config['training_params']['num_workers'] = args.workers
    if args.pretrained:
        config['pretrained'] = args.pretrained

    if args.debug:
        project_dir = os.path.join("experiments", "debug")
        os.makedirs(project_dir, exist_ok=True)
    else:
        project_dir = os.path.join("experiments", args.exper_name)
        resume_training = config.get('resume', False) and config.get('pretrained', False)
        exist_ok = resume_training
        if os.environ.get('LOCAL_RANK', '0') == '0':
            # os.makedirs(project_dir, exist_ok=exist_ok)
            os.makedirs(project_dir, exist_ok=True)
            print(f"Project directory: {project_dir}")

    TA = TrainAgent(config, project_dir, args.exper_name)
    TA.train()


if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    parser.add_argument('--exper_name', type=str, default='test', help='experiment name')
    parser.add_argument('--debug', action='store_true', default=False, help='turn on debugging mode')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--workers', type=int, help='number of workers')
    parser.add_argument('--pretrained', type=str, help='path to pretrained model')

    args = parser.parse_args()

    main(args)

# TODO: variable val images sizes + pre-train on pseudo-labels only, use target dataset as validation set, finetune on target dataset
# TODO: fix wilddash/mv val