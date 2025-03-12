from csv import excel

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset, WeightedRandomSampler, SequentialSampler, RandomSampler
import os


class DataLoaderFactory:

    def __call__(self, config, action, seg_task, debug=False, is_main_process=True):

        assert action in ['train', 'val', 'test', 'pseudo_label'], f'Unknown action: {action}'
        assert seg_task in ['semantic', 'instance', 'panoptic'], f'Unknown seg_task: {seg_task}'

        data_transform = transforms.Compose([transforms.ToTensor()])

        dataset, sampler = self._get_dataset(config, action=action, transform=data_transform, seg_task=seg_task, debug=debug, is_main_process=is_main_process)

        drop_last = action == 'train'

        batch_size = 1 if action in ['test', 'pseudo_label'] else config['training_params']['batch_size']
        num_workers = config.get('num_workers')
        if num_workers is None:
            num_workers = config['training_params']['num_workers']
        collate_fn = None if action == 'test' else self.collate_fn

        # print("!!!!!!!!! Random Sampler set to default !!!!!!!!!")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # sampler=sampler,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=self._worker_init_fn,
            drop_last=drop_last,
            collate_fn=collate_fn
        )

        return loader

    @staticmethod
    def _worker_init_fn(worker_id):
        """The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.

        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

        """
        base_seed = torch.IntTensor(1).random_().item()
        np.random.seed(base_seed + worker_id)

    def collate_fn(self, batch):

        # inputs_test = [sample.pop("test") for sample in batch]
        class_labels = {"class_labels": [sample.pop("class_labels") for sample in batch]}
        # ambiguity_map_reduced = {"ambiguity_map_reduced": [sample.pop("ambiguity_map_reduced") for sample in batch]}
        # ambiguity_map_full = {"ambiguity_map_full": [sample.pop("ambiguity_map_full") for sample in batch]}
        # mask_labels = {"mask_labels": [sample.pop("mask_labels") for sample in batch]}
        # ambiguity_mask_bool = {"ambiguity_mask_bool": [sample.pop("ambiguity_mask_bool") for sample in batch]}
        padding = {"padding": [sample.pop("padding") for sample in batch]}
        image_orig = {"image_orig": [sample.pop("image_orig") for sample in batch]} # only need it for tensorboard viz
        hard_labels = {"hard_labels": [sample.pop("hard_labels") for sample in batch]}
        # class_label_weights = {"class_label_weights": [sample.pop("class_label_weights") for sample in batch]}

        out_dict = (default_collate(batch) | class_labels | padding | image_orig | hard_labels) #| class_label_weights)# |
        return out_dict

    def _get_dataset(self, config, action, is_main_process=True, **kwargs):
        dataset_names = config['dataset']
        if not isinstance(dataset_names, list):
            dataset_names = [dataset_names]
        datasets = []
        dataset_weights = []
        dataset_weights_dict = config.get('training_params', {}).get('dataset_weights', {})

        for dataset_name in dataset_names:
            dataset_name = dataset_name.lower()
            if dataset_name == 'cityscapes':
                from DatasetClasses import Cityscapes
                dataset_dir = config.get('dataset_dir', 'datasets/cityscapes')
                datasets.append(Cityscapes(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('cityscapes', 1.0))
            elif dataset_name == 'cityscapesextra':
                from DatasetClasses import CityscapesExtra
                dataset_dir = config.get('dataset_dir', 'datasets/cityscapes')
                datasets.append(CityscapesExtra(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('cityscapesextra', 1.0))
            elif dataset_name == 'goose':
                from DatasetClasses import Goose
                dataset_dir = config.get('dataset_dir', 'datasets/goose2d')
                datasets.append(Goose(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('goose', 1.0))
            elif dataset_name == 'gooseurban':
                from DatasetClasses import GooseUrban
                dataset_dir = config.get('dataset_dir', 'datasets/goose2d')
                datasets.append(GooseUrban(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('goose', 1.0))
            elif dataset_name == 'rugd':
                from DatasetClasses import RUGD
                dataset_dir = config.get('dataset_dir', 'datasets/rugd')
                datasets.append(RUGD(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('rugd', 1.0))
            elif dataset_name == 'rellis3d':
                from DatasetClasses import Rellis3D
                dataset_dir = config.get('dataset_dir', 'datasets/rellis-3d')
                datasets.append(Rellis3D(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('rellis3d', 1.0))
            elif dataset_name == 'tas500':
                from DatasetClasses import TAS500
                dataset_dir = config.get('dataset_dir', 'datasets/tas500')
                datasets.append(TAS500(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('tas500', 1.0))
            elif dataset_name == 'yamaha':
                from DatasetClasses import Yamaha
                dataset_dir = config.get('dataset_dir', 'datasets/yamaha')
                datasets.append(Yamaha(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('yamaha', 1.0))
            elif dataset_name == 'coco':
                from DatasetClasses import COCO
                dataset_dir = config.get('dataset_dir', 'datasets/coco')
                datasets.append(COCO(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('coco', 1.0))
            elif dataset_name == 'lanes':
                from DatasetClasses import Lanes
                dataset_dir = config.get('dataset_dir', 'datasets/road_lane_segmentation')
                datasets.append(Lanes(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('lanes', 1.0))
            elif dataset_name == 'freiburgforest':
                from DatasetClasses import FreiburgForest
                dataset_dir = config.get('dataset_dir', 'datasets/freiburg-forest')
                datasets.append(FreiburgForest(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('freiburgforest', 1.0))
            elif dataset_name == 'apolloscape':
                from DatasetClasses import ApolloScape
                dataset_dir = config.get('dataset_dir', 'datasets/ApolloScape')
                datasets.append(ApolloScape(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('apolloscape', 1.0))
            elif dataset_name == 'indiadrivingdataset':
                from DatasetClasses import IndiaDrivingDataset
                dataset_dir = config.get('dataset_dir', 'datasets/IDD')
                datasets.append(IndiaDrivingDataset(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('idd', 1.0))
            elif dataset_name == 'berkeleydeepdrive':
                from DatasetClasses import BerkeleyDeepDrive
                dataset_dir = config.get('dataset_dir', 'datasets/BDD')
                datasets.append(BerkeleyDeepDrive(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('bdd', 1.0))
            elif dataset_name == 'kitti':
                from DatasetClasses import KITTI
                dataset_dir = config.get('dataset_dir', 'datasets/KITTI')
                datasets.append(KITTI(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('kitti', 1.0))
            elif dataset_name == 'a2d2':
                from DatasetClasses import A2D2
                dataset_dir = config.get('dataset_dir', 'datasets/a2d2')
                datasets.append(A2D2(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('a2d2', 1.0))
            elif dataset_name == 'nuimages':
                from DatasetClasses import NuImages
                dataset_dir = config.get('dataset_dir', 'datasets/nuimages')
                datasets.append(NuImages(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('nuimages', 1.0))
            elif dataset_name == 'wilddash':
                from DatasetClasses import WildDash
                dataset_dir = config.get('dataset_dir', 'datasets/wilddash')
                datasets.append(WildDash(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('wilddash', 1.0))
            elif dataset_name == 'mapillaryvistas':
                from DatasetClasses import MapillaryVistas
                dataset_dir = config.get('dataset_dir', 'datasets/mapillary-vistas')
                datasets.append(MapillaryVistas(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('mapillaryvistas', 1.0))
            elif dataset_name == 'camvid':
                from DatasetClasses import CamVid
                dataset_dir = config.get('dataset_dir', 'datasets/CamVid')
                datasets.append(CamVid(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('camvid', 1.0))
            elif dataset_name == 'waymo':
                from DatasetClasses import Waymo
                dataset_dir = config.get('dataset_dir', 'datasets/waymo')
                datasets.append(Waymo(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('waymo', 1.0))
            elif dataset_name == 'wildscenes':
                from DatasetClasses import WildScenes
                dataset_dir = config.get('dataset_dir', 'datasets/WildScenes')
                datasets.append(WildScenes(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('wildcenes', 1.0))
            elif dataset_name == 'greatoutdoors':
                from DatasetClasses import GreatOutdoors
                dataset_dir = config.get('dataset_dir', 'datasets/god')
                datasets.append(GreatOutdoors(dataset_dir=dataset_dir, action=action, config=config, is_main_process=is_main_process, **kwargs))
                dataset_weights.append(dataset_weights_dict.get('greatoutdoors', 1.0))
            elif os.path.isdir(dataset_name):
                from DatasetClasses import TestDataset
                datasets.append(TestDataset(dataset_dir=dataset_name, config=config))
            else:
                raise ValueError(f'Unknown dataset: {dataset_name}')

        cat_dataset = ConcatDataset(datasets)

        if not action == 'train':
            return cat_dataset, SequentialSampler(cat_dataset)
        else:
            weights = []
            for w, dataset in zip(dataset_weights, datasets):
                weights += [w] * len(dataset)
            num_samples = sum(int(len(dataset) * w) for dataset, w in zip(datasets, dataset_weights))
            for dataset in datasets:
                print(f'{dataset.__class__.__name__} {len(dataset)}')
            # print(f'Using {num_samples} samples for training')
            # print(f'len dataset {len(cat_dataset)}')
            sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=False)

            return cat_dataset, sampler
