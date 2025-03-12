import numpy as np
import cv2
import csv
from torch import nn
import torch.nn.functional as F
import torch
from collections import defaultdict
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import ListedColormap
import cProfile
import pstats
import io
from functools import wraps
import os
from transformers import Mask2FormerImageProcessor
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from collections import Counter
from copy import deepcopy
from datetime import datetime
from PIL import Image


def compute_iou(predicted_masks, ground_truth_masks):
    predicted_masks = predicted_masks.squeeze(1)
    predicted_masks[predicted_masks > 0] = 1
    predicted_masks[predicted_masks < 0] = 0

    intersection = (predicted_masks.squeeze(1) * ground_truth_masks).sum((1, 2))
    union = (predicted_masks + ground_truth_masks).sum((1, 2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def overlay_masks_on_image(input_image, predicted_mask, ground_truth_mask, points):
    """
    Overlays the predicted and ground truth masks on the input image.
    Args:
        input_image (ndarray): The input RGB image.
        predicted_mask (ndarray): The predicted binary mask.
        ground_truth_mask (ndarray): The ground truth binary mask.
    """

    # Create an RGB image for overlay
    overlay = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)

    # True Positives (Green): Both predicted and ground truth are positive
    true_positives = (predicted_mask == 1) & (ground_truth_mask == 1)
    overlay[true_positives] = [0, 255, 0]  # Green

    # False Positives (Yellow): Predicted is positive, but ground truth is not
    false_positives = (predicted_mask == 1) & (ground_truth_mask == 0)
    overlay[false_positives] = [0, 255, 255]  # Yellow

    # False Negatives (Red): Ground truth is positive, but predicted is not
    false_negatives = (predicted_mask == 0) & (ground_truth_mask == 1)
    overlay[false_negatives] = [0, 0, 255]  # Red

    # Convert the input image to grayscale and then back to BGR
    gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_input_image = cv2.cvtColor(gray_input_image, cv2.COLOR_GRAY2BGR)

    # Slightly desaturate the input image
    alpha = 0.5
    desaturated_image = cv2.addWeighted(input_image, alpha, gray_input_image, 1 - alpha, 0)

    # Overlay the colored masks onto the desaturated image
    overlay = cv2.addWeighted(desaturated_image, 1.0, overlay, 0.4, 0)

    # Draw points
    for point in points:
        cv2.circle(overlay, tuple(point), 3, (255, 0, 0), -1)

    return overlay


def scale_points(points, scale_factor_x, scale_factor_y):
    """
    Scale points by a given factor.
    Args:
        points (Tensor): The points to scale.
        scale_factor_x (float): The scale factor for the x-axis.
        scale_factor_y (float): The scale factor for the y-axis.
    """
    points[:, 0] *= scale_factor_x
    points[:, 1] *= scale_factor_y
    return points.astype(int)


class LabelMapper:
    def __init__(self, lists_dir, seg_task, master_labels="master_labels", verbose=False, dataset_name=""):
        assert os.path.isdir(lists_dir), f"File not found: {lists_dir}"
        labels_path = os.path.join(lists_dir, 'labels.csv')
        master_labels_path = os.path.join(lists_dir, f'{master_labels}.csv')

        self.name = dataset_name
        self.verbose = verbose
        self.seg_task = seg_task
        (self.label2master_label,
         self.master_id2master_label,
         self.master_label2master_id) = self._parse_master_labels_csv(master_labels_path)
        self.labels_dataset = self._parse_labels_csv(labels_path)  # labels.csv
        # self.master_labels_dataset = self.get_master_labels_dataset()
        # self.train_id2label = self.get_train_id2label() # trainId to category name    # trainId is deprecated
        self.id2master_label = self._get_id2master_label()  # id to category name
        # map each sub-label to one master label
        self.id2master_id = self._get_id2master_id()
        # self.id2master_id_merged = {k: min(v) for k, v in
        #                             self.id2master_id.items()}
        # map each master id to all ids
        self.master_id2id = self._get_master_id2id()

    def _parse_master_labels_csv(self, file_path):
        mapping_dict = {}
        id2category_label = {}
        category_label2id = {}

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Skip the header row

            for row in reader:
                desired_id = int(row[0])
                desired_name = row[1]  # Assuming the desired name is in the second column
                id2category_label[desired_id] = desired_name
                category_label2id[desired_name] = desired_id
                # Include the desired name mapping to itself
                # mapping_dict[desired_name] = desired_name
                # Iterate through all possible wrong names in the row

                for wrong_name in row[2:]:  # Skipping trainId and the correct name
                    if wrong_name:  # Ignore empty cells
                        if wrong_name in mapping_dict:
                            mapping_dict[wrong_name] += [desired_name]
                        else:
                            mapping_dict[wrong_name] = [desired_name]

        return mapping_dict, id2category_label, category_label2id

    def _parse_labels_csv(self, file_path):
        # Columns to extract
        columns = ['name', 'id', 'trainId', 'hasInstances']

        # Processed data as a dictionary of dictionaries
        data_dict = {}

        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Check if all required columns are present
            for col in columns:
                if col not in reader.fieldnames:
                    raise ValueError(f"Column '{col}' not found in the file")

            # Extract the required data and organize it into a dictionary
            for row in reader:
                # Convert 'id' and 'trainId' to integers, and 'hasInstance' to boolean
                id = int(row['id'])
                train_id = int(row['trainId'])
                has_instance = row['hasInstances'].lower() == 'true'

                # Create the dictionary for this row
                data_dict[id] = {
                    'name': row['name'],
                    'trainId': train_id,
                    'hasInstances': has_instance,
                    'id': id
                }

        # Get the trainIds
        train_ids = set([values['trainId'] for values in data_dict.values()])

        # Remove background class
        rm_ids = [255, -1]
        train_ids = [id_ for id_ in train_ids if id_ not in rm_ids]

        # Check that no instance classes were merged with another #TODO: make possible to merge instance classes
        if self.seg_task in ['instance', 'panoptic']:
            instance_ids = [id_ for id_, value in data_dict.items() if
                            value['hasInstances'] and value['trainId'] not in rm_ids]
            instance_train_ids = [data_dict[id_]['trainId'] for id_ in instance_ids]
            if len(set(instance_train_ids)) != len(instance_train_ids):
                raise NotImplementedError("Instance classes cannot be merged.")

        num_train_classes = len(train_ids)
        data_dict.update({'num_train_classes': num_train_classes})

        # Check for missing trainIds
        max_train_id = max(train_ids)
        missing_train_ids = [i for i in range(max_train_id + 1) if i not in train_ids]
        if len(missing_train_ids) and self.verbose:
            print(f"Warning: Missing trainIds: {missing_train_ids} in dataset {self.name}")

        return data_dict

    def _get_train_id2label(self):
        id2label = {v['trainId']: v['name'] for k, v in self.labels_dataset.items() if
                    k != "num_train_classes" and not v['trainId'] in {-1, 255}}

        # # re-map labels to category labels
        # id2label = {k: self.label2master_label[v] for k, v in id2label.items()}

        return id2label

    def _get_id2master_label(self):
        id2label = {v['id']: v['name'] for k, v in self.labels_dataset.items() if k != "num_train_classes"}

        # re-map labels to master labels
        # id2label = {k: self.label2master_label[v] for k, v in id2label.items()}

        for k, v in id2label.items():
            if v in self.label2master_label:
                id2label[k] = self.label2master_label[v]
            else:
                id2label[k] = ['void']
                if self.verbose:
                    print(f"Warning: label not found in master labels for dataset {self.name}: {v}. Assigning to void class.")

        return id2label

    def _get_id2master_id(self):
        return {k: [self.master_label2master_id[i] for i in v] for k, v in self.id2master_label.items()}

    def _get_master_id2id(self):
        master_id2id = {
            val: [k for k, v in self.id2master_id.items() if val in v]
            for sublist in self.id2master_id.values() for val in sublist
        }
        master_id2id.update({key: [] for key in self.master_id2master_label.keys() if key not in master_id2id})

        return master_id2id

    def _get_class_label_weights(self):
        class_label_weights = {m_id: 1 / len(ids) if len(ids) > 0 else 1 for m_id, ids in self.master_id2id.items()}
        return class_label_weights

    # deprecated
    def map_seg_classes(self, mask):
        # Create a new array for the modified mask
        modified_mask = np.zeros_like(mask)

        # Iterate over each pixel and replace with the corresponding trainId
        # Flatten instances to get only semantic labels
        for id, values in self.labels_dataset.items():
            if id == 'num_train_classes':
                continue
            if self.seg_task == 'panoptic' and values['hasInstances']:
                semantic_id_mask = mask // 1000 == id
                instance_id_mask = mask % 1000
                modified_mask[semantic_id_mask] = values['trainId'] * 1000 + instance_id_mask[semantic_id_mask]
                # trainId is deprecated
            else:
                modified_mask[mask == id] = values['trainId']

        return modified_mask

    # deprecated
    def map_seg_classes_soft(self, mask):
        # Create a new array for the modified mask
        # modified_mask = np.zeros((*mask.shape, len(self.id2master_id)), dtype=np.uint8)
        modified_mask = np.zeros_like(mask)
        # Iterate over each pixel and replace with the corresponding trainId
        # Flatten instances to get only semantic labels
        for id, values in self.labels_dataset.items():
            if id == 'num_train_classes':
                continue
            modified_mask[mask == id] = values['trainId']

        return modified_mask

    def map_classes_soft(self, mask, pixel_mask=None):
        # Create a new array for the modified mask
        modified_mask = np.zeros((len(self.master_id2id), *mask.shape), dtype=np.float32)
        class_labels = []

        for id, values in self.id2master_id.items():
            for val in values:
                idx = val if val != 255 else -1
                bool_mask = mask == id
                if np.any(bool_mask):
                    modified_mask[idx][mask == id] = 1
                    class_labels.extend(self.id2master_id[id])

        class_count = Counter(class_labels)
        class_ambiguity = [1/class_count[cls] if cls in class_count else 1. for cls in range(len(self.master_id2id))]
        # remove duplicates
        class_labels = sorted(list(set(class_labels)))  # remove 255 here for the time being
        if 255 in class_labels:
            class_labels.remove(255)

        # border pixels void
        if pixel_mask is not None:
            zero_mask = (pixel_mask == 0)
            modified_mask[:-1, zero_mask] = 0
            modified_mask[-1, zero_mask] = 1

        return modified_mask, class_labels, class_ambiguity

    def map_classes_hard(self, mask, pixel_mask=None):
        new_segmentation_map = np.zeros_like(mask)
        for new_label, old_labels in self.master_id2id.items():
            for old_label in old_labels:
                # Remap the old labels to the new label
                new_segmentation_map[mask == old_label] = new_label

        class_labels = sorted(list(set(new_segmentation_map.flatten())))
        return new_segmentation_map, class_labels

    def merge_labels2master_labels(self, mask):
        # converts original label mask to master label mask
        # automatically chooses lowest id when multiple master labels correspond to one label
        merged_mask = np.zeros_like(mask)
        for id_, master_ids in self.id2master_id.items():
            merged_mask[mask == id_] = min(master_ids)
        return merged_mask

    def merge_master_labels2labels(self, mask):
        # converts master labels (predictions) to original labels
        # automatically chooses lowest id when multiple master labels correspond to one label
        merged_mask = np.zeros_like(mask)
        for master_id, ids in self.master_id2id.items():
            merged_mask[mask == master_id] = min(ids)
        return merged_mask

    def merge_predictions(self, batch):
        """
        Sums channels of a tensor according to a mapping dictionary and reindexes them.
        Used to merge ambiguous classes, e.g. road and lane marking in a dataset that only segments road.

        Args:
        batch (list): List of torch tensors.
        index_map (dict): A dictionary mapping target indices to lists of original channel indices to be summed.

        Returns:
        torch.Tensor: Tensor after summing and reindexing channels.
        """
        # Create a list of summed channels using a list comprehension
        # Ensure the list has the correct size by sorting keys and processing in order
        # TODO: add a layer to list comprehension that does it for every tensor in the list!
        master_ids = sorted(self.master_id2id.keys())
        # new_channels = [batch[indices].sum(dim=0) for idx in master_ids for indices in [self.master_id2id[idx]]]
        reduced_batch = []
        for x in batch:
            new_channels = []
            for idx in master_ids:
                for indices in [self.id2master_id[idx]]:
                    new_channel = x[indices].sum(dim=0)
                    new_channels.append(new_channel)
            reduced_batch.append(torch.stack(new_channels, dim=0))

        # Stack all the new channels along the channel dimension to form the final tensor
        result = torch.stack(new_channels, dim=0)
        return result

    def make_labels_3D(self, mask):
        # Create a new array for the modified mask
        modified_mask = np.zeros((len(self.master_id2id), *mask.shape), dtype=float)

        for id, values in self.id2master_id.items():
            modified_mask[id][mask == id] = 1

        return modified_mask


def flatten_void_classes(mask, data_dict):
    # get void ids
    mask_copy = np.copy(mask)
    void_ids = [id for id, values in data_dict.items() if values['trainId'] == 0]

    # Iterate over each pixel and replace with 0
    for id in void_ids:
        mask_copy[mask_copy == id] = 0

    return mask_copy


def get_connected(binary_mask, associated_labels_mask_binary):
    """
    Returns the union of the binary mask and the connected components associated labels mask.
    Associated labels are ones to be merged with the class of the binary mask.
    :param binary_mask: Binary mask of one component of a selected class
    :param associated_labels_mask_binary: binary mask of all labels to be merged with the class of the binary mask
    """

    # Finding connected components in the original binary mask
    num_components_binary_mask, connected_components_binary_mask = cv2.connectedComponents(binary_mask)

    # Creating a union mask of the original binary mask and associated labels mask
    union = binary_mask.copy()
    union[associated_labels_mask_binary == 1] = 1

    # Finding connected components in the union mask
    num_components_union, connected_components_union = cv2.connectedComponents(union)

    # Initialize the result mask
    result_mask = np.zeros_like(binary_mask)

    # Iterate over each component in the original binary mask
    for i in range(1, num_components_binary_mask):
        # Find a pixel in the i-th component
        y, x = np.nonzero(connected_components_binary_mask == i)

        if len(y) > 0 and len(x) > 0:
            # Find the label of the connected component in the union mask corresponding to this pixel
            component_label = connected_components_union[y[0], x[0]]

            # Update the result mask with this component from the union mask
            result_mask[connected_components_union == component_label] = 1

    return result_mask


def remove_small_areas(mask, min_size):
    # Find all connected components (aka blobs or regions) in the mask
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(mask_uint8)

    for i in range(1, num_labels):  # Skip the background label 0
        if np.sum(labels_im == i) < min_size:
            mask[labels_im == i] = 0

    return mask


# loss functions:
class FocalLoss(nn.Module):
    """ Computes the Focal loss. """

    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs = inputs.flatten(0,2)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    """ Computes the Dice loss. """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        # inputs = inputs.flatten(0,2)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """ Computes a weighted sum of Focal and Dice loss. """

    def __init__(self, w_focal=20, w_dice=1, focal_params=None):
        super().__init__()
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.focal_loss = FocalLoss(**(focal_params or {}))
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        return self.w_focal * loss_focal + self.w_dice * loss_dice


def trim_whitespace(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find non-zero points (where there are whitespaces)
    points = np.argwhere(binary != 0)

    # Take the smallest and largest x and y indices
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    # Use the rectangle defined by these points to crop the image
    cropped_image = image[x_min:x_max + 1, y_min:y_max + 1]

    return cropped_image


class ColorMap:
    COLORMAP_CITYSCAPES = [
        (128, 64, 128),   # 0: road
        (244, 35, 232),   # 1: sidewalk
        (70, 70, 70),     # 2: building
        (102, 102, 156),  # 3: wall
        (190, 153, 153),  # 4: fence
        (153, 153, 153),  # 5: pole
        (250, 170, 30),   # 6: traffic light
        (220, 220, 0),    # 7: traffic sign
        (107, 142, 35),   # 8: vegetation
        (152, 251, 152),  # 9: terrain
        (70, 130, 180),   # 10: sky
        (220, 20, 60),    # 11: person
        (255, 0, 0),      # 12: rider
        (0, 0, 142),      # 13: car
        (0, 0, 70),       # 14: truck
        (0, 60, 100),     # 15: bus
        (0, 80, 100),     # 16: train
        (0, 0, 230),      # 17: motorcycle
        (119, 11, 32),    # 18: bicycle
        (0, 0, 0),        # 19: void
    ]

    COLORMAP_CITYSCAPES_LABELS = [
        (0, 0, 0),  # unlabeled
        (0, 0, 0),  # ego vehicle
        (0, 0, 0),  # rectification border
        (0, 0, 0),  # out of roi
        (0, 0, 0),  # static
        (111, 74, 0),  # dynamic
        (81, 0, 81),  # ground
        (128, 64, 128),  # road
        (244, 35, 232),  # sidewalk
        (250, 170, 160),  # parking
        (230, 150, 140),  # rail track
        (70, 70, 70),  # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (180, 165, 180),  # guard rail
        (150, 100, 100),  # bridge
        (150, 120, 90),  # tunnel
        (153, 153, 153),  # pole
        (153, 153, 153),  # polegroup
        (250, 170, 30),  # traffic light
        (220, 220, 0),  # traffic sign
        (107, 142, 35),  # vegetation
        (152, 251, 152),  # terrain
        (70, 130, 180),  # sky
        (220, 20, 60),  # person
        (255, 0, 0),  # rider
        (0, 0, 142),  # car
        (0, 0, 70),  # truck
        (0, 60, 100),  # bus
        (0, 0, 90),  # caravan
        (0, 0, 110),  # trailer
        (0, 80, 100),  # train
        (0, 0, 230),  # motorcycle
        (119, 11, 32),  # bicycle
        (0, 0, 142),  # license plate
    ]

    COLORMAP_APOLLOSCAPE_LABELS = [(147, 224, 248), # other
                                   (0,0,0),         # ego vehicle
                                   *15 * [(0, 0, 0)],
                                   (70, 130, 180),   # sky
                                   *15 * [(0, 0, 0)],
                                   (0, 0, 142), # car
                                   (0, 0, 230), # motorbike
                                   (119, 11, 32),   # bicycle
                                   (220, 20, 60),   # person
                                   (255, 0, 0), # rider
                                   (0, 0, 70),   # truck
                                   (0, 60, 100),   # bus
                                   (154, 14, 40),   # tricycle
                                   *8 * [(0, 0, 0)],
                                   (128, 64, 128),  # road
                                   (244, 35, 232),  # sidewalk
                                   *14 * [(0, 0, 0)],
                                   (247, 176, 170), # traffic cone
                                   (193, 233, 164), # road pile
                                   (190, 153, 153),  # fence
                              *13 * [(0, 0, 0)],
                                   (250, 170, 30),  # traffic light
                                   (153, 153, 153), # pole
                                   (220, 220, 0),  # traffic sign
                                   (102, 102, 156),  # wall
                              (51, 225, 51),            # dustbin
                                   (233, 7, 73),        # billboard
                                   *10 * [(0, 0, 0)],
                                   (70, 70, 70),        # building
                                   (150, 100, 100),     # bridge
                              (150, 120, 90),           # tunnel
                                   (194, 162, 249),     # overpass
                                   *12 * [(0, 0, 0)],
                                   (107, 142, 35),      # vegetation
                                   *47 * [(0, 0, 0)],
                                    (0, 0, 142),        # car group
                                   (0, 0, 230),       # motorbike group
                                   (119, 11, 32),      # bicycle group
                                   (220, 20, 60),        # person group
                                   (255, 0, 0),      # rider group
                              (191, 220, 74),           # truck group
                                   (0, 60, 100),      # bus group
                                   (107, 142, 35),      # tricycle group
                                   *86 * [(0, 0, 0)],
                                   (0,0,0)]             # unlabeled

    COLORMAP_MAPILLARY = [
        [165, 42, 42],  # bird
        [0, 192, 0],    # ground animal
        [196, 196, 196],# curb
        [190, 153, 153],# fence
        [180, 165, 180],
        [102, 102, 156],
        [102, 102, 156],   # wall
        [128, 64, 255],   # bike lane
        [140, 140, 200],
        [170, 170, 170],
        [250, 170, 160],
        [96, 96, 96],
        [230, 150, 140],
        [128, 64, 128], # road
        [110, 110, 110],
        [244, 35, 232], # sidewalk
        [150, 100, 100],# bridge
        [70, 70, 70],   # building
        [150, 120, 90], # tunnel
        [220, 20, 60],  # person
        [255, 0, 0],    # bicyclist
        [255, 0, 0],    # motorcyclist
        [255, 0, 0],    # rider other
        [200, 128, 128],
        [255, 255, 128], # marking general
        [64, 170, 64],
        [128, 64, 64],
        [70, 130, 180], # sky
        [255, 255, 255],
        [152, 251, 152],
        [107, 142, 35], # vegetation
        [0, 170, 30],
        [255, 255, 128],
        [250, 0, 30],
        [0, 0, 0],
        [220, 220, 220],
        [170, 170, 170],
        [222, 40, 40],
        [100, 170, 30],
        [40, 40, 40],
        [33, 33, 33],
        [170, 170, 170],
        [0, 0, 142],
        [170, 170, 170],
        [210, 170, 100],    # street light
        [153, 153, 153],# pole
        [128, 128, 128],# pole sign frame
        [0, 0, 142],    # utility pole
        [250, 170, 30], # traffic light
        [192, 192, 192],# traffic sign back
        [220, 220, 0],  # traffic sign front
        [180, 165, 180],
        [119, 11, 32],  # bicycle
        [0, 0, 142],
        [0, 60, 100],   # bus
        [0, 0, 142],    # car
        [0, 0, 90],
        [0, 0, 230],    # motorcycle
        [0, 80, 100],   # on rails
        [128, 64, 64],  # other vehicle
        [0, 0, 110],
        [0, 0, 70], # truck
        [0, 0, 192],
        [0, 0, 0],
    ]

    COLORMAP_OFFROAD = [
        (128, 64, 128),  # asphalt
        (219, 142, 70),  # rough drivable
        (155, 242, 239), # cobble
        (244, 35, 232),  # sidewalk
        (70, 70, 70),  # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (153, 153, 153),  # pole
        (250, 170, 30),  # traffic light
        (220, 220, 0),  # traffic sign
        (130, 200, 80), # tree
        (207, 190, 0),  # non-drivable vegetation
        (0,134,237), # crops
        (68, 60, 145), # water
        (0, 128, 4), # high_grass
        (136, 217, 65), # low_grass
        (194,255,237), # guard_rail
        (70, 130, 180),  # sky
        (220, 20, 60),  # person
        (255, 0, 0),  # rider
        (0, 0, 142),  # car
        (0, 0, 70),  # truck
        (0, 60, 100),  # bus
        (10,166,216), # trailer
        (48,0,24), # caravan
        (0, 80, 100),  # on rails
        (0, 0, 230),  # motorcycle
        (119, 11, 32),  # bicycle
        (246, 180, 100),  # road marking
        (64, 76, 245), # bridge
        (227, 117, 0), # tunnel
        (90, 107, 61), # rock
        (38, 255, 241), # snow
        (150, 93, 0), # animal
        (0, 0, 0),  # void
    ]

    COLORMAP_YAMAHA = [
        (128, 64, 128),  # asphalt
        (207, 190, 0),  # high_vegetation
        (136, 217, 65), # traversable_grass
        (219, 142, 70),  # smooth_trail
        (150, 93, 0), # obstacle
        (70, 130, 180),  # sky
        (150, 96, 45),  # smooth_trail
        (68, 60, 145), # water
        (113, 128, 0), # non_traversable_low_vegetation
    ]

    COLORMAP_RELLIS = [
        (0,0,0), # void
        (219, 142, 70),  # dirt
        (0, 0, 0),  # placeholder
        (136, 217, 65), # grass
        (130, 200, 80), # tree
        (153, 153, 153),  # pole
        (68, 60, 145),  # water
        (70, 130, 180),  # sky
        (0, 0, 70),  # vehicle
        (194, 255, 237),  # object
        (128, 64, 128),  # asphalt
        (0, 0, 0),  # placeholder
        (70, 70, 70),  # building
        (0, 0, 0),  # placeholder
        (0, 0, 0),  # placeholder
        (150, 93, 0),  # log
        (0, 0, 0),  # placeholder
        (220, 20, 60),  # person
        (190, 153, 153),  # fence
        (207, 190, 0),  # bush
        (0, 0, 0),  # placeholder
        (0, 0, 0),  # placeholder
        (0, 0, 0),  # placeholder
        (100, 242, 239),  # concrete
        (0, 0, 0),  # placeholder
        (0, 0, 0),  # placeholder
        (0, 0, 0),  # placeholder
        (122, 122, 156),  # barrier
        (0, 0, 0),  # placeholder
        (0, 0, 0),  # placeholder
        (0, 0, 0),  # placeholder
        (90, 77, 185),  # puddle
        (0, 0, 0),  # placeholder
        (200, 130, 66),  # mud
        (219, 100, 0),  # tunnel
    ]

    COLORMAP_CAMVID = [
        (0, 192, 0),  # animal
        (160, 160, 160),  # archway
        (255, 0, 0),  # bicyclist
        (150, 100, 100),  # bridge
        (70, 70, 70),  # building
        (0, 0, 142),  # car
        (81, 0, 181),  # cartluggagepram
        (237, 88, 180),  # child
        (153, 153, 153),  # Column_Pole
        (190, 153, 153),  # fence
        (246, 180, 100),  # lane markings drive
        (255, 84, 61),  # lane markings non drive
        (255, 244, 179),  # misc text
        (0, 0, 230),  # motorcycle scooter
        (179, 198, 255),  # OtherMoving
        (250, 170, 160),  # parking block
        (220, 20, 60),  # pedestrian
        (128, 64, 128),  # road
        (148, 180, 255),  # road shoulder
        (244, 35, 232),  # sidewalk
        (220, 220, 0),  # sign symbol
        (70, 130, 180),  # sky
        (17, 166, 131),  # suv
        (247, 176, 170),  # trafficcone
        (250, 170, 30),  # traffic light
        (0, 80, 100),  # train
        (130, 200, 80),  # tree
        (92, 11, 38),  # truckbus
        (150, 120, 90),  # tunnel
        (107, 142, 35),  # veg misc
        (0, 0, 0),  # void
        (102, 102, 156),  # wall
    ]


    def __init__(self):
        pass

    def get_cmap(self, name):
        if name == 'mapillary':
            return self.COLORMAP_MAPILLARY
        elif name == 'offroad':
            return self.COLORMAP_OFFROAD
        elif name == 'cityscapes':
            return self.COLORMAP_CITYSCAPES
        elif name == 'cityscapes_labels':
            return self.COLORMAP_CITYSCAPES_LABELS
        elif name == 'apolloscape_labels':
            return self.COLORMAP_APOLLOSCAPE_LABELS
        elif name == 'rellis_labels':
            return self.COLORMAP_RELLIS
        elif name == 'yamaha_labels':
            return self.COLORMAP_YAMAHA
        elif name == 'camvid_labels':
            return self.COLORMAP_CAMVID
        else:
            # Fallback to a standard colormap lookup using matplotlib
            try:
                return plt.get_cmap(name)
            except ValueError:
                # Default to 'viridis' if the provided colormap doesn't exist
                print(f"'{name}' colormap not found. Returning default colormap 'viridis'.")
                return plt.get_cmap('viridis')


class Plotter:
    def __init__(self, idx2label, colormap=None):
        num_classes = len(idx2label)
        if colormap is None:
            # generate colormap
            self.cmap = cm.get_cmap('viridis', num_classes)
        else:
            self.cmap = self._apply_custom_colormap(colormap)
        self.idx2label = idx2label
        self.fontsize = 'medium' if num_classes <= 20 else 'small' if num_classes <= 24 else 'x-small'
        self.ncol = 1 if num_classes <= 27 else 2

    @staticmethod
    def _apply_custom_colormap(color_list_255):
        color_list = [(r / 255.0, g / 255.0, b / 255.0, 0.7) for r, g, b in color_list_255]
        if color_list:
            color_list[-1] = (*color_list[-1][:3], 0.0)
        class_cmap = ListedColormap(color_list)
        return class_cmap

    def draw_semantic_segmentation(self, segmentation, rgb_image=None, alpha=0.3, figsize=(8, 6)):

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        if rgb_image is not None:
            ax.imshow(rgb_image)
            overlay = self.cmap(segmentation)
            ###################
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_filename = f"/home/anba/Desktop/ICCV/quality_pics/test/image_{timestamp}.png"
            # make untransparent
            overlay[:, :, 3] = 1
            pil_image = Image.fromarray((overlay * 255).astype(np.uint8))
            pil_image.save(unique_filename)
            ###################
            ax.imshow(overlay, alpha=alpha)
        else:
            ax.imshow(segmentation, cmap=self.cmap)

        handles = []
        class_labels = sorted(list(np.unique(segmentation)))
        for i, id in enumerate(class_labels):
            color = self.cmap(id)
            handles.append(mpatches.Patch(color=color, label=self.idx2label[id]))

        # Place legend outside the image
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=self.fontsize, ncol=self.ncol)
        plt.tight_layout()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot = trim_whitespace(image_from_plot)
        plt.close(fig)

        return image_from_plot

    # currently deprecated
    def draw_panoptic_segmentation(self, segmentation, segments, id2label, rgb_image=None, alpha=0.5):
        # get the used color map
        viridis = cm.get_cmap('viridis', np.max(segmentation) + 1)

        # Calculate the number of segments to determine the legend size
        num_segments = len(segments)

        # Adjust font size based on number of segments
        if num_segments <= 20:
            fontsize = 'medium'
            ncol = 1
        elif num_segments <= 24:
            fontsize = 'small'
            ncol = 1
        elif num_segments <= 27:
            fontsize = 'x-small'
            ncol = 1
        else:
            fontsize = 'x-small'
            ncol = 2

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Hide the axes
        ax.axis('off')

        if rgb_image is not None:
            # Display the RGB image
            ax.imshow(rgb_image)

            # Overlay the segmentation map on the RGB image with the given alpha transparency
            overlay = viridis(segmentation)
            ax.imshow(overlay, alpha=alpha)
        else:
            # Display the image
            # TODO: draw contours around instances
            ax.imshow(segmentation)

        instances_counter = defaultdict(int)
        handles = []

        # for each segment, draw its legend
        for segment in segments:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        # Place legend outside the image with dynamic font size
        legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize, ncol=ncol)

        # Use tight_layout to minimize white space
        plt.tight_layout()

        # Render the figure as a numpy array
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        image_from_plot = trim_whitespace(image_from_plot)

        # Close the figure to free memory
        plt.close(fig)

        return image_from_plot


def get_label_mask(mask, class_values, label_colors_list):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    counter = 0
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                counter += 1
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = counter
    label_mask = label_mask.astype(int)
    return label_mask


ENABLE_PROFILING = False


def profile(enabled=False, save_freq=1):
    """A decorator factory that profiles a function if enabled is True."""
    # Check and create directory once at decorator definition time
    if enabled:
        save_dir = os.path.join("experiments", "speed_tests")
        os.makedirs(save_dir, exist_ok=True)

    def decorator(func):
        if not enabled:
            return func

        call_count = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            with cProfile.Profile() as pr:
                retval = func(*args, **kwargs)

            if call_count % save_freq == 0:
                s = io.StringIO()

                ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
                ps.print_stats()

                func_name = func.__name__
                save_path = os.path.join(save_dir, f"{func_name}.prof")
                ps.dump_stats(save_path)

            return retval

        return wrapper

    return decorator


def plot_confusion_matrix(confusion_matrix, label_map, resolution=640, fontsize=14):
    resolution //= 10  # Adjust resolution for the figure DPI
    cm = confusion_matrix.cpu().numpy() if isinstance(confusion_matrix, torch.Tensor) else confusion_matrix
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=resolution)
    canvas = FigureCanvas(fig)
    cax = ax.matshow(np.log(cm + 1), cmap=plt.cm.Blues)  # Use logarithmic scale to enhance visibility
    fig.colorbar(cax)

    if label_map:
        labels = [label_map[i] for i in range(n_classes)]
    else:
        labels = [str(i) for i in range(n_classes)]

    # Set axis labels with the updated font size
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)

    # Increase the font size for the axis titles as well
    ax.set_xlabel('Prediction', fontsize=fontsize + 4)
    ax.set_ylabel('Ground Truth', fontsize=fontsize + 4)

    plt.tight_layout()

    # Render the plot onto the canvas and then convert it to an RGB image
    canvas.draw()
    width, height = fig.get_size_inches() * resolution
    rgb_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    plt.close(fig)

    return rgb_image


@torch.no_grad()
def get_semi_soft_labels(corrected_predictions, class_labels):
    """
    Generate semi-soft labels for loss calculation
    :param prediction_post:
    :param correction_mask:
    :param class_labels:
    :return:
    """
    corrected_predictions_sum = corrected_predictions.sum(dim=1, keepdim=True)
    # avoid division by zero (only relevant when there exists master_labels that cannot be mapped to any labels, e.g.
    # Rellis-3d: sidewalk, road_marking, ...)
    corrected_predictions_sum[corrected_predictions_sum == 0] = 255 # can be any value != 0
    corrected_predictions_norm = corrected_predictions / corrected_predictions_sum

    # unbind and reduce corrected_predictions
    corrected_predictions_norm = torch.unbind(corrected_predictions_norm, dim=0)
    corrected_predictions_norm = [x[class_labels[i]] for i, x in enumerate(corrected_predictions_norm)]

    return corrected_predictions_norm


@torch.no_grad()
def get_hard_labels(prediction_post, correction_mask):

    semi_soft_labels = prediction_post * correction_mask
    hard_labels = torch.argmax(semi_soft_labels, dim=1)

    return hard_labels


class SoftLabelImageProcessor(Mask2FormerImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_process_semantic_segmentation(
            self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None,
            return_softmax: bool = False,
            remove_null_class: bool = False
    ) -> "torch.Tensor":
        """
        Adapted from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mask2former/image_processing_mask2former.py
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_softmax (`bool`, *optional*, defaults to `False`):
                Whether to return the output as a soft prediction (logits) or as a hard prediction (argmax).
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        if remove_null_class:
            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        else:
            masks_classes = class_queries_logits.softmax(dim=-1)  # [..., :-1] # how does this affect torch.einsum(...

        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)    # Warning: segmentation.max() > 1
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # only necessary if target_sizes are different which they never are
            # semantic_segmentation = []
            # for idx in range(batch_size):
            #     resized_logits = torch.nn.functional.interpolate(
            #         segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            #     )
            #     semantic_map = resized_logits.squeeze().argmax(dim=0) if not return_softmax else resized_logits.squeeze()
            #     semantic_segmentation.append(semantic_map)


            resized_logits = torch.nn.functional.interpolate(
                segmentation, size=target_sizes[0], mode="bilinear", align_corners=False)
            semantic_segmentation = resized_logits.argmax(dim=1) if not return_softmax else resized_logits
        else:
            semantic_segmentation = segmentation.argmax(dim=1) if not return_softmax else segmentation

        return semantic_segmentation

    # def post_process_panoptic_segmentation(
    #     self,
    #     outputs,
    #     threshold: float = 0.5,
    #     mask_threshold: float = 0.5,
    #     overlap_mask_area_threshold: float = 0.8,
    #     label_ids_to_fuse: Optional[Set[int]] = None,
    #     target_sizes: Optional[List[Tuple[int, int]]] = None,
    # ) -> List[Dict]:
    #     """
    #     Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into image panoptic segmentation
    #     predictions. Only supports PyTorch.
    #
    #     Args:
    #         outputs ([`Mask2FormerForUniversalSegmentationOutput`]):
    #             The outputs from [`Mask2FormerForUniversalSegmentation`].
    #         threshold (`float`, *optional*, defaults to 0.5):
    #             The probability score threshold to keep predicted instance masks.
    #         mask_threshold (`float`, *optional*, defaults to 0.5):
    #             Threshold to use when turning the predicted masks into binary values.
    #         overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
    #             The overlap mask area threshold to merge or discard small disconnected parts within each binary
    #             instance mask.
    #         label_ids_to_fuse (`Set[int]`, *optional*):
    #             The labels in this state will have all their instances be fused together. For instance we could say
    #             there can only be one sky in an image, but several persons, so the label ID for sky would be in that
    #             set, but not the one for person.
    #         target_sizes (`List[Tuple]`, *optional*):
    #             List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
    #             final size (height, width) of each prediction in batch. If left to None, predictions will not be
    #             resized.
    #
    #     Returns:
    #         `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
    #         - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
    #           to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
    #           to the corresponding `target_sizes` entry.
    #         - **segments_info** -- A dictionary that contains additional information on each segment.
    #             - **id** -- an integer representing the `segment_id`.
    #             - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
    #             - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    #               Multiple instances of the same class / label were fused and assigned a single `segment_id`.
    #             - **score** -- Prediction score of segment with `segment_id`.
    #     """
    #
    #     if label_ids_to_fuse is None:
    #         logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
    #         label_ids_to_fuse = set()
    #
    #     class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    #     masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
    #
    #     # Scale back to preprocessed image size - (384, 384) for all models
    #     masks_queries_logits = torch.nn.functional.interpolate(
    #         masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
    #     )
    #
    #     batch_size = class_queries_logits.shape[0]
    #     num_labels = class_queries_logits.shape[-1] - 1
    #
    #     mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]
    #
    #     # Predicted label and score of each query (batch_size, num_queries)
    #     pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)
    #
    #     # Loop over items in batch size
    #     results: List[Dict[str, TensorType]] = []
    #
    #     for i in range(batch_size):
    #         mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
    #             mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
    #         )
    #
    #         # No mask found
    #         if mask_probs_item.shape[0] <= 0:
    #             height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
    #             segmentation = torch.zeros((height, width)) - 1
    #             results.append({"segmentation": segmentation, "segments_info": []})
    #             continue
    #
    #         # Get segmentation map and segment information of batch item
    #         target_size = target_sizes[i] if target_sizes is not None else None
    #         segmentation, segments = compute_segments(
    #             mask_probs=mask_probs_item,
    #             pred_scores=pred_scores_item,
    #             pred_labels=pred_labels_item,
    #             mask_threshold=mask_threshold,
    #             overlap_mask_area_threshold=overlap_mask_area_threshold,
    #             label_ids_to_fuse=label_ids_to_fuse,
    #             target_size=target_size,
    #         )
    #
    #         results.append({"segmentation": segmentation, "segments_info": segments})
    #     return results

def get_input_labels(corrected_predictions):
    flattened = corrected_predictions.argmax(dim=1).view(corrected_predictions.shape[0], -1)
    class_labels = [torch.unique(flattened[i]) for i in range(flattened.shape[0])]
    one_hot_indices = torch.argmax(corrected_predictions, dim=1, keepdim=True)
    corrected_predictions_hard = torch.zeros_like(corrected_predictions, device=corrected_predictions.device).scatter_(1, one_hot_indices, 1)
    input_labels = [x[class_labels[i]] for i, x in enumerate(corrected_predictions_hard)]
    return input_labels, class_labels

def post_process_semantic_segmentation(prediction, target_sizes, return_softmax=False):

    logits = prediction.logits

    if target_sizes is not None:
        target_size = target_sizes[0]
    else:
        target_size = logits.shape[-2:]

    # Resize the logits to the target sizes
    logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

    if return_softmax:
        # Compute softmax along the channel dimension (C)
        output = F.softmax(logits, dim=1)
    else:
        # Compute the argmax along the channel dimension (C)
        output = torch.argmax(logits, dim=1)

    return output

class PadCustom:
    def __init__(self, pad_x, pad_y, color_pad_value=0, grayscale_pad_value=0):
        self.color_pad_value = color_pad_value
        self.grayscale_pad_value = grayscale_pad_value
        self.pad_x = pad_x
        self.pad_y = pad_y

    def __call__(self, image):
        if len(image.shape) == 3:
            return np.pad(image, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x), (0, 0)), mode='constant',
                          constant_values=self.color_pad_value)
        else:
            return np.pad(image, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x)), mode='constant',
                          constant_values=self.grayscale_pad_value)

# https://github.com/huggingface/pytorch-image-models/blob/a6fe31b09670289dbc8e99a0cfae23de355534c9/timm/utils/model_ema.py#L16
class ModelEma(nn.Module):
    """ Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2/3,
            device: Optional[torch.device] = None,
            foreach: bool = True,
            exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1. - decay)
        else:
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v.to(device=self.device), weight=1. - decay)
                else:
                    ema_v.copy_(model_v.to(device=self.device))

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p.to(device=self.device), weight=1. - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def format_duration(seconds):
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    return f"{days}d {hours}h {minutes}m {seconds}s"