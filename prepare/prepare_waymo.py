import os
import warnings
import numpy as np
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
from waymo_open_dataset.utils import camera_segmentation_utils
import cv2

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

# List of dataset directories to process
dataset_dirs = [
    #'/mnt/ssd1/datasets/waymo/val',
    #'/mnt/ssd1/datasets/waymo/train',
    '/mnt/ssd1/datasets/waymo/test',
]

def read(path) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    return dd.read_parquet(path)

for dataset_dir in dataset_dirs:
    # Lazily read camera images and segmentation labels
    paths_camera_image = tf.io.gfile.glob(f'{dataset_dir}/camera_image/*.parquet')
    paths_segmentation = tf.io.gfile.glob(f'{dataset_dir}/camera_segmentation/*.parquet')

    for path_camera_image, path_segmentation in zip(paths_camera_image, paths_segmentation):

        cam_image_df = read(path_camera_image)
        cam_seg_df = read(path_segmentation)

        # Combine DataFrame for individual components into a single DataFrame.
        image_w_seg_df = v2.merge(cam_image_df, cam_seg_df, right_group=True)

        for i, (_, r) in enumerate(image_w_seg_df.iterrows()):
            # Create component dataclasses for the raw data
            cam_image = v2.CameraImageComponent.from_dict(r)
            cam_seg = v2.CameraSegmentationLabelComponent.from_dict(r)

            # Decode the image (assuming JPEG format, adjust as necessary)
            image_decoded = tf.io.decode_jpeg(cam_image.image).numpy()
            image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)

            # Extract the original name components
            context_name = cam_image.key.segment_context_name
            frame_timestamp = cam_image.key.frame_timestamp_micros
            camera_name = cam_image.key.camera_name

            # Create directories for output images if they don't exist
            output_dir_image = os.path.join(dataset_dir, "images")
            output_dir_seg = os.path.join(dataset_dir, "segmentation")
            os.makedirs(output_dir_image, exist_ok=True)
            os.makedirs(output_dir_seg, exist_ok=True)

            # Create filenames based on the original names
            image_filename = os.path.join(output_dir_image, f'{context_name}_{camera_name}_{frame_timestamp}.png')
            seg_filename = os.path.join(output_dir_seg, f'{context_name}_{camera_name}_{frame_timestamp}_seg.png')

            # Save the RGB image using OpenCV
            cv2.imwrite(image_filename, image_decoded)

            # Decode the panoptic segmentation label using Waymo utility
            panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(cam_seg)

            # Ensure the panoptic label is 2D
            if panoptic_label.ndim > 2:
                panoptic_label = np.squeeze(panoptic_label)

            # Save segmentation label as a 16-bit PNG image using OpenCV
            cv2.imwrite(seg_filename, panoptic_label.astype(np.uint16))  # Ensure 16-bit depth

            print(f'Processed image {i}: {image_filename} and {seg_filename}')

    print(f"Finished processing {dataset_dir}")
