import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 1. Initial Setup ---

# Import OpenCV for additional image reading options
import cv2

# Paths to the panoptic labels and the JSON file
panoptic_dir = '/mnt/ssd1/datasets/wilddash/wd_public_v2p0/panoptic'
json_file = '/mnt/ssd1/datasets/wilddash/wd_public_v2p0/panoptic.json'

# Output directory for semantic segmentation labels
output_dir = '/mnt/ssd1/datasets/wilddash/wd_public_v2p0/semantic_segmentation'
os.makedirs(output_dir, exist_ok=True)

# --- 2. Load Annotations ---

# Load the panoptic annotations
with open(json_file, 'r') as f:
    panoptic_data = json.load(f)

# Extract categories and create mappings
categories = panoptic_data['categories']

# Map category IDs to names (optional)
category_id_to_name = {category['id']: category['name'] for category in categories}

# Build initial segment ID to category ID mapping
segment_id_to_category_id = {}
for annotation in panoptic_data['annotations']:
    for segment_info in annotation['segments_info']:
        segment_id = segment_info['id']
        category_id = segment_info['category_id']
        segment_id_to_category_id[segment_id] = category_id

# --- 3. Debugging and Diagnosing ---

# 3.1 Verify Correspondence Between Images and Annotations

# Get list of image file names from annotations
annotation_file_names = [ann['file_name'] for ann in panoptic_data['annotations']]

# Get list of image file names in the panoptic directory
panoptic_image_files = os.listdir(panoptic_dir)

# Identify mismatches
missing_in_annotations = set(panoptic_image_files) - set(annotation_file_names)
missing_in_images = set(annotation_file_names) - set(panoptic_image_files)

print("Images missing in annotations:", missing_in_annotations)
print("Annotations missing in images:", missing_in_images)

# 3.2 Inspect Image Data

# Choose a sample image for inspection
sample_image_file = annotation_file_names[0]
image_path = os.path.join(panoptic_dir, sample_image_file)

# Load image using PIL
panoptic_img_pil = Image.open(image_path)
panoptic_np_pil = np.array(panoptic_img_pil)
print(f"Using PIL - Image shape: {panoptic_np_pil.shape}, dtype: {panoptic_np_pil.dtype}")

# Load image using OpenCV
panoptic_np_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
print(f"Using OpenCV - Image shape: {panoptic_np_cv.shape}, dtype: {panoptic_np_cv.dtype}")

# 3.3 Examine Segment ID Encoding

# Function to reconstruct segment IDs using different methods
def reconstruct_segment_ids(panoptic_np):
    segment_ids_methods = {}

    # Method 1: Standard COCO encoding (RGB)
    if panoptic_np.shape[2] >= 3:
        R = panoptic_np[:, :, 0].astype(np.uint32)
        G = panoptic_np[:, :, 1].astype(np.uint32)
        B = panoptic_np[:, :, 2].astype(np.uint32)
        segment_ids = (R << 16) + (G << 8) + B
        segment_ids_methods['COCO_RGB'] = segment_ids

    # Method 2: Alternative encoding (BGR)
    if panoptic_np.shape[2] >= 3:
        B = panoptic_np[:, :, 0].astype(np.uint32)
        G = panoptic_np[:, :, 1].astype(np.uint32)
        R = panoptic_np[:, :, 2].astype(np.uint32)
        segment_ids = (R << 16) + (G << 8) + B
        segment_ids_methods['COCO_BGR'] = segment_ids

    # Method 3: Including Alpha channel
    if panoptic_np.shape[2] == 4:
        A = panoptic_np[:, :, 3].astype(np.uint32)
        segment_ids = (A << 24) + (R << 16) + (G << 8) + B
        segment_ids_methods['COCO_RGBA'] = segment_ids

    # Method 4: Direct grayscale values (if single channel)
    if panoptic_np.ndim == 2 or panoptic_np.shape[2] == 1:
        segment_ids = panoptic_np.astype(np.uint32)
        segment_ids_methods['Grayscale'] = segment_ids

    return segment_ids_methods

# Reconstruct segment IDs using different methods
segment_ids_methods_pil = reconstruct_segment_ids(panoptic_np_pil)
segment_ids_methods_cv = reconstruct_segment_ids(panoptic_np_cv)

# Get segment IDs from annotations for the sample image
annotation = next((ann for ann in panoptic_data['annotations'] if ann['file_name'] == sample_image_file), None)
segment_ids_in_annotation = [segment_info['id'] for segment_info in annotation['segments_info']]

print(f"Segment IDs in annotation: {segment_ids_in_annotation}")

# Compare reconstructed segment IDs with those in annotations
for method_name, segment_ids in segment_ids_methods_pil.items():
    unique_segment_ids_in_image = np.unique(segment_ids)
    ids_in_image_not_in_annotation = set(unique_segment_ids_in_image) - set(segment_ids_in_annotation)
    print(f"Method {method_name} - IDs in image not in annotation: {ids_in_image_not_in_annotation}")

# --- 4. Adjust Segment ID Reconstruction ---

# Based on the above diagnostics, select the method that matches the segment IDs in the annotations
# For example, if 'COCO_BGR' matches, we will use that method

# Function to reconstruct segment IDs for processing (adjust as needed)
def get_segment_ids(panoptic_np):
    # Adjust this function based on which method works
    # Assuming 'COCO_BGR' works best
    if panoptic_np.shape[2] >= 3:
        B = panoptic_np[:, :, 0].astype(np.uint32)
        G = panoptic_np[:, :, 1].astype(np.uint32)
        R = panoptic_np[:, :, 2].astype(np.uint32)
        segment_ids = (R << 16) + (G << 8) + B
    else:
        # Handle grayscale images or images with fewer channels
        segment_ids = panoptic_np.astype(np.uint32)
    return segment_ids

# --- 5. Process Images ---

# Determine the appropriate NumPy data type based on the maximum category ID
max_category_id = max(category_id_to_name.keys())
if max_category_id < 256:
    dtype = np.uint8
elif max_category_id < 65536:
    dtype = np.uint16
else:
    dtype = np.uint32

# Process each panoptic PNG image
for annotation in tqdm(panoptic_data['annotations'], desc='Processing images'):
    file_name = annotation['file_name']  # Corresponds to the PNG file name
    image_path = os.path.join(panoptic_dir, file_name)

    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found.")
        continue

    # Load the panoptic segmentation image using OpenCV
    panoptic_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if panoptic_np is None:
        print(f"Failed to read image {image_path}")
        continue

    # Reconstruct segment IDs using the selected method
    segment_ids = get_segment_ids(panoptic_np)

    # Initialize semantic segmentation label map
    semantic_label = np.zeros(segment_ids.shape, dtype=dtype)

    # Map segment IDs to category IDs
    unique_segment_ids = np.unique(segment_ids)
    for segment_id in unique_segment_ids:
        if segment_id == 0:
            # Background or unlabeled regions
            semantic_label[segment_ids == segment_id] = 0  # Assign background ID
            continue
        category_id = segment_id_to_category_id.get(segment_id, None)
        if category_id is None:
            print(f"Warning: Segment ID {segment_id} not found in annotations for image {file_name}.")
            # Assign a default category ID or skip
            continue
        semantic_label[segment_ids == segment_id] = category_id

    # Save the semantic label map as a NumPy array
    output_file_name = os.path.splitext(file_name)[0] + '_label.npy'
    output_path = os.path.join(output_dir, output_file_name)
    np.save(output_path, semantic_label)

# --- 6. Save Results ---

# Saving is handled within the processing loop

print("Processing completed.")
