import os
import json
import cv2
import numpy as np
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm  # Import tqdm

# Define dataset directories and paths
dataset_dir = "datasets/a2d2"
json_path = "datasets/a2d2/lists/class_list.json"
label_paths = glob(os.path.join(dataset_dir, "camera_lidar_semantic", "*", "label", "*", "*.png"))

# Load the color to class mapping
with open(json_path, 'r') as f:
    color_to_class = json.load(f)

# Group similar classes into one general class and assign unique class ids
merged_class_map = {
    "Car": ["Car 1", "Car 2", "Car 3", "Car 4"],
    "Bicycle": ["Bicycle 1", "Bicycle 2", "Bicycle 3", "Bicycle 4"],
    "Pedestrian": ["Pedestrian 1", "Pedestrian 2", "Pedestrian 3"],
    "Truck": ["Truck 1", "Truck 2", "Truck 3"],
    "Small vehicles": ["Small vehicles 1", "Small vehicles 2", "Small vehicles 3"],
    "Traffic signal": ["Traffic signal 1", "Traffic signal 2", "Traffic signal 3"],
    "Traffic sign": ["Traffic sign 1", "Traffic sign 2", "Traffic sign 3"],
    "Utility vehicle": ["Utility vehicle 1", "Utility vehicle 2"],
    # Add more class groupings here if needed
}

# Create a reverse mapping from specific class to merged class
class_to_merged_class = {}
for merged_class, specific_classes in merged_class_map.items():
    for specific_class in specific_classes:
        class_to_merged_class[specific_class] = merged_class

# Normalize the color keys and assign unique class ids based on merged classes
color_to_class_map = {}
class_id_map = {}
current_id = 1

for color, specific_class_name in color_to_class.items():
    # Find the merged class name
    merged_class_name = class_to_merged_class.get(specific_class_name, specific_class_name)

    if merged_class_name not in class_id_map:
        class_id_map[merged_class_name] = current_id
        current_id += 1

    # Convert hex to RGB tuple, but since OpenCV uses BGR, we reverse the order
    color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))  # Convert hex to RGB tuple
    color_bgr = color_rgb[::-1]  # Reverse to BGR

    color_to_class_map[color_bgr] = class_id_map[merged_class_name]

# Function to process a single label image
def process_label_image(label_path):
    # Read the image
    img = cv2.imread(label_path)

    # Prepare the output image
    label_id_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # For each color in the image, assign the corresponding class id
    for color, class_id in color_to_class_map.items():
        mask = np.all(img == color, axis=-1)
        label_id_img[mask] = class_id

    # Create new output path by replacing "label" with "label_id" in the path
    output_path = label_path.replace("label", "label_id")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the black-and-white label image
    cv2.imwrite(output_path, label_id_img)

# Multi-processing function to speed up processing
def process_dataset_in_parallel(label_paths):
    with Pool() as pool:
        # Wrap label paths with tqdm to show progress
        for _ in tqdm(pool.imap_unordered(process_label_image, label_paths), total=len(label_paths), desc="Processing Images"):
            pass

if __name__ == "__main__":
    # Print the ID to label mapping
    print("ID to Label Mapping:")
    for label, class_id in class_id_map.items():
        print(f"ID: {class_id}, Label: {label}")

    # Process the dataset in parallel with tqdm progress bar
    process_dataset_in_parallel(label_paths)
