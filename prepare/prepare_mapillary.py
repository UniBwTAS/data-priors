import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count
import cv2
from tqdm import tqdm

# Load the config file
config_path = "/mnt/ssd1/datasets/mapillary-vistas/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Create a mapping from RGB colors to label IDs and a mapping from IDs to label names
color_to_id = {}
id_to_label = {}

for idx, label in enumerate(config['labels']):
    color_tuple = tuple(label['color'])  # Convert the list to a tuple to use as a key
    color_to_id[color_tuple] = idx  # Map the RGB color to the label ID
    id_to_label[idx] = label['name']  # Map the label ID to the label name

# Print the dictionary mapping class IDs to their corresponding labels
print("Class ID to Label Mapping:")
for class_id, label_name in id_to_label.items():
    print(f"{class_id}: {label_name}")


# Function to convert an RGB image to an ID image
def convert_rgb_to_id(image, color_to_id):
    # Create an array that can map RGB to ID
    id_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Flatten the color image for easier mapping
    image_flat = image.reshape(-1, 3)

    # Map the RGB colors to their respective IDs using a vectorized approach
    for color, id_ in color_to_id.items():
        mask = np.all(image_flat == color, axis=-1)
        id_image.flat[mask] = id_

    return id_image


# Function to process a single file
def process_file(args):
    input_dir, output_dir, filename = args
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if filename.endswith(".png"):
        # Load the image using OpenCV
        image_cv = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # Convert to ID image
        id_image_array = convert_rgb_to_id(image_cv_rgb, color_to_id)

        # Save the ID image directly using OpenCV
        cv2.imwrite(output_path, id_image_array)


# Main processing function
def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Create a list of all files to process
    file_list = [(input_dir, output_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".png")]

    # Use tqdm for progress tracking with multiprocessing
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_file, file_list), total=len(file_list), desc=f"Processing {input_dir}"))


# Directories
input_dirs = {
    "val": "/mnt/ssd1/datasets/mapillary-vistas/val/labels",
    "train": "/mnt/ssd1/datasets/mapillary-vistas/train/labels"
}
output_dirs = {
    "val": "/mnt/ssd1/datasets/mapillary-vistas/val/labels_id",
    "train": "/mnt/ssd1/datasets/mapillary-vistas/train/labels_id"
}

if __name__ == "__main__":
    # Print the class ID to label mapping
    print("Class ID to Label Mapping:")
    for class_id, label_name in id_to_label.items():
        print(f"{class_id}: {label_name}")

    # Process each directory
    for key in input_dirs:
        process_directory(input_dirs[key], output_dirs[key])

    print("Conversion completed.")
