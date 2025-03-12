import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

# Load the CSV file to extract the RGB to ID mapping
csv_path = 'datasets/CamVid/lists/class_dict.csv'
class_dict = pd.read_csv(csv_path)

# Define the directories to process
directories = ["datasets/CamVid/test_labels/",
               "datasets/CamVid/train_labels/",
               "datasets/CamVid/val_labels/"]

# Create a colormap where RGB values map to class IDs
colormap = {}
for idx, row in class_dict.iterrows():
    rgb = (row['r'], row['g'], row['b'])
    colormap[rgb] = idx  # Use the row index as the class ID


def process_image(path):
    # Load the image
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create an empty array for the grayscale image (class IDs)
    img_id = np.zeros(img.shape[:2], dtype=np.uint8)

    # Map the RGB values to class IDs
    for rgb, class_id in colormap.items():
        img_id[(img == rgb).all(axis=-1)] = class_id

    # Modify the filename to add '_id' before the extension
    directory, filename = os.path.split(path)
    filename_base, extension = os.path.splitext(filename)
    new_filename = f"{filename_base}_id{extension}"
    new_path = os.path.join(directory, new_filename)

    # Save the processed image as a grayscale PNG
    cv2.imwrite(new_path, img_id)


# Get the list of all label image paths from the directories
paths = []
for directory in directories:
    paths.extend(glob(os.path.join(directory, "*.png")))

# Process the images using multiprocessing
num_processes = os.cpu_count()
with Pool(num_processes) as pool:
    list(tqdm(pool.imap(process_image, paths), total=len(paths)))
