from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool

path_glob = "datasets/rugd/RUGD_annotations/**/*.png"
paths = sorted(glob(path_glob, recursive=True))

class_map_txt = "datasets/rugd/RUGD_annotations/RUGD_annotation-colormap.txt"

def parse_colormap(file_path):
    colormap = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure the line has exactly 5 parts
                category_id, category_name, r, g, b = parts
                # Handle categories with '/' by replacing with '_'
                category_name = category_name.replace('/', '_')
                # colormap[category_name] = np.array([int(r), int(g), int(b)])
                colormap[int(category_id)] = np.array([int(r), int(g), int(b)])
    return colormap

def modify_and_process_image(path):
    new_path = modify_path_with_bw_before_number(path)
    new_path = os.path.join("datasets/rugd/RUGD_annotations/", new_path)
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_id = np.zeros(img.shape[:2], dtype=np.uint8)
    # convert colors to class labels
    for key, value in cmap.items():
        img_id[(img == value).all(axis=-1)] = int(key)
    # cv2.imshow("img", img_id)
    cv2.imwrite(new_path, img_id)

def modify_path_with_bw_before_number(original_path):
    # Split the path into components
    parts = original_path.split('/')

    # Extract the directory name and filename
    directory_name = parts[-2]
    filename = parts[-1]

    # Identify the part before the numeric sequence and insert '_bw'
    filename_base, extension = filename.rsplit('.', 1)
    prefix, numeric_sequence = filename_base.rsplit('_', 1)
    modified_filename = f"{prefix}_bw_{numeric_sequence}.{extension}"

    # Reassemble the modified path
    modified_path = f"{directory_name}/{modified_filename}"

    return modified_path

cmap = parse_colormap(class_map_txt)

# Define the number of processes to use
num_processes = os.cpu_count()

# Create a multiprocessing pool
with Pool(num_processes) as pool:
    list(tqdm(pool.imap(modify_and_process_image, paths), total=len(paths)))
