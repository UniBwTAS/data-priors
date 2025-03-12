from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool

path_glob = "/mnt/ssd1/datasets/freiburg-forest/download/freiburg_forest_annotated/train/GT_color/*.png"
paths = sorted(glob(path_glob, recursive=True))

color_map = {
    0: [0, 0, 0],
    1: [170, 170, 170],
    2: [0, 255, 0],
    3: [102, 102, 51],
    4: [0, 60, 0],
    5: [0, 120, 255],
}

def modify_and_process_image(path):
    new_path = path.replace("GT_color", "GT_id").replace("Clipped", "mask")
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_id = np.zeros(img.shape[:2], dtype=np.uint8)
    # convert colors to class labels
    for key, value in cmap.items():
        img_id[(img == value).all(axis=-1)] = int(key)
    # cv2.imshow("img", img_id)
    # cv2.waitKey(0)
    cv2.imwrite(new_path, img_id)

cmap = color_map

# Define the number of processes to use
num_processes = os.cpu_count()

# Create a multiprocessing pool
with Pool(num_processes) as pool:
    list(tqdm(pool.imap(modify_and_process_image, paths), total=len(paths)))
