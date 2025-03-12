from glob import glob
from pathlib import Path
import os

# Define the glob patterns for images and labels
imgs_path_glob = "datasets/rellis-3d/Rellis_3D_pylon_camera_node/Rellis-3D/**/*.jpg"
labels_path_glob = "datasets/rellis-3d/Rellis_3D_pylon_camera_node_label_id/Rellis-3D/**/*png"

# Get the sorted list of image and label paths
rgb_paths = sorted(glob(imgs_path_glob, recursive=True))
label_paths = sorted(glob(labels_path_glob, recursive=True))

# Create a dictionary of RGB images for fast lookup
rgb_dict = {Path(rgb).stem: rgb for rgb in rgb_paths}

# Function to check if corresponding RGB image exists
def has_corresponding_rgb(label_path):
    label_filename = Path(label_path).stem
    return label_filename in rgb_dict

if __name__ == "__main__":
    # Iterate over each label path and delete those without corresponding RGB images
    for label_path in label_paths:
        if not has_corresponding_rgb(label_path):
            print(f"Deleting {label_path} because no corresponding RGB image exists.")
            # os.remove(label_path)  # Delete the label
