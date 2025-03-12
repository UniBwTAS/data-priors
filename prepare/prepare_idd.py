import os
import json
from PIL import Image, ImageDraw
import numpy as np
from multiprocessing import Pool, Manager
import glob
from tqdm import tqdm


def create_label_image(params):
    json_file_path, img_file_path, output_path, label_to_id = params

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Load the image to get dimensions
    img = Image.open(img_file_path)
    img_shape = np.array(img).shape

    # Create a blank image (grayscale)
    label_image = Image.new('L', (img_shape[1], img_shape[0]), 0)
    draw = ImageDraw.Draw(label_image)

    # Draw each polygon on the image
    for obj in data['objects']:
        label = obj['label']
        polygon = [(point[0], point[1]) for point in obj['polygon']]

        # Ensure that the polygon has at least 2 coordinates
        if len(polygon) >= 2:
            if label not in label_to_id:
                with label_to_id_lock:
                    if label not in label_to_id:
                        label_to_id[label] = len(label_to_id) + 1  # Assign a new ID
            draw.polygon(polygon, fill=label_to_id[label])

    # Save the image
    label_image.save(output_path, 'PNG')


def process_dataset(json_dir, img_dir, output_dir, label_to_id, img_extension):
    tasks = []

    # Use glob to collect all JSON file paths
    json_files = glob.glob(os.path.join(json_dir, '**', '*_polygons.json'), recursive=True)

    for json_path in json_files:
        # Determine corresponding image file path
        image_file_name = os.path.basename(json_path).replace('_gtFine_polygons.json', f'_leftImg8bit.{img_extension}')
        img_path = os.path.join(img_dir, os.path.relpath(json_path, json_dir).replace('_gtFine_polygons.json',
                                                                                      f'_leftImg8bit.{img_extension}'))

        # Adjust the image path based on the folder structure
        img_path = os.path.join(img_dir, os.path.dirname(os.path.relpath(json_path, json_dir)), image_file_name)

        # Output path for the label image
        output_file_name = image_file_name.replace(f'_leftImg8bit.{img_extension}', '_label.png')
        output_path = os.path.join(output_dir,
                                   os.path.relpath(json_path, json_dir).replace('_gtFine_polygons.json', '_label.png'))

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare the parameters for each task
        tasks.append((json_path, img_path, output_path, label_to_id))

    # Use multiprocessing with tqdm to show progress
    with Pool() as pool:
        list(tqdm(pool.imap_unordered(create_label_image, tasks), total=len(tasks), desc="Processing images"))


if __name__ == "__main__":
    # Define directories for both datasets
    datasets_info = [
        {
            "base_dir": "/mnt/ssd1/datasets/IndiaDrivingDataset/idd-segmentation/IDD_Segmentation",
            "img_extension": "png"
        },
        {
            "base_dir": "/mnt/ssd1/datasets/IndiaDrivingDataset/idd-20k-II/idd20kII",
            "img_extension": "jpg"
        }
    ]
    label_dirs = ["val", "train"]

    manager = Manager()
    label_to_id = manager.dict()  # Shared dictionary for label-to-ID mapping
    global label_to_id_lock
    label_to_id_lock = manager.Lock()  # Lock for safely updating the shared dictionary

    for dataset_info in datasets_info:
        base_dir = dataset_info["base_dir"]
        img_extension = dataset_info["img_extension"]

        for label_dir in label_dirs:
            json_dir = os.path.join(base_dir, "gtFine", label_dir)
            img_dir = os.path.join(base_dir, "leftImg8bit", label_dir)
            output_dir = os.path.join(base_dir, "gtFine", "labels", label_dir)

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Process the dataset
            process_dataset(json_dir, img_dir, output_dir, label_to_id, img_extension)

    # Print out the final label to ID mapping
    print("Label to ID mapping:")
    for label, id in label_to_id.items():
        print(f"{label}: {id}")


#[90.55800763 93.78552729 92.17880896]
#[69.26074431 71.61879521 75.98489896]