import json
import os
import numpy as np
import cv2
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def draw_polygon(image, points, value):
    """
    Draws a polygon on the image using OpenCV.
    """
    # Convert points to a NumPy array and ensure they are integers
    polygon = np.array(points, np.int32)

    if len(polygon) > 2:  # Ensure there are enough points for a polygon
        # Draw filled polygon using OpenCV
        cv2.fillPoly(image, [polygon], color=value)
    else:
        print("Invalid polygon: Not enough points.")


def convert_supervisely_to_png(json_path, output_path):
    try:
        # Load the JSON annotation file
        with open(json_path, 'r') as f:
            annotation = json.load(f)

        # Get image size from the annotation
        image_size = (annotation['size']['width'], annotation['size']['height'])

        # Full label map from meta file
        label_map = {
            "adult": 1,
            "ambulance": 2,
            "animal": 3,
            "barrier": 4,
            "bendy": 5,
            "bicycle": 6,
            "bicycle rack": 7,
            "car": 8,
            "child": 9,
            "construction": 10,
            "construction worker": 11,
            "debris": 12,
            "driveable surface": 13,
            "ego": 14,
            "motorcycle": 15,
            "personal mobility": 16,
            "police": 17,
            "police officer": 18,
            "pushable pullable": 19,
            "rigid": 20,
            "stroller": 21,
            "trafficcone": 22,
            "trailer": 23,
            "truck": 24,
            "wheelchair": 25
        }

        # Initialize an empty mask (black image)
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)  # OpenCV uses (height, width)

        # Loop through all objects in the JSON file
        for obj in annotation['objects']:
            class_title = obj['classTitle']
            if class_title in label_map:
                # Draw polygons based on exterior points
                if obj['geometryType'] == 'polygon':
                    exterior_points = obj['points']['exterior']
                    draw_polygon(mask, exterior_points, label_map[class_title])

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the mask as a PNG file using OpenCV
        cv2.imwrite(output_path, mask)
        return f"Processed: {output_path}"

    except Exception as e:
        return f"Failed to process {json_path}: {e}"


def process_file(file_info):
    json_file_path, output_png_path = file_info
    return convert_supervisely_to_png(json_file_path, output_png_path)


def main():
    # Get list of json files to process
    json_file_paths = glob("datasets/nuimages/*/ann/*jpg.json")
    file_info_list = [(json_file_path, json_file_path.replace(".jpg.json", ".png").replace("ann", "masks")) for
                      json_file_path in json_file_paths]

    # Use multiprocessing to process files in parallel
    num_workers = max(1, cpu_count() - 1)  # Leave 1 core free
    print(f"Using {num_workers} workers")

    # Create a pool and process the files
    with Pool(num_workers) as pool:
        # Use tqdm to add a progress bar
        results = list(tqdm(pool.imap(process_file, file_info_list), total=len(file_info_list)))

    # Optionally print the results
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
