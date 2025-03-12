import numpy as np
import cv2
import glob
from collections import OrderedDict

# Define the label-to-color mapping and the reverse
LABEL_TO_COLOR = OrderedDict({
    "background": [0, 0, 0],
    "road": [128, 0, 0],
    "lane_mark_solid": [0,128,0],
    "lane_mark_dashed": [128,128,0],
})

COLOR_TO_LABEL = {tuple(value): idx for idx, value in enumerate(LABEL_TO_COLOR.values())}


def convert_rgb_to_labels(rgb_image):
    # Create an empty array for the label ids, same height and width as input image
    label_ids = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    # Iterate over each color mapping and update the label ids
    for rgb, label_id in COLOR_TO_LABEL.items():
        # Find pixels in the image that match this color
        mask = cv2.inRange(rgb_image, np.array(rgb), np.array(rgb))
        # Set these pixels to the corresponding label ID
        label_ids[mask > 0] = label_id

    return label_ids


def main():
    # Specify the pattern for the input images
    file_pattern = '/mnt/ssd1/datasets/road_lane_segmentation/val/masks/*.png'
    # Use glob to find all files matching the pattern
    files = glob.glob(file_pattern, recursive=True)

    for file_path in files:
        # Read the image using OpenCV
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert RGB to label IDs
        label_image = convert_rgb_to_labels(image)
        # Construct the output path
        output_path = file_path[:-4] + '_id.png'
        # Save the label image using OpenCV
        cv2.imwrite(output_path, label_image)
        print(f'Saved {output_path}')


if __name__ == "__main__":
    main()
