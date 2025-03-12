import json
import numpy as np
from PIL import Image, ImageDraw
import os
from glob import glob
from pathlib import Path

# coarse data
folders = glob("/mnt/ssd1/datasets/cityscapes/extra/gtCoarse/gtCoarse/train_extra/*")

# video data
folders = [glob(f"/mnt/ssd1/datasets/cityscapes/gtFine/{action}/*") for action in ['train', 'val', 'test']]
folders = [item for sublist in folders for item in sublist]
json_paths = []
for folder in folders:
    json_path = glob(folder + "/*.json")[0]
    json_paths.append(json_path)

# Load the JSON file
for json_file in json_paths:
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Create a blank mask image
    img_width = data['imgWidth']
    img_height = data['imgHeight']
    mask = np.ones((img_height, img_width), dtype=np.uint8)

    # Draw polygons for specified labels
    def draw_polygon(draw, polygon, value):
        # Convert coordinates to tuples
        polygon = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon, fill=value)

    # Create a PIL image for drawing
    mask_image = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_image)

    # Iterate over objects and draw polygons
    for obj in data['objects']:
        label = obj['label'].lower()
        polygon = obj['polygon']
        if label in ['ego vehicle', 'out of roi']:
            draw_polygon(draw, polygon, 0)

    # Save the mask as a PNG
    json_path = Path(json_file)
    city = json_path.parts[-2]
    output_file = json_path.parent.parent.parent.parent / "extra" / f"{city}.png"
    mask_image.save(output_file)
    print(f"Label mask saved to {output_file}")

# coarse data

json_paths = []
for folder in folders:
    json_path = glob(folder + "/*.json")[0]
    json_paths.append(json_path)

# Load the JSON file
for json_file in json_paths:
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Create a blank mask image
    img_width = data['imgWidth']
    img_height = data['imgHeight']
    mask = np.ones((img_height, img_width), dtype=np.uint8)

    # Draw polygons for specified labels
    def draw_polygon(draw, polygon, value):
        # Convert coordinates to tuples
        polygon = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon, fill=value)

    # Create a PIL image for drawing
    mask_image = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_image)

    # Iterate over objects and draw polygons
    for obj in data['objects']:
        label = obj['label'].lower()
        polygon = obj['polygon']
        if label in ['ego vehicle', 'out of roi']:
            draw_polygon(draw, polygon, 0)

    # Save the mask as a PNG
    json_path = Path(json_file)
    city = json_path.parts[-2]
    output_file = json_path.parent.parent.parent.parent.parent / f"{city}.png"
    mask_image.save(output_file)
    print(f"Label mask saved to {output_file}")
