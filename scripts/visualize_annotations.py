#!/usr/bin/env python3
"""
Visualize annotations by drawing bounding boxes on images.
Creates visual verification of character detection.
"""

import cv2
import json
from pathlib import Path
import numpy as np

def draw_annotations(image_path, annotations_data, output_path):
    """Draw bounding boxes on image for verification."""
    img = cv2.imread(image_path)
    if img is None:
        return False

    height, width = img.shape[:2]

    # Find annotation for this image
    filename = Path(image_path).name
    image_annotation = None

    for img_data in annotations_data['images']:
        if img_data['filename'] == filename:
            image_annotation = img_data
            break

    if image_annotation is None:
        return False

    # Draw each detected character
    for obj in image_annotation['objects']:
        xmin = obj['xmin']
        ymin = obj['ymin']
        xmax = obj['xmax']
        ymax = obj['ymax']
        confidence = obj['confidence']

        # Draw bounding box
        color = (0, 255, 0)  # Green for character
        thickness = 2

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        # Draw confidence label
        label = f"char {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (0, 255, 0)

        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = xmin
        text_y = max(ymin - 5, 20)

        # Draw text background
        cv2.rectangle(img,
                     (text_x - 2, text_y - text_size[1] - 4),
                     (text_x + text_size[0] + 2, text_y + 2),
                     (0, 0, 0), -1)

        # Draw text
        cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Write image info on image
    info_text = f"{image_annotation['filename']} - {image_annotation['num_characters']} characters"
    cv2.putText(img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Save annotated image
    cv2.imwrite(output_path, img)
    return True

def main():
    # Paths
    annotations_file = '/'
    images_dir = Path('/')
    output_dir = Path('/')

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)

    # Find diverse examples
    examples = []

    # Find 1 with many characters
    for img in annotations_data['images']:
        if img['num_characters'] >= 10 and img['image_type'] == 'gameplay':
            examples.append(img['filename'])
            break

    # Find 1 with few characters
    for img in annotations_data['images']:
        if 3 <= img['num_characters'] <= 6 and img['image_type'] == 'gameplay':
            examples.append(img['filename'])
            break

    # Find 1 with single character
    for img in annotations_data['images']:
        if img['num_characters'] == 1 and img['image_type'] == 'gameplay':
            examples.append(img['filename'])
            break

    # Find 1 with no characters
    for img in annotations_data['images']:
        if img['num_characters'] == 0 and img['image_type'] == 'gameplay':
            examples.append(img['filename'])
            break

    print("Creating visualization samples...")
    print()

    for filename in examples[:4]:
        img_path = images_dir / filename
        output_path = output_dir / f"annotated_{filename}"

        if draw_annotations(str(img_path), annotations_data, str(output_path)):
            # Get annotation info
            for img_data in annotations_data['images']:
                if img_data['filename'] == filename:
                    print(f"✓ {filename}")
                    print(f"  Characters: {img_data['num_characters']}")
                    print(f"  Type: {img_data['image_type']}")
                    print(f"  Output: annotated_{filename}")
                    print()
                    break

    print(f"Annotated samples saved to: {output_dir}")

if __name__ == '__main__':
    main()
