#!/usr/bin/env python3
"""
Convert Pascal VOC XML annotations to YOLO format.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from loguru import logger


def convert_box(size, box):
    """
    Convert Pascal VOC bbox to YOLO format.

    Args:
        size: (width, height) of image
        box: (xmin, ymin, xmax, ymax) in pixels

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    # Calculate center point
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0

    # Calculate width and height
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Normalize
    x_center = x_center * dw
    y_center = y_center * dh
    width = width * dw
    height = height * dh

    return (x_center, y_center, width, height)


def convert_xml_to_yolo(xml_file, output_file, class_mapping):
    """
    Convert a single XML file to YOLO format.

    Args:
        xml_file: Path to XML file
        output_file: Path to output TXT file
        class_mapping: Dict mapping class names to IDs

    Returns:
        Number of objects converted
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # Process objects
    lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text

        if class_name not in class_mapping:
            logger.warning(f"Unknown class '{class_name}' in {xml_file.name}")
            continue

        class_id = class_mapping[class_name]

        # Get bbox
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Convert to YOLO format
        box = (xmin, ymin, xmax, ymax)
        yolo_box = convert_box((img_width, img_height), box)

        # Create YOLO line: class_id x_center y_center width height
        line = f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
        lines.append(line)

    # Write to file
    if lines:
        output_file.write_text('\n'.join(lines) + '\n')
    else:
        output_file.write_text('')

    return len(lines)


def convert_dataset(source_dir, output_dir=None, classes=None):
    """
    Convert all XML annotations in a directory to YOLO format.

    Args:
        source_dir: Directory with XML files
        output_dir: Output directory (default: same as source)
        classes: List of class names (default: auto-detect)

    Returns:
        Statistics dict
    """
    source_path = Path(source_dir)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = source_path

    # Get all XML files
    xml_files = list(source_path.glob("*.xml"))
    logger.info(f"Found {len(xml_files)} XML files")

    # Auto-detect classes if not provided
    if classes is None:
        classes_set = set()
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes_set.add(class_name)
        classes = sorted(list(classes_set))
        logger.info(f"Auto-detected classes: {classes}")

    # Create class mapping
    class_mapping = {name: idx for idx, name in enumerate(classes)}

    # Convert each file
    stats = {
        'files_processed': 0,
        'total_objects': 0,
        'classes': classes
    }

    for xml_file in xml_files:
        # Determine output filename
        output_file = output_path / xml_file.with_suffix('.txt').name

        # Convert
        num_objects = convert_xml_to_yolo(xml_file, output_file, class_mapping)

        stats['files_processed'] += 1
        stats['total_objects'] += num_objects

    logger.info(f"\n=== Conversion Report ===")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total objects: {stats['total_objects']}")
    logger.info(f"Classes: {', '.join(classes)}")
    logger.info(f"Output directory: {output_path}")

    # Save class names file
    classes_file = output_path / 'classes.txt'
    classes_file.write_text('\n'.join(classes) + '\n')
    logger.info(f"Saved class names to {classes_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML to YOLO format"
    )
    parser.add_argument(
        "source_dir",
        help="Directory with XML annotations"
    )
    parser.add_argument(
        "--output",
        help="Output directory (default: same as source)"
    )
    parser.add_argument(
        "--classes",
        nargs='+',
        help="Class names in order (default: auto-detect)"
    )

    args = parser.parse_args()

    convert_dataset(args.source_dir, args.output, args.classes)

    logger.success("✓ Conversion complete!")


if __name__ == "__main__":
    main()
