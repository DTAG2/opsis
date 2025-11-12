#!/usr/bin/env python3
"""
Dataset splitting utility.
Splits annotated data into train/val sets.
"""

import shutil
from pathlib import Path
import random
import argparse
from loguru import logger


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    copy_files: bool = True,
    label_format: str = 'xml'
):
    """
    Split dataset into train and validation sets.

    Args:
        source_dir: Source directory with images and labels
        output_dir: Output directory for dataset
        train_ratio: Ratio of training data (0-1)
        copy_files: If True, copy files; if False, move files
        label_format: Format of labels ('xml' or 'txt')
    """
    source = Path(source_dir)
    output = Path(output_dir)

    # Create output directories
    for split in ['train', 'val']:
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(source.glob(f'*{ext}'))

    if not image_files:
        logger.error(f"No images found in {source}")
        return False

    logger.info(f"Found {len(image_files)} images")

    # Validate and filter image-label pairs
    label_extension = f'.{label_format}'
    valid_pairs = []
    skipped_files = []

    for img_file in image_files:
        label_file = img_file.with_suffix(label_extension)

        if label_file.exists():
            valid_pairs.append(img_file)
        else:
            skipped_files.append(img_file.name)
            logger.warning(f"Skipping {img_file.name} - no matching {label_format} annotation")

    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} files without valid annotations")

    if not valid_pairs:
        logger.error(f"No valid image-{label_format} pairs found in {source}")
        return False

    logger.info(f"Found {len(valid_pairs)} valid image-annotation pairs")

    # Shuffle and split
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)

    train_files = valid_pairs[:split_idx]
    val_files = valid_pairs[split_idx:]

    logger.info(f"Splitting: {len(train_files)} train, {len(val_files)} val")

    # Copy/move files
    transfer_func = shutil.copy2 if copy_files else shutil.move

    # Process training files
    for img_file in train_files:
        label_file = img_file.with_suffix(label_extension)

        # Copy image
        transfer_func(img_file, output / 'images' / 'train' / img_file.name)

        # Copy label
        transfer_func(label_file, output / 'labels' / 'train' / label_file.name)

    # Process validation files
    for img_file in val_files:
        label_file = img_file.with_suffix(label_extension)

        # Copy image
        transfer_func(img_file, output / 'images' / 'val' / img_file.name)

        # Copy label
        transfer_func(label_file, output / 'labels' / 'val' / label_file.name)

    logger.info("Dataset split complete!")
    logger.info(f"Output directory: {output}")
    logger.info(f"Train: {len(train_files)} samples, Val: {len(val_files)} samples")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Split dataset into train/val")
    parser.add_argument(
        "source",
        type=str,
        help="Source directory with images and labels"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/datasets/<source_name>)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio (default: 0.8)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying"
    )
    parser.add_argument(
        "--label-format",
        type=str,
        default="xml",
        choices=["xml", "txt"],
        help="Label file format: 'xml' for Pascal VOC or 'txt' for YOLO (default: xml)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        source_name = Path(args.source).name
        args.output = f"data/datasets/{source_name}"

    logger.info(f"Source: {args.source}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Train ratio: {args.train_ratio}")
    logger.info(f"Label format: {args.label_format}")
    logger.info(f"Mode: {'move' if args.move else 'copy'}")

    success = split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        copy_files=not args.move,
        label_format=args.label_format
    )

    if success:
        print(f"\n✅ Dataset split successfully!")
        print(f"📁 Output: {args.output}")
        print(f"\nNext steps:")
        print(f"1. Create data YAML: data/datasets/<name>.yaml")
        print(f"2. Run training: python scripts/train.py --data <yaml_file>")
    else:
        print(f"\n❌ Dataset split failed!")


if __name__ == "__main__":
    main()
