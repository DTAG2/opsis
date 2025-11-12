#!/usr/bin/env python3
"""
Model training script for YOLOv8 object detection.
Trains custom models on annotated datasets.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
from datetime import datetime
from ultralytics import YOLO
from loguru import logger


class ModelTrainer:
    """Trains YOLO models on custom datasets."""

    def __init__(
        self,
        data_yaml: str,
        model: str = "yolov8n.pt",
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        device: str = "cuda",
        project: str = "models/runs",
        name: str = "train"
    ):
        """
        Initialize model trainer.

        Args:
            data_yaml: Path to data configuration YAML file
            model: Base model to use (yolov8n.pt, yolov8s.pt, etc.)
            epochs: Number of training epochs
            batch_size: Batch size for training
            image_size: Input image size
            device: Device to train on ('cuda', 'cpu', 'mps')
            project: Project directory for saving runs
            name: Name of this training run
        """
        self.data_yaml = Path(data_yaml)
        self.model_name = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.project = project
        self.name = name

        self.model = None

        logger.info(f"ModelTrainer initialized: model={model}, epochs={epochs}")

    def validate_dataset(self) -> bool:
        """
        Validate that dataset configuration is correct.

        Returns:
            True if dataset is valid
        """
        if not self.data_yaml.exists():
            logger.error(f"Data YAML not found: {self.data_yaml}")
            return False

        try:
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)

            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    logger.error(f"Missing required field in data YAML: {field}")
                    return False

            # Check paths exist
            yaml_dir = self.data_yaml.parent
            train_path = yaml_dir / data_config['train']
            val_path = yaml_dir / data_config['val']

            if not train_path.exists():
                logger.error(f"Training data path not found: {train_path}")
                return False

            if not val_path.exists():
                logger.warning(f"Validation data path not found: {val_path}")

            logger.info(f"Dataset validation passed")
            logger.info(f"Classes: {data_config['names']}")
            logger.info(f"Number of classes: {data_config['nc']}")

            return True

        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False

    def train(self):
        """Train the model."""
        logger.info("Starting model training...")

        # Validate dataset
        if not self.validate_dataset():
            logger.error("Dataset validation failed, aborting training")
            return False

        try:
            # Load base model
            logger.info(f"Loading base model: {self.model_name}")
            self.model = YOLO(self.model_name)

            # Start training
            logger.info(f"Training for {self.epochs} epochs...")

            results = self.model.train(
                data=str(self.data_yaml),
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.image_size,
                device=self.device,
                project=self.project,
                name=self.name,
                verbose=True,
                patience=50,  # Early stopping patience
                save=True,
                save_period=10,  # Save checkpoint every 10 epochs
                plots=True,  # Generate training plots
                augment=True,  # Use data augmentation
                mosaic=1.0,  # Mosaic augmentation probability
                mixup=0.1,  # Mixup augmentation probability
                hsv_h=0.015,  # HSV-Hue augmentation
                hsv_s=0.7,  # HSV-Saturation augmentation
                hsv_v=0.4,  # HSV-Value augmentation
                degrees=0.0,  # Rotation augmentation
                translate=0.1,  # Translation augmentation
                scale=0.5,  # Scale augmentation
                fliplr=0.5,  # Horizontal flip probability
            )

            logger.info("Training completed!")
            self._print_results(results)

            return True

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def _print_results(self, results):
        """Print training results summary."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING RESULTS")
        logger.info("="*60)

        try:
            # Get final metrics
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}

            if metrics:
                logger.info(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
                logger.info(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
                logger.info(f"Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
                logger.info(f"Recall: {metrics.get('metrics/recall(B)', 'N/A')}")

            # Get save location
            if hasattr(results, 'save_dir'):
                logger.info(f"\nModel saved to: {results.save_dir}")
                logger.info(f"Best weights: {results.save_dir}/weights/best.pt")
                logger.info(f"Last weights: {results.save_dir}/weights/last.pt")

        except Exception as e:
            logger.warning(f"Could not print detailed results: {e}")

        logger.info("="*60)

    def export_model(self, output_path: str, format: str = "onnx"):
        """
        Export trained model to different format.

        Args:
            output_path: Output path for exported model
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
        """
        if self.model is None:
            logger.error("No model loaded, train a model first")
            return False

        try:
            logger.info(f"Exporting model to {format} format...")
            self.model.export(format=format)
            logger.info(f"Model exported successfully")
            return True

        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return False


def create_example_data_yaml(output_path: str = "data/datasets/example.yaml"):
    """Create an example data YAML file for reference."""
    example_yaml = """# Dataset configuration for YOLO training

# Paths (relative to this YAML file or absolute)
path: ../datasets/my_game  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val  # Validation images (relative to 'path')
test: images/test  # Test images (optional)

# Classes
nc: 3  # Number of classes
names: ['head', 'body', 'character']  # Class names

# Optional: Download URL (if using auto-download)
# download: https://example.com/dataset.zip
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(example_yaml)

    logger.info(f"Example data YAML created at: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on custom dataset")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of training run"
    )
    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create example data YAML file and exit"
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    # Create example if requested
    if args.create_example:
        create_example_data_yaml()
        return

    # Generate run name if not provided
    if args.name is None:
        args.name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Print configuration
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    print(f"Data YAML: {args.data}")
    print(f"Base Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.img_size}")
    print(f"Device: {args.device}")
    print(f"Run Name: {args.name}")
    print("="*60 + "\n")

    # Create trainer and start training
    trainer = ModelTrainer(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.img_size,
        device=args.device,
        name=args.name
    )

    success = trainer.train()

    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
