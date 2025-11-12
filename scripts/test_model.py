#!/usr/bin/env python3
"""
Test trained YOLOv8 model on images or video.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
from ultralytics import YOLO
from loguru import logger


def test_on_images(model_path: str, image_dir: str, conf_threshold: float = 0.25):
    """
    Test model on images in a directory.

    Args:
        model_path: Path to trained model weights
        image_dir: Directory containing test images
        conf_threshold: Confidence threshold for detections
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # Get images
    image_path = Path(image_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(image_path.glob(ext))

    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return

    logger.info(f"Found {len(image_files)} images")

    # Process each image
    for img_file in image_files:
        logger.info(f"Processing {img_file.name}")

        # Run inference
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            save=True,
            show_labels=True,
            show_conf=True,
            line_width=2
        )

        # Print detections
        for result in results:
            boxes = result.boxes
            logger.info(f"  Detected {len(boxes)} objects")

            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                logger.info(f"    - Class {class_id}, Conf: {conf:.2f}, Box: {coords}")

    logger.success("Testing complete! Results saved to runs/detect/predict*")


def test_live_capture(model_path: str, conf_threshold: float = 0.25):
    """
    Test model on live screen capture.

    Args:
        model_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
    """
    from src.capture.screen_grabber import ScreenGrabber

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # Create screen grabber
    grabber = ScreenGrabber()

    logger.info("Starting live detection (press 'q' to quit)")

    try:
        while True:
            # Capture screen
            frame = grabber.capture()

            if frame is None:
                continue

            # Run inference
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False
            )

            # Draw results
            for result in results:
                annotated_frame = result.plot()

                # Display
                cv2.imshow('Character Detection', annotated_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        logger.info("Live detection stopped")


def benchmark_model(model_path: str, image_path: str, iterations: int = 100):
    """
    Benchmark model inference speed.

    Args:
        model_path: Path to trained model weights
        image_path: Path to test image
        iterations: Number of iterations to run
    """
    import time

    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    logger.info(f"Running {iterations} iterations...")

    times = []
    for i in range(iterations):
        start = time.time()
        results = model.predict(source=image_path, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{iterations}")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    logger.info(f"\n=== Benchmark Results ===")
    logger.info(f"Average inference time: {avg_time*1000:.2f}ms")
    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"Min time: {min(times)*1000:.2f}ms")
    logger.info(f"Max time: {max(times)*1000:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Test trained YOLOv8 model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (best.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path to image directory or single image"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live detection on screen capture"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark model inference speed"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )

    args = parser.parse_args()

    # Validate model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    if args.live:
        test_live_capture(args.model, args.conf)
    elif args.benchmark:
        if not args.source:
            logger.error("--source required for benchmarking")
            sys.exit(1)
        benchmark_model(args.model, args.source, args.iterations)
    else:
        if not args.source:
            logger.error("--source required for image testing")
            sys.exit(1)
        test_on_images(args.model, args.source, args.conf)


if __name__ == "__main__":
    main()
