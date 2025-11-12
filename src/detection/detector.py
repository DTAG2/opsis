"""
Object detector for real-time character/object detection.
Processes frames and returns detection results with bounding boxes.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from loguru import logger
from .model_loader import ModelLoader


class Detection:
    """Represents a single object detection."""

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int, class_name: str):
        """
        Initialize detection object.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            confidence: Detection confidence score (0-1)
            class_id: Class ID from model
            class_name: Class name string
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

        # Calculate center point and critical points
        self.center = self._calculate_center()
        self.head_point = self._calculate_head_point()
        self.body_point = self._calculate_body_point()

    def _calculate_center(self) -> Tuple[int, int]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _calculate_head_point(self) -> Tuple[int, int]:
        """Calculate head/top target point (upper third of bbox)."""
        x1, y1, x2, y2 = self.bbox
        center_x = int((x1 + x2) / 2)
        head_y = int(y1 + (y2 - y1) * 0.2)  # 20% from top
        return (center_x, head_y)

    def _calculate_body_point(self) -> Tuple[int, int]:
        """Calculate body/center target point."""
        return self.center

    def get_width(self) -> int:
        """Get bounding box width."""
        return self.bbox[2] - self.bbox[0]

    def get_height(self) -> int:
        """Get bounding box height."""
        return self.bbox[3] - self.bbox[1]

    def get_area(self) -> int:
        """Get bounding box area."""
        return self.get_width() * self.get_height()

    def __repr__(self) -> str:
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


class Detector:
    """Real-time object detector using YOLO."""

    def __init__(
        self,
        model_loader: ModelLoader,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        target_classes: Optional[List[str]] = None
    ):
        """
        Initialize detector.

        Args:
            model_loader: ModelLoader instance with loaded model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            image_size: Input image size for model
            target_classes: List of class names to detect, or None for all
        """
        self.model_loader = model_loader
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.target_classes = target_classes

        self.inference_time = 0.0
        self.fps = 0.0

        logger.info(
            f"Detector initialized: conf={confidence_threshold}, "
            f"iou={iou_threshold}, size={image_size}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects
        """
        start_time = time.time()

        try:
            model = self.model_loader.get_model()
            if model is None:
                logger.warning("Model not loaded")
                return []

            # Run inference
            results = model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                verbose=False,
                half=self.model_loader.half_precision
            )

            # Parse results
            detections = self._parse_results(results)

            # Filter by target classes if specified
            if self.target_classes:
                detections = [
                    det for det in detections
                    if det.class_name in self.target_classes
                ]

            # Update timing
            self.inference_time = time.time() - start_time
            self.fps = 1.0 / self.inference_time if self.inference_time > 0 else 0

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def _parse_results(self, results) -> List[Detection]:
        """
        Parse YOLO results into Detection objects.

        Args:
            results: YOLO results object

        Returns:
            List of Detection objects
        """
        detections = []

        try:
            # Get the first result (single image)
            result = results[0]

            # Check if there are any boxes
            if result.boxes is None or len(result.boxes) == 0:
                return []

            # Get boxes, confidences, and class IDs
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Get class names
            names = result.names

            # Create Detection objects
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2, y2)
                class_name = names[cls_id]

                detection = Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name
                )
                detections.append(detection)

        except Exception as e:
            logger.error(f"Error parsing results: {e}")

        return detections

    def get_best_target(self, detections: List[Detection], priority: str = "head") -> Optional[Tuple[int, int]]:
        """
        Get the best target point from detections.

        Args:
            detections: List of Detection objects
            priority: Target priority ('head', 'body', 'center', 'closest')

        Returns:
            (x, y) tuple of target point, or None
        """
        if not detections:
            return None

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        # Get best detection
        best_detection = sorted_detections[0]

        # Return appropriate target point
        if priority == "head":
            return best_detection.head_point
        elif priority == "body" or priority == "center":
            return best_detection.body_point
        else:
            return best_detection.center

    def get_closest_target(self, detections: List[Detection], screen_center: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Get the target closest to screen center.

        Args:
            detections: List of Detection objects
            screen_center: (x, y) of screen center

        Returns:
            (x, y) tuple of target point, or None
        """
        if not detections:
            return None

        def distance_to_center(det: Detection) -> float:
            dx = det.center[0] - screen_center[0]
            dy = det.center[1] - screen_center[1]
            return (dx ** 2 + dy ** 2) ** 0.5

        closest = min(detections, key=distance_to_center)
        return closest.head_point

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold."""
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold set to {threshold}")

    def set_target_classes(self, classes: Optional[List[str]]):
        """Set target classes to detect."""
        self.target_classes = classes
        logger.info(f"Target classes set to {classes}")

    def get_inference_time(self) -> float:
        """Get last inference time in seconds."""
        return self.inference_time

    def get_fps(self) -> float:
        """Get inference FPS."""
        return self.fps

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "inference_time_ms": self.inference_time * 1000,
            "fps": self.fps,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "target_classes": self.target_classes
        }
