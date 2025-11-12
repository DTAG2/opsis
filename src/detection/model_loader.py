"""
Model loader for YOLO object detection models.
Handles loading, validation, and basic model operations.
"""

import torch
from pathlib import Path
from typing import Optional, List
from ultralytics import YOLO
from loguru import logger


class ModelLoader:
    """Manages YOLO model loading and configuration."""

    def __init__(self, model_path: str, device: str = "cuda", half_precision: bool = True):
        """
        Initialize model loader.

        Args:
            model_path: Path to YOLO model file (.pt)
            device: Device for inference ('cuda', 'cpu', 'mps')
            half_precision: Use FP16 for faster inference (GPU only)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.half_precision = half_precision and device == "cuda"
        self.model: Optional[YOLO] = None
        self.is_loaded = False

        logger.info(f"ModelLoader initialized: path={model_path}, device={device}, fp16={self.half_precision}")

    def load(self) -> bool:
        """
        Load the YOLO model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                logger.info("Please train a model or place a pre-trained model in the models/ directory")
                return False

            logger.info(f"Loading model from {self.model_path}...")

            # Load YOLO model
            self.model = YOLO(str(self.model_path))

            # Move to device
            self.model.to(self.device)

            # Enable half precision if requested
            if self.half_precision:
                logger.info("Enabling FP16 (half precision) mode")

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

            # Log model info
            self._log_model_info()

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def _log_model_info(self):
        """Log model information."""
        if self.model is None:
            return

        try:
            # Get class names
            names = self.model.names
            logger.info(f"Model classes: {names}")
            logger.info(f"Number of classes: {len(names)}")

        except Exception as e:
            logger.warning(f"Could not retrieve model info: {e}")

    def get_model(self) -> Optional[YOLO]:
        """
        Get the loaded model.

        Returns:
            YOLO model instance or None
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, call load() first")
            return None
        return self.model

    def get_class_names(self) -> List[str]:
        """
        Get list of class names from model.

        Returns:
            List of class names
        """
        if self.model is None:
            return []
        return list(self.model.names.values())

    def reload(self, model_path: Optional[str] = None) -> bool:
        """
        Reload model, optionally from a different path.

        Args:
            model_path: New model path, or None to reload current

        Returns:
            True if reload successful
        """
        if model_path:
            self.model_path = Path(model_path)

        logger.info(f"Reloading model from {self.model_path}")
        self.is_loaded = False
        self.model = None

        return self.load()

    def validate_device(self) -> bool:
        """
        Validate that the requested device is available.

        Returns:
            True if device is available
        """
        if self.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.half_precision = False
                return False
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return True

        elif self.device == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS not available, falling back to CPU")
                self.device = "cpu"
                return False
            logger.info("MPS (Apple Silicon) available")
            return True

        elif self.device == "cpu":
            logger.info("Using CPU for inference")
            return True

        else:
            logger.error(f"Unknown device: {self.device}")
            self.device = "cpu"
            return False


def check_gpu_availability() -> dict:
    """
    Check GPU availability and return system information.

    Returns:
        Dictionary with GPU information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

    logger.info(f"GPU Info: {info}")
    return info
