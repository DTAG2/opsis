"""
Configuration loader and manager for the CV Overlay System.
Handles loading, saving, and validation of configuration settings.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class ConfigLoader:
    """Manages application configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "settings.json"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Dictionary containing configuration settings.
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                self._create_default_config()

            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def save(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"Configuration saved to {self.config_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'detection.confidence_threshold')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Config key '{key_path}' not found, using default: {default}")
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config_dict = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]

        # Set the value
        config_dict[keys[-1]] = value
        logger.debug(f"Config '{key_path}' set to: {value}")

    def reload(self) -> None:
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self.load()

    def _create_default_config(self) -> None:
        """Create a default configuration file if none exists."""
        logger.info("Creating default configuration file...")

        default_config = {
            "general": {
                "app_name": "CV Overlay System",
                "version": "1.0.0",
                "debug_mode": False
            },
            "screen_capture": {
                "target_fps": 60,
                "capture_region": None,
                "capture_monitor": 0,
                "use_gpu": True
            },
            "detection": {
                "model_path": "models/default.pt",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "target_classes": ["head", "body", "character"],
                "inference_device": "cuda",
                "image_size": 640,
                "half_precision": True
            },
            "overlay": {
                "enabled": True,
                "box_color": [0, 255, 0],
                "box_thickness": 2,
                "show_labels": True,
                "show_confidence": True,
                "transparency": 0.7
            },
            "mouse_control": {
                "enabled": False,
                "target_priority": "head",
                "smoothing_factor": 0.3,
                "max_move_speed": 100,
                "enable_stabilization": True
            },
            "hotkeys": {
                "toggle_overlay": "F1",
                "toggle_mouse_control": "F2",
                "toggle_menu": "F3",
                "exit_program": "F12"
            }
        }

        self.config = default_config
        self.save()

    def validate(self) -> bool:
        """
        Validate the current configuration.

        Returns:
            True if configuration is valid, False otherwise.
        """
        required_sections = ['general', 'screen_capture', 'detection', 'overlay', 'mouse_control', 'hotkeys']

        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required config section: {section}")
                return False

        logger.info("Configuration validation passed")
        return True


# Global config instance
_config_instance: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """
    Get the global configuration instance.

    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance
