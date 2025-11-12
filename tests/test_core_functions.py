#!/usr/bin/env python3
"""
Comprehensive unit tests for Opsis CV Overlay System core functionality.
Run from opsis directory: python -m pytest tests/test_core_functions.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import Detection
from src.utils.config_loader import ConfigLoader
from src.control.mouse_controller import SmoothMouseController


class TestDetection(unittest.TestCase):
    """Test Detection class functionality."""

    def test_detection_initialization(self):
        """Test Detection object creation."""
        det = Detection(
            bbox=(100, 200, 300, 400),
            confidence=0.95,
            class_id=0,
            class_name="character"
        )

        self.assertEqual(det.bbox, (100, 200, 300, 400))
        self.assertEqual(det.confidence, 0.95)
        self.assertEqual(det.class_id, 0)
        self.assertEqual(det.class_name, "character")

    def test_center_calculation(self):
        """Test center point calculation."""
        det = Detection(
            bbox=(100, 200, 300, 400),
            confidence=0.95,
            class_id=0,
            class_name="character"
        )

        # Center should be ((100+300)/2, (200+400)/2) = (200, 300)
        self.assertEqual(det.center, (200, 300))

    def test_head_point_calculation(self):
        """Test head point calculation (upper portion of bbox)."""
        det = Detection(
            bbox=(100, 200, 300, 400),
            confidence=0.95,
            class_id=0,
            class_name="character"
        )

        # Head should be at center_x=200, y at 20% from top
        # Height = 200, 20% = 40, so y = 200 + 40 = 240
        self.assertEqual(det.head_point[0], 200)  # center x
        self.assertEqual(det.head_point[1], 240)  # 20% from top

    def test_dimensions(self):
        """Test width, height, and area calculations."""
        det = Detection(
            bbox=(100, 200, 300, 400),
            confidence=0.95,
            class_id=0,
            class_name="character"
        )

        self.assertEqual(det.get_width(), 200)  # 300 - 100
        self.assertEqual(det.get_height(), 200)  # 400 - 200
        self.assertEqual(det.get_area(), 40000)  # 200 * 200


class TestConfigLoader(unittest.TestCase):
    """Test configuration loading and management."""

    def setUp(self):
        """Set up test configuration."""
        self.test_config = {
            "detection": {
                "confidence_threshold": 0.5,
                "model_path": "test.pt"
            },
            "mouse_control": {
                "enabled": True,
                "smoothing_factor": 0.6
            }
        }

    @patch('src.utils.config_loader.Path.exists')
    @patch('builtins.open')
    def test_config_load(self, mock_open, mock_exists):
        """Test configuration loading from JSON."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(self.test_config)

        config = ConfigLoader()
        config.load()

        self.assertEqual(config.get("detection", {}).get("confidence_threshold"), 0.5)
        self.assertEqual(config.get("mouse_control", {}).get("enabled"), True)

    def test_config_get_nested(self):
        """Test getting nested configuration values."""
        config = ConfigLoader()
        config.config = self.test_config

        # Test nested get
        self.assertEqual(config.get("detection", {}).get("confidence_threshold"), 0.5)

        # Test default values
        self.assertEqual(config.get("nonexistent", {"default": "value"}), {"default": "value"})

    def test_config_set(self):
        """Test setting configuration values."""
        config = ConfigLoader()
        config.config = self.test_config.copy()

        # Set new value
        config.set("detection.new_key", "new_value")
        self.assertEqual(config.get("detection", {}).get("new_key"), "new_value")

        # Update existing value
        config.set("detection.confidence_threshold", 0.7)
        self.assertEqual(config.get("detection", {}).get("confidence_threshold"), 0.7)


class TestMouseController(unittest.TestCase):
    """Test mouse controller functionality."""

    def setUp(self):
        """Set up mouse controller for testing."""
        self.controller = SmoothMouseController(
            smoothing_factor=0.5,
            max_move_speed=100,
            enable_stabilization=True,
            stabilization_strength=0.5,
            aim_offset=(0, 0)
        )

    def test_initialization(self):
        """Test mouse controller initialization."""
        self.assertEqual(self.controller.smoothing_factor, 0.5)
        self.assertEqual(self.controller.max_move_speed, 100)
        self.assertEqual(self.controller.enable_stabilization, True)
        self.assertFalse(self.controller.is_enabled)

    def test_enable_disable(self):
        """Test enabling and disabling mouse control."""
        # Initially disabled
        self.assertFalse(self.controller.is_enabled)

        # Enable
        self.controller.enable()
        self.assertTrue(self.controller.is_enabled)

        # Disable
        self.controller.disable()
        self.assertFalse(self.controller.is_enabled)

        # Toggle
        self.controller.toggle()
        self.assertTrue(self.controller.is_enabled)
        self.controller.toggle()
        self.assertFalse(self.controller.is_enabled)

    @patch('pynput.mouse.Controller')
    def test_get_position(self, mock_mouse):
        """Test getting current mouse position."""
        mock_mouse.return_value.position = (500, 300)
        controller = SmoothMouseController()
        controller.mouse = mock_mouse.return_value

        pos = controller.get_current_position()
        self.assertEqual(pos, (500, 300))

    def test_calculate_smooth_delta(self):
        """Test smooth movement calculation."""
        # Test movement calculation
        current = (100, 100)
        target = (200, 150)

        dx, dy = self.controller._calculate_smooth_delta(current, target)

        # With smoothing_factor=0.5, movement should be half the distance
        # But also capped by max_move_speed
        self.assertIsInstance(dx, (int, float))
        self.assertIsInstance(dy, (int, float))

        # Movement should be in the right direction
        self.assertGreater(dx, 0)  # Moving right
        self.assertGreater(dy, 0)  # Moving down

    def test_clamp_movement(self):
        """Test movement speed clamping."""
        controller = SmoothMouseController(max_move_speed=10)

        # Test clamping large movements
        dx, dy = controller._clamp_movement(50, 30)

        # Total magnitude should not exceed max_move_speed
        magnitude = (dx**2 + dy**2)**0.5
        self.assertLessEqual(magnitude, 10.1)  # Small tolerance for float math

    def test_movement_history(self):
        """Test movement history tracking."""
        self.controller.is_enabled = True

        # Add some movements to history
        self.controller.movement_history.append((10, 5))
        self.controller.movement_history.append((8, 6))
        self.controller.movement_history.append((12, 4))

        # History should be limited by maxlen
        self.assertLessEqual(len(self.controller.movement_history), 10)

    def test_set_parameters(self):
        """Test parameter setters."""
        # Test smoothing setter
        self.controller.set_smoothing(0.8)
        self.assertEqual(self.controller.smoothing_factor, 0.8)

        # Test max speed setter
        self.controller.set_max_speed(50)
        self.assertEqual(self.controller.max_move_speed, 50)

        # Test stabilization setter
        self.controller.set_stabilization_strength(0.7)
        self.assertEqual(self.controller.stabilization_strength, 0.7)


class TestCoordinateConversion(unittest.TestCase):
    """Test coordinate conversion from frame to screen space."""

    def test_frame_to_screen_conversion(self):
        """Test converting frame coordinates to screen coordinates."""
        # Monitor at position (1920, 0)
        monitor_offset_x = 1920
        monitor_offset_y = 0

        # Target in frame space
        frame_target = (500, 300)

        # Convert to screen space
        screen_target = (
            frame_target[0] + monitor_offset_x,
            frame_target[1] + monitor_offset_y
        )

        self.assertEqual(screen_target, (2420, 300))

    def test_multi_monitor_offsets(self):
        """Test coordinate conversion for different monitor configurations."""
        test_cases = [
            # (monitor_offset, frame_coord, expected_screen_coord)
            ((0, 0), (500, 300), (500, 300)),  # Primary monitor
            ((1920, 0), (500, 300), (2420, 300)),  # Right monitor
            ((0, 1080), (500, 300), (500, 1380)),  # Bottom monitor
            ((-1920, 0), (500, 300), (-1420, 300)),  # Left monitor
        ]

        for monitor_offset, frame_coord, expected in test_cases:
            screen_coord = (
                frame_coord[0] + monitor_offset[0],
                frame_coord[1] + monitor_offset[1]
            )
            self.assertEqual(screen_coord, expected)


class TestHotkeyConfiguration(unittest.TestCase):
    """Test hotkey configuration consistency."""

    def test_hotkey_values(self):
        """Test that hotkey configuration uses correct values."""
        # These should match what's in config/settings.json
        expected_hotkeys = {
            "toggle_overlay": "F1",
            "toggle_mouse_control": "F2",
            "toggle_menu": "F3",
            "reload_config": "F4",
            "exit_program": "F5"  # Should be F5, not F12
        }

        # Load actual config
        config_path = Path(__file__).parent.parent / "config" / "settings.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                actual_hotkeys = config.get("hotkeys", {})

                for key, expected_value in expected_hotkeys.items():
                    self.assertEqual(
                        actual_hotkeys.get(key),
                        expected_value,
                        f"Hotkey {key} should be {expected_value}"
                    )


class TestBoundingBoxExpansion(unittest.TestCase):
    """Test bounding box expansion for annotation processing."""

    def test_expand_box_centered(self):
        """Test expanding a box from its center."""
        # Original box: 100x100 at (100, 100) to (200, 200)
        xmin, ymin, xmax, ymax = 100, 100, 200, 200

        # Expand by 2x in both dimensions
        width_mult = 2.0
        height_mult = 2.0
        img_width = 1000
        img_height = 1000

        # Import the function from expand_annotations
        from scripts.expand_annotations import expand_box

        new_xmin, new_ymin, new_xmax, new_ymax = expand_box(
            xmin, ymin, xmax, ymax,
            width_mult, height_mult,
            img_width, img_height
        )

        # New box should be 200x200, centered at same point
        # Center was at (150, 150)
        self.assertEqual(new_xmin, 50)   # 150 - 100
        self.assertEqual(new_ymin, 50)   # 150 - 100
        self.assertEqual(new_xmax, 250)  # 150 + 100
        self.assertEqual(new_ymax, 250)  # 150 + 100

    def test_expand_box_with_boundaries(self):
        """Test that expanded boxes respect image boundaries."""
        from scripts.expand_annotations import expand_box

        # Box near edge: 50x50 at (10, 10) to (60, 60)
        xmin, ymin, xmax, ymax = 10, 10, 60, 60
        width_mult = 5.0  # Would expand to 250x250
        height_mult = 5.0
        img_width = 200
        img_height = 200

        new_xmin, new_ymin, new_xmax, new_ymax = expand_box(
            xmin, ymin, xmax, ymax,
            width_mult, height_mult,
            img_width, img_height
        )

        # Should be clamped to image boundaries
        self.assertGreaterEqual(new_xmin, 0)
        self.assertGreaterEqual(new_ymin, 0)
        self.assertLessEqual(new_xmax, img_width)
        self.assertLessEqual(new_ymax, img_height)


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestMouseController))
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestHotkeyConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundingBoxExpansion))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit(run_tests())