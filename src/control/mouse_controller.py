"""
Mouse controller with smooth movement and weapon stabilization.
Provides automated mouse control with configurable smoothing and recoil compensation.
"""

import time
import math
from typing import Tuple, Optional, List
from collections import deque
import numpy as np
from pynput.mouse import Controller as MouseController
from loguru import logger


class SmoothMouseController:
    """Mouse controller with smooth movement and stabilization."""

    def __init__(
        self,
        smoothing_factor: float = 0.3,
        max_move_speed: float = 100.0,
        enable_stabilization: bool = True,
        stabilization_strength: float = 0.5,
        aim_offset: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize mouse controller.

        Args:
            smoothing_factor: Movement smoothing (0-1, higher = smoother but slower)
            max_move_speed: Maximum pixels per movement
            enable_stabilization: Enable weapon stabilization/recoil compensation
            stabilization_strength: Strength of stabilization (0-1)
            aim_offset: Fixed offset to apply to target (x, y)
        """
        self.mouse = MouseController()
        self.smoothing_factor = smoothing_factor
        self.max_move_speed = max_move_speed
        self.enable_stabilization = enable_stabilization
        self.stabilization_strength = stabilization_strength
        self.aim_offset = aim_offset

        self.is_enabled = False
        self.last_target: Optional[Tuple[int, int]] = None
        self.movement_history: deque = deque(maxlen=10)

        # For stabilization
        self.recoil_pattern: List[Tuple[float, float]] = []
        self.current_recoil_index = 0
        self.is_firing = False

        logger.info(
            f"MouseController initialized: smoothing={smoothing_factor}, "
            f"max_speed={max_move_speed}, stabilization={enable_stabilization}"
        )

    def get_current_position(self) -> Tuple[int, int]:
        """
        Get current mouse position.

        Returns:
            (x, y) tuple of current position
        """
        pos = self.mouse.position
        return (int(pos[0]), int(pos[1]))

    def move_to_target(self, target: Tuple[int, int], smooth: bool = True):
        """
        Move mouse to target position.

        Args:
            target: Target (x, y) coordinates
            smooth: Apply smooth movement
        """
        if not self.is_enabled:
            return

        # Apply aim offset
        target = (target[0] + self.aim_offset[0], target[1] + self.aim_offset[1])

        current_pos = self.get_current_position()

        if smooth:
            self._smooth_move(current_pos, target)
        else:
            self.mouse.position = target

        self.last_target = target

    def _smooth_move(self, start: Tuple[int, int], target: Tuple[int, int]):
        """
        Perform smooth mouse movement using exponential smoothing.

        Args:
            start: Starting position
            target: Target position
        """
        # Calculate distance
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)

        # If very close to target, snap to it
        if distance < 2:
            self.mouse.position = target
            return

        # Limit maximum movement per frame
        if distance > self.max_move_speed:
            scale = self.max_move_speed / distance
            dx *= scale
            dy *= scale
            distance = self.max_move_speed

        # Apply smoothing - but ensure we eventually reach target
        # Use inverse smoothing: lower values = faster movement
        move_factor = 1.0 - self.smoothing_factor

        # For small distances, reduce smoothing to ensure we reach target
        if distance < 10:
            move_factor = max(move_factor, 0.5)

        move_x = dx * move_factor
        move_y = dy * move_factor

        # Calculate new position
        new_x = start[0] + move_x
        new_y = start[1] + move_y

        # Move mouse
        self.mouse.position = (int(new_x), int(new_y))

        # Record movement for stabilization
        self.movement_history.append((move_x, move_y))

    def apply_stabilization(self):
        """Apply weapon stabilization/recoil compensation."""
        if not self.enable_stabilization or not self.is_firing:
            return

        if not self.recoil_pattern:
            # Use default stabilization (pull down)
            self._apply_default_stabilization()
        else:
            # Use learned recoil pattern
            self._apply_pattern_stabilization()

    def _apply_default_stabilization(self):
        """Apply default downward stabilization."""
        current_pos = self.get_current_position()

        # Pull down slightly
        compensation_y = -2 * self.stabilization_strength

        new_pos = (current_pos[0], int(current_pos[1] + compensation_y))
        self.mouse.position = new_pos

    def _apply_pattern_stabilization(self):
        """Apply learned recoil pattern compensation."""
        if self.current_recoil_index >= len(self.recoil_pattern):
            self.current_recoil_index = 0

        # Get recoil compensation for current shot
        comp_x, comp_y = self.recoil_pattern[self.current_recoil_index]

        # Apply strength multiplier
        comp_x *= self.stabilization_strength
        comp_y *= self.stabilization_strength

        # Move mouse
        current_pos = self.get_current_position()
        new_pos = (
            int(current_pos[0] + comp_x),
            int(current_pos[1] + comp_y)
        )
        self.mouse.position = new_pos

        self.current_recoil_index += 1

    def track_target(self, target: Tuple[int, int]):
        """
        Continuously track a moving target.

        Args:
            target: Target position to track
        """
        if not self.is_enabled:
            return

        self.move_to_target(target, smooth=True)

        # Apply stabilization if firing
        if self.is_firing:
            self.apply_stabilization()

    def start_firing(self):
        """Signal that firing has started (for stabilization)."""
        self.is_firing = True
        self.current_recoil_index = 0
        logger.debug("Firing started, stabilization active")

    def stop_firing(self):
        """Signal that firing has stopped."""
        self.is_firing = False
        self.current_recoil_index = 0
        logger.debug("Firing stopped, stabilization inactive")

    def learn_recoil_pattern(self, pattern: List[Tuple[float, float]]):
        """
        Set a learned recoil pattern for stabilization.

        Args:
            pattern: List of (x, y) compensation values for each shot
        """
        self.recoil_pattern = pattern
        logger.info(f"Recoil pattern learned: {len(pattern)} shots")

    def enable(self):
        """Enable mouse control."""
        self.is_enabled = True
        logger.info("Mouse control enabled")

    def disable(self):
        """Disable mouse control."""
        self.is_enabled = False
        logger.info("Mouse control disabled")

    def toggle(self):
        """Toggle mouse control."""
        self.is_enabled = not self.is_enabled
        status = "enabled" if self.is_enabled else "disabled"
        logger.info(f"Mouse control {status}")

    def set_smoothing(self, smoothing: float):
        """
        Set smoothing factor.

        Args:
            smoothing: Smoothing factor (0-1)
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing))
        logger.info(f"Smoothing set to {self.smoothing_factor}")

    def set_max_speed(self, speed: float):
        """
        Set maximum movement speed.

        Args:
            speed: Maximum pixels per movement
        """
        self.max_move_speed = speed
        logger.info(f"Max speed set to {speed}")

    def set_stabilization_strength(self, strength: float):
        """
        Set stabilization strength.

        Args:
            strength: Stabilization strength (0-1)
        """
        self.stabilization_strength = max(0.0, min(1.0, strength))
        logger.info(f"Stabilization strength set to {self.stabilization_strength}")

    def set_aim_offset(self, offset: Tuple[int, int]):
        """
        Set aim offset.

        Args:
            offset: (x, y) offset in pixels
        """
        self.aim_offset = offset
        logger.info(f"Aim offset set to {offset}")

    def get_average_movement(self) -> Tuple[float, float]:
        """
        Get average movement from history.

        Returns:
            (avg_x, avg_y) average movement
        """
        if not self.movement_history:
            return (0.0, 0.0)

        movements = np.array(list(self.movement_history))
        avg_x = np.mean(movements[:, 0])
        avg_y = np.mean(movements[:, 1])

        return (float(avg_x), float(avg_y))

    def is_active(self) -> bool:
        """Check if mouse control is active."""
        return self.is_enabled

    def reset(self):
        """Reset controller state."""
        self.last_target = None
        self.movement_history.clear()
        self.current_recoil_index = 0
        self.is_firing = False
        logger.info("Mouse controller reset")


def calculate_smooth_path(start: Tuple[int, int], end: Tuple[int, int], steps: int = 10) -> List[Tuple[int, int]]:
    """
    Calculate smooth bezier path between two points.

    Args:
        start: Starting position
        end: Ending position
        steps: Number of intermediate points

    Returns:
        List of (x, y) positions along the path
    """
    path = []

    # Create bezier curve with control point
    control_x = (start[0] + end[0]) / 2
    control_y = (start[1] + end[1]) / 2

    for i in range(steps + 1):
        t = i / steps

        # Quadratic bezier formula
        x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * control_x + t**2 * end[0]
        y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * control_y + t**2 * end[1]

        path.append((int(x), int(y)))

    return path
