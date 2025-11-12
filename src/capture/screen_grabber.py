"""
High-performance screen capture module.
Supports real-time screen grabbing with configurable FPS and regions.
"""

import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import mss
import cv2
from threading import Thread, Lock
from loguru import logger


class ScreenGrabber:
    """High-performance screen capture using MSS library."""

    def __init__(self, monitor: int = 0, target_fps: int = 60, region: Optional[Dict[str, int]] = None):
        """
        Initialize screen grabber.

        Args:
            monitor: Monitor index to capture (0 for all monitors, 1+ for specific monitor)
            target_fps: Target frames per second for capture
            region: Optional region dict with keys: left, top, width, height
        """
        self.monitor = monitor
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.region = region

        self.sct = mss.mss()
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = Lock()

        self.is_running = False
        self.capture_thread: Optional[Thread] = None

        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.current_fps = 0

        logger.info(f"ScreenGrabber initialized: monitor={monitor}, target_fps={target_fps}")

    def get_monitor_info(self) -> Dict[str, Any]:
        """
        Get information about available monitors.

        Returns:
            Dictionary with monitor information
        """
        monitors = self.sct.monitors
        logger.info(f"Available monitors: {len(monitors) - 1}")  # -1 because index 0 is all monitors
        return {
            'count': len(monitors) - 1,
            'monitors': monitors
        }

    def _get_capture_region(self) -> Dict[str, int]:
        """
        Get the capture region based on configuration.

        Returns:
            Dictionary defining the capture region
        """
        if self.region:
            return self.region

        # Use specified monitor or all monitors
        monitor = self.sct.monitors[self.monitor]
        return {
            "top": monitor["top"],
            "left": monitor["left"],
            "width": monitor["width"],
            "height": monitor["height"]
        }

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the screen.

        Returns:
            Numpy array containing the frame in BGR format (OpenCV format)
        """
        region = self._get_capture_region()

        # Capture screenshot
        screenshot = self.sct.grab(region)

        # Convert to numpy array
        frame = np.array(screenshot)

        # Convert from BGRA to BGR (remove alpha channel)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def _capture_loop(self):
        """Internal capture loop running in separate thread."""
        logger.info("Screen capture loop started")

        while self.is_running:
            loop_start = time.time()

            try:
                # Capture frame
                frame = self.capture_frame()

                # Update current frame (thread-safe)
                with self.frame_lock:
                    self.current_frame = frame

                # Update FPS counter
                self._update_fps()

                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Prevent tight error loop

        logger.info("Screen capture loop stopped")

    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_last_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_last_time = current_time

    def start(self):
        """Start continuous screen capture in background thread."""
        if self.is_running:
            logger.warning("Screen capture already running")
            return

        self.is_running = True
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Screen capture started")

    def stop(self):
        """Stop screen capture."""
        if not self.is_running:
            return

        logger.info("Stopping screen capture...")
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None

        logger.info("Screen capture stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent captured frame.

        Returns:
            Numpy array containing the frame, or None if no frame available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_fps(self) -> int:
        """
        Get current capture FPS.

        Returns:
            Current FPS
        """
        return self.current_fps

    def set_region(self, region: Optional[Dict[str, int]]):
        """
        Set capture region.

        Args:
            region: Region dict with keys: left, top, width, height, or None for full screen
        """
        self.region = region
        logger.info(f"Capture region updated: {region}")

    def set_target_fps(self, fps: int):
        """
        Set target capture FPS.

        Args:
            fps: Target frames per second
        """
        self.target_fps = fps
        self.frame_time = 1.0 / fps
        logger.info(f"Target FPS updated: {fps}")

    def get_frame_dimensions(self) -> Optional[Tuple[int, int]]:
        """
        Get dimensions of captured frames.

        Returns:
            Tuple of (width, height) or None
        """
        with self.frame_lock:
            if self.current_frame is not None:
                h, w = self.current_frame.shape[:2]
                return (w, h)
        return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        if self.sct:
            self.sct.close()
