#!/usr/bin/env python3
"""
Data collection script for gathering training images.
Captures screenshots at regular intervals for model training.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import argparse
from datetime import datetime
from pynput import keyboard
from loguru import logger

from src.capture.screen_grabber import ScreenGrabber


class DataCollector:
    """Collects screenshots for training data."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        game_name: str = "default",
        capture_interval: float = 1.0,
        monitor: int = 0
    ):
        """
        Initialize data collector.

        Args:
            output_dir: Directory to save screenshots
            game_name: Name of game/scenario for organization
            capture_interval: Seconds between captures
            monitor: Monitor to capture from
        """
        self.output_dir = Path(output_dir) / game_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.game_name = game_name
        self.capture_interval = capture_interval
        self.monitor = monitor

        self.screen_grabber = ScreenGrabber(monitor=monitor)
        self.is_collecting = False
        self.is_paused = False

        self.image_count = 0
        self.session_start = None

        logger.info(f"DataCollector initialized: output={self.output_dir}")

    def start_collection(self):
        """Start collecting screenshots."""
        logger.info("Starting data collection...")
        logger.info("Press SPACE to pause/resume, ESC to stop")

        self.is_collecting = True
        self.is_paused = False
        self.session_start = datetime.now()
        self.image_count = 0

        # Start screen grabber
        self.screen_grabber.start()

        # Setup keyboard listener
        listener = keyboard.Listener(
            on_press=self._on_key_press
        )
        listener.start()

        # Collection loop
        last_capture = 0
        try:
            while self.is_collecting:
                current_time = time.time()

                if not self.is_paused and (current_time - last_capture) >= self.capture_interval:
                    self._capture_and_save()
                    last_capture = current_time

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")

        finally:
            self.screen_grabber.stop()
            listener.stop()

        self._print_summary()

    def _capture_and_save(self):
        """Capture and save a screenshot."""
        frame = self.screen_grabber.get_frame()

        if frame is None:
            return

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.game_name}_{timestamp}.jpg"
        filepath = self.output_dir / filename

        # Save image
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.image_count += 1
        logger.info(f"Captured image {self.image_count}: {filename}")

    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.space:
                self.is_paused = not self.is_paused
                status = "paused" if self.is_paused else "resumed"
                logger.info(f"Collection {status}")

            elif key == keyboard.Key.esc:
                logger.info("Stopping collection...")
                self.is_collecting = False

        except AttributeError:
            pass

    def _print_summary(self):
        """Print collection summary."""
        if self.session_start:
            duration = (datetime.now() - self.session_start).total_seconds()
            logger.info("\n" + "="*50)
            logger.info("DATA COLLECTION SUMMARY")
            logger.info("="*50)
            logger.info(f"Game/Scenario: {self.game_name}")
            logger.info(f"Images collected: {self.image_count}")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect training data screenshots")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for screenshots"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="default",
        help="Game/scenario name for organization"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Capture interval in seconds"
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=0,
        help="Monitor to capture (0 for all monitors)"
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    # Print instructions
    print("\n" + "="*60)
    print("DATA COLLECTION TOOL")
    print("="*60)
    print(f"Game/Scenario: {args.game}")
    print(f"Capture Interval: {args.interval}s")
    print(f"Output Directory: {args.output}/{args.game}")
    print("\nControls:")
    print("  SPACE - Pause/Resume collection")
    print("  ESC   - Stop and exit")
    print("="*60 + "\n")

    input("Press ENTER to start collection...")

    # Create collector and start
    collector = DataCollector(
        output_dir=args.output,
        game_name=args.game,
        capture_interval=args.interval,
        monitor=args.monitor
    )

    collector.start_collection()


if __name__ == "__main__":
    main()
