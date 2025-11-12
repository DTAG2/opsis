#!/usr/bin/env python3
"""
CV Overlay System - Main Entry Point
Real-time computer vision with transparent overlay and mouse control.
"""

import os
import sys
import time
import queue
import platform
import argparse
from pathlib import Path
from threading import Thread
from loguru import logger

from src.utils.config_loader import get_config
from src.utils.hotkey_manager import create_hotkey_manager
from src.capture.screen_grabber import ScreenGrabber
from src.detection.model_loader import ModelLoader, check_gpu_availability
from src.detection.detector import Detector
from src.overlay.transparent_window import TransparentOverlay
from src.control.mouse_controller import SmoothMouseController
from src.gui.menu import SettingsMenu


class CVOverlaySystem:
    """Main system orchestrator."""

    def __init__(self, config_path: str = None, monitor: int = None, quiet_mode: bool = False):
        """Initialize CV overlay system."""
        self.config = get_config()
        if config_path:
            self.config.config_path = Path(config_path)
            self.config.load()

        self.monitor_override = monitor
        self.quiet_mode = quiet_mode  # Suppress verbose logging

        self.screen_grabber: ScreenGrabber = None
        self.model_loader: ModelLoader = None
        self.detector: Detector = None
        self.overlay: TransparentOverlay = None
        self.mouse_controller: SmoothMouseController = None
        self.menu: SettingsMenu = None
        self.hotkey_manager = None
        self.main_queue = queue.Queue()
        self.gui_queue = queue.Queue()  # Separate queue for GUI operations

        self.is_running = False
        self.overlay_enabled = True
        self.mouse_control_enabled = False

        # Store monitor offset for coordinate conversion
        self.monitor_offset_x = 0
        self.monitor_offset_y = 0

        # Platform detection for cross-platform compatibility
        self.os_type = platform.system()  # 'Windows', 'Linux', 'Darwin' (macOS)
        self.is_vscode = 'TERM_PROGRAM' in os.environ and 'vscode' in os.environ.get('TERM_PROGRAM', '').lower()

        if not self.quiet_mode:
            logger.info(f"System initialized on {self.os_type}")

    def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            if not self.quiet_mode:
                logger.info("Initializing components...")

            # Check GPU availability
            gpu_info = check_gpu_availability()
            if not self.quiet_mode:
                logger.debug(f"GPU Info: {gpu_info}")

            # Initialize screen grabber
            capture_config = self.config.get("screen_capture", {})
            monitor_index = self.monitor_override if self.monitor_override is not None else capture_config.get("capture_monitor", 1)

            self.screen_grabber = ScreenGrabber(
                monitor=monitor_index,
                target_fps=capture_config.get("target_fps", 30),
                region=capture_config.get("capture_region")
            )

            # Get monitor info for overlay positioning
            monitor_info = self.screen_grabber.get_monitor_info()
            monitors = monitor_info['monitors']

            if not self.quiet_mode:
                logger.debug(f"Available monitors: {len(monitors) - 1}")

            if monitor_index >= len(monitors):
                logger.warning(f"Monitor {monitor_index} not found, using monitor 1")
                monitor_index = 1

            selected_monitor = monitors[monitor_index]

            # Store monitor offset for coordinate conversion
            self.monitor_offset_x = selected_monitor['left']
            self.monitor_offset_y = selected_monitor['top']
            if not self.quiet_mode:
                logger.debug(f"Monitor offset: ({self.monitor_offset_x}, {self.monitor_offset_y})")

            # Initialize model loader
            detection_config = self.config.get("detection", {})
            model_path = detection_config.get("model_path", "models/default.pt")

            if not Path(model_path).exists():
                logger.warning(f"Model not found at {model_path}")
                logger.warning("Train a model first or update model_path in config/settings.json")
                return False

            self.model_loader = ModelLoader(
                model_path=model_path,
                device=detection_config.get("inference_device", "cuda"),
                half_precision=detection_config.get("half_precision", False)
            )

            if not self.model_loader.load():
                logger.error("Failed to load model")
                return False

            # Initialize detector
            self.detector = Detector(
                model_loader=self.model_loader,
                confidence_threshold=detection_config.get("confidence_threshold", 0.5),
                iou_threshold=detection_config.get("iou_threshold", 0.45),
                image_size=detection_config.get("image_size", 640),
                target_classes=detection_config.get("target_classes")
            )

            # Initialize overlay with monitor positioning
            overlay_config = self.config.get("overlay", {})

            self.overlay = TransparentOverlay(
                width=selected_monitor["width"],
                height=selected_monitor["height"],
                monitor_x=selected_monitor["left"],
                monitor_y=selected_monitor["top"],
                box_color=tuple(overlay_config.get("box_color", [0, 255, 0])),
                box_thickness=overlay_config.get("box_thickness", 2),
                show_labels=overlay_config.get("show_labels", True),
                show_confidence=overlay_config.get("show_confidence", True),
                target_point_color=tuple(overlay_config.get("target_point_color", [255, 0, 0])),
                target_point_size=overlay_config.get("target_point_size", 5)
            )

            # Note: Global keyboard shortcuts set up in start() using pynput

            # Initialize mouse controller
            mouse_config = self.config.get("mouse_control", {})
            self.mouse_controller = SmoothMouseController(
                smoothing_factor=mouse_config.get("smoothing_factor", 0.3),
                max_move_speed=mouse_config.get("max_move_speed", 100),
                enable_stabilization=mouse_config.get("enable_stabilization", True),
                stabilization_strength=mouse_config.get("stabilization_strength", 0.5),
                aim_offset=(mouse_config.get("aim_offset_x", 0), mouse_config.get("aim_offset_y", 0))
            )

            # Set initial state
            self.overlay_enabled = overlay_config.get("enabled", True)
            self.mouse_control_enabled = mouse_config.get("enabled", False)

            if self.mouse_control_enabled:
                self.mouse_controller.enable()

            # Initialize menu (but don't create window yet - must be done on main thread)
            self.menu = SettingsMenu(
                config=self.config.config,
                on_settings_changed=self._on_settings_changed
            )

            if not self.quiet_mode:
                logger.info("Components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            if not self.quiet_mode:
                import traceback
                traceback.print_exc()
            return False

    def start(self):
        """Start the system."""
        if not self.quiet_mode:
            logger.info("Starting...")

        if not self.initialize_components():
            logger.error("Failed to initialize")
            return

        self.is_running = True
        self.screen_grabber.start()

        # Setup improved hotkey manager for better VSCode compatibility
        hotkeys = {
            'f1': (self._toggle_overlay, "Toggle Overlay"),
            'f2': (self._toggle_mouse_control, "Toggle Mouse Control"),
            'f3': (self._toggle_menu, "Settings Menu"),
            'f4': (self._reload_config, "Reload Config"),
        }

        # Use F5 on macOS, F12 on Windows/Linux for exit
        if self.os_type == 'Darwin':
            hotkeys['f5'] = (self._exit_program, "Exit Program")
        else:
            hotkeys['f12'] = (self._exit_program, "Exit Program")

        self.hotkey_manager = create_hotkey_manager(hotkeys, self.main_queue)
        self.hotkey_manager.start()

        print()
        print("=" * 70)
        print("★★★ CV OVERLAY SYSTEM READY ★★★")
        print()
        print("  F1  →  Toggle Overlay")
        print("  F2  →  Toggle Mouse Control")
        print("  F3  →  Settings Menu")
        print("  F4  →  Reload Config")
        if self.os_type == 'Darwin':
            print("  F5  →  Exit")
        else:
            print("  F12 →  Exit")

        if self.is_vscode:
            print()
            print("  ⚠️  VSCode terminal detected - hotkeys may work better in native terminal")

        print()
        print("=" * 70)
        print()

        # Start main detection loop
        main_thread = Thread(target=self._main_loop, daemon=True)
        main_thread.start()

        # Start hotkey queue processor
        queue_thread = Thread(target=self._process_hotkey_queue, daemon=True)
        queue_thread.start()

        if not self.quiet_mode:
            logger.info("System ready!")

        # Process GUI events on main thread before starting overlay
        self._setup_gui_processor()

        # Run overlay on main thread (tkinter requirement)
        self.overlay.start()
        self.stop()

    def stop(self):
        """Stop the system."""
        if not self.is_running:
            return

        if not self.quiet_mode:
            logger.info("Stopping...")
        self.is_running = False

        if self.hotkey_manager:
            self.hotkey_manager.stop()
        if self.screen_grabber:
            self.screen_grabber.stop()
        if self.overlay:
            self.overlay.stop()
        if self.menu:
            self.menu.destroy()

        if not self.quiet_mode:
            logger.info("System stopped")

    def _setup_gui_processor(self):
        """Setup GUI event processing on main thread."""
        if self.overlay and self.overlay.root:
            # Pre-create settings menu window on main thread
            if self.menu and not self.menu.window:
                try:
                    self.menu.create_window()
                    self.menu.hide()  # Hide it immediately
                except Exception as e:
                    logger.error(f"Error creating settings window: {e}")

            # Schedule regular GUI queue processing
            self._process_gui_queue()

    def _process_gui_queue(self):
        """Process GUI operations on main thread."""
        if not self.is_running or not self.overlay or not self.overlay.root:
            return

        try:
            # Process all pending GUI operations
            while True:
                try:
                    operation = self.gui_queue.get_nowait()
                    if operation:
                        operation()
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"GUI queue processing error: {e}")
            import traceback
            traceback.print_exc()

        # Schedule next check
        if self.overlay and self.overlay.root:
            self.overlay.root.after(50, self._process_gui_queue)  # Check every 50ms

    def _process_hotkey_queue(self):
        """Process hotkey callbacks from queue (thread-safe)."""
        while self.is_running:
            try:
                # Get callback from queue with timeout
                callback = self.main_queue.get(timeout=0.1)
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error executing hotkey callback: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                if not self.quiet_mode:
                    logger.error(f"Queue processor error: {e}")

    def _main_loop(self):
        """Main detection loop."""
        if not self.quiet_mode:
            logger.debug("Detection loop started")

        frame_count = 0
        detection_count = 0
        last_status_time = time.time()

        while self.is_running:
            try:
                frame = self.screen_grabber.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                detections = self.detector.detect(frame)

                if detections:
                    detection_count += 1
                    # Debug: Log when detections are found
                    if not self.quiet_mode:
                        logger.debug(f"Found {len(detections)} detections")

                target_point = None
                if detections:
                    priority = self.config.get("mouse_control", {}).get("target_priority", "head")
                    target_point = self.detector.get_best_target(detections, priority)

                if self.overlay_enabled:
                    self.overlay.update_detections(detections, target_point)
                    self.overlay.set_fps(self.detector.get_fps())

                if self.mouse_control_enabled and target_point:
                    # Convert from frame-relative to absolute screen coordinates
                    absolute_target = (
                        target_point[0] + self.monitor_offset_x,
                        target_point[1] + self.monitor_offset_y
                    )
                    self.mouse_controller.track_target(absolute_target)

                # Only show status every 30 seconds in quiet mode, or 5 seconds in normal mode
                current_time = time.time()
                status_interval = 30 if self.quiet_mode else 5
                if current_time - last_status_time > status_interval:
                    if not self.quiet_mode:
                        logger.debug(f"Status: FPS={self.detector.get_fps():.1f}, Detections={detection_count}")
                    last_status_time = current_time

            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                if not self.quiet_mode:
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)

        if not self.quiet_mode:
            logger.debug("Detection loop stopped")

    def _toggle_overlay(self):
        """Toggle overlay visibility."""
        try:
            self.overlay_enabled = not self.overlay_enabled
            status = "ON" if self.overlay_enabled else "OFF"
            print(f"\n→ Overlay {status}")

            # Queue GUI operation for main thread
            if self.overlay:
                self.gui_queue.put(lambda: self.overlay.set_visibility(self.overlay_enabled))
        except Exception as e:
            logger.error(f"Error toggling overlay: {e}")

    def _toggle_mouse_control(self):
        """Toggle mouse control."""
        try:
            self.mouse_control_enabled = not self.mouse_control_enabled
            if self.mouse_controller:
                if self.mouse_control_enabled:
                    self.mouse_controller.enable()
                else:
                    self.mouse_controller.disable()
            status = "ON" if self.mouse_control_enabled else "OFF"
            print(f"\n→ Mouse Control {status}")
        except Exception as e:
            logger.error(f"Error toggling mouse control: {e}")

    def _toggle_menu(self):
        """Toggle settings menu."""
        print(f"\n→ Settings Menu toggled")

        # Queue GUI operation for main thread
        if self.menu:
            self.gui_queue.put(lambda: self.menu.toggle())

    def _reload_config(self):
        """Reload configuration."""
        print(f"\n→ Reloading configuration...")
        self.config.reload()
        self._apply_config()
        print(f"→ Configuration reloaded")

    def _exit_program(self):
        """Exit program."""
        print(f"\n→ Exiting...")

        # Queue stop operation for main thread
        if self.overlay and self.overlay.root:
            self.gui_queue.put(lambda: self.stop())
        else:
            self.stop()

    def _on_settings_changed(self, key: str, value):
        """Handle settings changes."""
        if key == "save":
            self.config.save()
            if not self.quiet_mode:
                logger.debug("Config saved")
        elif key == "reload":
            self.config.reload()
            self._apply_config()
        else:
            self.config.set(key, value)
            self._apply_config()

    def _apply_config(self):
        """Apply config changes."""
        try:
            if self.detector:
                conf = self.config.get("detection", {}).get("confidence_threshold", 0.5)
                self.detector.set_confidence_threshold(conf)

            if self.mouse_controller:
                mc = self.config.get("mouse_control", {})
                self.mouse_controller.set_smoothing(mc.get("smoothing_factor", 0.3))
                self.mouse_controller.set_max_speed(mc.get("max_move_speed", 100))
                self.mouse_controller.set_stabilization_strength(mc.get("stabilization_strength", 0.5))

            if self.overlay:
                color = tuple(self.config.get("overlay", {}).get("box_color", [0, 255, 0]))
                self.overlay.set_box_color(color)

            if not self.quiet_mode:
                logger.debug("Config applied")
        except Exception as e:
            logger.error(f"Config error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CV Overlay System - Cross-platform computer vision overlay")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--monitor", type=int, help="Monitor to use (1=primary, 2=secondary, etc)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up logging with appropriate level
    quiet_mode = not (args.verbose or args.debug)
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")

    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level
    )

    # Platform-specific setup instructions
    os_type = platform.system()

    if os_type == 'Darwin':  # macOS
        setup_hint = "• System Preferences → Security & Privacy → Accessibility\n  • Add Terminal/iTerm and enable it"
    elif os_type == 'Windows':
        setup_hint = "• Run as Administrator for best performance\n  • Windows Defender may need exception for this app"
    else:  # Linux
        setup_hint = "• May require sudo for input device access\n  • Install python3-tk if overlay doesn't appear"

    print()
    print("="*70)
    print("               CV OVERLAY SYSTEM")
    print("="*70)
    print()

    if 'TERM_PROGRAM' in os.environ and 'vscode' in os.environ.get('TERM_PROGRAM', '').lower():
        print("⚠️  VSCode Terminal Detected")
        print("   For best hotkey support, consider using native terminal")
        print()

    print(f"Platform: {os_type}")
    print()
    print("SETUP:")
    print(setup_hint)
    print("="*70)

    try:
        system = CVOverlaySystem(config_path=args.config, monitor=args.monitor, quiet_mode=quiet_mode)
        system.start()
    except KeyboardInterrupt:
        if not quiet_mode:
            logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
