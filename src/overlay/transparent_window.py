"""
Transparent overlay window for drawing detection boxes and target points.
Creates an always-on-top transparent window using tkinter.
FIXED VERSION: Enhanced for terminal execution on macOS.
"""

import platform
import tkinter as tk
from typing import List, Tuple, Optional, Dict, Any
from threading import Thread, Lock
import time
from loguru import logger

from ..detection.detector import Detection


class TransparentOverlay:
    """Transparent overlay window for visual feedback."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2,
        show_labels: bool = True,
        show_confidence: bool = True,
        target_point_color: Tuple[int, int, int] = (255, 0, 0),
        target_point_size: int = 5,
        monitor_x: int = 0,
        monitor_y: int = 0
    ):
        """
        Initialize transparent overlay.

        Args:
            width: Window width
            height: Window height
            box_color: RGB color for bounding boxes (0-255)
            box_thickness: Thickness of bounding box lines
            show_labels: Show class labels
            show_confidence: Show confidence scores
            target_point_color: RGB color for target points
            target_point_size: Size of target point markers
            monitor_x: X offset for monitor positioning
            monitor_y: Y offset for monitor positioning
        """
        self.width = width
        self.height = height
        self.monitor_x = monitor_x
        self.monitor_y = monitor_y
        self.box_color = self._rgb_to_hex(box_color)
        self.box_thickness = box_thickness
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.target_point_color = self._rgb_to_hex(target_point_color)
        self.target_point_size = target_point_size

        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.is_running = False
        self.is_visible = True

        self.detections: List[Detection] = []
        self.target_point: Optional[Tuple[int, int]] = None
        self.detections_lock = Lock()

        self.fps_display = True
        self.current_fps = 0
        self.show_help = True
        self.help_fade_time = 10  # Show help for 10 seconds
        self.start_time = None

        self.key_callbacks = {}

        logger.info("TransparentOverlay initialized")

    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color string."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _create_window(self):
        """Create the transparent overlay window."""
        self.root = tk.Tk()
        self.root.title("CV Overlay")

        # Remove window decorations
        self.root.overrideredirect(True)

        # Set window size and position
        self.root.geometry(f"{self.width}x{self.height}+{self.monitor_x}+{self.monitor_y}")

        # Platform-specific transparency and fullscreen overlay settings
        os_type = platform.system()

        if os_type == 'Darwin':  # macOS
            # ENHANCED: Force window to absolute top level
            self.root.attributes("-topmost", True)

            # Make fully transparent background
            self.root.wm_attributes("-transparent", True)
            self.root.config(bg='systemTransparent')
            canvas_bg = 'systemTransparent'

            # Multiple methods to ensure window stays on top
            self.root.lift()
            self.root.wm_attributes("-topmost", 1)

            # Force window to highest level (above fullscreen apps)
            self.root.call('wm', 'attributes', '.', '-topmost', '1')

            # ENHANCED: Set window level to floating panel (higher than normal windows)
            try:
                # Use Cocoa to set window level to floating panel
                from AppKit import NSApp, NSFloatingWindowLevel, NSScreenSaverWindowLevel
                from Cocoa import NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary, NSWindowCollectionBehaviorIgnoresCycle
                import objc

                # Update window multiple times to ensure it's created
                for _ in range(5):
                    self.root.update()
                    time.sleep(0.05)

                # Find our window and set its level
                for window in NSApp.windows():
                    window_title = window.title()
                    if window_title == "CV Overlay" or "CV Overlay" in str(window_title):
                        # Set window to floating panel level (stays above normal windows)
                        # Use NSFloatingWindowLevel for normal overlay
                        # Use NSScreenSaverWindowLevel for absolute top (even above menu bar)
                        window.setLevel_(NSScreenSaverWindowLevel)  # Maximum priority for Terminal.app

                        # Keep window on all spaces and stationary
                        window.setCollectionBehavior_(
                            NSWindowCollectionBehaviorCanJoinAllSpaces |
                            NSWindowCollectionBehaviorStationary |
                            NSWindowCollectionBehaviorIgnoresCycle
                        )

                        # Make sure window stays visible
                        window.orderFrontRegardless()

                        logger.debug("Window level set to floating panel")
                        break
            except ImportError:
                logger.warning("PyObjC not installed - overlay may not stay on top")
                logger.warning("Install with: pip install pyobjc-framework-Cocoa")
            except Exception as e:
                logger.error(f"Could not set window level: {e}")

        elif os_type == 'Windows':
            # Windows transparency and always on top
            self.root.attributes("-topmost", True)
            self.root.attributes("-alpha", 0.01)  # Near-transparent but not fully
            self.root.attributes("-transparentcolor", "black")
            self.root.config(bg='black')
            canvas_bg = 'black'

            # Windows-specific: Set to HWND_TOPMOST
            try:
                import ctypes
                import ctypes.wintypes
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                HWND_TOPMOST = -1
                SWP_NOSIZE = 0x0001
                ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE)
            except:
                pass

        else:  # Linux
            # Linux transparency (requires compositor)
            self.root.attributes("-topmost", True)
            try:
                self.root.wait_visibility()
                self.root.attributes("-alpha", 0.01)
            except:
                pass
            self.root.config(bg='black')
            canvas_bg = 'black'

        # Create canvas with transparent background
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=canvas_bg,
            highlightthickness=0
        )
        self.canvas.pack()

        # Make window click-through (platform-specific)
        self._make_clickthrough()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        logger.info(f"Transparent overlay created at +{self.monitor_x}+{self.monitor_y}")

    def _make_clickthrough(self):
        """Make window click-through (platform-specific)."""
        os_type = platform.system()

        if os_type == 'Darwin':  # macOS
            try:
                from AppKit import NSApp
                from Cocoa import NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary, NSWindowCollectionBehaviorIgnoresCycle
                import time

                # Update multiple times to ensure window is created
                for _ in range(5):
                    self.root.update()
                    time.sleep(0.05)

                # Try to find our window by title
                found = False
                for window in NSApp.windows():
                    window_title = window.title()

                    if window_title == "CV Overlay" or "CV Overlay" in str(window_title):
                        # Make window ignore mouse events (click-through)
                        window.setIgnoresMouseEvents_(True)

                        # Keep window on all spaces and stationary
                        window.setCollectionBehavior_(
                            NSWindowCollectionBehaviorCanJoinAllSpaces |
                            NSWindowCollectionBehaviorStationary |
                            NSWindowCollectionBehaviorIgnoresCycle
                        )

                        logger.debug("Overlay set to click-through mode")
                        found = True
                        break

                if not found:
                    logger.warning("Could not find CV Overlay window - click-through not enabled")

            except ImportError:
                logger.warning("PyObjC not installed - overlay will capture mouse clicks")
                logger.warning("Install with: pip install pyobjc-framework-Cocoa")
            except Exception as e:
                logger.error(f"Could not make overlay click-through: {e}")

        elif os_type == 'Windows':
            # Windows: Use win32api to make click-through
            try:
                import win32gui
                import win32con
                import win32api

                # Get window handle
                hwnd = self.root.winfo_id()

                # Set window to be click-through
                styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                styles |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)

                logger.debug("Overlay set to click-through mode (Windows)")

            except ImportError:
                logger.warning("pywin32 not installed - overlay will capture mouse clicks")
                logger.warning("Install with: pip install pywin32")
            except Exception as e:
                logger.error(f"Could not make overlay click-through: {e}")

        else:  # Linux
            # Linux: Limited click-through support
            try:
                # Some window managers support this
                self.root.attributes("-type", "notification")
            except:
                pass
            logger.info("Linux: Click-through support is limited, depends on window manager")

    def _draw_loop(self):
        """Main drawing loop."""
        if self.root is None or self.canvas is None:
            return

        while self.is_running:
            try:
                if self.is_visible:
                    self._update_canvas()

                # ENHANCED: Keep lifting window to top periodically
                if hasattr(self.root, 'lift'):
                    self.root.lift()
                    self.root.attributes("-topmost", True)

                # Run tkinter update
                self.root.update()
                time.sleep(0.016)  # ~60 FPS

            except tk.TclError:
                logger.info("Window closed")
                break
            except Exception as e:
                logger.error(f"Error in draw loop: {e}")
                break

    def _update_canvas(self):
        """Update canvas with current detections."""
        if self.canvas is None:
            return

        # Clear canvas (no background fill - stays transparent)
        self.canvas.delete("all")

        # Keep window on top (important for fullscreen apps)
        if hasattr(self.root, 'lift'):
            self.root.lift()

        with self.detections_lock:
            # Draw bounding boxes
            for detection in self.detections:
                self._draw_detection(detection)

            # Draw target point
            if self.target_point:
                self._draw_target_point(self.target_point)

        # Draw FPS
        if self.fps_display and self.current_fps > 0:
            self._draw_fps()

        # Draw help text (fades after 10 seconds)
        if self.show_help and self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed < self.help_fade_time:
                self._draw_help()
            else:
                self.show_help = False


    def _draw_detection(self, detection: Detection):
        """Draw a single detection."""
        if self.canvas is None:
            return

        x1, y1, x2, y2 = detection.bbox

        # Draw bounding box
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=self.box_color,
            width=self.box_thickness
        )

        # Draw label
        if self.show_labels or self.show_confidence:
            label_parts = []
            if self.show_labels:
                label_parts.append(detection.class_name)
            if self.show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")

            label = " ".join(label_parts)

            # Draw label background
            self.canvas.create_rectangle(
                x1, y1 - 20, x1 + len(label) * 8, y1,
                fill=self.box_color,
                outline=""
            )

            # Draw label text
            self.canvas.create_text(
                x1 + 5, y1 - 10,
                text=label,
                fill="black",
                anchor="w",
                font=("Arial", 10, "bold")
            )

    def _draw_target_point(self, point: Tuple[int, int]):
        """Draw target point marker."""
        if self.canvas is None:
            return

        x, y = point
        size = self.target_point_size

        # Draw crosshair
        self.canvas.create_line(
            x - size, y, x + size, y,
            fill=self.target_point_color,
            width=2
        )
        self.canvas.create_line(
            x, y - size, x, y + size,
            fill=self.target_point_color,
            width=2
        )

        # Draw circle
        self.canvas.create_oval(
            x - size, y - size, x + size, y + size,
            outline=self.target_point_color,
            width=2
        )

    def _draw_fps(self):
        """Draw FPS counter and status."""
        if self.canvas is None:
            return

        # FPS counter
        fps_text = f"FPS: {self.current_fps:.1f}"
        self.canvas.create_text(
            10, 10,
            text=fps_text,
            fill=self.box_color,
            anchor="nw",
            font=("Arial", 12, "bold")
        )

        # Detection count
        detection_text = f"Objects: {len(self.detections)}"
        self.canvas.create_text(
            10, 30,
            text=detection_text,
            fill=self.box_color,
            anchor="nw",
            font=("Arial", 12, "bold")
        )

        # Status indicator (always visible corner dot)
        self.canvas.create_oval(
            self.width - 20, 10, self.width - 10, 20,
            fill="#00ff00",
            outline=""
        )

    def _draw_help(self):
        """Draw hotkey help text."""
        if self.canvas is None:
            return

        help_text = [
            "GLOBAL HOTKEYS:",
            "F1 - Toggle Overlay",
            "F2 - Mouse Control",
            "F3 - Settings",
            "F5 - Quit"
        ]

        y_offset = 50
        for line in help_text:
            self.canvas.create_text(
                10, y_offset,
                text=line,
                fill="#00ff00" if "HOTKEYS" in line else "#ffffff",
                anchor="nw",
                font=("Arial", 11, "bold" if "HOTKEYS" in line else "normal")
            )
            y_offset += 20

    def start(self):
        """Start the overlay window."""
        if self.is_running:
            logger.warning("Overlay already running")
            return

        self.is_running = True
        self.start_time = time.time()

        # Create window in main thread
        self._create_window()

        # Start draw loop
        self._draw_loop()

    def stop(self):
        """Stop the overlay window."""
        if not self.is_running:
            return

        logger.info("Stopping overlay...")
        self.is_running = False

        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

        logger.info("Overlay stopped")

    def update_detections(self, detections: List[Detection], target_point: Optional[Tuple[int, int]] = None):
        """
        Update detections to display.

        Args:
            detections: List of Detection objects
            target_point: Optional target point coordinates
        """
        with self.detections_lock:
            self.detections = detections
            self.target_point = target_point

    def set_visibility(self, visible: bool):
        """Set overlay visibility."""
        self.is_visible = visible
        if self.root:
            if visible:
                self.root.deiconify()
            else:
                self.root.withdraw()

    def toggle_visibility(self):
        """Toggle overlay visibility."""
        self.is_visible = not self.is_visible
        self.set_visibility(self.is_visible)

    def set_fps(self, fps: float):
        """Set FPS for display."""
        self.current_fps = fps

    def set_box_color(self, color: Tuple[int, int, int]):
        """Set bounding box color."""
        self.box_color = self._rgb_to_hex(color)

    def set_target_point_color(self, color: Tuple[int, int, int]):
        """Set target point color."""
        self.target_point_color = self._rgb_to_hex(color)

    def is_active(self) -> bool:
        """Check if overlay is active."""
        return self.is_running and self.root is not None

    def bind_key(self, key: str, callback):
        """
        Bind keyboard shortcut to callback.

        Args:
            key: Key string (e.g., '<F1>', '<F2>')
            callback: Function to call when key is pressed
        """
        self.key_callbacks[key] = callback
        logger.info(f"Bound key {key} to callback")

    def _handle_keypress(self, event):
        """Handle keyboard events."""
        # Map tkinter keysyms to standard format
        key_map = {
            'F1': '<F1>',
            'F2': '<F2>',
            'F3': '<F3>',
            'F4': '<F4>',
            'F12': '<F12>'
        }

        key_str = key_map.get(event.keysym)
        if key_str and key_str in self.key_callbacks:
            logger.debug(f"Key pressed: {key_str}")
            try:
                self.key_callbacks[key_str]()
            except Exception as e:
                logger.error(f"Error in key callback: {e}")


def run_overlay_in_thread(overlay: TransparentOverlay):
    """
    Run overlay in a separate thread (for testing).

    Args:
        overlay: TransparentOverlay instance
    """
    thread = Thread(target=overlay.start, daemon=True)
    thread.start()
    return thread