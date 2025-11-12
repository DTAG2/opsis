"""
Improved hotkey manager with fallback mechanisms for VSCode compatibility.
Uses multiple approaches to ensure hotkeys work in all environments.
"""

import sys
import os
import threading
import queue
import platform
from typing import Dict, Callable, Optional
from loguru import logger

# Try multiple hotkey libraries for better compatibility
hotkey_backend = None

# Try pynput first (most compatible)
try:
    from pynput import keyboard
    hotkey_backend = "pynput"
except ImportError:
    pass

# Try keyboard as secondary option (better for VSCode)
if not hotkey_backend:
    try:
        import keyboard as kb
        hotkey_backend = "keyboard"
    except ImportError:
        pass

# Try system_hotkey as fallback (uses native APIs)
if not hotkey_backend:
    try:
        from system_hotkey import SystemHotkey
        hotkey_backend = "system_hotkey"
    except ImportError:
        pass


class HotkeyManager:
    """
    Robust hotkey manager with multiple backends and VSCode compatibility.
    """

    def __init__(self, main_thread_queue: Optional[queue.Queue] = None):
        """
        Initialize hotkey manager.

        Args:
            main_thread_queue: Queue for thread-safe communication with main thread
        """
        self.callbacks: Dict[str, Callable] = {}
        self.listener = None
        self.is_running = False
        self.main_queue = main_thread_queue or queue.Queue()
        self.backend = hotkey_backend
        self.keyboard_hooks = []  # For keyboard backend

        # For VSCode compatibility - track if we're in VSCode terminal
        self.in_vscode = 'VSCODE_PID' in os.environ or ('TERM_PROGRAM' in os.environ and 'vscode' in os.environ.get('TERM_PROGRAM', '').lower())

        # Prefer keyboard backend for VSCode
        if self.in_vscode and hotkey_backend == "pynput":
            try:
                import keyboard as kb
                self.backend = "keyboard"
                logger.info("VSCode detected - switching to keyboard backend for better compatibility")
            except ImportError:
                pass

        logger.info(f"HotkeyManager initialized with backend: {self.backend}")

    def register_hotkey(self, key_combo: str, callback: Callable, description: str = ""):
        """
        Register a hotkey with its callback.

        Args:
            key_combo: Key combination (e.g., 'f1', 'ctrl+shift+a')
            callback: Function to call when hotkey pressed
            description: Human-readable description
        """
        self.callbacks[key_combo] = {
            'callback': callback,
            'description': description
        }
        logger.debug(f"Registered hotkey: {key_combo} - {description}")

    def _queue_callback(self, callback: Callable):
        """Queue a callback for execution on main thread."""
        def wrapped():
            try:
                # Put callback in queue for main thread to execute
                self.main_queue.put(callback)
            except Exception as e:
                logger.error(f"Error queuing callback: {e}")
        return wrapped

    def start_pynput_listener(self):
        """Start pynput keyboard listener (most compatible)."""
        from pynput import keyboard

        # Map of pynput keys to our key names
        key_map = {
            keyboard.Key.f1: 'f1',
            keyboard.Key.f2: 'f2',
            keyboard.Key.f3: 'f3',
            keyboard.Key.f4: 'f4',
            keyboard.Key.f5: 'f5',
            keyboard.Key.f6: 'f6',
            keyboard.Key.f7: 'f7',
            keyboard.Key.f8: 'f8',
            keyboard.Key.f9: 'f9',
            keyboard.Key.f10: 'f10',
            keyboard.Key.f11: 'f11',
            keyboard.Key.f12: 'f12',
        }

        def on_press(key):
            try:
                # Check if it's a function key
                if key in key_map:
                    key_name = key_map[key]
                    if key_name in self.callbacks:
                        callback_info = self.callbacks[key_name]
                        # Queue for main thread execution
                        self.main_queue.put(callback_info['callback'])

            except Exception as e:
                logger.error(f"Error in hotkey handler: {e}")

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        logger.info("Pynput listener started")

    def start_system_hotkey_listener(self):
        """Start system_hotkey listener (uses native APIs)."""
        from system_hotkey import SystemHotkey

        self.listener = SystemHotkey()

        # Register all hotkeys
        for key_combo, callback_info in self.callbacks.items():
            try:
                # Convert key format
                if key_combo.startswith('f') and key_combo[1:].isdigit():
                    # Function keys
                    native_combo = [key_combo]
                else:
                    # Parse modifiers
                    native_combo = key_combo.split('+')

                # Register with native API
                self.listener.register(
                    native_combo,
                    callback=lambda x: self.main_queue.put(callback_info['callback'])
                )
                logger.debug(f"Registered native hotkey: {native_combo}")

            except Exception as e:
                logger.warning(f"Failed to register {key_combo}: {e}")

    def start_keyboard_listener(self):
        """Start keyboard backend listener (better for VSCode)."""
        import keyboard as kb

        # Register all hotkeys
        for key_combo, callback_info in self.callbacks.items():
            try:
                # Register with keyboard module
                hook_id = kb.add_hotkey(key_combo, lambda cb=callback_info['callback']: self.main_queue.put(cb))
                self.keyboard_hooks.append(hook_id)
                logger.debug(f"Registered keyboard hotkey: {key_combo}")
            except Exception as e:
                logger.warning(f"Failed to register {key_combo}: {e}")

        logger.info("Keyboard backend listener started")

    def start(self):
        """Start the appropriate hotkey listener."""
        if self.is_running:
            logger.warning("HotkeyManager already running")
            return

        self.is_running = True

        if self.backend == "pynput":
            self.start_pynput_listener()
        elif self.backend == "keyboard":
            self.start_keyboard_listener()
        elif self.backend == "system_hotkey":
            self.start_system_hotkey_listener()
        else:
            logger.error("No hotkey backend available!")
            logger.error("Install pynput: pip install pynput")
            logger.error("Or install keyboard: pip install keyboard")
            return

        # Start queue processor thread
        self.queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_thread.start()

        logger.info("HotkeyManager started successfully")

    def _process_queue(self):
        """Process callbacks from queue (runs in separate thread)."""
        while self.is_running:
            try:
                # Get callback from queue with timeout
                callback = self.main_queue.get(timeout=0.1)
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error executing callback: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processor error: {e}")

    def stop(self):
        """Stop the hotkey listener."""
        if not self.is_running:
            return

        self.is_running = False

        if self.backend == "pynput" and self.listener:
            self.listener.stop()
        elif self.backend == "keyboard":
            # Unregister keyboard hooks
            import keyboard as kb
            for hook_id in self.keyboard_hooks:
                try:
                    kb.remove_hotkey(hook_id)
                except:
                    pass
            self.keyboard_hooks.clear()
        elif self.backend == "system_hotkey" and self.listener:
            # Unregister all hotkeys
            for key_combo in self.callbacks:
                try:
                    self.listener.unregister(key_combo.split('+'))
                except:
                    pass

        logger.info("HotkeyManager stopped")

    def get_registered_hotkeys(self) -> Dict[str, str]:
        """Get list of registered hotkeys and their descriptions."""
        return {
            key: info['description']
            for key, info in self.callbacks.items()
        }


def create_hotkey_manager(callbacks: Dict[str, tuple], main_queue: Optional[queue.Queue] = None) -> HotkeyManager:
    """
    Convenience function to create and configure a hotkey manager.

    Args:
        callbacks: Dict of key_combo -> (callback, description) tuples
        main_queue: Optional queue for main thread communication

    Returns:
        Configured HotkeyManager instance
    """
    manager = HotkeyManager(main_queue)

    for key_combo, (callback, description) in callbacks.items():
        manager.register_hotkey(key_combo, callback, description)

    return manager