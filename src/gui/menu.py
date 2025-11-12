"""
GUI menu system for configuring and controlling the CV overlay.
Provides a user-friendly interface for adjusting settings.
"""

import customtkinter as ctk
from typing import Callable, Dict, Any, Optional
from pathlib import Path
from loguru import logger


class SettingsMenu:
    """Settings menu GUI using customtkinter."""

    def __init__(
        self,
        config: Dict[str, Any],
        on_settings_changed: Optional[Callable[[str, Any], None]] = None
    ):
        """
        Initialize settings menu.

        Args:
            config: Configuration dictionary
            on_settings_changed: Callback when settings change (key, value)
        """
        self.config = config
        self.on_settings_changed = on_settings_changed

        self.window: Optional[ctk.CTk] = None
        self.is_visible = False

        # Setting widgets
        self.widgets: Dict[str, Any] = {}

        logger.info("SettingsMenu initialized")

    def create_window(self):
        """Create the settings window."""
        if self.window is not None:
            return

        # Create window
        self.window = ctk.CTk()
        self.window.title("CV Overlay - Settings")
        self.window.geometry("600x800")

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create main container with scrollbar
        self.main_frame = ctk.CTkScrollableFrame(self.window, width=560, height=760)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Create sections
        self._create_general_section()
        self._create_detection_section()
        self._create_overlay_section()
        self._create_mouse_control_section()
        self._create_capture_section()

        # Create action buttons
        self._create_action_buttons()

        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.hide)

        logger.info("Settings window created")

    def _create_section_label(self, text: str):
        """Create a section label."""
        label = ctk.CTkLabel(
            self.main_frame,
            text=text,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        label.pack(anchor="w", padx=10, pady=(20, 10))
        return label

    def _create_general_section(self):
        """Create general settings section."""
        self._create_section_label("General Settings")

        # Debug mode
        debug_var = ctk.BooleanVar(value=self.config.get("general", {}).get("debug_mode", False))
        debug_check = ctk.CTkCheckBox(
            self.main_frame,
            text="Debug Mode",
            variable=debug_var,
            command=lambda: self._on_setting_changed("general.debug_mode", debug_var.get())
        )
        debug_check.pack(anchor="w", padx=20, pady=5)
        self.widgets["debug_mode"] = debug_var

    def _create_detection_section(self):
        """Create detection settings section."""
        self._create_section_label("Detection Settings")

        # Confidence threshold
        conf_label = ctk.CTkLabel(self.main_frame, text="Confidence Threshold:")
        conf_label.pack(anchor="w", padx=20, pady=(10, 0))

        conf_value = self.config.get("detection", {}).get("confidence_threshold", 0.5)
        conf_slider = ctk.CTkSlider(
            self.main_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=90,
            command=lambda v: self._on_setting_changed("detection.confidence_threshold", v)
        )
        conf_slider.set(conf_value)
        conf_slider.pack(anchor="w", padx=20, pady=5, fill="x")
        self.widgets["confidence"] = conf_slider

        # Confidence value label
        conf_value_label = ctk.CTkLabel(self.main_frame, text=f"{conf_value:.2f}")
        conf_value_label.pack(anchor="w", padx=20)
        self.widgets["confidence_label"] = conf_value_label

        # Model path
        model_label = ctk.CTkLabel(self.main_frame, text="Model Path:")
        model_label.pack(anchor="w", padx=20, pady=(10, 0))

        model_value = self.config.get("detection", {}).get("model_path", "models/default.pt")
        model_entry = ctk.CTkEntry(self.main_frame, width=400)
        model_entry.insert(0, model_value)
        model_entry.pack(anchor="w", padx=20, pady=5)
        self.widgets["model_path"] = model_entry

        model_button = ctk.CTkButton(
            self.main_frame,
            text="Browse",
            width=100,
            command=self._browse_model
        )
        model_button.pack(anchor="w", padx=20, pady=5)

    def _create_overlay_section(self):
        """Create overlay settings section."""
        self._create_section_label("Overlay Settings")

        # Enable overlay
        overlay_var = ctk.BooleanVar(value=self.config.get("overlay", {}).get("enabled", True))
        overlay_check = ctk.CTkCheckBox(
            self.main_frame,
            text="Enable Overlay",
            variable=overlay_var,
            command=lambda: self._on_setting_changed("overlay.enabled", overlay_var.get())
        )
        overlay_check.pack(anchor="w", padx=20, pady=5)
        self.widgets["overlay_enabled"] = overlay_var

        # Show labels
        labels_var = ctk.BooleanVar(value=self.config.get("overlay", {}).get("show_labels", True))
        labels_check = ctk.CTkCheckBox(
            self.main_frame,
            text="Show Labels",
            variable=labels_var,
            command=lambda: self._on_setting_changed("overlay.show_labels", labels_var.get())
        )
        labels_check.pack(anchor="w", padx=20, pady=5)
        self.widgets["show_labels"] = labels_var

        # Show confidence
        conf_var = ctk.BooleanVar(value=self.config.get("overlay", {}).get("show_confidence", True))
        conf_check = ctk.CTkCheckBox(
            self.main_frame,
            text="Show Confidence",
            variable=conf_var,
            command=lambda: self._on_setting_changed("overlay.show_confidence", conf_var.get())
        )
        conf_check.pack(anchor="w", padx=20, pady=5)
        self.widgets["show_confidence"] = conf_var

        # Box thickness
        thickness_label = ctk.CTkLabel(self.main_frame, text="Box Thickness:")
        thickness_label.pack(anchor="w", padx=20, pady=(10, 0))

        thickness_value = self.config.get("overlay", {}).get("box_thickness", 2)
        thickness_slider = ctk.CTkSlider(
            self.main_frame,
            from_=1,
            to=5,
            number_of_steps=4,
            command=lambda v: self._on_setting_changed("overlay.box_thickness", int(v))
        )
        thickness_slider.set(thickness_value)
        thickness_slider.pack(anchor="w", padx=20, pady=5, fill="x")
        self.widgets["box_thickness"] = thickness_slider

    def _create_mouse_control_section(self):
        """Create mouse control settings section."""
        self._create_section_label("Mouse Control Settings")

        # Enable mouse control
        mouse_var = ctk.BooleanVar(value=self.config.get("mouse_control", {}).get("enabled", False))
        mouse_check = ctk.CTkCheckBox(
            self.main_frame,
            text="Enable Mouse Control (Use with caution!)",
            variable=mouse_var,
            command=lambda: self._on_setting_changed("mouse_control.enabled", mouse_var.get())
        )
        mouse_check.pack(anchor="w", padx=20, pady=5)
        self.widgets["mouse_enabled"] = mouse_var

        # Smoothing factor
        smooth_label = ctk.CTkLabel(self.main_frame, text="Smoothing Factor:")
        smooth_label.pack(anchor="w", padx=20, pady=(10, 0))

        smooth_value = self.config.get("mouse_control", {}).get("smoothing_factor", 0.3)
        smooth_slider = ctk.CTkSlider(
            self.main_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=lambda v: self._on_setting_changed("mouse_control.smoothing_factor", v)
        )
        smooth_slider.set(smooth_value)
        smooth_slider.pack(anchor="w", padx=20, pady=5, fill="x")
        self.widgets["smoothing"] = smooth_slider

        # Max move speed
        speed_label = ctk.CTkLabel(self.main_frame, text="Max Move Speed (pixels):")
        speed_label.pack(anchor="w", padx=20, pady=(10, 0))

        speed_value = self.config.get("mouse_control", {}).get("max_move_speed", 100)
        speed_slider = ctk.CTkSlider(
            self.main_frame,
            from_=10,
            to=500,
            number_of_steps=49,
            command=lambda v: self._on_setting_changed("mouse_control.max_move_speed", int(v))
        )
        speed_slider.set(speed_value)
        speed_slider.pack(anchor="w", padx=20, pady=5, fill="x")
        self.widgets["max_speed"] = speed_slider

        # Enable stabilization
        stab_var = ctk.BooleanVar(value=self.config.get("mouse_control", {}).get("enable_stabilization", True))
        stab_check = ctk.CTkCheckBox(
            self.main_frame,
            text="Enable Weapon Stabilization",
            variable=stab_var,
            command=lambda: self._on_setting_changed("mouse_control.enable_stabilization", stab_var.get())
        )
        stab_check.pack(anchor="w", padx=20, pady=5)
        self.widgets["stabilization"] = stab_var

        # Stabilization strength
        stab_strength_label = ctk.CTkLabel(self.main_frame, text="Stabilization Strength:")
        stab_strength_label.pack(anchor="w", padx=20, pady=(10, 0))

        stab_strength_value = self.config.get("mouse_control", {}).get("stabilization_strength", 0.5)
        stab_strength_slider = ctk.CTkSlider(
            self.main_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=lambda v: self._on_setting_changed("mouse_control.stabilization_strength", v)
        )
        stab_strength_slider.set(stab_strength_value)
        stab_strength_slider.pack(anchor="w", padx=20, pady=5, fill="x")
        self.widgets["stab_strength"] = stab_strength_slider

    def _create_capture_section(self):
        """Create screen capture settings section."""
        self._create_section_label("Screen Capture Settings")

        # Target FPS
        fps_label = ctk.CTkLabel(self.main_frame, text="Target FPS:")
        fps_label.pack(anchor="w", padx=20, pady=(10, 0))

        fps_value = self.config.get("screen_capture", {}).get("target_fps", 60)
        fps_slider = ctk.CTkSlider(
            self.main_frame,
            from_=30,
            to=144,
            number_of_steps=114,
            command=lambda v: self._on_setting_changed("screen_capture.target_fps", int(v))
        )
        fps_slider.set(fps_value)
        fps_slider.pack(anchor="w", padx=20, pady=5, fill="x")
        self.widgets["target_fps"] = fps_slider

    def _create_action_buttons(self):
        """Create action buttons."""
        button_frame = ctk.CTkFrame(self.main_frame)
        button_frame.pack(pady=20, fill="x")

        save_button = ctk.CTkButton(
            button_frame,
            text="Save Settings",
            command=self._save_settings,
            width=150
        )
        save_button.pack(side="left", padx=10)

        reload_button = ctk.CTkButton(
            button_frame,
            text="Reload Config",
            command=self._reload_config,
            width=150
        )
        reload_button.pack(side="left", padx=10)

        close_button = ctk.CTkButton(
            button_frame,
            text="Close",
            command=self.hide,
            width=150
        )
        close_button.pack(side="left", padx=10)

    def _on_setting_changed(self, key: str, value: Any):
        """Handle setting change."""
        logger.debug(f"Setting changed: {key} = {value}")

        # Update confidence label if needed
        if key == "detection.confidence_threshold" and "confidence_label" in self.widgets:
            self.widgets["confidence_label"].configure(text=f"{value:.2f}")

        if self.on_settings_changed:
            self.on_settings_changed(key, value)

    def _browse_model(self):
        """Browse for model file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.widgets["model_path"].delete(0, "end")
            self.widgets["model_path"].insert(0, filename)
            self._on_setting_changed("detection.model_path", filename)

    def _save_settings(self):
        """Save settings to config."""
        logger.info("Saving settings...")
        if self.on_settings_changed:
            self.on_settings_changed("save", None)

    def _reload_config(self):
        """Reload configuration."""
        logger.info("Reloading configuration...")
        if self.on_settings_changed:
            self.on_settings_changed("reload", None)

    def show(self):
        """Show the menu."""
        if self.window is None:
            self.create_window()

        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self.is_visible = True
        logger.info("Menu shown")

    def hide(self):
        """Hide the menu."""
        if self.window:
            self.window.withdraw()
            self.is_visible = False
            logger.info("Menu hidden")

    def toggle(self):
        """Toggle menu visibility (thread-safe)."""
        try:
            if self.is_visible:
                self.hide()
            else:
                self.show()
        except Exception as e:
            logger.error(f"Error toggling menu: {e}")
            # If we're not on the main thread, defer to next GUI update
            if self.window is None:
                logger.warning("Menu window not created yet - deferring creation")

    def run(self):
        """Run the menu (blocking)."""
        if self.window is None:
            self.create_window()

        self.window.mainloop()

    def destroy(self):
        """Destroy the menu window."""
        if self.window:
            self.window.destroy()
            self.window = None
            logger.info("Menu destroyed")
