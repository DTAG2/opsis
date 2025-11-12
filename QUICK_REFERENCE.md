# Opsis Quick Reference

## Running the System

```bash
# Navigate to opsis directory
cd /path/to/opsis

# Activate environment
source venv/bin/activate

# Run on primary monitor
python main.py

# Run on specific monitor
python main.py --monitor 2

# Run with debug output
python main.py --debug
```

## Global Hotkeys (Work from ANY window!)

| Hotkey | Action | Description |
|--------|--------|-------------|
| **F1** | Toggle Overlay | Show/hide detection boxes |
| **F2** | Toggle Mouse | Enable/disable mouse tracking |
| **F3** | Settings Menu | Open configuration GUI |
| **F4** | Reload Config | Apply config changes |
| **F5** | Exit Program | Quit cleanly |

## First-Time Setup

### macOS Accessibility Permissions

When you run the program for the first time, you must grant permissions:

1. **System Preferences** → **Security & Privacy** → **Privacy** → **Accessibility**
2. Click the lock to make changes
3. Add your terminal app:
   - **Terminal.app** (default macOS terminal)
   - **iTerm2** (if you use iTerm)
   - **Visual Studio Code** (if using VSCode terminal)
4. Check the box to enable
5. Restart the terminal and program

## What You Should See

### Terminal Output
```
=======================================
           CV OVERLAY SYSTEM
=======================================

★★★ HOTKEYS READY ★★★

  F1  →  Toggle Overlay
  F2  →  Toggle Mouse Control
  F3  →  Settings Menu
  F4  →  Reload Config
  F5  →  Quit
```

### Overlay Display
- **FPS Counter** (top-left corner)
- **Object Count** (below FPS)
- **Green Boxes** around detected objects
- **Red Crosshair** at target points
- **Green Dot** (top-right) indicates overlay is active

### Status Messages
When pressing hotkeys, you'll see:
```
★★★ OVERLAY ON ★★★
★★★ MOUSE CONTROL ON ★★★
★★★ CONFIG RELOADED ★★★
```

## Configuration

Edit `config/settings.json`:

```json
{
  "detection": {
    "model_path": "models/runs/my_game_character_detection/weights/best.pt",
    "confidence_threshold": 0.4,
    "inference_device": "mps"  // Use "cuda" for NVIDIA, "cpu" for CPU
  },
  "mouse_control": {
    "enabled": false,
    "smoothing_factor": 0.6,  // Higher = smoother movement
    "max_move_speed": 50      // Lower = gentler tracking
  },
  "overlay": {
    "enabled": true,
    "box_color": [0, 255, 0],  // RGB green
    "show_labels": true
  }
}
```

## Troubleshooting

### Hotkeys Not Working
1. **Check Permissions**: Ensure terminal has Accessibility permissions
2. **Restart Terminal**: Completely quit (Cmd+Q) and reopen
3. **VSCode Users**: Add Visual Studio Code to Accessibility permissions
4. **Test Keys**: Press F1 - you should see `★★★ OVERLAY OFF/ON ★★★`

### No Detections
- **Check Model**: Ensure `best.pt` exists in `models/runs/my_game_character_detection/weights/`
- **Lower Threshold**: Set `confidence_threshold` to 0.25 in config
- **Wrong Monitor**: Try different monitor with `--monitor 1`, `--monitor 2`, etc.

### Overlay Issues
- **Not Visible**: Press F1 to toggle on
- **Wrong Monitor**: Use `--monitor` flag
- **Blocks Clicks**: Update code - overlay should be click-through

### Mouse Control Issues
- **Moving Wrong Way**: Fixed - coordinates now properly converted
- **Too Aggressive**: Increase `smoothing_factor`, decrease `max_move_speed`
- **Not Working**: Press F2 to enable

## Testing Without a Model

```bash
# Test system setup
python tests/test_setup.py

# Test keyboard detection
python tests/keyboard/test_keyboard.py

# Check macOS permissions
python tests/setup/check_permissions.py
```

## Tips

- **Performance**: Use `yolov8n.pt` for fastest detection
- **Multiple Monitors**: Overlay only appears on selected monitor
- **Live Reload**: Press F4 after editing config - no restart needed
- **Debug Mode**: Use `--debug` flag for detailed logging