#!/usr/bin/env python3
"""
Quick test script to verify system setup without requiring a trained model.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.capture.screen_grabber import ScreenGrabber
from src.detection.model_loader import check_gpu_availability

def test_gpu():
    """Test GPU availability."""
    print("\n=== GPU Test ===")
    gpu_info = check_gpu_availability()
    print(f"GPU Info: {gpu_info}")
    return True

def test_screen_capture():
    """Test screen capture."""
    print("\n=== Screen Capture Test ===")

    grabber = ScreenGrabber(monitor=1, target_fps=1)
    monitor_info = grabber.get_monitor_info()

    print(f"Available monitors: {monitor_info['count']}")
    for i in range(1, len(monitor_info['monitors'])):
        m = monitor_info['monitors'][i]
        print(f"  Monitor {i}: {m['width']}x{m['height']} at position ({m['left']}, {m['top']})")

    print("\nCapturing 1 frame...")
    grabber.start()

    import time
    time.sleep(1)

    frame = grabber.get_frame()
    grabber.stop()

    if frame is not None:
        print(f"✓ Successfully captured frame: {frame.shape}")
        return True
    else:
        print("✗ Failed to capture frame")
        return False

def test_imports():
    """Test all critical imports."""
    print("\n=== Import Test ===")

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")

        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")

        from ultralytics import YOLO
        print(f"✓ Ultralytics YOLO")

        import mss
        print(f"✓ MSS (screen capture)")

        from pynput import keyboard
        print(f"✓ pynput (global hotkeys)")

        import tkinter
        print(f"✓ tkinter (overlay)")

        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*50)
    print("CV OVERLAY SYSTEM - SETUP TEST")
    print("="*50)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("GPU", test_gpu()))
    results.append(("Screen Capture", test_screen_capture()))

    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Train a model (see README.md)")
        print("2. Run: python main.py --monitor 1")
    else:
        print("\n✗ Some tests failed. Check errors above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
