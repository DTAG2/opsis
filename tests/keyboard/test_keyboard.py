#!/usr/bin/env python3
"""
Test if keyboard listener works - SIMPLE TEST
Press any key and you should see it printed.
Press ESC to quit.
"""

from pynput.keyboard import Listener, Key

print("=" * 60)
print("KEYBOARD TEST - Press ANY key to test")
print("Press ESC to quit")
print("=" * 60)
print()

def on_press(key):
    print(f"✓ KEY DETECTED: {key}")
    if key == Key.esc:
        print("\nESC pressed - exiting...")
        return False

def on_release(key):
    pass

# Start listener
with Listener(on_press=on_press, on_release=on_release) as listener:
    print("Listener started. Type something now...")
    listener.join()

print("\nTest complete!")
