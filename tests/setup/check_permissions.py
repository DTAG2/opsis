#!/usr/bin/env python3
"""Check if we have the necessary permissions for keyboard monitoring."""

import sys

print("=" * 60)
print("CHECKING ACCESSIBILITY PERMISSIONS")
print("=" * 60)
print()

# Test 1: Can we import pynput?
print("1. Testing pynput import...")
try:
    from pynput.keyboard import Listener, Key
    print("   ✓ pynput imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import pynput: {e}")
    sys.exit(1)

# Test 2: Can we create a listener?
print("2. Testing listener creation...")
try:
    def dummy_press(key):
        pass

    listener = Listener(on_press=dummy_press)
    print("   ✓ Listener created successfully")
except Exception as e:
    print(f"   ✗ Failed to create listener: {e}")
    sys.exit(1)

# Test 3: Can we start the listener?
print("3. Testing listener start...")
try:
    listener.start()
    print("   ✓ Listener started successfully")
    print()
    print("=" * 60)
    print("LISTENER IS RUNNING!")
    print("Press ANY key now (you should see output below)")
    print("Press ESC to exit")
    print("=" * 60)
    print()

    # Now test actual key detection
    key_detected = False

    def on_press(key):
        global key_detected
        key_detected = True
        print(f"✓✓✓ KEY DETECTED: {key} ✓✓✓")
        if key == Key.esc:
            print("\nESC pressed - stopping...")
            return False

    # Stop the first listener and start a new one with our handler
    listener.stop()

    with Listener(on_press=on_press) as listener:
        listener.join()

    if key_detected:
        print("\n✓✓✓ SUCCESS! Keyboard detection is working! ✓✓✓")
    else:
        print("\n✗ No keys were detected")
        print("\nThis means Accessibility permissions are NOT granted.")
        print("\nFIX:")
        print("1. Open System Preferences")
        print("2. Security & Privacy → Privacy → Accessibility")
        print("3. Make sure Terminal is checked")
        print("4. If Terminal is already there, REMOVE it and ADD it again")
        print("5. Restart Terminal completely (Cmd+Q, then reopen)")

except Exception as e:
    print(f"   ✗ Failed to start listener: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
