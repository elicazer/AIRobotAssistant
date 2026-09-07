#!/usr/bin/env python3
"""
Test Jaw Servo on Channel 8
Quick diagnostic to check if jaw servo is working
"""

import os
import time

# Initialize FT232H for servo control
os.environ['BLINKA_FT232H'] = '1'

try:
    from adafruit_servokit import ServoKit
    
    print("=" * 60)
    print("JAW SERVO TEST - Channel 8")
    print("=" * 60)
    print()
    
    # Initialize servo controller
    print("🔧 Initializing servo controller at address 0x40...")
    servo_kit = ServoKit(channels=16, address=0x40)
    print("✅ Servo controller initialized")
    print()
    
    JAW_CHANNEL = 8
    
    # Test sequence
    print(f"🧪 Testing jaw servo on channel {JAW_CHANNEL}")
    print()
    
    # Test 1: Move to 0 degrees (closed)
    print("Test 1: Moving to 0° (closed)...")
    servo_kit.servo[JAW_CHANNEL].angle = 0
    time.sleep(2)
    print("✅ Should be closed")
    print()
    
    # Test 2: Move to 90 degrees (half open)
    print("Test 2: Moving to 90° (half open)...")
    servo_kit.servo[JAW_CHANNEL].angle = 90
    time.sleep(2)
    print("✅ Should be half open")
    print()
    
    # Test 3: Move to 100 degrees (open)
    print("Test 3: Moving to 100° (open)...")
    servo_kit.servo[JAW_CHANNEL].angle = 100
    time.sleep(2)
    print("✅ Should be open")
    print()
    
    # Test 4: Back to closed
    print("Test 4: Moving back to 0° (closed)...")
    servo_kit.servo[JAW_CHANNEL].angle = 0
    time.sleep(2)
    print("✅ Should be closed")
    print()
    
    # Test 5: Sweep test
    print("Test 5: Smooth sweep from 0° to 100° and back...")
    for angle in range(0, 101, 5):
        servo_kit.servo[JAW_CHANNEL].angle = angle
        print(f"  Angle: {angle}°")
        time.sleep(0.1)
    
    time.sleep(0.5)
    
    for angle in range(100, -1, -5):
        servo_kit.servo[JAW_CHANNEL].angle = angle
        print(f"  Angle: {angle}°")
        time.sleep(0.1)
    
    print("✅ Sweep complete")
    print()
    
    print("=" * 60)
    print("✅ JAW SERVO TEST COMPLETE")
    print("=" * 60)
    print()
    print("If the jaw didn't move, check:")
    print("  1. Servo is connected to channel 8 on PCA9685")
    print("  2. PCA9685 power supply is connected (5-6V)")
    print("  3. FT232H is connected to PCA9685 I2C pins (SDA/SCL)")
    print("  4. I2C address is 0x40 (default)")
    print()

except ImportError as e:
    print(f"❌ Error: Required libraries not installed")
    print(f"   {e}")
    print()
    print("Install with: pip install adafruit-circuitpython-servokit")

except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Possible issues:")
    print("  1. FT232H not connected")
    print("  2. PCA9685 not connected or wrong I2C address")
    print("  3. No power to PCA9685")
    print("  4. Wrong I2C pins (should be SDA/SCL)")
