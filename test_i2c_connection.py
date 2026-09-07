#!/usr/bin/env python3
"""
I2C Connection Diagnostic Tool
Tests FT232H and PCA9685 connection
"""

import os
import sys

# Enable FT232H mode
os.environ['BLINKA_FT232H'] = '1'

print("=" * 60)
print("I2C CONNECTION DIAGNOSTIC")
print("=" * 60)
print()

# Step 1: Check if libraries are installed
print("Step 1: Checking libraries...")
try:
    import board
    import busio
    from adafruit_servokit import ServoKit
    print("✅ All required libraries installed")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("   Install with: pip install adafruit-circuitpython-servokit")
    sys.exit(1)

print()

# Step 2: Check FT232H connection
print("Step 2: Checking FT232H connection...")
try:
    import board
    print(f"✅ FT232H detected")
    print(f"   Available pins: {dir(board)}")
except Exception as e:
    print(f"❌ FT232H not detected: {e}")
    print("   Make sure FT232H is connected via USB")
    sys.exit(1)

print()

# Step 3: Initialize I2C bus
print("Step 3: Initializing I2C bus...")
print("   Using pins: D0 (SCL), D1 (SDA)")
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    print("✅ I2C bus initialized")
except Exception as e:
    print(f"❌ I2C initialization failed: {e}")
    sys.exit(1)

print()

# Step 4: Scan for I2C devices
print("Step 4: Scanning for I2C devices...")
while not i2c.try_lock():
    pass

try:
    devices = i2c.scan()
    print(f"   Found {len(devices)} device(s)")
    
    if devices:
        print("   Device addresses:")
        for device in devices:
            print(f"      • 0x{device:02X} ({device})")
        
        # Check for PCA9685 at 0x40
        if 0x40 in devices:
            print()
            print("✅ PCA9685 found at address 0x40 (default)")
        else:
            print()
            print("⚠️  PCA9685 NOT found at 0x40")
            print("   Your PCA9685 might be at a different address")
            print("   Check the solder jumpers on your PCA9685 board")
    else:
        print("❌ No I2C devices found")
        print("   Check:")
        print("   1. PCA9685 is connected to FT232H")
        print("   2. Wiring: FT232H D0 → PCA9685 SCL")
        print("   3. Wiring: FT232H D1 → PCA9685 SDA")
        print("   4. Wiring: FT232H GND → PCA9685 GND")
        print("   5. PCA9685 has external 5-6V power")
finally:
    i2c.unlock()
    i2c.deinit()

print()

# Step 5: Test ServoKit initialization
if devices and 0x40 in devices:
    print("Step 5: Testing ServoKit initialization...")
    try:
        servo_kit = ServoKit(channels=16, address=0x40)
        print("✅ ServoKit initialized successfully")
        print()
        
        # Step 6: Test jaw servo (channel 8)
        print("Step 6: Testing jaw servo on channel 8...")
        print("   Moving to 0° (closed)...")
        servo_kit.servo[8].angle = 0
        import time
        time.sleep(1)
        
        print("   Moving to 90° (half open)...")
        servo_kit.servo[8].angle = 90
        time.sleep(1)
        
        print("   Moving back to 0° (closed)...")
        servo_kit.servo[8].angle = 0
        time.sleep(1)
        
        print("✅ Jaw servo test complete")
        print()
        print("If the servo didn't move:")
        print("   1. Check servo is connected to channel 8")
        print("   2. Check PCA9685 has external power (5-6V)")
        print("   3. Check servo is functional (try different channel)")
        
    except Exception as e:
        print(f"❌ ServoKit initialization failed: {e}")

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
