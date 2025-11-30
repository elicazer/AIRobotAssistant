"""
Platform Detection and Configuration Module
Automatically detects Raspberry Pi, Jetson Nano, or Mac and configures accordingly
"""

import platform
import os
import sys

class PlatformConfig:
    """Detect and configure for different platforms"""
    
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    MAC = "mac"
    LINUX = "linux"
    UNKNOWN = "unknown"
    
    def __init__(self):
        self.platform_type = self._detect_platform()
        self.gpio_available = False
        self.camera_config = self._get_camera_config()
        
        print(f"Detected platform: {self.platform_type}")
    
    def _detect_platform(self):
        """Detect which platform we're running on"""
        system = platform.system()
        machine = platform.machine()
        
        # Check for Mac
        if system == "Darwin":
            return self.MAC
        
        # Check for Jetson Nano
        if os.path.exists('/etc/nv_tegra_release'):
            return self.JETSON_NANO
        
        # Check for Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                    return self.RASPBERRY_PI
        except:
            pass
        
        # Generic Linux
        if system == "Linux":
            return self.LINUX
        
        return self.UNKNOWN
    
    def _get_camera_config(self):
        """Get camera configuration based on platform"""
        if self.platform_type == self.MAC:
            return {
                'device': 0,
                'backend': None,
                'use_gstreamer': False,
                'width': 640,
                'height': 480,
                'fps': 30
            }
        
        elif self.platform_type == self.JETSON_NANO:
            return {
                'device': self._get_jetson_gstreamer_source(),
                'backend': 'CAP_GSTREAMER',
                'use_gstreamer': True,
                'width': 640,
                'height': 480,
                'fps': 20
            }
        
        elif self.platform_type == self.RASPBERRY_PI:
            return {
                'device': '/dev/video0',
                'backend': 'CAP_V4L2',
                'use_gstreamer': False,
                'width': 640,
                'height': 480,
                'fps': 20
            }
        
        else:  # Generic Linux or Unknown
            return {
                'device': 0,
                'backend': None,
                'use_gstreamer': False,
                'width': 640,
                'height': 480,
                'fps': 30
            }
    
    def _get_jetson_gstreamer_source(self, capture_width=640, capture_height=480, 
                                     display_width=640, display_height=480, 
                                     framerate=20, flip_method=0):
        """Return GStreamer pipeline for Jetson Nano CSI camera"""
        return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), '
            f'width=(int){capture_width}, height=(int){capture_height}, '
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
            f'nvvidconv flip-method={flip_method} ! '
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! '
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
        )
    
    def setup_gpio(self):
        """Setup GPIO based on platform"""
        if self.platform_type == self.JETSON_NANO:
            try:
                import Jetson.GPIO as GPIO
                self.gpio_available = True
                return GPIO
            except ImportError:
                print("Warning: Jetson.GPIO not available")
                return None
        
        elif self.platform_type == self.RASPBERRY_PI:
            try:
                import RPi.GPIO as GPIO
                self.gpio_available = True
                return GPIO
            except ImportError:
                print("Warning: RPi.GPIO not available")
                return None
        
        else:
            print(f"GPIO not available on {self.platform_type}")
            return None
    
    def get_i2c_bus(self):
        """Get I2C bus number based on platform"""
        if self.platform_type == self.MAC:
            # For USB-to-I2C adapters on Mac, we don't specify a bus number
            # The adapter handles this automatically
            return None
        
        elif self.platform_type in [self.JETSON_NANO, self.RASPBERRY_PI]:
            return 1  # Standard I2C bus
        
        else:
            return 1  # Default
    
    def _test_i2c_bus(self, bus_num):
        """Test if I2C bus is available"""
        try:
            import board
            import busio
            i2c = busio.I2C(board.SCL, board.SDA)
            i2c.deinit()
            return True
        except:
            return False
    
    def setup_usb_i2c(self):
        """Setup USB-to-I2C adapter (FT232H, CH341A, etc.)"""
        if not self.is_mac():
            return False
        
        import os
        # Enable FT232H mode for Blinka
        os.environ['BLINKA_FT232H'] = '1'
        
        try:
            import board
            import busio
            
            # Try to initialize I2C
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Scan for devices
            while not i2c.try_lock():
                pass
            
            try:
                devices = i2c.scan()
                print(f"I2C devices found: {[hex(d) for d in devices]}")
                if devices:
                    print(f"Found {len(devices)} I2C device(s)")
                    return True
                else:
                    print("No I2C devices detected")
                    return False
            finally:
                i2c.unlock()
                i2c.deinit()
        
        except Exception as e:
            print(f"USB-to-I2C setup failed: {e}")
            return False
    
    def is_jetson(self):
        return self.platform_type == self.JETSON_NANO
    
    def is_raspberry_pi(self):
        return self.platform_type == self.RASPBERRY_PI
    
    def is_mac(self):
        return self.platform_type == self.MAC
    
    def is_linux(self):
        return self.platform_type in [self.LINUX, self.JETSON_NANO, self.RASPBERRY_PI]


# Global instance
config = PlatformConfig()
