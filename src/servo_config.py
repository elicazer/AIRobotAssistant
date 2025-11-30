"""
Servo Configuration for Different Robot Heads
Supports multiple servo layouts with easy switching
"""

class ServoConfig:
    """Base class for servo configurations"""
    
    def __init__(self):
        self.name = "Base Config"
        self.servo_count = 0
        self.channels = {}
        self.default_angles = {}
        self.angle_ranges = {}
    
    def get_channel(self, servo_name):
        """Get channel number for a servo"""
        return self.channels.get(servo_name)
    
    def get_default_angle(self, servo_name):
        """Get default/center angle for a servo"""
        return self.default_angles.get(servo_name, 90)
    
    def get_angle_range(self, servo_name):
        """Get min/max angle range for a servo"""
        return self.angle_ranges.get(servo_name, (0, 180))


class InMoovHeadConfig(ServoConfig):
    """
    InMoov Head Configuration
    8 servos total: 4 per eye (X, Y, Upper Lid, Lower Lid)
    """
    
    def __init__(self):
        super().__init__()
        self.name = "InMoov Head"
        self.servo_count = 8
        
        # Servo channel assignments
        self.channels = {
            # Left Eye
            'left_eye_x': 0,
            'left_eye_y': 1,
            'left_upper_lid': 2,
            'left_lower_lid': 3,
            
            # Right Eye
            'right_eye_x': 4,
            'right_eye_y': 5,
            'right_upper_lid': 6,
            'right_lower_lid': 7,
        }
        
        # Default center positions
        self.default_angles = {
            'left_eye_x': 90,
            'left_eye_y': 90,
            'left_upper_lid': 70,
            'left_lower_lid': 100,
            
            'right_eye_x': 90,
            'right_eye_y': 90,
            'right_upper_lid': 120,
            'right_lower_lid': 90,
        }
        
        # Angle ranges (min, max)
        self.angle_ranges = {
            'left_eye_x': (57, 145),
            'left_eye_y': (52, 112),
            'left_upper_lid': (70, 180),
            'left_lower_lid': (10, 100),
            
            'right_eye_x': (57, 145),
            'right_eye_y': (52, 112),
            'right_upper_lid': (10, 120),
            'right_lower_lid': (90, 180),
        }
        
        # Blink positions (closed)
        self.blink_angles = {
            'left_upper_lid': 180,
            'left_lower_lid': 10,
            'right_upper_lid': 10,
            'right_lower_lid': 180,
        }


class OriginalConfig(ServoConfig):
    """
    Original Configuration (from facerec.py)
    6 servos: Shared X/Y axes, separate eyelids
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Original (Shared Axes)"
        self.servo_count = 6
        
        # Servo channel assignments
        self.channels = {
            # Shared axes for both eyes
            'eyes_x': 0,  # Both eyes X-axis
            'eyes_y': 1,  # Both eyes Y-axis
            
            # Individual eyelids
            'left_upper_lid': 2,
            'right_upper_lid': 3,
            'left_lower_lid': 4,
            'right_lower_lid': 5,
        }
        
        # Default center positions
        self.default_angles = {
            'eyes_x': 100,
            'eyes_y': 80,
            'left_upper_lid': 70,
            'right_upper_lid': 120,
            'left_lower_lid': 100,
            'right_lower_lid': 90,
        }
        
        # Angle ranges (min, max)
        self.angle_ranges = {
            'eyes_x': (57, 145),
            'eyes_y': (52, 112),
            'left_upper_lid': (70, 180),
            'right_upper_lid': (10, 120),
            'left_lower_lid': (10, 100),
            'right_lower_lid': (90, 180),
        }
        
        # Blink positions (closed)
        self.blink_angles = {
            'left_upper_lid': 180,
            'right_upper_lid': 10,
            'left_lower_lid': 10,
            'right_lower_lid': 180,
        }


class SimpleConfig(ServoConfig):
    """
    Simple Configuration
    2 servos: Just X and Y movement, no eyelids
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Simple (X/Y Only)"
        self.servo_count = 2
        
        # Servo channel assignments
        self.channels = {
            'eyes_x': 0,
            'eyes_y': 1,
        }
        
        # Default center positions
        self.default_angles = {
            'eyes_x': 90,
            'eyes_y': 90,
        }
        
        # Angle ranges (min, max)
        self.angle_ranges = {
            'eyes_x': (0, 180),
            'eyes_y': (0, 180),
        }


# Available configurations
CONFIGS = {
    'inmoov': InMoovHeadConfig,
    'original': OriginalConfig,
    'simple': SimpleConfig,
}


def get_config(config_name='inmoov'):
    """
    Get servo configuration by name
    
    Args:
        config_name: 'inmoov', 'original', or 'simple'
    
    Returns:
        ServoConfig instance
    """
    config_class = CONFIGS.get(config_name.lower())
    if config_class is None:
        print(f"Warning: Unknown config '{config_name}', using 'inmoov'")
        config_class = InMoovHeadConfig
    
    return config_class()


def list_configs():
    """List all available configurations"""
    print("Available servo configurations:")
    for name, config_class in CONFIGS.items():
        config = config_class()
        print(f"  - {name}: {config.name} ({config.servo_count} servos)")


# Default configuration
DEFAULT_CONFIG = 'inmoov'
