"""
Eye Controller Module
Controls eye servos for tracking and blinking
"""

import time
import random
from typing import Optional, Dict


def map_value(value, in_min, in_max, out_min, out_max):
    """Map value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class EyeController:
    """Controls eye servos for tracking and expressions"""
    
    def __init__(self, servo_kit, servo_config, position_callback=None):
        """
        Initialize eye controller
        
        Args:
            servo_kit: ServoKit instance
            servo_config: Servo configuration object
            position_callback: Optional callback function to send position updates
        """
        self.servo_kit = servo_kit
        self.config = servo_config
        self.position_callback = position_callback
        
        # Blinking state
        self.last_blink_time = 0
        self.blink_interval = 5  # seconds between blinks
        self.blink_probability = 0.01  # 1% chance per frame
        
        # Tracking state
        self.current_angles = {}
        
        # Eye alignment offsets (adjust if eyes don't look straight)
        self.right_eye_x_offset = 0  # Negative = look more left, Positive = look more right
        self.right_eye_y_offset = 10    # Negative = look more down, Positive = look more up
        
    def set_servo_angle(self, servo_name: str, angle: float):
        """Set servo angle with bounds checking"""
        if self.servo_kit is None:
            return
        
        channel = self.config.get_channel(servo_name)
        if channel is None:
            return
        
        # Constrain angle to valid range
        min_angle, max_angle = self.config.get_angle_range(servo_name)
        angle = max(min_angle, min(max_angle, angle))
        
        try:
            self.servo_kit.servo[channel].angle = angle
            self.current_angles[servo_name] = angle
            
            # Send position update to web UI
            if self.position_callback:
                self.position_callback(self.current_angles)
        except Exception as e:
            print(f"⚠️  Error setting {servo_name} to {angle}°: {e}")
    
    def center_eyes(self):
        """Center both eyes to default position"""
        if self.servo_kit is None:
            return
        
        print("👀 Centering eyes...")
        try:
            for servo_name in self.config.channels.keys():
                angle = self.config.get_default_angle(servo_name)
                self.set_servo_angle(servo_name, angle)
            time.sleep(0.1)
        except Exception as e:
            print(f"⚠️  Error centering eyes: {e}")
    
    def blink_eyes(self, force=False):
        """
        Perform blinking behavior
        
        Args:
            force: Force blink regardless of timing
        """
        if self.servo_kit is None:
            return
        
        current_time = time.time()
        
        # Check if it's time to blink
        should_blink = force or (
            (current_time - self.last_blink_time) > self.blink_interval and
            random.random() < self.blink_probability
        )
        
        if not should_blink:
            return
        
        try:
            # Close eyelids
            if hasattr(self.config, 'blink_angles'):
                for servo_name, angle in self.config.blink_angles.items():
                    self.set_servo_angle(servo_name, angle)
                
                time.sleep(0.15)
                
                # Open eyelids back to default
                for servo_name in self.config.blink_angles.keys():
                    angle = self.config.get_default_angle(servo_name)
                    self.set_servo_angle(servo_name, angle)
            
            self.last_blink_time = current_time
        except Exception as e:
            print(f"⚠️  Error blinking eyes: {e}")
    
    def track_position(self, face_x: int, face_y: int, frame_width: int, frame_height: int):
        """
        Move eyes to track face position
        
        Args:
            face_x: Face center X coordinate
            face_y: Face center Y coordinate
            frame_width: Video frame width
            frame_height: Video frame height
        """
        if self.servo_kit is None:
            return
        
        try:
            # Calculate all angles first (batch calculation)
            angles_to_set = {}
            
            # Left eye X-axis
            if 'left_eye_x' in self.config.channels:
                x_min, x_max = self.config.get_angle_range('left_eye_x')
                left_x_angle = map_value(face_x, 0, frame_width, x_max, x_min)
                angles_to_set['left_eye_x'] = left_x_angle
            
            # Left eye Y-axis (inverted mapping)
            if 'left_eye_y' in self.config.channels:
                y_min, y_max = self.config.get_angle_range('left_eye_y')
                # Inverted: face at top (0) = y_min, face at bottom (height) = y_max
                left_y_angle = map_value(face_y, 0, frame_height, y_min, y_max)
                angles_to_set['left_eye_y'] = left_y_angle
            
            # Right eye X-axis (with offset correction)
            if 'right_eye_x' in self.config.channels:
                x_min, x_max = self.config.get_angle_range('right_eye_x')
                right_x_angle = map_value(face_x, 0, frame_width, x_max, x_min)
                # Apply offset to correct alignment
                right_x_angle = max(x_min, min(x_max, right_x_angle + self.right_eye_x_offset))
                angles_to_set['right_eye_x'] = right_x_angle
            
            # Right eye Y-axis (with offset correction)
            if 'right_eye_y' in self.config.channels:
                y_min, y_max = self.config.get_angle_range('right_eye_y')
                right_y_angle = map_value(face_y, 0, frame_height, y_max, y_min)
                # Apply offset to correct alignment
                right_y_angle = max(y_min, min(y_max, right_y_angle + self.right_eye_y_offset))
                angles_to_set['right_eye_y'] = right_y_angle
            
            # Shared X-axis (for simpler configs)
            if 'eye_x' in self.config.channels:
                x_min, x_max = self.config.get_angle_range('eye_x')
                x_angle = map_value(face_x, 0, frame_width, x_max, x_min)
                angles_to_set['eye_x'] = x_angle
            
            # Shared Y-axis (for simpler configs)
            if 'eye_y' in self.config.channels:
                y_min, y_max = self.config.get_angle_range('eye_y')
                y_angle = map_value(face_y, 0, frame_height, y_max, y_min)
                angles_to_set['eye_y'] = y_angle
            
            # Set all servos simultaneously (no delays between)
            for servo_name, angle in angles_to_set.items():
                channel = self.config.get_channel(servo_name)
                if channel is not None:
                    min_angle, max_angle = self.config.get_angle_range(servo_name)
                    angle = max(min_angle, min(max_angle, angle))
                    self.servo_kit.servo[channel].angle = angle
                self.current_angles[servo_name] = angle  # Update even if no physical servo
            
            # Send single position update after all angles are calculated
            # This works for both physical servos and virtual display
            if self.position_callback and angles_to_set:
                self.position_callback(angles_to_set)
                
        except Exception as e:
            print(f"⚠️  Error tracking position: {e}")
    
    def get_current_angles(self) -> Dict[str, float]:
        """Get current servo angles"""
        return self.current_angles.copy()
