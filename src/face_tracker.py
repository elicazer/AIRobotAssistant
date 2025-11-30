"""
Face Tracking Module
Handles face detection and eye tracking for the robot
"""

import cv2
import numpy as np
import time
import random
from typing import Optional, Tuple, Dict


class FaceTracker:
    """Face detection and tracking using OpenCV"""
    
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            # Default to cascade file in src directory
            import os
            cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        """Initialize face detector"""
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.camera = None
        self.is_running = False
        
        # Tracking state
        self.last_face_position = None
        self.face_lost_time = None
        self.face_lost_threshold = 2.0  # seconds before considering face truly lost
        
    def start_camera(self, camera_index=0) -> bool:
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                print("❌ Failed to open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("✅ Camera started")
            self.is_running = True
            return True
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()
        print("📷 Camera stopped")
    
    def detect_faces(self, frame) -> list:
        """
        Detect faces in frame
        Returns list of (x, y, w, h) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def get_closest_face(self, faces, frame_width, frame_height) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the face closest to center of frame
        Returns (x, y, w, h) or None
        """
        if len(faces) == 0:
            return None
        
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        closest_face = None
        min_distance = float('inf')
        
        for (x, y, w, h) in faces:
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            distance = np.sqrt(
                (face_center_x - frame_center_x) ** 2 +
                (face_center_y - frame_center_y) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_face = (x, y, w, h)
        
        return closest_face
    
    def get_face_center(self, face: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of face"""
        x, y, w, h = face
        return (x + w // 2, y + h // 2)
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from camera"""
        if not self.camera or not self.is_running:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        return frame
    
    def track_face(self) -> Optional[Dict]:
        """
        Track face and return tracking data
        Returns dict with face info or None
        """
        frame = self.read_frame()
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        faces = self.detect_faces(frame)
        closest_face = self.get_closest_face(faces, width, height)
        
        current_time = time.time()
        
        if closest_face is not None:
            # Face found
            center_x, center_y = self.get_face_center(closest_face)
            self.last_face_position = (center_x, center_y)
            self.face_lost_time = None
            
            return {
                'found': True,
                'center_x': center_x,
                'center_y': center_y,
                'frame_width': width,
                'frame_height': height,
                'face_rect': closest_face,
                'frame': frame,
                'num_faces': len(faces)
            }
        else:
            # No face found
            if self.face_lost_time is None:
                self.face_lost_time = current_time
            
            # Check if face has been lost for too long
            time_lost = current_time - self.face_lost_time
            
            return {
                'found': False,
                'last_position': self.last_face_position,
                'time_lost': time_lost,
                'frame': frame,
                'frame_width': width,
                'frame_height': height
            }
