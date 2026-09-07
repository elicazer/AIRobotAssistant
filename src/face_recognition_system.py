"""
Face Recognition System Module
Manages face encoding generation, database persistence, enrollment, and identification.
Uses DeepFace library for face encoding generation.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """Face recognition using DeepFace library"""

    def __init__(self, database_path: str = "config/face_database.json",
                 confidence_threshold: float = 0.6):
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.face_database: Dict[str, Dict] = {}
        self._load_database()

    # --- Encoding ---
    def generate_encoding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate face encoding from a cropped face image.
        Returns None if image is too small (<50x50) or encoding fails."""
        if face_image is None:
            return None

        height, width = face_image.shape[:2]
        if width < 50 or height < 50:
            return None

        try:
            from deepface import DeepFace
            representations = DeepFace.represent(
                img_path=face_image,
                model_name="Facenet",
                enforce_detection=False
            )
            if representations and len(representations) > 0:
                embedding = representations[0].get("embedding")
                if embedding is not None:
                    return np.array(embedding)
                else:
                    print(f"   ⚠️ DeepFace returned no embedding, keys: {representations[0].keys()}")
            else:
                print(f"   ⚠️ DeepFace returned empty representations: {representations}")
            return None
        except Exception as e:
            print(f"   ❌ DeepFace.represent() exception: {e}")
            logger.debug(f"Face encoding generation failed: {e}")
            return None

    # --- Database ---
    def _load_database(self) -> None:
        """Load face database from disk. Initialize empty on corruption."""
        if not os.path.exists(self.database_path):
            self.face_database = {}
            return

        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict) or "people" not in data:
                logger.error("Face database has invalid schema, initializing empty")
                self.face_database = {}
                return

            self.face_database = {}
            for name, person_data in data["people"].items():
                encodings_list = person_data.get("encodings", [])
                self.face_database[name] = {
                    "encodings": [np.array(enc) for enc in encodings_list],
                    "enrolled_at": person_data.get("enrolled_at", ""),
                    "last_seen": person_data.get("last_seen", "")
                }
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, OSError) as e:
            logger.error(f"Face database corrupted or unreadable, initializing empty: {e}")
            self.face_database = {}

    def _save_database(self) -> None:
        """Persist face database to disk as JSON."""
        data = {
            "version": 1,
            "people": {}
        }

        for name, person_data in self.face_database.items():
            data["people"][name] = {
                "encodings": [enc.tolist() for enc in person_data["encodings"]],
                "enrolled_at": person_data.get("enrolled_at", ""),
                "last_seen": person_data.get("last_seen", "")
            }

        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save face database: {e}")

    def list_enrolled(self) -> List[str]:
        """Return list of enrolled person names."""
        return list(self.face_database.keys())

    def remove_person(self, name: str) -> bool:
        """Remove all encodings for a person. Returns True if found."""
        if name in self.face_database:
            del self.face_database[name]
            self._save_database()
            return True
        return False

    # --- Enrollment ---
    def enroll(self, name: str, camera_source, num_captures: int = 3,
               timeout: float = 10.0) -> Tuple[bool, str]:
        """Capture face encodings from camera and store under name.
        Returns (success, message)."""
        if not name or not name.strip():
            return (False, "Name cannot be empty")

        encodings_captured = []
        start_time = time.time()

        while len(encodings_captured) < num_captures:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                if len(encodings_captured) == 0:
                    return (False, f"Enrollment failed: no face detected within {timeout} seconds")
                else:
                    return (False, f"Enrollment incomplete: captured {len(encodings_captured)}/{num_captures} encodings before timeout")

            ret, frame = camera_source.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # Use DeepFace directly on the full frame — it handles face detection internally
            encoding = self.generate_encoding(frame)
            if encoding is not None:
                encodings_captured.append(encoding)
                # Wait for a fresh frame before next capture
                time.sleep(0.5)

        # Store encodings
        now = datetime.now(timezone.utc).isoformat()
        if name in self.face_database:
            # Re-enrollment: append to existing encodings
            self.face_database[name]["encodings"].extend(encodings_captured)
            self.face_database[name]["last_seen"] = now
        else:
            self.face_database[name] = {
                "encodings": encodings_captured,
                "enrolled_at": now,
                "last_seen": now
            }

        self._save_database()
        return (True, f"Successfully enrolled {name} with {len(encodings_captured)} face captures")

    # --- Identification ---
    def identify(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Compare encoding against database.
        Returns (name, confidence) or (None, 0.0) for unknown."""
        encoding = self.generate_encoding(face_image)
        if encoding is None:
            return (None, 0.0)

        if not self.face_database:
            return (None, 0.0)

        best_name = None
        best_confidence = 0.0

        for name, person_data in self.face_database.items():
            for stored_encoding in person_data["encodings"]:
                confidence = self._cosine_similarity(encoding, stored_encoding)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_name = name

        if best_confidence > self.confidence_threshold:
            # Update last_seen
            if best_name in self.face_database:
                self.face_database[best_name]["last_seen"] = (
                    datetime.now(timezone.utc).isoformat()
                )
            return (best_name, best_confidence)
        else:
            return (None, 0.0)

    def identify_multiple(self, frame: np.ndarray,
                          face_locations: List[Tuple]) -> List[Tuple[Optional[str], float]]:
        """Identify multiple faces in a single frame."""
        results = []
        for location in face_locations:
            x, y, w, h = location
            face_region = frame[y:y+h, x:x+w]
            result = self.identify(face_region)
            results.append(result)
        return results

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        Confidence = 1 - cosine_distance = cosine_similarity.
        Returns value in [0, 1] where 1 is identical."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        # Clamp to [0, 1] — face embeddings are typically non-negative
        # so cosine similarity is usually in [0, 1] already
        return float(max(0.0, min(1.0, similarity)))
