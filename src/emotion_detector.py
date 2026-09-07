"""
Emotion Detector Module
Analyzes facial expressions and provides thread-safe access to the current
emotional state.

Two backends are supported:
  - "hsemotion" (default): the HSEmotion AffectNet model (enet_b0_8), which is
    far more accurate on real webcam faces than the legacy FER model.
  - "deepface": DeepFace's built-in FER emotion model (fallback). Kept for
    environments where HSEmotion/onnxruntime is unavailable.

Both backends produce a per-class probability map over this module's emotion
vocabulary; a common decision stage then applies a confidence floor and a
"must beat neutral" margin to suppress noise.
"""

import logging
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# HSEmotion (AffectNet) class labels -> this module's emotion vocabulary.
# "Contempt" has no counterpart here and is folded into neutral.
_HSEMOTION_TO_APP = {
    "anger": "angry",
    "contempt": "neutral",
    "disgust": "disgusted",
    "fear": "fearful",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
}

# DeepFace FER labels -> this module's emotion vocabulary.
_DEEPFACE_TO_APP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "surprise": "surprised",
    "disgust": "disgusted",
    "fear": "fearful",
    "neutral": "neutral",
}


class EmotionDetector:
    """Facial emotion detection with a pluggable backend."""

    SUPPORTED_EMOTIONS = [
        "happy", "sad", "angry", "surprised",
        "disgusted", "fearful", "neutral"
    ]

    # Minimum face size for reliable emotion analysis
    MIN_FACE_SIZE = 48

    DEFAULT_HSEMOTION_MODEL = "enet_b0_8_best_afew"

    def __init__(self, min_confidence: float = 0.0, neutral_bias: float = 0.0,
                 backend: str = "hsemotion",
                 hsemotion_model: Optional[str] = None):
        """Create an emotion detector.

        Args:
            min_confidence: Minimum dominant-emotion probability [0,1] required
                to report a non-neutral emotion; lower hits become neutral.
            neutral_bias: The dominant emotion must beat the model's own neutral
                probability by at least this margin [0,1]; otherwise neutral.
            backend: "hsemotion" (preferred) or "deepface".
            hsemotion_model: Optional HSEmotion model name override.
        """
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.neutral_bias = max(0.0, min(1.0, neutral_bias))
        self.current_emotion: str = "neutral"
        self.current_confidence: float = 0.0
        self._lock = threading.Lock()

        self.backend = "deepface"
        self._hse = None
        if backend == "hsemotion":
            self._hse = self._init_hsemotion(hsemotion_model or self.DEFAULT_HSEMOTION_MODEL)
            if self._hse is not None:
                self.backend = "hsemotion"
            else:
                logger.warning("HSEmotion unavailable; falling back to DeepFace emotion model")
        logger.info(f"EmotionDetector backend: {self.backend}")

    @staticmethod
    def _init_hsemotion(model_name: str):
        """Load the HSEmotion recognizer, or return None if unavailable."""
        try:
            # hsemotion_onnx downloads its model via urllib.request.urlretrieve
            # but forgets to import urllib.request; importing it here registers
            # the submodule so the download works.
            import urllib.request  # noqa: F401
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            return HSEmotionRecognizer(model_name=model_name)
        except Exception as e:
            logger.warning(f"Failed to initialize HSEmotion ({e})")
            return None

    def analyze(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Analyze emotion from a BGR face crop.

        Returns (emotion, confidence) where emotion is one of
        SUPPORTED_EMOTIONS and confidence is in [0.0, 1.0]. Returns
        ('neutral', 0.1) if the face is too small or analysis fails.
        """
        if face_image is None:
            return ("neutral", 0.1)

        height, width = face_image.shape[:2]
        if width < self.MIN_FACE_SIZE or height < self.MIN_FACE_SIZE:
            return ("neutral", 0.1)

        try:
            if self.backend == "hsemotion":
                probs = self._probs_hsemotion(face_image)
            else:
                probs = self._probs_deepface(face_image)
        except Exception as e:
            logger.debug(f"Emotion analysis failed: {e}")
            return ("neutral", 0.1)

        if not probs:
            return ("neutral", 0.1)

        # Dominant emotion + confidence, then noise suppression.
        dominant = max(probs, key=probs.get)
        confidence = max(0.0, min(1.0, probs[dominant]))
        neutral_conf = max(0.0, min(1.0, probs.get("neutral", 0.0)))

        if dominant != "neutral":
            if confidence < self.min_confidence or (confidence - neutral_conf) < self.neutral_bias:
                return ("neutral", max(confidence, neutral_conf))

        return (dominant, confidence)

    def _probs_hsemotion(self, face_bgr: np.ndarray) -> Dict[str, float]:
        """Run HSEmotion, returning {app_emotion: probability}."""
        import cv2
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # logits=True returns raw scores; we softmax them ourselves.
        _label, scores = self._hse.predict_emotions(face_rgb, logits=True)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        scores = scores - scores.max()
        exp = np.exp(scores)
        soft = exp / exp.sum()

        idx_to_class = self._hse.idx_to_class  # {0:'Anger', ...}
        probs: Dict[str, float] = {}
        for idx, cls in idx_to_class.items():
            app = _HSEMOTION_TO_APP.get(str(cls).lower(), "neutral")
            probs[app] = probs.get(app, 0.0) + float(soft[idx])
        return probs

    def _probs_deepface(self, face_bgr: np.ndarray) -> Dict[str, float]:
        """Run DeepFace emotion, returning {app_emotion: probability}."""
        from deepface import DeepFace
        results = DeepFace.analyze(
            img_path=face_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if not results:
            return {}
        analysis = results[0] if isinstance(results, list) else results
        emotion_scores = analysis.get("emotion", {})
        if not emotion_scores:
            return {}
        probs: Dict[str, float] = {}
        for cls, pct in emotion_scores.items():
            app = _DEEPFACE_TO_APP.get(str(cls).lower(), "neutral")
            probs[app] = probs.get(app, 0.0) + float(pct) / 100.0
        return probs

    def get_current_state(self) -> Dict[str, Any]:
        """Thread-safe getter for current emotion state.

        Returns dict with 'emotion' and 'confidence' keys.
        """
        with self._lock:
            return {
                "emotion": self.current_emotion,
                "confidence": self.current_confidence
            }

    def update(self, emotion: str, confidence: float) -> bool:
        """Thread-safe update of current emotion state.

        Args:
            emotion: One of SUPPORTED_EMOTIONS.
            confidence: Confidence score in [0.0, 1.0].

        Returns True if emotion changed from previous state.
        """
        with self._lock:
            changed = (emotion != self.current_emotion)
            self.current_emotion = emotion
            self.current_confidence = confidence
            return changed
