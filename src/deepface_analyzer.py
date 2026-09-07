"""
DeepFace Analyzer Module
Background thread that runs face recognition and emotion detection at 1-2 second intervals.
Provides thread-safe shared state for identity and emotion, and emits callbacks on changes.
"""

import logging
import threading
import time
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from emotion_detector import EmotionDetector
from face_recognition_system import FaceRecognitionSystem

logger = logging.getLogger(__name__)


class DeepFaceAnalyzer:
    """Background thread for face recognition and emotion detection.

    Runs DeepFace analysis every `analysis_interval` seconds on the latest
    frame provided by the main thread. Updates shared state with current
    identity and emotion, and emits callbacks on changes.
    """

    # Time threshold for considering a face "lost" (seconds)
    FACE_LOST_THRESHOLD = 5.0

    # Minimum time between duplicate identity events for the same person (seconds)
    IDENTITY_EVENT_DEDUP_WINDOW = 5.0

    def __init__(self, face_recognition: FaceRecognitionSystem,
                 emotion_detector: EmotionDetector,
                 analysis_interval: float = 1.5,
                 emotion_smoothing_window: int = 1):
        """Initialize the DeepFace Analyzer.

        Args:
            face_recognition: FaceRecognitionSystem instance for identification.
            emotion_detector: EmotionDetector instance for emotion analysis.
            analysis_interval: Seconds between analysis cycles (default 1.5s).
            emotion_smoothing_window: Number of recent cycles to majority-vote
                over before committing an emotion. Smooths out single-frame
                flips (1 = no smoothing).
        """
        self.face_recognition = face_recognition
        self.emotion_detector = emotion_detector
        self.analysis_interval = analysis_interval

        # Temporal smoothing for emotion: keep the last N raw results and commit
        # the majority so a single noisy frame can't flip the displayed emotion.
        self.emotion_smoothing_window = max(1, int(emotion_smoothing_window))
        self._emotion_history: deque = deque(maxlen=self.emotion_smoothing_window)

        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Frame buffer (written by main thread, read by background thread)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Shared state (written by background thread, read by anyone)
        self._identity_lock = threading.Lock()
        self._current_identity: Optional[str] = None
        self._identity_confidence: float = 0.0

        # Identity event tracking
        self._last_identity_event_time: float = 0.0
        self._last_identity_event_name: Optional[str] = None
        self._face_last_seen_time: Optional[float] = None
        self._face_lost_time: Optional[float] = None
        self._session_started: bool = False

        # Callbacks
        self.on_identity_changed: Optional[Callable[[str, float], None]] = None
        self.on_emotion_changed: Optional[Callable[[str, float], None]] = None

    # --- Frame Update (called from main thread) ---

    def update_frame(self, frame: np.ndarray) -> None:
        """Provide the latest camera frame for background analysis.

        Called from the main face tracking thread. This method is non-blocking
        and simply stores a copy of the frame for the background thread to pick up.

        Args:
            frame: BGR image from the camera (numpy array).
        """
        with self._frame_lock:
            self._latest_frame = frame.copy()

    # --- Thread Lifecycle ---

    def start(self) -> None:
        """Start the background analysis thread."""
        if self._running:
            logger.warning("DeepFaceAnalyzer is already running")
            return

        self._running = True
        self._session_started = False
        self._thread = threading.Thread(
            target=self._analysis_loop,
            name="DeepFaceAnalyzer",
            daemon=True
        )
        self._thread.start()
        logger.info("DeepFaceAnalyzer background thread started "
                    f"(interval={self.analysis_interval}s)")

    def stop(self) -> None:
        """Stop the background analysis thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.analysis_interval * 2)
            if self._thread.is_alive():
                logger.warning("DeepFaceAnalyzer thread did not stop cleanly")
            self._thread = None
        logger.info("DeepFaceAnalyzer background thread stopped")

    # --- Thread-Safe Getters ---

    def get_identity(self) -> Tuple[Optional[str], float]:
        """Get the current recognized identity (thread-safe).

        Returns:
            Tuple of (person_name, confidence) or (None, 0.0) if no one recognized.
        """
        with self._identity_lock:
            return (self._current_identity, self._identity_confidence)

    def get_emotion(self) -> Dict[str, Any]:
        """Get the current emotion state (thread-safe).

        Returns:
            Dict with 'emotion' and 'confidence' keys.
        """
        return self.emotion_detector.get_current_state()

    # --- Background Analysis Loop ---

    def _analysis_loop(self) -> None:
        """Main analysis loop running in the background thread.

        Grabs the latest frame, runs face detection, recognition, and emotion
        analysis, then updates shared state and emits callbacks as needed.
        """
        logger.debug("DeepFaceAnalyzer analysis loop started")

        while self._running:
            loop_start = time.time()

            try:
                self._run_analysis_cycle()
            except Exception as e:
                logger.error(f"DeepFaceAnalyzer analysis cycle error: {e}")

            # Sleep for the remainder of the interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.analysis_interval - elapsed)
            if sleep_time > 0:
                # Use short sleep intervals so we can stop quickly
                end_time = time.time() + sleep_time
                while self._running and time.time() < end_time:
                    time.sleep(min(0.1, end_time - time.time()))

        logger.debug("DeepFaceAnalyzer analysis loop exited")

    def _run_analysis_cycle(self) -> None:
        """Execute a single analysis cycle: detect, recognize, analyze emotion."""
        # Grab the latest frame
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None  # Consume the frame

        if frame is None:
            # No new frame available — check for face lost
            self._handle_no_frame()
            return

        # Detect faces using Haar cascade (lightweight detection)
        face_regions = self._detect_faces(frame)

        if not face_regions:
            # No face detected in this cycle
            self._handle_face_lost()
            return

        # Use the largest face for recognition and emotion
        largest_face = max(face_regions, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_face
        face_image = frame[y:y+h, x:x+w]

        # Mark face as present
        current_time = time.time()
        self._face_last_seen_time = current_time

        # --- Face Recognition ---
        identity, confidence = self.face_recognition.identify(face_image)
        self._update_identity(identity, confidence, current_time)

        # --- Emotion Detection ---
        raw_emotion, raw_confidence = self.emotion_detector.analyze(face_image)
        # Temporal smoothing: commit the majority emotion over the recent window
        # so a single noisy frame can't flip the displayed state.
        self._emotion_history.append((raw_emotion, raw_confidence))
        emotion, emotion_confidence = self._smoothed_emotion()
        emotion_changed = self.emotion_detector.update(emotion, emotion_confidence)

        if emotion != "neutral" or emotion_changed:
            print(f"   🎭 Emotion: {emotion} ({emotion_confidence:.2f}) changed={emotion_changed}")

        if emotion_changed and self.on_emotion_changed is not None:
            try:
                self.on_emotion_changed(emotion, emotion_confidence)
            except Exception as e:
                logger.error(f"on_emotion_changed callback error: {e}")

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame using Haar cascade.

        Returns list of (x, y, w, h) tuples for detected faces.
        """
        import os
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cascade_path = os.path.join(
            os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'
        )
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if faces is None or len(faces) == 0:
            return []

        return [tuple(f) for f in faces]

    # --- Identity Event Logic ---

    def _update_identity(self, identity: Optional[str], confidence: float,
                         current_time: float) -> None:
        """Update identity state and emit events as appropriate.

        Handles:
        - First detection in a session
        - Identity change (different person)
        - Re-detection after absence (>5 seconds)
        - Deduplication (no duplicate events within 5-second window)
        """
        previous_identity = self._current_identity
        was_face_lost = self._face_lost_time is not None

        # Calculate absence duration if face was lost
        absence_duration = 0.0
        if was_face_lost and self._face_lost_time is not None:
            absence_duration = current_time - self._face_lost_time

        # Clear face-lost state since we see a face now
        self._face_lost_time = None

        # Update shared state
        with self._identity_lock:
            self._current_identity = identity
            self._identity_confidence = confidence

        # Determine if we should emit an identity event
        should_emit = False
        reason = ""

        if not self._session_started:
            # First detection in this session (known or unknown)
            self._session_started = True
            should_emit = True
            reason = "first_detection"
        elif identity != previous_identity:
            # Identity changed (different person, or known→unknown, or unknown→known)
            should_emit = True
            reason = "identity_changed"
        elif identity is not None and was_face_lost and absence_duration > self.FACE_LOST_THRESHOLD:
            # Same person re-detected after extended absence
            should_emit = True
            reason = "re_detection_after_absence"

        if should_emit:
            self._try_emit_identity_event(identity, confidence, current_time, reason)

    def _try_emit_identity_event(self, identity: Optional[str], confidence: float,
                                 current_time: float, reason: str) -> None:
        """Emit an identity event if deduplication allows it.

        Deduplication rule: no duplicate events for the same person within
        a 5-second window.
        """
        # Check deduplication
        if (self._last_identity_event_name == identity and
                (current_time - self._last_identity_event_time) < self.IDENTITY_EVENT_DEDUP_WINDOW):
            logger.debug(f"Identity event deduplicated for '{identity}' "
                         f"(within {self.IDENTITY_EVENT_DEDUP_WINDOW}s window)")
            return

        # Emit the event
        self._last_identity_event_time = current_time
        self._last_identity_event_name = identity

        logger.info(f"Identity event: '{identity}' (confidence={confidence:.2f}, "
                    f"reason={reason})")

        if self.on_identity_changed is not None:
            try:
                self.on_identity_changed(identity, confidence)
            except Exception as e:
                logger.error(f"on_identity_changed callback error: {e}")

    def _handle_face_lost(self) -> None:
        """Handle the case where no face is detected in the current cycle."""
        current_time = time.time()

        # Clear emotion history so a returning face starts fresh (and relax to
        # neutral rather than holding a stale emotion while nobody is present).
        self._emotion_history.clear()

        if self._face_lost_time is None and self._face_last_seen_time is not None:
            # Face just disappeared — record the time
            self._face_lost_time = current_time
            logger.debug("Face lost — starting absence timer")

    def _smoothed_emotion(self) -> Tuple[str, float]:
        """Majority-vote the recent emotion window.

        Returns the most common emotion across the window and the average
        confidence of the votes that agree with it. With a window of 1 this is
        just the latest raw result.
        """
        if not self._emotion_history:
            return ("neutral", 0.0)
        labels = [e for e, _ in self._emotion_history]
        winner, _count = Counter(labels).most_common(1)[0]
        confs = [c for e, c in self._emotion_history if e == winner]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return (winner, avg_conf)

    def _handle_no_frame(self) -> None:
        """Handle the case where no new frame is available.

        This doesn't necessarily mean the face is lost — it could just mean
        the main thread hasn't provided a new frame yet. We don't update
        face-lost state here.
        """
        pass
