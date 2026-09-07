"""
Real-time audio-to-viseme driver.

This is a faithful Python port of met4citizen/HeadAudio (MIT License,
Copyright (c) 2025 Mika Suominen), which performs audio-driven viseme
detection using MFCC feature vectors, Gaussian prototypes, and a
Mahalanobis-distance classifier. The original runs as a browser
AudioWorklet; this port lets the Python side own the phoneme/viseme
command stream so both the on-screen VRM avatar and the physical robot
servos can consume the same events.

The port matches HeadAudio's DSP parameters and algorithm so the bundled
pretrained model (``model-en-mixed.bin``, 15 Oculus visemes) can be used
directly. Verified against the original JavaScript via Node.js on
identical inputs (see tests).

Design:
    - ``VisemeFrame``: the abstract command emitted per analysis frame.
    - ``Analyzer``: base interface (push PCM16 -> list of VisemeFrame).
    - ``HeadAudioAnalyzer``: the ported MFCC + Mahalanobis classifier.
    - ``AmplitudeAnalyzer``: a dependency-light energy-only fallback
      (the project's original behaviour), used when the model or numpy
      is unavailable.

Content derived from HeadAudio was reimplemented in Python; see LICENSE
notes in vendor/headaudio/.
"""

from __future__ import annotations

import math
import os
import struct
from array import array
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised only without numpy
    _HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# Parameters (ported from HeadAudio modules/parameters.mjs)
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 16000
AUDIO_DOWNSAMPLE_FILTER_N = 32
AUDIO_DOWNSAMPLE_PHASE_N = 64
AUDIO_PREEMPHASIS_ENABLED = True
AUDIO_PREEMPHASIS_ALPHA = 0.97

MFCC_SAMPLES_N = 512
MFCC_SAMPLES_HOP = 256
MFCC_COEFF_N = 12
MFCC_MEL_BANDS_N = 40
MFCC_LIFTER = 22
MFCC_COMPRESSION_ENABLED = True
MFCC_COMPRESSION_TANH_R = 1.0

MODEL_VISEMES_N = 15
MODEL_VISEME_SIL = 14

# Binary record layout (bytes). Header is 8 bytes: packed phoneme (big-endian
# uint32), reserved, group (byte 5), reserved, viseme (byte 7). Then mu
# (12 x float32, little-endian) and sigmaInvLower (78 x float32, little-endian).
_RECORD_MU_OFFSET = 8
_RECORD_MU_LEN = MFCC_COEFF_N
_RECORD_SIGMA_OFFSET = _RECORD_MU_OFFSET + _RECORD_MU_LEN * 4
_RECORD_SIGMA_LEN = MFCC_COEFF_N * (MFCC_COEFF_N + 1) // 2  # 78
_RECORD_BYTES = _RECORD_SIGMA_OFFSET + _RECORD_SIGMA_LEN * 4  # 368

# Oculus viseme IDs -> names (HeadAudio Appendix A)
OCULUS_VISEME_NAMES = [
    "aa", "E", "I", "O", "U", "PP", "SS", "TH",
    "DD", "FF", "kk", "nn", "RR", "CH", "sil",
]

# Oculus viseme -> (VRM expression name, mouth-open weight scale).
# VRM 1.0 exposes only five vowel visemes (aa/ih/ou/ee/oh) plus neutral, so the
# 15 Oculus visemes are reduced. Consonants keep a small opening so the mouth
# still articulates; closures collapse to neutral.
VISEME_TO_VRM = {
    "aa": ("aa", 1.00),
    "E":  ("ee", 0.90),
    "I":  ("ih", 0.85),
    "O":  ("oh", 0.95),
    "U":  ("ou", 0.90),
    "PP": ("neutral", 0.00),   # bilabial closure
    "SS": ("ih", 0.45),
    "TH": ("ih", 0.40),
    "DD": ("ih", 0.45),
    "FF": ("ih", 0.40),
    "kk": ("aa", 0.45),
    "nn": ("neutral", 0.10),
    "RR": ("oh", 0.50),
    "CH": ("ou", 0.60),
    "sil": ("neutral", 0.00),
}

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vendor", "headaudio", "model-en-mixed.bin",
)


@dataclass
class VisemeFrame:
    """A single viseme command in the abstract control contract.

    Attributes:
        vrm: VRM expression name (``aa``/``ih``/``ou``/``ee``/``oh``/``neutral``).
        weight: Expression weight in [0, 1].
        oculus: Source Oculus viseme name (diagnostic), or None.
        speaking: Whether speech is currently active.
    """
    vrm: str
    weight: float
    oculus: Optional[str] = None
    speaking: bool = True


# ---------------------------------------------------------------------------
# MFCC (ported from HeadAudio modules/mfcc.mjs)
# ---------------------------------------------------------------------------

class MFCC:
    """Mel-frequency cepstral coefficients, matching HeadAudio's pipeline."""

    def __init__(self, speaker_mean_hz: float = 150.0):
        n = MFCC_SAMPLES_N
        # Hamming window (length N, denominator N-1)
        self.hamming = np.array(
            [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)],
            dtype=np.float64,
        )
        self.mel_filters = self._build_mel_filterbank(speaker_mean_hz)
        self.dct_matrix = self._build_dct_matrix()   # rows 1..12 (12 x 40)
        self.lifter = self._build_lifter()

    def _build_mel_filterbank(self, f0: float) -> "np.ndarray":
        n = MFCC_SAMPLES_N
        half = n // 2
        f0_ref = 150.0
        warp = min(max(f0 / f0_ref, 0.6), 1.8)
        low_freq, high_freq = 30.0, 7800.0

        def mel(f):
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_inv(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_low = mel(low_freq)
        mel_high = mel(high_freq)
        mel_range = mel_high - mel_low

        mel_points = []
        for i in range(MFCC_MEL_BANDS_N + 2):
            m = mel_low + (mel_range / (MFCC_MEL_BANDS_N + 1)) * i
            m_warped = mel_low + (m - mel_low) * warp
            mel_points.append(mel_inv(m_warped))

        bins = [int(math.floor((f / AUDIO_SAMPLE_RATE) * n)) for f in mel_points]

        filters = np.zeros((MFCC_MEL_BANDS_N, half), dtype=np.float64)
        for i in range(MFCC_MEL_BANDS_N):
            for k in range(bins[i], bins[i + 1]):
                if 0 <= k < half:
                    filters[i, k] = (k - bins[i]) / (bins[i + 1] - bins[i])
            for k in range(bins[i + 1], bins[i + 2]):
                if 0 <= k < half:
                    filters[i, k] = (bins[i + 2] - k) / (bins[i + 2] - bins[i + 1])
        return filters

    def _build_dct_matrix(self) -> "np.ndarray":
        scale = math.sqrt(2.0 / MFCC_MEL_BANDS_N)
        # HeadAudio builds rows 0..MFCC_COEFF_N and uses rows 1..MFCC_COEFF_N.
        rows = []
        for i in range(1, MFCC_COEFF_N + 1):
            row = [
                scale * math.cos((math.pi * i * (j + 0.5)) / MFCC_MEL_BANDS_N)
                for j in range(MFCC_MEL_BANDS_N)
            ]
            rows.append(row)
        return np.array(rows, dtype=np.float64)  # (12, 40)

    def _build_lifter(self) -> "np.ndarray":
        return np.array(
            [
                1.0 + (MFCC_LIFTER / 2.0) * math.sin((math.pi * i) / MFCC_LIFTER)
                for i in range(1, MFCC_COEFF_N + 1)
            ],
            dtype=np.float64,
        )

    def compute(self, block: "np.ndarray"):
        """Compute (log_energy, mfcc[12]) for a 512-sample block."""
        half = MFCC_SAMPLES_N // 2
        windowed = block * self.hamming
        spec = np.fft.rfft(windowed, n=MFCC_SAMPLES_N)  # length half+1
        # HeadAudio uses bins 0..half-1 and normalizes by N.
        power = (np.abs(spec[:half]) ** 2) / MFCC_SAMPLES_N
        total_energy = float(np.sum(power))
        le = math.log10(total_energy + 1e-10)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mel_energies = np.log10(self.mel_filters @ power + 1e-10)  # (40,)
        mfcc = (self.dct_matrix @ mel_energies) * self.lifter           # (12,)
        return le, mfcc


# ---------------------------------------------------------------------------
# Polyphase downsampler + pre-emphasis (ported from processor.mjs)
# ---------------------------------------------------------------------------

class _Downsampler:
    """Streaming polyphase resampler from ``input_rate`` to 16 kHz with
    pre-emphasis, matching HeadAudio's per-sample algorithm."""

    def __init__(self, input_rate: int):
        self.ratio = input_rate / AUDIO_SAMPLE_RATE
        self.acc = 0.0
        self.preemph_prev = 0.0
        self.poly = self._design_polyphase()          # (PHASE_N, FILTER_N)
        self.ring = [0.0] * AUDIO_DOWNSAMPLE_FILTER_N  # circular tap buffer
        self.ring_head = 0                             # index of oldest tap

    @staticmethod
    def _design_polyphase() -> "np.ndarray":
        table = np.zeros((AUDIO_DOWNSAMPLE_PHASE_N, AUDIO_DOWNSAMPLE_FILTER_N),
                         dtype=np.float64)
        mid = (AUDIO_DOWNSAMPLE_FILTER_N - 1) / 2.0
        cutoff = 0.45
        for p in range(AUDIO_DOWNSAMPLE_PHASE_N):
            phase = p / AUDIO_DOWNSAMPLE_PHASE_N
            for i in range(AUDIO_DOWNSAMPLE_FILTER_N):
                x = i - mid - phase
                if x == 0:
                    c = 2 * cutoff
                else:
                    c = math.sin(2 * math.pi * cutoff * x) / (math.pi * x)
                c *= 0.54 - 0.46 * math.cos(
                    (2 * math.pi * i) / (AUDIO_DOWNSAMPLE_FILTER_N - 1)
                )
                table[p, i] = c
        return table

    def process(self, samples) -> List[float]:
        """Feed float samples, return list of 16 kHz output samples."""
        out: List[float] = []
        ring = self.ring
        n = AUDIO_DOWNSAMPLE_FILTER_N
        acc = self.acc
        prev = self.preemph_prev
        head = self.ring_head
        inv_ratio = 1.0 / self.ratio
        poly = self.poly

        for d in samples:
            if AUDIO_PREEMPHASIS_ENABLED:
                orig = d
                d = d - AUDIO_PREEMPHASIS_ALPHA * prev
                prev = orig
            # push d into ring (overwrite oldest)
            ring[head] = d
            head = (head + 1) % n
            acc += inv_ratio
            while acc >= 1.0:
                phase_fraction = (acc - 1.0) * AUDIO_DOWNSAMPLE_PHASE_N
                phase_index = int(math.floor(phase_fraction))
                if phase_index >= AUDIO_DOWNSAMPLE_PHASE_N:
                    phase_index = AUDIO_DOWNSAMPLE_PHASE_N - 1
                coeffs = poly[phase_index]
                # getHead(j): j=0 is oldest tap == ring[head] (next-to-write is
                # oldest after wrap). Oldest element index is `head`.
                acc_sum = 0.0
                idx = head
                for j in range(n):
                    acc_sum += ring[idx] * coeffs[j]
                    idx += 1
                    if idx == n:
                        idx = 0
                out.append(acc_sum)
                acc -= 1.0
        self.acc = acc
        self.preemph_prev = prev
        self.ring_head = head
        return out


# ---------------------------------------------------------------------------
# Gaussian prototype model + Mahalanobis classifier (ported from classifier.mjs)
# ---------------------------------------------------------------------------

@dataclass
class _Prototype:
    group: int
    viseme: int
    mu: "np.ndarray"       # (12,)
    inv_cov: "np.ndarray"  # (12, 12) full symmetric


def load_model(path: str = DEFAULT_MODEL_PATH) -> List[_Prototype]:
    """Parse a HeadAudio ``.bin`` Gaussian prototype model."""
    with open(path, "rb") as f:
        data = f.read()

    # Lower-triangular packing index map: index runs i=0.., j=0..i.
    idx_map = {}
    counter = 0
    for i in range(MFCC_COEFF_N):
        for j in range(i + 1):
            idx_map[(i, j)] = counter
            counter += 1

    protos: List[_Prototype] = []
    pos = 0
    n = len(data)
    while pos + _RECORD_BYTES <= n:
        rec = data[pos:pos + _RECORD_BYTES]
        group = rec[5]
        viseme = rec[7]
        mu = np.frombuffer(rec, dtype="<f4", count=_RECORD_MU_LEN,
                           offset=_RECORD_MU_OFFSET).astype(np.float64)
        sigma_lower = np.frombuffer(rec, dtype="<f4", count=_RECORD_SIGMA_LEN,
                                    offset=_RECORD_SIGMA_OFFSET).astype(np.float64)
        # Reconstruct full symmetric inverse-covariance matrix.
        inv_cov = np.zeros((MFCC_COEFF_N, MFCC_COEFF_N), dtype=np.float64)
        for i in range(MFCC_COEFF_N):
            for j in range(i + 1):
                v = sigma_lower[idx_map[(i, j)]]
                inv_cov[i, j] = v
                inv_cov[j, i] = v
        protos.append(_Prototype(group=group, viseme=viseme, mu=mu, inv_cov=inv_cov))
        pos += _RECORD_BYTES
    return protos


class Classifier:
    """Mahalanobis nearest-prototype classifier with majority voting."""

    def __init__(self, prototypes: List[_Prototype], sil_sensitivity: float = 1.2,
                 vote_window: int = 6):
        self.prototypes = prototypes
        self.sil_sensitivity = sil_sensitivity
        self.ring = deque([MODEL_VISEME_SIL] * vote_window, maxlen=vote_window)
        self.prediction_last = MODEL_VISEME_SIL
        # Raw (pre-vote) nearest-prototype viseme from the most recent predict().
        # The majority vote deliberately lags to stay stable; exposing the raw
        # decision lets the analyzer react instantly at speech onset without
        # changing the (verified) voted output.
        self.raw_last = MODEL_VISEME_SIL

    def predict(self, vector: "np.ndarray") -> Optional[int]:
        if not self.prototypes:
            return None
        min_d = math.inf
        min_v = MODEL_VISEME_SIL
        for p in self.prototypes:
            diff = vector - p.mu
            d = float(diff @ (p.inv_cov @ diff))
            if p.viseme == MODEL_VISEME_SIL:
                d /= self.sil_sensitivity
            if d <= min_d:
                min_d = d
                min_v = p.viseme

        self.raw_last = min_v
        self.ring.append(min_v)
        counts = [0] * MODEL_VISEMES_N
        for v in self.ring:
            counts[v] += 1
        viseme = 0
        max_count = 0
        for i in range(MODEL_VISEMES_N):
            if counts[i] >= max_count:
                viseme = i
                max_count = counts[i]

        if viseme == self.prediction_last and viseme == MODEL_VISEME_SIL:
            return None
        self.prediction_last = viseme
        return viseme


# ---------------------------------------------------------------------------
# Analyzer interface
# ---------------------------------------------------------------------------

class Analyzer:
    """Base analyzer: consumes PCM16 mono bytes, emits VisemeFrame list."""

    def push_pcm16(self, pcm16_bytes: bytes) -> List[VisemeFrame]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class HeadAudioAnalyzer(Analyzer):
    """Ported HeadAudio MFCC + Mahalanobis viseme detection.

    Args:
        input_rate: Sample rate of the incoming PCM16 audio (e.g. 24000 for
            OpenAI Realtime output).
        model_path: Path to the HeadAudio ``.bin`` prototype model.
        speaker_mean_hz: Speaker mean frequency (mel warp).
        smoothing: Exponential smoothing factor for output weight [0,1].
    """

    def __init__(self, input_rate: int = 24000, model_path: str = DEFAULT_MODEL_PATH,
                 speaker_mean_hz: float = 150.0, smoothing: float = 0.5,
                 attack: Optional[float] = None, fast_attack: bool = True):
        if not _HAVE_NUMPY:
            raise RuntimeError("HeadAudioAnalyzer requires numpy")
        self.input_rate = input_rate
        self.downsampler = _Downsampler(input_rate)
        self.mfcc = MFCC(speaker_mean_hz)
        self.classifier = Classifier(load_model(model_path))
        self.smoothing = max(0.0, min(1.0, smoothing))

        # Asymmetric weight smoothing: open the mouth quickly (attack) but close
        # it gently (release) so speech onsets are crisp without the shape
        # flickering. `smoothing` is the release rate; attack defaults faster.
        self.release = self.smoothing
        self.attack = (max(0.0, min(1.0, attack)) if attack is not None
                       else max(self.smoothing, 0.9))
        # When True, react to the classifier's raw prediction while not yet
        # speaking so the mouth doesn't wait for the majority vote to fill at
        # onset (~50 ms of lag). Steady-state still uses the voted viseme.
        self.fast_attack = fast_attack
        self._speaking = False

        # 16 kHz sample ring for framing (512 window, 256 hop).
        self._buf = array("d")
        self._smoothed_weight = 0.0
        self._last_vrm = "neutral"

        # Optional per-frame feature capture for verification/testing. When set
        # to a list, each post-compression MFCC vector is appended.
        self.debug_vectors: Optional[list] = None

    def reset(self) -> None:
        self.downsampler = _Downsampler(self.input_rate)
        del self._buf[:]
        self._smoothed_weight = 0.0
        self._last_vrm = "neutral"
        self._speaking = False
        self.classifier.ring.clear()
        for _ in range(self.classifier.ring.maxlen):
            self.classifier.ring.append(MODEL_VISEME_SIL)
        self.classifier.prediction_last = MODEL_VISEME_SIL
        self.classifier.raw_last = MODEL_VISEME_SIL

    def push_pcm16(self, pcm16_bytes: bytes) -> List[VisemeFrame]:
        if not pcm16_bytes:
            return []
        samples = np.frombuffer(pcm16_bytes, dtype="<i2").astype(np.float64) / 32768.0
        down = self.downsampler.process(samples.tolist())
        self._buf.extend(down)

        frames: List[VisemeFrame] = []
        buf = self._buf
        while len(buf) >= MFCC_SAMPLES_N:
            block = np.frombuffer(buf[:MFCC_SAMPLES_N], dtype=np.float64).copy()
            # advance by hop
            del buf[:MFCC_SAMPLES_HOP]

            le, mfcc = self.mfcc.compute(block)
            if MFCC_COMPRESSION_ENABLED:
                mfcc = MFCC_COMPRESSION_TANH_R * np.tanh(mfcc / MFCC_COMPRESSION_TANH_R)

            if self.debug_vectors is not None:
                self.debug_vectors.append(mfcc.copy())

            viseme_id = self.classifier.predict(mfcc)

            # Fast attack: while not already speaking, adopt the classifier's
            # raw (pre-vote) decision so the mouth opens on the first speaking
            # frame instead of waiting for the majority vote to fill. Once
            # speaking, use the stable voted viseme.
            use_id = viseme_id
            if self.fast_attack and not self._speaking:
                raw_id = self.classifier.raw_last
                if raw_id is not None and raw_id != MODEL_VISEME_SIL:
                    use_id = raw_id

            if use_id is None or use_id == MODEL_VISEME_SIL:
                target_vrm, target_w = "neutral", 0.0
                oculus = "sil"
                speaking = False
            else:
                oculus = OCULUS_VISEME_NAMES[use_id]
                target_vrm, target_w = VISEME_TO_VRM.get(oculus, ("neutral", 0.0))
                speaking = target_vrm != "neutral"
            self._speaking = speaking

            # Asymmetric smoothing: fast to open, gentle to close.
            alpha = self.attack if target_w > self._smoothed_weight else self.release
            self._smoothed_weight += alpha * (target_w - self._smoothed_weight)
            self._last_vrm = target_vrm if target_w > 0 else self._last_vrm
            frames.append(VisemeFrame(
                vrm=target_vrm,
                weight=round(self._smoothed_weight, 4),
                oculus=oculus,
                speaking=speaking,
            ))
        return frames


class AmplitudeAnalyzer(Analyzer):
    """Dependency-light fallback: maps short-time energy to an open-vowel
    weight. Mirrors the project's original amplitude behaviour and needs no
    model or numpy."""

    def __init__(self, input_rate: int = 24000, min_threshold: float = 0.015,
                 max_threshold: float = 0.25, smoothing: float = 0.4):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.smoothing = smoothing
        self._current = 0.0

    def push_pcm16(self, pcm16_bytes: bytes) -> List[VisemeFrame]:
        if not pcm16_bytes:
            return []
        samples = array("h")
        samples.frombytes(pcm16_bytes[: len(pcm16_bytes) // 2 * 2])
        if not samples:
            return []
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
        if rms < self.min_threshold:
            target = 0.0
        else:
            norm = (rms - self.min_threshold) / (self.max_threshold - self.min_threshold)
            target = max(0.0, min(1.0, norm)) ** 0.8
        self._current += self.smoothing * (target - self._current)
        weight = round(self._current, 4)
        vrm = "aa" if weight > 0.05 else "neutral"
        return [VisemeFrame(vrm=vrm, weight=weight if vrm == "aa" else 0.0,
                            oculus=None, speaking=weight > 0.05)]

    def reset(self) -> None:
        self._current = 0.0


def create_analyzer(kind: str = "headaudio", input_rate: int = 24000,
                    model_path: str = DEFAULT_MODEL_PATH) -> Analyzer:
    """Factory: return the requested analyzer, falling back to amplitude if
    the HeadAudio model or numpy is unavailable."""
    if kind == "headaudio" and _HAVE_NUMPY and os.path.isfile(model_path):
        try:
            return HeadAudioAnalyzer(input_rate=input_rate, model_path=model_path)
        except Exception:
            pass
    return AmplitudeAnalyzer(input_rate=input_rate)
