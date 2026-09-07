"""Unit tests for the viseme_driver library (ported from HeadAudio)."""
import math
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import viseme_driver as vd  # noqa: E402


def _pcm16(samples):
    """Pack a float array in [-1,1] into little-endian PCM16 bytes."""
    clipped = np.clip(np.asarray(samples), -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def _tone(freq, seconds, rate=24000, amp=0.4):
    t = np.arange(int(rate * seconds)) / rate
    return amp * np.sin(2 * math.pi * freq * t)


# --- model parsing -------------------------------------------------------

def test_model_parses_expected_prototype_count():
    protos = vd.load_model()
    assert len(protos) == 39
    for p in protos:
        assert 0 <= p.viseme < vd.MODEL_VISEMES_N
        assert p.mu.shape == (vd.MFCC_COEFF_N,)
        assert p.inv_cov.shape == (vd.MFCC_COEFF_N, vd.MFCC_COEFF_N)
        # inverse covariance must be symmetric
        assert np.allclose(p.inv_cov, p.inv_cov.T)


def test_model_has_silence_prototype():
    protos = vd.load_model()
    assert any(p.viseme == vd.MODEL_VISEME_SIL for p in protos)


# --- MFCC ----------------------------------------------------------------

def test_mfcc_shape_and_finiteness():
    mfcc = vd.MFCC()
    block = np.sin(2 * math.pi * 300 * np.arange(vd.MFCC_SAMPLES_N) / 16000)
    le, v = mfcc.compute(block)
    assert v.shape == (vd.MFCC_COEFF_N,)
    assert np.all(np.isfinite(v))
    assert math.isfinite(le)


def test_mfcc_is_deterministic():
    block = np.linspace(-0.5, 0.5, vd.MFCC_SAMPLES_N)
    a = vd.MFCC().compute(block)[1]
    b = vd.MFCC().compute(block)[1]
    assert np.array_equal(a, b)


# --- viseme mapping ------------------------------------------------------

def test_all_oculus_visemes_map_to_valid_vrm():
    valid = {"aa", "ih", "ou", "ee", "oh", "neutral"}
    assert set(vd.OCULUS_VISEME_NAMES) == set(vd.VISEME_TO_VRM.keys())
    for name, (vrm, weight) in vd.VISEME_TO_VRM.items():
        assert vrm in valid
        assert 0.0 <= weight <= 1.0


# --- HeadAudioAnalyzer ---------------------------------------------------

def test_headaudio_analyzer_silence_is_neutral():
    an = vd.HeadAudioAnalyzer(input_rate=24000, smoothing=1.0)
    frames = an.push_pcm16(_pcm16(np.zeros(24000)))  # 1s silence
    assert frames, "expected frames from 1s of audio"
    # Silence should not drive an open vowel.
    assert all(f.weight <= 0.2 for f in frames)


def test_headaudio_analyzer_speech_produces_visemes():
    an = vd.HeadAudioAnalyzer(input_rate=24000, smoothing=1.0)
    # A voiced-vowel-like tone complex should yield speaking frames.
    sig = _tone(180, 1.0) + 0.3 * _tone(800, 1.0)
    frames = an.push_pcm16(_pcm16(sig))
    assert frames
    for f in frames:
        assert f.vrm in {"aa", "ih", "ou", "ee", "oh", "neutral"}
        assert 0.0 <= f.weight <= 1.0
    assert any(f.speaking for f in frames), "expected at least one speaking frame"


def test_headaudio_analyzer_frame_rate_is_reasonable():
    an = vd.HeadAudioAnalyzer(input_rate=24000, smoothing=1.0)
    frames = an.push_pcm16(_pcm16(_tone(220, 1.0)))
    # ~62.5 Hz at 16 kHz after 1 s of audio; allow warm-up slack.
    assert 55 <= len(frames) <= 70


def test_reset_clears_state():
    an = vd.HeadAudioAnalyzer(input_rate=24000)
    an.push_pcm16(_pcm16(_tone(300, 0.5)))
    an.reset()
    assert len(an._buf) == 0
    assert an._smoothed_weight == 0.0


# --- AmplitudeAnalyzer + factory ----------------------------------------

def test_amplitude_analyzer_reacts_to_energy():
    an = vd.AmplitudeAnalyzer(input_rate=24000, smoothing=1.0)
    quiet = an.push_pcm16(_pcm16(np.zeros(2400)))
    loud = an.push_pcm16(_pcm16(_tone(300, 0.1, amp=0.9)))
    assert quiet[0].weight == 0.0
    assert loud[0].weight > quiet[0].weight


def test_create_analyzer_returns_headaudio_when_model_present():
    an = vd.create_analyzer("headaudio", input_rate=24000)
    assert isinstance(an, vd.HeadAudioAnalyzer)


def test_create_analyzer_falls_back_on_missing_model():
    an = vd.create_analyzer("headaudio", input_rate=24000, model_path="/nonexistent.bin")
    assert isinstance(an, vd.AmplitudeAnalyzer)
