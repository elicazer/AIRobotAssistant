"""Dual FT232H + PCA9685 animatronic head driver for the Will Cogley head.

Shared, safe hardware layer used by the voice assistant (and reusable by tools).
It loads the calibrated actuator map (config/servo_calibration.json) and the named
poses (config/poses.json), and exposes high-level control:

  - arm() / disarm(): open/close the USB link. UNARMED BY DEFAULT — no method moves
    a servo until arm() has been called, and disarm() releases every channel.
  - set_norm(key, v): drive one logical actuator to a calibration-relative value
    in [-1, 1] (0=home, +1=up/open/left/out, -1=down/closed/right). Jaw resolves to
    its linked servo pair automatically.
  - apply_pose(kind, name): strike a whole expression/viseme pose at once.
  - set_jaw_open(frac): lip-sync helper, frac 0..1 -> closed..speech-open.
  - set_eyes(x, y): gaze helper for face tracking (normalized -1..1).
  - blink(): quick eyelid close/open.
  - release_all() / shutdown(): stop driving all channels.

Every value is clamped to each actuator's calibrated travel. Pose values are
resolved against the CURRENT calibration at call time, so recalibrating a servo (or
swapping one and re-capturing its named positions) carries through automatically.

This module owns the FT232H boards while armed; the calibration web server must not
be running at the same time (only one process can open the adapters).
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CALIB_PATH = os.path.join(_ROOT, "config", "servo_calibration.json")
POSES_PATH = os.path.join(_ROOT, "config", "poses.json")

PCA_ADDR = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
OSC_HZ = 25_000_000.0
TARGET_HZ = 50.0

# Absolute pulse backstop (matches the calibration tool's widened range).
ABS_MIN_US = 400
ABS_MAX_US = 2600

# How far the jaw opens at full speech amplitude, as a normalized jaw value.
# Small on purpose: natural speech barely drops the jaw. Tunable via settings.
DEFAULT_JAW_SPEECH_OPEN = 0.18


class _Board:
    """One FT232H + PCA9685 on a stable serial URL. Serialized by a lock."""

    def __init__(self, url):
        self.url = url
        self._lock = threading.Lock()
        self._ctrl = None
        self._port = None
        self._prescale = None

    def open(self):
        from pyftdi.i2c import I2cController
        self._ctrl = I2cController()
        self._ctrl.configure(self.url, frequency=400000)
        self._port = self._ctrl.get_port(PCA_ADDR)
        self._ensure_awake_50hz()

    def close(self):
        try:
            if self._ctrl is not None:
                self._ctrl.terminate()
        except Exception:
            pass
        self._ctrl = None
        self._port = None

    def _rb(self, reg):
        return self._port.read_from(reg, 1)[0]

    def _ensure_awake_50hz(self):
        want = int(round(OSC_HZ / (4096.0 * TARGET_HZ))) - 1
        prescale = self._rb(PRESCALE)
        mode1 = self._rb(MODE1)
        if prescale != want:
            self._port.write_to(MODE1, [(mode1 & 0x7F) | 0x10])
            self._port.write_to(PRESCALE, [want])
            wake = mode1 & ~0x10 & 0xFF
            self._port.write_to(MODE1, [wake])
            time.sleep(0.005)
            self._port.write_to(MODE1, [wake | 0xA0])
            prescale = want
        elif mode1 & 0x10:
            wake = mode1 & ~0x10 & 0xFF
            self._port.write_to(MODE1, [wake])
            time.sleep(0.005)
            self._port.write_to(MODE1, [wake | 0xA0])
        self._prescale = prescale

    def _us_per_tick(self):
        return (self._prescale + 1) / (OSC_HZ / 1_000_000.0)

    def set_us(self, channel, us):
        us = max(ABS_MIN_US, min(ABS_MAX_US, int(us)))
        ticks = max(0, min(4095, int(round(us / self._us_per_tick()))))
        base = LED0_ON_L + 4 * channel
        with self._lock:
            self._port.write_to(base, [0x00, 0x00, ticks & 0xFF, (ticks >> 8) & 0x0F])
        return us

    def release(self, channel):
        base = LED0_ON_L + 4 * channel
        with self._lock:
            self._port.write_to(base, [0x00, 0x00, 0x00, 0x10])

    def release_all(self):
        with self._lock:
            for ch in range(16):
                base = LED0_ON_L + 4 * ch
                self._port.write_to(base, [0x00, 0x00, 0x00, 0x10])


# Actuators that make up the mouth (driven by visemes). Everything else — eyes,
# eyelids, eyebrows — is owned by gaze/expression so visemes never fight tracking.
MOUTH_KEYS = {
    "jaw", "lip_top_left", "lip_top_right", "lip_bottom_left", "lip_bottom_right",
    "mouth_corner_left_top", "mouth_corner_left_bottom",
    "mouth_corner_right_top", "mouth_corner_right_bottom", "tongue",
}
# Gaze axes are owned by face tracking; expressions must not fight them.
GAZE_KEYS = {"eye_x", "eye_y"}


def _clamp_actuator(a, us):
    lo, hi = a.get("min_us", ABS_MIN_US), a.get("max_us", ABS_MAX_US)
    lo, hi = min(lo, hi), max(lo, hi)
    return max(lo, min(hi, int(us)))


def _axis(a):
    pos = a.get("positions", {}) or {}
    home = pos.get("home", a.get("home_us"))
    for lo, hi in (("down", "up"), ("closed", "open"), ("right", "left")):
        if lo in pos and hi in pos:
            return home, pos[lo], pos[hi]
    if "out" in pos:
        return home, home, pos["out"]
    return home, a.get("min_us", home), a.get("max_us", home)


def _jaw_axis(jaw):
    prim = jaw.get("primary", {})
    pos = prim.get("positions", {})
    home = pos.get("home", prim.get("home_us"))
    return home, pos.get("closed", home), pos.get("open", home)


def _resolve_norm(home, low, high, v):
    v = max(-1.0, min(1.0, float(v)))
    return round(home + (high - home) * v) if v >= 0 else round(home + (low - home) * (-v))


class HeadHardware:
    """High-level, safe controller for the dual-board animatronic head."""

    def __init__(self, calib_path=CALIB_PATH, poses_path=POSES_PATH,
                 jaw_speech_open=DEFAULT_JAW_SPEECH_OPEN):
        self.calib_path = calib_path
        self.poses_path = poses_path
        self.jaw_speech_open = jaw_speech_open
        self.armed = False
        self.available = False
        self._boards = {}          # board number -> _Board
        self._lock = threading.Lock()
        self._load_config()

    # ---- config ----
    def _load_config(self):
        with open(self.calib_path) as f:
            self.calib = json.load(f)
        self.actuators = self.calib.get("actuators", {})
        self.board_urls = {int(k): v["ftdi_url"] for k, v in self.calib.get("boards", {}).items()}
        try:
            with open(self.poses_path) as f:
                self.poses = json.load(f)
        except FileNotFoundError:
            self.poses = {"expressions": {}, "visemes": {}}

    def reload_config(self):
        """Re-read calibration + poses from disk (pick up live edits)."""
        self._load_config()

    # ---- arming ----
    def arm(self):
        """Open both FT232H boards. Does not move any servo."""
        with self._lock:
            if self.armed:
                return True
            try:
                for num, url in self.board_urls.items():
                    b = _Board(url)
                    b.open()
                    self._boards[num] = b
                self.armed = True
                self.available = True
                logger.info("Head hardware armed (%d boards)", len(self._boards))
                return True
            except Exception as e:
                logger.warning("Head hardware arm failed: %s", e)
                self._close_boards()
                self.armed = False
                self.available = False
                return False

    def disarm(self):
        """Release every channel and close the USB link."""
        with self._lock:
            if self._boards:
                for b in self._boards.values():
                    try:
                        b.release_all()
                    except Exception:
                        pass
            self._close_boards()
            self.armed = False

    def _close_boards(self):
        for b in self._boards.values():
            b.close()
        self._boards = {}

    def _board(self, num):
        return self._boards.get(int(num))

    # ---- low-level logical control ----
    def set_norm(self, key, v):
        """Drive one actuator to normalized value v in [-1,1]. No-op unless armed."""
        if not self.armed:
            return None
        a = self.actuators.get(key)
        if not a:
            return None
        try:
            if a.get("type") == "linked_pair":
                return self._drive_jaw_norm(v)
            home, low, high = _axis(a)
            us = _clamp_actuator(a, _resolve_norm(home, low, high, v))
            b = self._board(a["board"])
            if b:
                b.set_us(int(a["channel"]), us)
            return us
        except Exception as e:
            logger.warning("set_norm(%s) failed: %s", key, e)
            return None

    def _drive_jaw_norm(self, v):
        jaw = self.actuators.get("jaw")
        if not jaw:
            return None
        home, low, high = _jaw_axis(jaw)
        pri = _clamp_actuator(jaw, _resolve_norm(home, low, high, v))
        gain = jaw.get("gain", -1.0)
        pri_home = jaw.get("primary", {}).get("home_us", 1500)
        sec_home = jaw.get("secondary", {}).get("home_us", 1500)
        sec = max(ABS_MIN_US, min(ABS_MAX_US, round(sec_home + gain * (pri - pri_home))))
        b = self._board(jaw.get("board", 2))
        if b:
            b.set_us(int(jaw["primary_channel"]), pri)
            b.set_us(int(jaw["secondary_channel"]), sec)
        return {"primary": pri, "secondary": sec}

    # ---- poses ----
    def apply_pose(self, kind, name, only_keys=None, skip_keys=None):
        """Apply a saved expression/viseme pose. No-op unless armed.

        only_keys: if set, apply just these actuator keys.
        skip_keys: if set, skip these actuator keys."""
        if not self.armed:
            return False
        targets = (self.poses.get(kind) or {}).get(name)
        if targets is None:
            return False
        for key, v in targets.items():
            if only_keys is not None and key not in only_keys:
                continue
            if skip_keys is not None and key in skip_keys:
                continue
            self.set_norm(key, v)
        return True

    def apply_viseme(self, name):
        """Apply a viseme pose to the MOUTH ONLY (never touches eyes/brows/lids),
        so lip-sync doesn't fight face tracking. No-op unless armed."""
        return self.apply_pose("visemes", name, only_keys=MOUTH_KEYS)

    def apply_expression(self, emotion, weight=1.0):
        """Apply an emotion pose by name, blending toward neutral by (1-weight).
        Skips the gaze axes (eye_x/eye_y) so face tracking keeps owning the eyes."""
        if not self.armed:
            return False
        expr = self.poses.get("expressions", {})
        name = emotion if emotion in expr else ("neutral" if "neutral" in expr else None)
        if name is None:
            return False
        neutral = expr.get("neutral", {})
        target = expr[name]
        w = max(0.0, min(1.0, float(weight)))
        for key, v in target.items():
            if key in GAZE_KEYS:
                continue
            base = neutral.get(key, 0.0)
            self.set_norm(key, base + (v - base) * w)
        return True

    # ---- lip-sync + gaze helpers ----
    def set_jaw_open(self, frac):
        """Lip-sync: frac 0..1 maps closed(home)..speech-open. No-op unless armed."""
        if not self.armed:
            return None
        frac = max(0.0, min(1.0, float(frac)))
        return self._drive_jaw_norm(frac * self.jaw_speech_open)

    def set_eyes(self, x_norm, y_norm):
        """Gaze: x_norm/y_norm in [-1,1] (calibration-relative). No-op unless armed."""
        if not self.armed:
            return
        self.set_norm("eye_x", x_norm)
        self.set_norm("eye_y", y_norm)

    def blink(self, closed_ms=90):
        """Quick blink: close all eyelids, then reopen to home. No-op unless armed."""
        if not self.armed:
            return
        lids = ["eyelid_left_top", "eyelid_left_bottom", "eyelid_right_top", "eyelid_right_bottom"]
        for k in lids:
            self.set_norm(k, -1.0)   # closed
        time.sleep(closed_ms / 1000.0)
        for k in lids:
            self.set_norm(k, 0.0)    # home (open rest)

    # ---- teardown ----
    def release_all(self):
        if not self.armed:
            return
        for b in self._boards.values():
            try:
                b.release_all()
            except Exception:
                pass

    def shutdown(self):
        """Release servos and close the link. Safe to call multiple times."""
        self.disarm()
