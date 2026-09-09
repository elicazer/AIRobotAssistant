#!/usr/bin/env python3
"""Standalone servo calibration web UI for the dual FT232H + PCA9685 Will Cogley head.

SAFETY MODEL (intentionally conservative):
  * This is a SEPARATE app from the voice assistant. It never starts face tracking,
    never auto-centers, and never drives a channel on its own.
  * A channel only moves when YOU move its slider / press its button in the browser.
  * Every pulse is hard-clamped to an absolute safe window (600-2400 us). The UI
    defaults to a narrow 1000-2000 us range; you can opt into the wide range per page.
  * "RELEASE ALL" immediately stops PWM on every channel on both boards (servos go limp).
  * Startup only ensures the PCA is awake at 50 Hz; it does NOT rewrite channel outputs,
    so anything already holding a calibrated pose is left as-is.
  * Bound to 127.0.0.1 only (local machine). No authentication, so do not expose it.

Run:
  venv/bin/python scripts/servo_calibration_server.py
Then open http://127.0.0.1:8600
"""

import json
import os
import threading

from flask import Flask, jsonify, request, render_template
from pyftdi.i2c import I2cController, I2cNackError

# --- Paths -----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CALIB_PATH = os.path.join(ROOT, "config", "servo_calibration.json")
POSES_PATH = os.path.join(ROOT, "config", "poses.json")
TEMPLATES = os.path.join(ROOT, "templates")

# --- Board identity (verified by disconnect test) --------------------------------
BOARD_URLS = {
    1: "ftdi://ftdi:232h:FTAEUZGR/1",
    2: "ftdi://ftdi:232h:FTAEUMUV/1",
}
PCA_ADDR = 0x40

# --- Pulse safety limits ---------------------------------------------------------
# Backstop against typos, not the mechanical limiter. ~400-2600 us is near the
# practical extreme for MG90S/MG996R; watch for the internal stop near these ends.
# (Widened so the linked jaw can close further: primary < 500 needs secondary > 2500.)
ABS_MIN_US = 400
ABS_MAX_US = 2600

# --- PCA9685 registers -----------------------------------------------------------
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
OSC_HZ = 25_000_000.0
TARGET_HZ = 50.0

# --- Logical actuator map (ROBOT perspective; confirmed with user) ---------------
ACTUATORS = [
    # board, channel, key, label, group
    (1, 0, "eyebrow_right_front", "Right eyebrow - front", "Eyebrows"),
    (1, 1, "eyebrow_right_back", "Right eyebrow - back", "Eyebrows"),
    (1, 2, "eyebrow_left_front", "Left eyebrow - front", "Eyebrows"),
    (1, 3, "eyebrow_left_back", "Left eyebrow - back", "Eyebrows"),
    (1, 4, "eyelid_right_top", "Right eyelid - top", "Eyelids"),
    (1, 5, "eyelid_right_bottom", "Right eyelid - bottom", "Eyelids"),
    (1, 6, "eyelid_left_top", "Left eyelid - top", "Eyelids"),
    (1, 7, "eyelid_left_bottom", "Left eyelid - bottom", "Eyelids"),
    (1, 8, "eye_y", "Eyes - vertical (Y)", "Eyes"),
    (1, 9, "eye_x", "Eyes - horizontal (X)", "Eyes"),
    # Board-2 default channels below are only STARTING GUESSES. The real head is wired
    # differently (e.g. bottom-left lip is on Ch5, Ch3 appears empty). Use the per-card
    # channel selector in the UI to point each function at the channel it truly drives;
    # the saved calibration records the chosen channel.
    (2, 0, "jaw", "Jaw (MG996R)", "Jaw"),
    (2, 1, "lip_top_right", "Top lip - right", "Lips"),
    (2, 2, "lip_top_left", "Top lip - left", "Lips"),
    (2, 4, "lip_bottom_right", "Bottom lip - right", "Lips"),
    (2, 5, "lip_bottom_left", "Bottom lip - left", "Lips"),
    (2, 6, "mouth_corner_left_top", "Left mouth corner - top", "Mouth corners"),
    (2, 7, "mouth_corner_left_bottom", "Left mouth corner - bottom", "Mouth corners"),
    (2, 8, "mouth_corner_right_top", "Right mouth corner - top", "Mouth corners"),
    (2, 9, "mouth_corner_right_bottom", "Right mouth corner - bottom", "Mouth corners"),
    (2, 10, "tongue", "Tongue", "Tongue"),
]
SERVO_TYPE = {(2, 0): "MG996R", (2, 3): "MG996R"}  # both jaw servos are MG996R; rest MG90S


def servo_type(board, channel):
    return SERVO_TYPE.get((board, channel), "MG90S")


# --- Named position schemes ------------------------------------------------------
# Each actuator captures a small set of semantically named positions (in us).
# The safe min/max travel bounds are derived automatically from the extreme
# captured positions, so there is no separate min/max to enter.
GROUP_SCHEME = {
    "Eyebrows": ["down", "home", "up"],
    "Eyelids": ["closed", "home", "open"],
    "Jaw": ["closed", "home", "open"],
    "Lips": ["home", "up", "down"],
    "Mouth corners": ["home", "up", "down"],
    "Tongue": ["home", "out"],
}
KEY_SCHEME = {
    "eye_x": ["right", "home", "left"],
    "eye_y": ["down", "home", "up"],
}


def positions_for(key, group):
    return KEY_SCHEME.get(key) or GROUP_SCHEME.get(group, ["home"])


# --- Linked jaw (two coupled servos moving in opposite directions) ---------------
# The jaw is driven by TWO servos that must move together. The secondary follows:
#   secondary_us = sec_home + gain * (primary_us - pri_home)
# gain is typically -1.0 (opposite direction); flip the sign or magnitude if they fight.
JAW_PRIMARY_CH = 0
JAW_SECONDARY_CH = 3
JAW_DEFAULTS = {"secondary_channel": JAW_SECONDARY_CH, "gain": -1.0,
                "pri_home": 1500, "sec_home": 1500}


class BoardController:
    """Owns one FT232H + PCA9685. All access serialized by an internal lock."""

    def __init__(self, board, url):
        self.board = board
        self.url = url
        self._lock = threading.Lock()
        self._ctrl = None
        self._port = None
        self._prescale = None
        self.connected = False
        try:
            self._open()
        except Exception as e:
            # Board may be powered down at startup; come up degraded and let the
            # user recover with the Reconnect button once power is restored.
            print(f"  WARN board {board} not ready: {e}. Use Reconnect later.")

    def _open(self):
        self._ctrl = I2cController()
        self._ctrl.configure(self.url, frequency=400000)
        self._port = self._ctrl.get_port(PCA_ADDR)
        self._ensure_awake_50hz()
        self.connected = True

    def reopen(self):
        """Re-establish the USB/I2C link after the board was power-cycled."""
        with self._lock:
            self.connected = False
            try:
                if self._ctrl is not None:
                    self._ctrl.terminate()
            except Exception:
                pass
            self._open()

    def _rb(self, reg):
        return self._port.read_from(reg, 1)[0]

    def _ensure_awake_50hz(self):
        prescale = self._rb(PRESCALE)
        mode1 = self._rb(MODE1)
        want = int(round(OSC_HZ / (4096.0 * TARGET_HZ))) - 1
        if prescale != want:
            # Reconfigure prescale (requires sleep). This briefly stops outputs.
            self._port.write_to(MODE1, [(mode1 & 0x7F) | 0x10])
            self._port.write_to(PRESCALE, [want])
            wake = mode1 & ~0x10 & 0xFF
            self._port.write_to(MODE1, [wake])
            import time
            time.sleep(0.005)
            self._port.write_to(MODE1, [wake | 0xA0])
            prescale = want
        elif mode1 & 0x10:
            # Same freq but asleep: just wake it (does not move outputs still FULL_OFF).
            wake = mode1 & ~0x10 & 0xFF
            self._port.write_to(MODE1, [wake])
            import time
            time.sleep(0.005)
            self._port.write_to(MODE1, [wake | 0xA0])
        self._prescale = prescale

    def us_per_tick(self):
        return (self._prescale + 1) / (OSC_HZ / 1_000_000.0)

    def _retry(self, fn):
        """Run a locked op; if the USB link is stale (board power-cycled), reopen once."""
        try:
            return fn()
        except (I2cNackError,):
            raise
        except Exception:
            self.reopen()
            return fn()

    def set_us(self, channel, us):
        us = max(ABS_MIN_US, min(ABS_MAX_US, int(us)))
        base = LED0_ON_L + 4 * channel

        def op():
            ticks = int(round(us / self.us_per_tick()))
            ticks = max(0, min(4095, ticks))
            with self._lock:
                self._port.write_to(base, [0x00, 0x00, ticks & 0xFF, (ticks >> 8) & 0x0F])
            return us
        return self._retry(op)

    def release(self, channel):
        base = LED0_ON_L + 4 * channel

        def op():
            with self._lock:
                self._port.write_to(base, [0x00, 0x00, 0x00, 0x10])
        return self._retry(op)

    def release_all(self):
        def op():
            with self._lock:
                for ch in range(16):
                    base = LED0_ON_L + 4 * ch
                    self._port.write_to(base, [0x00, 0x00, 0x00, 0x10])
        return self._retry(op)

    def read_us(self, channel):
        base = LED0_ON_L + 4 * channel

        def op():
            with self._lock:
                return [self._rb(base + k) for k in range(4)]
        on_l, on_h, off_l, off_h = self._retry(op)
        if off_h & 0x10:
            return None  # full off / released
        on = on_l | ((on_h & 0x0F) << 8)
        off = off_l | ((off_h & 0x0F) << 8)
        ticks = (off - on) & 0x0FFF
        return round(ticks * self.us_per_tick())


# --- Calibration persistence -----------------------------------------------------
_calib_lock = threading.Lock()


def load_calibration():
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            return json.load(f)
    return {
        "_comment": "Dual FT232H + PCA9685 Will Cogley head calibration. Robot perspective. Microseconds at 50 Hz.",
        "pwm_frequency_hz": 50,
        "boards": {
            "1": {"name": "eyes/eyelids/eyebrows", "ftdi_url": BOARD_URLS[1], "pca_address": "0x40"},
            "2": {"name": "jaw/lips/mouth/tongue", "ftdi_url": BOARD_URLS[2], "pca_address": "0x40"},
        },
        "actuators": {},
    }


def save_actuator(key, board, channel, positions):
    """positions: dict of {name: us}. Derives home/min/max from it."""
    clean = {}
    for name, val in positions.items():
        if val in (None, "", "null"):
            continue
        us = max(ABS_MIN_US, min(ABS_MAX_US, int(val)))
        clean[name] = us
    if not clean:
        raise ValueError("no positions provided")
    values = list(clean.values())
    home_us = clean.get("home", round(sum(values) / len(values)))
    with _calib_lock:
        data = load_calibration()
        data.setdefault("actuators", {})[key] = {
            "board": board,
            "channel": channel,
            "servo_type": servo_type(board, channel),
            "positions": clean,
            "home_us": home_us,
            "min_us": min(values),
            "max_us": max(values),
            "calibrated": True,
        }
        tmp = CALIB_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, CALIB_PATH)
        return data["actuators"][key]


# --- Flask app -------------------------------------------------------------------
app = Flask(__name__, template_folder=TEMPLATES)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
_boards = {}


def get_board(board):
    b = int(board)
    if b not in _boards:
        raise KeyError(f"unknown board {board}")
    return _boards[b]


@app.route("/")
def index():
    return render_template(
        "servo_calibration.html",
        actuators=ACTUATORS,
        abs_min=ABS_MIN_US,
        abs_max=ABS_MAX_US,
    )


@app.route("/api/state")
def api_state():
    calib = load_calibration().get("actuators", {})
    out = []
    for board, default_channel, key, label, group in ACTUATORS:
        saved = calib.get(key)
        # Use the channel the user previously chose/saved for this role, if any.
        channel = int(saved["channel"]) if saved and "channel" in saved else default_channel
        try:
            current = get_board(board).read_us(channel)
        except Exception:
            current = None
        entry = {
            "board": board, "channel": channel, "default_channel": default_channel,
            "key": key, "label": label, "group": group,
            "servo_type": servo_type(board, channel),
            "positions": positions_for(key, group),
            "channels": list(range(16)),
            "current_us": current, "calibration": saved,
        }
        if key == "jaw":
            link = dict(JAW_DEFAULTS)
            if saved:
                link["secondary_channel"] = saved.get("secondary_channel", link["secondary_channel"])
                link["gain"] = saved.get("gain", link["gain"])
                link["pri_home"] = (saved.get("primary") or {}).get("home_us", link["pri_home"])
                link["sec_home"] = (saved.get("secondary") or {}).get("home_us", link["sec_home"])
            entry["linked"] = link
        out.append(entry)
    return jsonify({"actuators": out, "abs_min": ABS_MIN_US, "abs_max": ABS_MAX_US})


@app.route("/api/set", methods=["POST"])
def api_set():
    d = request.get_json(force=True)
    try:
        actual = get_board(d["board"]).set_us(int(d["channel"]), int(d["us"]))
        return jsonify({"ok": True, "us": actual})
    except I2cNackError:
        return jsonify({"ok": False, "error": "I2C NACK (check power/wiring)"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/release", methods=["POST"])
def api_release():
    d = request.get_json(force=True)
    try:
        get_board(d["board"]).release(int(d["channel"]))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# --- Poses (expressions + visemes) -----------------------------------------------
_poses_lock = threading.Lock()


def load_poses():
    if os.path.exists(POSES_PATH):
        with open(POSES_PATH) as f:
            return json.load(f)
    return {"expressions": {}, "visemes": {}}


def _clamp_actuator(a, us):
    lo = a.get("min_us", ABS_MIN_US)
    hi = a.get("max_us", ABS_MAX_US)
    lo, hi = min(lo, hi), max(lo, hi)
    return max(lo, min(hi, int(us)))


# --- Calibration-relative pose values -------------------------------------------
# Pose targets are stored NORMALIZED in [-1, +1] along each actuator's own axis:
#   0  -> calibrated home
#  +1  -> the "high" named extreme (up / open / left / out)
#  -1  -> the "low" named extreme  (down / closed / right)
# Resolved against the CURRENT calibration at apply time, so recalibrating a servo
# (or swapping one and re-capturing its named positions) moves every pose with it.
def _axis(a):
    """(home, low, high) microseconds for a non-jaw actuator, from named positions."""
    pos = a.get("positions", {}) or {}
    home = pos.get("home", a.get("home_us"))
    for lo, hi in (("down", "up"), ("closed", "open"), ("right", "left")):
        if lo in pos and hi in pos:
            return home, pos[lo], pos[hi]
    if "out" in pos:  # tongue: home..out only (no negative side)
        return home, home, pos["out"]
    return home, a.get("min_us", home), a.get("max_us", home)


def _jaw_axis(jaw):
    prim = jaw.get("primary", {})
    pos = prim.get("positions", {})
    home = pos.get("home", prim.get("home_us"))
    return home, pos.get("closed", home), pos.get("open", home)


def resolve_norm(home, low, high, v):
    v = max(-1.0, min(1.0, float(v)))
    return round(home + (high - home) * v) if v >= 0 else round(home + (low - home) * (-v))


def to_norm(home, low, high, us):
    d, dh, dl = us - home, high - home, low - home
    if d == 0:
        return 0.0
    if dh != 0 and (d > 0) == (dh > 0):
        return round(max(-1.0, min(1.0, d / dh)), 4)
    if dl != 0 and (d > 0) == (dl > 0):
        return round(-max(-1.0, min(1.0, d / dl)), 4)
    return 0.0


def apply_targets(targets):
    """Drive every actuator in a pose. Values are normalized [-1,1] and resolved
    against current calibration; jaw resolves to its linked servo pair. Everything
    is clamped to calibrated travel for safety."""
    calib = load_calibration().get("actuators", {})
    applied = {}
    for key, v in targets.items():
        if key == "jaw":
            jaw = calib.get("jaw")
            if not jaw:
                continue
            home, low, high = _jaw_axis(jaw)
            pri = _clamp_actuator(jaw, resolve_norm(home, low, high, v))
            gain = jaw.get("gain", -1.0)
            pri_home = jaw.get("primary", {}).get("home_us", 1500)
            sec_home = jaw.get("secondary", {}).get("home_us", 1500)
            sec = max(ABS_MIN_US, min(ABS_MAX_US, round(sec_home + gain * (pri - pri_home))))
            b = get_board(jaw.get("board", 2))
            b.set_us(int(jaw["primary_channel"]), pri)
            b.set_us(int(jaw["secondary_channel"]), sec)
            applied["jaw"] = {"primary": pri, "secondary": sec}
        else:
            a = calib.get(key)
            if not a:
                continue
            home, low, high = _axis(a)
            val = _clamp_actuator(a, resolve_norm(home, low, high, v))
            get_board(a["board"]).set_us(int(a["channel"]), val)
            applied[key] = val
    return applied


@app.route("/api/poses")
def api_poses():
    calib = load_calibration().get("actuators", {})
    summary = {}
    for key, a in calib.items():
        if a.get("type") == "linked_pair":
            home, low, high = _jaw_axis(a)
        else:
            home, low, high = _axis(a)
        summary[key] = {
            "min_us": a.get("min_us"), "max_us": a.get("max_us"),
            "home_us": a.get("home_us"),
            "home": home, "low": low, "high": high,
            "board": a.get("board"), "channel": a.get("channel"),
            "linked": a.get("type") == "linked_pair",
        }
    return jsonify({"poses": load_poses(), "actuators": summary})


@app.route("/api/apply_pose", methods=["POST"])
def api_apply_pose():
    d = request.get_json(force=True)
    kind, name = d.get("kind"), d.get("name")
    # Allow applying ad-hoc targets directly, or a named saved pose.
    targets = d.get("targets")
    try:
        if targets is None:
            poses = load_poses()
            targets = poses.get(kind, {}).get(name)
            if targets is None:
                return jsonify({"ok": False, "error": f"unknown pose {kind}/{name}"}), 404
        applied = apply_targets(targets)
        return jsonify({"ok": True, "applied": applied})
    except I2cNackError:
        return jsonify({"ok": False, "error": "I2C NACK (check power/wiring)"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/save_pose", methods=["POST"])
def api_save_pose():
    d = request.get_json(force=True)
    kind, name = d.get("kind"), d.get("name")
    targets = d.get("targets", {})
    if kind not in ("expressions", "visemes") or not name:
        return jsonify({"ok": False, "error": "kind must be expressions/visemes and name required"}), 400
    # Values are normalized [-1, 1]; store as floats.
    clean = {k: max(-1.0, min(1.0, float(v))) for k, v in targets.items()
             if v not in (None, "", "null")}
    with _poses_lock:
        poses = load_poses()
        poses.setdefault(kind, {})[name] = clean
        tmp = POSES_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(poses, f, indent=2)
        os.replace(tmp, POSES_PATH)
    return jsonify({"ok": True, "saved": {"kind": kind, "name": name, "targets": clean}})


@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    results = {}
    for board, ctl in _boards.items():
        try:
            ctl.reopen()
            results[board] = "ok"
        except Exception as e:
            results[board] = f"failed: {e}"
    ok = all(v == "ok" for v in results.values())
    return jsonify({"ok": ok, "results": results})


@app.route("/api/release_all", methods=["POST"])
def api_release_all():
    errors = []
    for b in _boards.values():
        try:
            b.release_all()
        except Exception as e:
            errors.append(str(e))
    return jsonify({"ok": not errors, "errors": errors})


@app.route("/api/save", methods=["POST"])
def api_save():
    d = request.get_json(force=True)
    try:
        rec = save_actuator(
            d["key"], int(d["board"]), int(d["channel"]), d.get("positions", {}),
        )
        return jsonify({"ok": True, "saved": rec})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/save_jaw", methods=["POST"])
def api_save_jaw():
    d = request.get_json(force=True)
    try:
        positions = {}
        for name, val in d.get("positions", {}).items():
            if val in (None, "", "null"):
                continue
            positions[name] = max(ABS_MIN_US, min(ABS_MAX_US, int(val)))
        if not positions:
            raise ValueError("capture at least the primary closed/home/open first")
        pri_ch = int(d["primary_channel"])
        sec_ch = int(d["secondary_channel"])
        pri_home = max(ABS_MIN_US, min(ABS_MAX_US, int(d["pri_home"])))
        sec_home = max(ABS_MIN_US, min(ABS_MAX_US, int(d["sec_home"])))
        gain = float(d["gain"])
        values = list(positions.values())
        with _calib_lock:
            data = load_calibration()
            data.setdefault("actuators", {})["jaw"] = {
                "type": "linked_pair",
                "board": 2,
                "primary_channel": pri_ch,
                "secondary_channel": sec_ch,
                "gain": gain,
                "primary": {"servo_type": "MG996R", "home_us": pri_home, "positions": positions},
                "secondary": {"servo_type": "MG996R", "home_us": sec_home,
                              "formula": "sec_us = sec_home + gain * (pri_us - pri_home)"},
                "home_us": positions.get("home", pri_home),
                "min_us": min(values),
                "max_us": max(values),
                "calibrated": True,
            }
            tmp = CALIB_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, CALIB_PATH)
            return jsonify({"ok": True, "saved": data["actuators"]["jaw"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


def main():
    for board, url in BOARD_URLS.items():
        print(f"Opening board {board}: {url}")
        _boards[board] = BoardController(board, url)
    print("All boards awake at 50 Hz. No channel is being driven.")
    print("Open http://127.0.0.1:8600  (Ctrl-C to stop)")
    app.run(host="127.0.0.1", port=8600, threaded=True, debug=False)


if __name__ == "__main__":
    main()
