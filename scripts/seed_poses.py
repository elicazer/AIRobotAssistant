#!/usr/bin/env python3
"""Generate an initial config/poses.json from the calibrated servo positions.

Poses are stored as absolute microseconds per actuator (jaw stored as its PRIMARY us;
the secondary follows the linked formula at apply time). Values are seeded from the
calibrated named positions so they're safe (within each actuator's travel). Tweak them
live in the calibration page and re-save; this script only seeds sensible starting poses.

Run once:  venv/bin/python scripts/seed_poses.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(ROOT, "config", "servo_calibration.json")
OUT = os.path.join(ROOT, "config", "poses.json")

with open(CALIB) as f:
    cal = json.load(f)
A = cal["actuators"]

# Actuator groups
EYEBROWS = ["eyebrow_left_front", "eyebrow_left_back", "eyebrow_right_front", "eyebrow_right_back"]
LIDS_TOP = ["eyelid_left_top", "eyelid_right_top"]
LIDS_BOT = ["eyelid_left_bottom", "eyelid_right_bottom"]
LIDS = LIDS_TOP + LIDS_BOT
EYES = ["eye_x", "eye_y"]
CORNERS = ["mouth_corner_left_top", "mouth_corner_left_bottom",
           "mouth_corner_right_top", "mouth_corner_right_bottom"]
LIPS = ["lip_top_left", "lip_top_right", "lip_bottom_left", "lip_bottom_right"]
LIPS_BOT = ["lip_bottom_left", "lip_bottom_right"]
NON_JAW = EYEBROWS + LIDS + EYES + CORNERS + LIPS + ["tongue"]


def pos(key, name):
    """Named-position pulse for an actuator."""
    return A[key]["positions"][name]


def home(key):
    p = A[key].get("positions", {})
    return p.get("home", A[key].get("home_us"))


def lerp(key, a_name, b_name, t):
    a, b = pos(key, a_name), pos(key, b_name)
    return round(a + (b - a) * t)


def jaw_frac(t):
    """Interpolate jaw PRIMARY between closed and open by fraction t (0..1)."""
    j = A["jaw"]["primary"]["positions"]
    return round(j["closed"] + (j["open"] - j["closed"]) * t)


def base(jaw_us=None):
    """Everything at home; jaw at its home (near-closed) unless overridden."""
    d = {k: home(k) for k in NON_JAW}
    d["jaw"] = jaw_us if jaw_us is not None else home("jaw")
    return d


def build(overrides, jaw_us=None):
    d = base(jaw_us)
    d.update(overrides)
    return d


# --- Expressions ----------------------------------------------------------------
expressions = {
    "neutral": base(),
    "happy": build({k: pos(k, "up") for k in CORNERS}),
    "sad": build({**{k: pos(k, "down") for k in CORNERS},
                  **{k: pos(k, "down") for k in EYEBROWS}}),
    "angry": build({**{k: pos(k, "down") for k in EYEBROWS},
                    **{k: pos(k, "down") for k in CORNERS},
                    **{k: lerp(k, "home", "closed", 0.4) for k in LIDS_TOP}}),
    "surprised": build({**{k: pos(k, "up") for k in EYEBROWS},
                        **{k: pos(k, "open") for k in LIDS}},
                       jaw_us=jaw_frac(0.4)),
    "blink": build({k: pos(k, "closed") for k in LIDS}),
}

# --- Visemes (phoneme groups) ---------------------------------------------------
# Approximate mouth shapes; refine live. Each is jaw openness + lip/corner/tongue hints.
visemes = {
    # Jaw openness tuned for natural speech (subtle). Even the widest vowel barely drops.
    "rest": base(jaw_us=jaw_frac(0.0)),
    "MBP": base(jaw_us=jaw_frac(0.0)),  # bilabial closed (m,b,p)
    "FV": build({k: pos(k, "up") for k in LIPS_BOT}, jaw_us=jaw_frac(0.0)),  # f,v: closed, lip tuck
    # A and O share the same (subtle) jaw opening; cheeks move in OPPOSITE directions.
    "AI": build({k: pos(k, "down") for k in CORNERS}, jaw_us=jaw_frac(0.106)),  # "ah / eye"
    "E": build({k: pos(k, "up") for k in CORNERS}, jaw_us=jaw_frac(0.0)),   # "eh/ee": closed, wide
    "O": build({k: pos(k, "up") for k in CORNERS}, jaw_us=jaw_frac(0.106)),   # round "oh"
    "U": build({k: pos(k, "down") for k in CORNERS}, jaw_us=jaw_frac(0.078)),  # round "oo"
    "L": build({"tongue": pos("tongue", "out")}, jaw_us=jaw_frac(0.108)),      # l,th (tongue)
    "WQ": build({k: pos(k, "up") for k in CORNERS}, jaw_us=jaw_frac(0.06)),   # w,s,z
}

# --- Convert absolute-us poses to calibration-relative normalized [-1,1] ---------
def _axis(key):
    a = A[key]
    if a.get("type") == "linked_pair":
        pr = a["primary"]["positions"]
        h = pr.get("home", A[key]["primary"].get("home_us"))
        return h, pr.get("closed", h), pr.get("open", h)
    p = a.get("positions", {})
    h = p.get("home", a.get("home_us"))
    for lo, hi in (("down", "up"), ("closed", "open"), ("right", "left")):
        if lo in p and hi in p:
            return h, p[lo], p[hi]
    if "out" in p:
        return h, h, p["out"]
    return h, a.get("min_us", h), a.get("max_us", h)


def to_norm(key, us):
    h, low, high = _axis(key)
    d, dh, dl = us - h, high - h, low - h
    if d == 0:
        return 0.0
    if dh != 0 and (d > 0) == (dh > 0):
        return round(max(-1.0, min(1.0, d / dh)), 4)
    if dl != 0 and (d > 0) == (dl > 0):
        return round(-max(-1.0, min(1.0, d / dl)), 4)
    return 0.0


def normalize(poses):
    return {name: {k: to_norm(k, us) for k, us in pose.items()} for name, pose in poses.items()}


out = {
    "_comment": "Poses stored CALIBRATION-RELATIVE: each value is normalized [-1,1] along the actuator's axis (0=home, +1=high extreme up/open/left/out, -1=low extreme down/closed/right). Resolved against current calibration at apply time. Jaw resolves then drives its linked pair.",
    "format": "normalized",
    "expressions": normalize(expressions),
    "visemes": normalize(visemes),
}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {OUT}")
print(f"  expressions: {', '.join(expressions)}")
print(f"  visemes: {', '.join(visemes)}")
