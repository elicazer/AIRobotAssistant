#!/usr/bin/env python3
"""Minimal, safe, single-channel servo calibration tool for the dual FT232H + PCA9685 head.

Design goals (safety first):
  * Talks to ONE named board (by stable FT232H serial URL) per invocation.
  * Touches ONLY the channel you name. All other channels are left untouched.
  * Every movement is an explicit microsecond value you pass on the command line.
    There are no sweeps, no loops, no "center everything", no auto-tracking.
  * Pulses are hard-clamped to an absolute safe window so a typo cannot command
    a wildly out-of-range pulse.
  * Setting the PWM frequency does NOT move any servo (outputs stay off until you
    explicitly `set` a channel).

This tool deliberately uses pyftdi directly (not Blinka/ServoKit) so it can select a
specific FT232H by serial and drive exactly one PCA9685 channel with no side effects.

Examples:
  # Read current registers (no movement):
  python scripts/servo_calibrate.py --board 1 read

  # Set the servo PWM frequency to 50 Hz (no movement):
  python scripts/servo_calibrate.py --board 1 freq

  # Drive board 1, channel 9 (eye X) to 1500 us and hold:
  python scripts/servo_calibrate.py --board 1 set --channel 9 --us 1500

  # Release (stop driving) a single channel -> servo goes limp:
  python scripts/servo_calibrate.py --board 1 release --channel 9

  # Release every channel on a board:
  python scripts/servo_calibrate.py --board 1 release-all
"""

import argparse
import sys
import time

from pyftdi.i2c import I2cController, I2cNackError

# --- Board identity (verified by disconnect test) --------------------------------
BOARDS = {
    "1": ("Board 1 (eyes/lids/brows)", "ftdi://ftdi:232h:FTAEUZGR/1"),
    "2": ("Board 2 (jaw/lips/mouth)", "ftdi://ftdi:232h:FTAEUMUV/1"),
}
PCA_ADDR = 0x40

# --- PCA9685 registers -----------------------------------------------------------
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

# --- Absolute safety clamp -------------------------------------------------------
# A standard hobby servo pulse is ~1000-2000 us (some tolerate 500-2500).
# We hard-clamp every commanded pulse to this window. Nothing can command outside it.
ABS_MIN_US = 600
ABS_MAX_US = 2400
# Default "conservative" window used unless --wide is passed. Keeps early calibration
# near center so a constrained linkage is less likely to slam a hard stop.
SAFE_MIN_US = 1100
SAFE_MAX_US = 1900

OSC_HZ = 25_000_000.0


def open_port(url):
    ctrl = I2cController()
    ctrl.configure(url, frequency=100000)
    return ctrl, ctrl.get_port(PCA_ADDR)


def read_reg(port, reg):
    return port.read_from(reg, 1)[0]


def prescale_for(freq_hz):
    return int(round(OSC_HZ / (4096.0 * freq_hz))) - 1


def us_per_tick(prescale):
    return (prescale + 1) / (OSC_HZ / 1_000_000.0)


def cmd_read(port):
    mode1 = read_reg(port, MODE1)
    prescale = read_reg(port, PRESCALE)
    sleeping = bool(mode1 & 0x10)
    freq = OSC_HZ / (4096.0 * (prescale + 1))
    upt = us_per_tick(prescale)
    print(f"MODE1=0x{mode1:02X} (SLEEP={sleeping})  PRESCALE=0x{prescale:02X} "
          f"-> {freq:.1f} Hz, {upt:.3f} us/tick")
    for ch in range(16):
        base = LED0_ON_L + 4 * ch
        on_l, on_h, off_l, off_h = (read_reg(port, base + k) for k in range(4))
        on = on_l | ((on_h & 0x0F) << 8)
        off = off_l | ((off_h & 0x0F) << 8)
        full_off = bool(off_h & 0x10)
        ticks = (off - on) & 0x0FFF
        pulse = ticks * upt
        tag = "FULL_OFF" if full_off else f"{pulse:7.1f} us"
        print(f"  CH{ch:2d}: ON={on:4d} OFF={off:4d}  {tag}")


def set_frequency(port, freq_hz):
    """Set the PWM prescale for the given frequency. Does not move servos."""
    prescale = prescale_for(freq_hz)
    old = read_reg(port, MODE1)
    sleep = (old & 0x7F) | 0x10          # set SLEEP (bit4), clear RESTART (bit7)
    port.write_to(MODE1, [sleep])        # sleep so prescale can be written
    port.write_to(PRESCALE, [prescale])
    wake = old & ~0x10 & 0xFF            # clear SLEEP bit
    port.write_to(MODE1, [wake])         # wake
    time.sleep(0.005)                    # oscillator needs >500us to stabilize
    port.write_to(MODE1, [wake | 0xA0])  # RESTART (bit7) + auto-increment (bit5)
    actual = OSC_HZ / (4096.0 * (prescale + 1))
    print(f"Frequency set to ~{actual:.2f} Hz (prescale=0x{prescale:02X}). No servo moved.")


def clamp(us, wide):
    lo, hi = (ABS_MIN_US, ABS_MAX_US) if wide else (SAFE_MIN_US, SAFE_MAX_US)
    c = max(lo, min(hi, us))
    # Absolute clamp always applies as a final backstop.
    c = max(ABS_MIN_US, min(ABS_MAX_US, c))
    if c != us:
        print(f"  (clamped {us} us -> {c} us; window {lo}-{hi})")
    return c


def set_channel(port, channel, us, wide):
    prescale = read_reg(port, PRESCALE)
    if read_reg(port, MODE1) & 0x10:
        print("REFUSING: PCA is asleep. Run the `freq` command first to wake it.",
              file=sys.stderr)
        sys.exit(3)
    us = clamp(us, wide)
    upt = us_per_tick(prescale)
    ticks = int(round(us / upt))
    ticks = max(0, min(4095, ticks))
    base = LED0_ON_L + 4 * channel
    # ON=0, OFF=ticks (auto-increment write of the 4 LEDn registers)
    port.write_to(base, [0x00, 0x00, ticks & 0xFF, (ticks >> 8) & 0x0F])
    print(f"CH{channel} <- {us} us ({ticks} ticks). Holding. Other channels untouched.")


def release_channel(port, channel):
    base = LED0_ON_L + 4 * channel
    port.write_to(base, [0x00, 0x00, 0x00, 0x10])  # full-off bit
    print(f"CH{channel} released (FULL_OFF). Servo is no longer driven.")


def release_all(port):
    for ch in range(16):
        base = LED0_ON_L + 4 * ch
        port.write_to(base, [0x00, 0x00, 0x00, 0x10])
    print("All 16 channels released (FULL_OFF).")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", required=True, choices=sorted(BOARDS),
                   help="1 = eyes/lids/brows, 2 = jaw/lips/mouth")
    p.add_argument("--wide", action="store_true",
                   help=f"allow the extended pulse window "
                        f"{ABS_MIN_US}-{ABS_MAX_US} us instead of the conservative "
                        f"{SAFE_MIN_US}-{SAFE_MAX_US} us")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("read", help="read registers, no movement")
    fp = sub.add_parser("freq", help="set PWM frequency (no movement)")
    fp.add_argument("--hz", type=float, default=50.0)
    sp = sub.add_parser("set", help="drive one channel to a pulse and hold")
    sp.add_argument("--channel", type=int, required=True, choices=range(0, 16))
    sp.add_argument("--us", type=int, required=True)
    rp = sub.add_parser("release", help="stop driving one channel")
    rp.add_argument("--channel", type=int, required=True, choices=range(0, 16))
    sub.add_parser("release-all", help="stop driving all channels on the board")

    args = p.parse_args()
    name, url = BOARDS[args.board]
    print(f"== {name} :: {url} ==")

    ctrl = None
    try:
        ctrl, port = open_port(url)
        if args.action == "read":
            cmd_read(port)
        elif args.action == "freq":
            set_frequency(port, args.hz)
        elif args.action == "set":
            set_channel(port, args.channel, args.us, args.wide)
        elif args.action == "release":
            release_channel(port, args.channel)
        elif args.action == "release-all":
            release_all(port)
    except I2cNackError:
        print("I2C NACK: PCA9685 did not acknowledge. Check power/wiring/AD1-AD2 bridge.",
              file=sys.stderr)
        sys.exit(2)
    finally:
        if ctrl is not None:
            ctrl.terminate()


if __name__ == "__main__":
    main()
