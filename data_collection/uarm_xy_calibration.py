"""
uarm_xy_calibration.py
======================
Manual jog of the uArm over Bluetooth LE (uarm_ble.py) to locate the sensor and
coil centres in arm coordinates (no USB cable, no uFactory dongle).

Notes:
  - SwiftAPI(port=COM4) -> SwiftBLE (keep uarm_ble.py in the same folder)
  - SDK-only helpers are done with raw G-code over Bluetooth:
        reset            -> M17 + G0 X200 Y0 Z150 (the SDK's reset position)
        power status     -> P2234  ("ok V1" = 12 V power connected)
        servo attach     -> M17,  attach check -> M2203 N0..N2
  - Every jog waits until the arm has stopped before reading the position,
    so "actual position now" is the final position, not a mid-move reading.
  - Ctrl+C exits cleanly: marked points are still printed and Bluetooth is
    disconnected.

Before use: pip install bleak, uArm user button DOWN, no phone / other program
connected to the arm.
"""

import time

from uarm_ble import SwiftBLE

UARM_BLE_ADDRESS = "7C:01:0A:5E:93:53"   # None = find by name
START_Z      = 110.0
DEFAULT_STEP = 1
SPEED        = 10
RESET_XYZ    = (200.0, 0.0, 150.0)       # SDK reset() position
MOVE_TIMEOUT = 60


def wait_move(swift, x, y, z, speed=SPEED):
    """Move and block until the arm has stopped. Returns the reply ('ok' / 'E..')."""
    try:
        reply = swift.set_position(x=x, y=y, z=z, speed=speed, wait=True, timeout=MOVE_TIMEOUT)
    except RuntimeError as exc:             # E.. reply, e.g. unreachable point
        return str(exc)
    swift.flush_cmd(timeout=MOVE_TIMEOUT, wait_stop=True)
    return reply

def init_uarm():
    print("Connecting to uArm Swift Pro over Bluetooth ...")
    swift = SwiftBLE(address=UARM_BLE_ADDRESS)
    swift.waiting_ready(timeout=5)
    swift.set_mode(mode=0)
    swift.set_report_position(interval=0)

    # reset: attach motors, go to the SDK reset position
    swift.send_cmd_sync("M17")
    print(f"Reset move: {wait_move(swift, *RESET_XYZ)}")
    time.sleep(1)

    power_ok = None
    try:
        raw = swift.send_cmd_sync("P2234")
        print(f"Power status (raw firmware P2234): {raw}")
        if raw.startswith("ok"):
            power_ok = "V1" in raw
    except Exception as e:
        print(f"  (Could not query raw power status: {e})")

    swift.send_cmd_sync("M17")
    time.sleep(0.2)
    states = []
    for n in range(3):
        try:
            states.append(f"joint {n}: {swift.send_cmd_sync(f'M2203 N{n}')}")
        except Exception as e:
            states.append(f"joint {n}: ({e})")
    print("Servo attach state: " + ", ".join(states))

    return swift, power_ok


def get_pos(swift, retries=3, delay=0.3):
    """Retry on empty/garbled replies instead of crashing."""
    for _ in range(retries):
        pos = swift.get_position()
        if pos is None:
            time.sleep(delay)
            continue
        try:
            return float(pos[0]), float(pos[1]), float(pos[2])
        except (ValueError, TypeError, IndexError):
            time.sleep(delay)
    print("  WARNING: get_position() failed after retries. Returning last-known values.")
    return None


def move_and_report(swift, x, y, z):
    result = wait_move(swift, x, y, z)
    print(f"  set_position result: {result}")
    pos = get_pos(swift)
    if pos is None:
        return x, y, z  # keep previous known-good values, don't crash
    cx, cy, cz = pos
    print(f"  -> actual position now: x={cx:.2f}, y={cy:.2f}, z={cz:.2f}")
    return cx, cy, cz


def print_marked(marked):
    print(f"\n{'-'*50}")
    print("Marked points:")
    for i, (mx, my, mz) in enumerate(marked):
        print(f"  #{i+1}: x={mx:.2f}, y={my:.2f}, z={mz:.2f}")
    if marked:
        xs = [p[0] for p in marked]
        ys = [p[1] for p in marked]
        print("\nSuggested bounds based on marked points:")
        print(f"  X_MIN, X_MAX = {min(xs):.2f}, {max(xs):.2f}")
        print(f"  Y_MIN, Y_MAX = {min(ys):.2f}, {max(ys):.2f}")
    print(f"{'-'*50}")


def main():
    swift, power_ok = init_uarm()
    marked = []
    try:
        if power_ok is False:
            print("\n*** STOPPING: servo power not detected. ***")
            print("Fix the DC adapter / connection, then restart this script.\n")
            return

        pos = get_pos(swift)
        if pos is None:
            print("Could not get a valid starting position. Aborting.")
            return
        x, y, z = pos
        z = START_Z
        x, y, z = move_and_report(swift, x, y, z)

        step = DEFAULT_STEP
        print(f"\nJog started at x={x:.2f}, y={y:.2f}, z={z:.2f}  step={step} mm")
        print("Commands: w/s=y+/y-  a/d=x-/x+  q/e=z+/z-  "
              "+/-=step size  p=print  m=mark  x=exit  (Ctrl+C also exits)\n")

        while True:
            cmd = input(f"[step={step:.1f}mm] > ").strip().lower()

            if cmd == 'w':
                y += step
            elif cmd == 's':
                y -= step
            elif cmd == 'a':
                x -= step
            elif cmd == 'd':
                x += step
            elif cmd == 'q':
                z += step
            elif cmd == 'e':
                z -= step
            elif cmd == '+':
                step *= 2
                print(f"  step size -> {step:.2f} mm")
                continue
            elif cmd == '-':
                step /= 2
                print(f"  step size -> {step:.2f} mm")
                continue
            elif cmd == 'p':
                pos = get_pos(swift)
                if pos:
                    print(f"  Current position: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                continue
            elif cmd == 'm':
                pos = get_pos(swift)
                if pos:
                    marked.append(pos)
                    print(f"  Marked point #{len(marked)}: "
                          f"x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                continue
            elif cmd == 'x':
                break
            else:
                print("  Unknown command.")
                continue

            x, y, z = move_and_report(swift, x, y, z)

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted by user.")
    finally:
        print_marked(marked)
        swift.disconnect()


if __name__ == "__main__":
    main()