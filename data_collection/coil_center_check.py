"""
coil_center_check.py
====================
Visual check of the sensor centre and the five coil centres with the uArm.

1) Type the sensor centre (x y). The arm goes there and stops VIEW_GAP_MM above the
   surface. Press Enter to confirm, or type a new "x y" to move and look again.
2) Each Enter then moves the arm to the next coil centre (C1 .. C5), using
   COIL_OFFSETS_MM around the confirmed centre. Between points the arm lifts to hover
   height, moves across, and lowers to the viewing height again.

Keys at each coil (type + Enter):
    Enter     next coil
    b         back to the previous coil
    c         back to the sensor centre
    x y       go to a typed position instead (e.g. "247.0 18.5"), then Enter continues
    q         quit (the arm lifts)
At the end the positions you looked at are printed.

Settings (Bluetooth, Z_SURFACE, SURFACE_PLANE, hover gap, speeds) come from
uarm_random_indentation.py. Heights are taken from the local surface (SURFACE_PLANE,
which corrects the arm's reach-dependent z), so the tip is VIEW_GAP_MM above the
surface at every coil.

    python coil_center_check.py
    python coil_center_check.py --center 228.5 0.0
"""

import argparse

import numpy as np

import uarm_random_indentation as uri

VIEW_GAP_MM = 1.5           # tip height above the local surface while you look
DEFAULT_CENTER = (228.0, 0.0)

# Coil centres relative to the sensor centre (mm), as marked in the dL maps
COIL_OFFSETS_MM = {         # measured with this script
    "C1": (0.0, 0.0),
    "C2": (-17.0, 17.5),    # top-left     (-x, +y)
    "C3": (17.5, 17.5),     # top-right    (+x, +y)
    "C4": (-17.5, -19.0),   # bottom-left  (-x, -y)
    "C5": (17.5, -18.5),    # bottom-right (+x, -y)
}

# Soft limits: anything outside is refused (typo guard)
X_LIMITS = (150.0, 300.0)
Y_LIMITS = (-100.0, 100.0)



def local_surface(x, y):
    f = getattr(uri, "surface_z", None)
    return f(x, y) if f is not None else uri.Z_SURFACE


HOVER_Z = getattr(uri, "HOVER_Z", uri.Z_SURFACE + uri.HOVER_GAP_MM)   # safe height everywhere


def parse_xy(text):
    try:
        v = [float(t) for t in text.replace(",", " ").split()]
    except ValueError:
        return None
    return (v[0], v[1]) if len(v) == 2 else None


def safe(x, y):
    ok = X_LIMITS[0] <= x <= X_LIMITS[1] and Y_LIMITS[0] <= y <= Y_LIMITS[1]
    if not ok:
        print(f"  refused: ({x:.1f}, {y:.1f}) is outside x {X_LIMITS}, y {Y_LIMITS}")
    return ok


def go_view(swift, x, y):
    """Lift to hover height, move across, lower to the viewing height."""
    if not safe(x, y):
        return False
    cx, cy, cz = uri.query_position(swift)
    if np.isnan(cx):
        print("  could not read the arm position; not moving")
        return False
    if cz < HOVER_Z:
        uri.send_move(swift, cx, cy, HOVER_Z, uri.INDENT_SPEED)
    uri.send_move(swift, x, y, HOVER_Z, uri.TRAVEL_SPEED)
    uri.send_move(swift, x, y, local_surface(x, y) + VIEW_GAP_MM, uri.INDENT_SPEED)
    rx, ry, rz = uri.query_position(swift)
    print(f"  at ({x:.1f}, {y:.1f}), {VIEW_GAP_MM:.1f} mm above the local surface   "
          f"(arm reports {rx:.2f}, {ry:.2f}, {rz:.2f})")
    return True


def main():
    ap = argparse.ArgumentParser(description="Visual check of the sensor centre and coil centres")
    ap.add_argument("--center", nargs=2, type=float, default=None, help="sensor centre x y")
    args = ap.parse_args()

    if args.center is not None:
        center = tuple(args.center)
    else:
        txt = input(f"Sensor centre x y [Enter = {DEFAULT_CENTER[0]} {DEFAULT_CENTER[1]}]: ")
        center = parse_xy(txt) if txt.strip() else DEFAULT_CENTER
        if center is None:
            print("Type two numbers, e.g. 228.5 0.0")
            return

    from uarm_ble import SwiftBLE
    swift = None
    visited = {}                     # name -> (x, y), last visit kept
    try:
        swift = SwiftBLE(address=uri.UARM_BLE_ADDRESS, name_prefix=uri.UARM_BLE_NAME_PREFIX,
                         scan_timeout=uri.UARM_BLE_SCAN_TIMEOUT_S)
        swift.waiting_ready(timeout=5)
        swift.set_mode(0)
        swift.set_speed_factor(1)
        swift.set_report_position(interval=0)

        # 1) Sensor centre: confirm or re-type
        print(f"\nSensor centre ({center[0]:.1f}, {center[1]:.1f}):")
        if not go_view(swift, *center):
            return
        while True:
            txt = input("Enter = confirm centre, x y = move there instead, q = quit: ").strip().lower()
            if txt == "q":
                return
            if not txt:
                break
            xy = parse_xy(txt)
            if xy is None:
                print("  type two numbers, e.g. 228.0 0.5")
            elif go_view(swift, *xy):
                center = xy
        print(f"Centre confirmed: ({center[0]:.1f}, {center[1]:.1f})\n")

        # 2) Coil centres, one per Enter
        names = list(COIL_OFFSETS_MM)
        k, quit_ = 0, False
        while 0 <= k < len(names):
            name = names[k]
            dx, dy = COIL_OFFSETS_MM[name]
            x, y = center[0] + dx, center[1] + dy
            print(f"{name}: centre + ({dx:+.1f}, {dy:+.1f}) = ({x:.1f}, {y:.1f})")
            if go_view(swift, x, y):
                visited[name] = (x, y)
            while True:
                txt = input("  Enter = next, b = back, c = centre, x y = go there, q = quit: ").strip().lower()
                if txt == "q":
                    quit_ = True
                    k = len(names)
                    break
                if not txt:
                    k += 1
                    break
                if txt == "b":
                    k = max(k - 1, 0)
                    break
                if txt == "c":
                    go_view(swift, *center)
                    continue
                xy = parse_xy(txt)
                if xy is None:
                    print("  type Enter, b, c, q, or two numbers")
                elif go_view(swift, *xy):
                    visited[f"{name} (typed)"] = xy
                    print(f"  = centre + ({xy[0] - center[0]:+.1f}, {xy[1] - center[1]:+.1f})")
        if not quit_:
            print("\nAll coils shown.")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if swift is not None:
            try:
                cx, cy, cz = uri.query_position(swift)
                if not np.isnan(cx) and cz < HOVER_Z + 10.0:
                    uri.send_move(swift, cx, cy, HOVER_Z + 10.0, uri.INDENT_SPEED)
            except Exception as exc:
                print(f"uArm cleanup warning: {exc}")
            swift.disconnect()
        if visited:
            print(f"\nCentre: ({center[0]:.1f}, {center[1]:.1f})")
            print("Positions visited (x, y) and offset from the centre:")
            for name, (x, y) in visited.items():
                print(f"  {name:12s} ({x:.1f}, {y:.1f})   offset ({x - center[0]:+.1f}, {y - center[1]:+.1f})")


if __name__ == "__main__":
    main()