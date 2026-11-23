"""
uarm_random_indentation.py
==========================
3D random indentation WITH force: indents the FME at random (x, y, depth) inside a
36 x 36 x 4.5 mm volume (depth DEPTH_RANGE_MM, default 0.5..4.5 mm) with the uArm
Swift Pro ball-tip indenter while logging force (TAL221/HX711) and the 5 LDC1614
coils (Arduino: five_channel_force_daq.ino).

No acceleration setting is sent to the arm (no set_acceleration); the firmware's
default acceleration is used for every move.

Surface plane (SURFACE_PLANE): the arm's real z drifts with reach, so the same
commanded z is not the same height everywhere (measured with sparse_tilt_check.py:
about 2.5 mm across x on a level sensor). Every height is therefore taken from the
local surface
    z_surface(x, y) = Z_SURFACE + dz0 + bx * (x - x0) + by * (y - y0)
target z = z_surface(x, y) - depth, hover = z_surface(x, y) + HOVER_GAP_MM, and the
z_surface column of the CSV holds the local value (so the extraction's z = z_rep -
z_surface is the depth below the local surface). Set SURFACE_PLANE = None to use the
flat Z_SURFACE everywhere.

BLUETOOTH VERSION: the uArm is controlled over Bluetooth LE with uarm_ble.py
(SwiftBLE, same methods as the SDK's SwiftAPI), so no uArm USB cable and no uFactory
dongle are needed. The Arduino (sensor + load cell) is still on its USB serial port.
    * pip install bleak pyserial numpy pandas
    * uarm_ble.py must be in the same folder
    * uArm user-defined button DOWN (Bluetooth mode) before power-on
    * No phone / other program connected to the arm (one BLE connection at a time)
    * Test the link first:  python uarm_ble.py

Per indentation point the CSV contains:
    1) N_BASELINE rows with row_type == "baseline"  (arm stopped HOVER_GAP_MM above the surface)
    2) N_HOLD     rows with row_type == "hold"      (arm stopped at the target depth)
Nothing is logged while the arm moves.

Sampling (fixed rate, default 10 Hz):
    * One "SAMPLE" request per sample on a fixed schedule; a late reply makes the
      next request go out immediately and the real time is logged.
    * The Arduino caches the latest HX711 conversion, so a SAMPLE never waits for
      the load cell. force_seq shows which conversion each row holds.
    * A SAMPLE that gets no reply within SAMPLE_REPLY_TIMEOUT_S is written as a NaN row.

Arm position:
    * Sampling only happens while the arm is stopped. The position is queried with
      get_position() right before and right after each sampling phase; the arm's
      serial line is idle during sampling (position reports are off).
    * x_rep, y_rep, z_rep = before; x_rep_end, y_rep_end, z_rep_end = after.

Points:
    * N_POINTS points from a fixed seed: x, y uniform in XY_BOUNDS, depth uniform in
      DEPTH_RANGE_MM (equal ends = fixed depth), all rounded to 0.1 mm. The same seed
      always gives the same list, which is what makes batch / resume work.

Batch / resume:
    * --start / --end run only points [start, end) of the list.
    * --resume: if --out already exists, the script reads it, keeps every point that
      is complete (all baseline and hold rows present), removes the rows of a point
      that was interrupted half-way (a backup <out>.bak is written first), and
      continues with the next point, appending to the same file.
    * Several CSVs (e.g. one per batch) can be passed together to the extraction.

Clock information:
    * At start-up the script sends "INFO" and saves the Arduino's clock source and
      divider settings to <out>_arduino_info.txt (needed to recompute L later).

Output CSV columns:
    pc_time_s, pc_elapsed_s, row_type, point_id, sample_idx, t_since_stop_s,
    target_x, target_y, target_z, depth_cmd_mm, z_surface,
    rak_t_ms, force_g, force_t_ms, force_seq, L1..L5, raw1..raw5,
    x_rep, y_rep, z_rep, x_rep_end, y_rep_end, z_rep_end

Example (VS Code terminal):
    python uarm_random_indentation.py --rate-test
    python uarm_random_indentation.py --out 3d_force_6000_v2.csv                # dataset used in the paper
    python uarm_random_indentation.py --out 3d_force_6000_v2.csv --resume       # after an interruption
    python uarm_random_indentation.py --start 0 --end 2000 --out batch1.csv     # batches
"""

import argparse
import csv
import os
import shutil
import sys
import time

import numpy as np

try:
    import serial
except ImportError:
    print("This script requires pyserial. Install with: pip install pyserial")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 115200
SERIAL_TIMEOUT_S = 1.0
SAMPLE_REPLY_TIMEOUT_S = 0.5

# uArm over Bluetooth LE (uarm_ble.py). The address is the one nRF Connect showed;
# if it is not found, the first device whose name starts with UARM_BLE_NAME_PREFIX is used.
UARM_BLE_ADDRESS = "7C:01:0A:5E:93:53"   # None = find by name only (needed on macOS)
UARM_BLE_NAME_PREFIX = "UARM"
UARM_BLE_SCAN_TIMEOUT_S = 15.0

# Indentation area in arm coordinates (mm): sensor centre +/- 18
X_CENTER = 228.0                 # sensor centre, confirmed with coil_center_check.py
Y_CENTER = 0.0
XY_HALF = 18.0
XY_BOUNDS = {"x_min": X_CENTER - XY_HALF, "x_max": X_CENTER + XY_HALF,
             "y_min": Y_CENTER - XY_HALF, "y_max": Y_CENTER + XY_HALF}

Z_SURFACE = 99.0               # reference height: KEEP the value used in sparse_tilt_check.py
# Local surface = Z_SURFACE + dz0 + bx*(x - x0) + by*(y - y0)   (from sparse_tilt_check.py;
# it prints this dict ready to paste). None = flat Z_SURFACE everywhere.
SURFACE_PLANE = {'x0': 228.0, 'y0': 0.0, 'dz0': 0.129, 'bx': -0.0811, 'by': -0.0062}
HOVER_GAP_MM = 5.0               # baseline height above the surface


def surface_z(x, y):
    """Local surface height (arm z) at (x, y)."""
    p = SURFACE_PLANE
    if p is None:
        return Z_SURFACE
    return Z_SURFACE + p["dz0"] + p["bx"] * (x - p["x0"]) + p["by"] * (y - p["y0"])


# Highest surface over the indentation area + gap: safe height for start-up / clean-up
HOVER_Z = max(surface_z(x, y) for x in (X_CENTER - XY_HALF, X_CENTER + XY_HALF)
              for y in (Y_CENTER - XY_HALF, Y_CENTER + XY_HALF)) + HOVER_GAP_MM
DEPTH_RANGE_MM = (0.5, 4.5)      # indentation depth range (uniform); equal ends = fixed depth
DEPTH_DECIMALS = 1
MAX_DEPTH_ALLOWED_MM = 5.0       # hard safety limit below the local surface

# Speeds passed to set_position are "SDK speeds"; F = SDK speed * SPEED_FACTOR.
SPEED_FACTOR = 0.1
F_MIN = 1.0
F_MAX = 250.0
TRAVEL_SPEED = 100.0             # moves at hover height (F = 10)
INDENT_SPEED = 50.0              # into / out of the material (F = 5); same in the loading test

MOVE_CMD = "G0"
COORD_DECIMALS = 1
MOVE_TIMEOUT_S = 120

SAMPLE_RATE_HZ = 10.0
N_POINTS_DEFAULT = 6000
N_BASELINE_DEFAULT = 20          # median window (same in every test)
N_HOLD_DEFAULT = 20

HOVER_SETTLE_S = 0.5             # arm stopped at hover -> start of baseline
HOLD_SETTLE_S = 0.5              # arm stopped at depth -> start of hold sampling
POS_QUERY_RETRIES = 3
POS_QUERY_TIMEOUT_S = 2.0

OUTPUT_CSV = "3d_force_6000_v2.csv"
RNG_SEED = 42


# --------------------------------------------------------------------------- #
# Speed check
# --------------------------------------------------------------------------- #
def check_sdk_speed(sdk_speed, name="speed"):
    f = sdk_speed * SPEED_FACTOR
    if not (F_MIN <= f <= F_MAX):
        print(f"WARNING: {name} = {sdk_speed} gives F = {f:.2f}, outside {F_MIN}..{F_MAX}.")
    return f


# --------------------------------------------------------------------------- #
# Random indentation points
# --------------------------------------------------------------------------- #
def generate_random_points(n, bounds, depth_range, seed):
    """List of (x, y, depth) in arm coordinates (mm), rounded. Same seed -> same list."""
    rng = np.random.default_rng(seed)
    xs = np.round(rng.uniform(bounds["x_min"], bounds["x_max"], n), COORD_DECIMALS)
    ys = np.round(rng.uniform(bounds["y_min"], bounds["y_max"], n), COORD_DECIMALS)
    # Rounding can push a point just outside bounds that are not on the 0.1 mm grid
    # (e.g. y_min = -14.03 -> y = -14.0 is fine, but x_max = 170.25 -> 170.3 is not).
    # Clip to the bounds rounded inward. Bounds already on the grid are unaffected.
    q = 10 ** COORD_DECIMALS
    xs = np.clip(xs, np.ceil(bounds["x_min"] * q - 1e-9) / q, np.floor(bounds["x_max"] * q + 1e-9) / q)
    ys = np.clip(ys, np.ceil(bounds["y_min"] * q - 1e-9) / q, np.floor(bounds["y_max"] * q + 1e-9) / q)
    ds = np.round(rng.uniform(depth_range[0], depth_range[1], n), DEPTH_DECIMALS)
    return list(zip(xs.tolist(), ys.tolist(), ds.tolist()))


# --------------------------------------------------------------------------- #
# Arduino sensor reader
# --------------------------------------------------------------------------- #
class SensorReader:
    """Request/reply reader for five_channel_force_daq.ino."""

    N_COMMAS = 13       # millis,force_g,force_t_ms,force_seq,L1..L5,raw1..raw5 (14 fields)
    KEYS = ["rak_t_ms", "force_g", "force_t_ms", "force_seq",
            "L1", "L2", "L3", "L4", "L5", "raw1", "raw2", "raw3", "raw4", "raw5"]

    def __init__(self, port, baud, timeout):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)
        self._flush_boot_messages()

    def _flush_boot_messages(self):
        t_end = time.time() + 2.0
        while time.time() < t_end:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"[Arduino boot] {line}")

    def info(self, timeout_s=3.0):
        """Ask for the clock source / divider settings. Returns the INFO lines."""
        self.ser.reset_input_buffer()
        self.ser.write(b"INFO\n")
        lines, t_end = [], time.time() + timeout_s
        while time.time() < t_end:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line.startswith("# INFO_END"):
                break
            if line.startswith("# INFO"):
                lines.append(line[2:])
        if not lines:
            print("WARNING: no INFO reply (is five_channel_force_daq.ino uploaded?)")
        return lines

    def tare(self, timeout_s=5.0):
        self.ser.reset_input_buffer()
        self.ser.write(b"TARE\n")
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            line = self.ser.readline().decode(errors="ignore").strip()
            if "TARE_ACK" in line:
                print("[Arduino] load cell tared")
                return True
        print("WARNING: no TARE_ACK received.")
        return False

    def _read_data_line(self, timeout_s):
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            if line.count(",") == self.N_COMMAS:
                return line
        return None

    def sample_once(self):
        t_send = time.time()
        self.ser.write(b"SAMPLE\n")
        line = self._read_data_line(SAMPLE_REPLY_TIMEOUT_S)
        t_recv = time.time()
        if line is None:
            return None
        try:
            vals = [float(p) for p in line.split(",")]
        except ValueError:
            return None
        s = dict(zip(self.KEYS, vals))
        s["pc_time_s"] = 0.5 * (t_send + t_recv)
        s["round_trip_s"] = t_recv - t_send
        return s

    def collect(self, n, rate_hz):
        """n samples on a fixed schedule. A failed SAMPLE becomes a NaN row."""
        period = 1.0 / rate_hz
        self.ser.reset_input_buffer()
        out, n_fail, n_late = [], 0, 0
        t_start = time.perf_counter()
        for k in range(n):
            t_target = t_start + k * period
            _sleep_until(t_target)
            if time.perf_counter() - t_target > 0.5 * period:
                n_late += 1
            s = self.sample_once()
            if s is None:
                n_fail += 1
                self.ser.reset_input_buffer()
                s = {key: float("nan") for key in self.KEYS}
                s["pc_time_s"] = time.time()
                s["round_trip_s"] = float("nan")
            s["sample_idx"] = k
            out.append(s)
        dur = time.perf_counter() - t_start
        stats = {"n": n, "n_fail": n_fail, "n_late": n_late,
                 "rate_hz": (n - 1) / dur if n > 1 and dur > 0 else float("nan")}
        return out, stats

    def close(self):
        self.ser.close()


def _sleep_until(t_target):
    while True:
        dt = t_target - time.perf_counter()
        if dt <= 0:
            return
        if dt > 0.003:
            time.sleep(dt - 0.002)


def save_info(lines, out_csv):
    path = os.path.splitext(out_csv)[0] + "_arduino_info.txt"
    with open(path, "w") as f:
        f.write(f"# saved {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for ln in lines:
            f.write(ln + "\n")
    print(f"[info] clock settings -> {path}")
    for ln in lines:
        print(f"       {ln}")


# --------------------------------------------------------------------------- #
# Arm helpers
# --------------------------------------------------------------------------- #
def f_of(speed):
    return round(speed * SPEED_FACTOR, 2)


XY_TOL_MM = 0.2      # a point at most this far outside XY_BOUNDS (rounding) is moved onto the edge


def clamp_xy(x, y, tol=XY_TOL_MM):
    """Return (x, y) moved onto XY_BOUNDS if it is outside by <= tol (rounding).
    Raises ValueError if it is further outside (the point is then skipped)."""
    q = 10 ** COORD_DECIMALS
    out = []
    for name, v, lo, hi in (("x", x, XY_BOUNDS["x_min"], XY_BOUNDS["x_max"]),
                            ("y", y, XY_BOUNDS["y_min"], XY_BOUNDS["y_max"])):
        if v < lo - tol or v > hi + tol:
            raise ValueError(f"{name}={v:.2f} is more than {tol} mm outside [{lo:.2f}, {hi:.2f}]")
        # edges rounded inward to the 0.1 mm grid, so the rounded command stays inside
        lo_g = np.ceil(lo * q - 1e-9) / q
        hi_g = np.floor(hi * q + 1e-9) / q
        out.append(float(min(max(v, lo_g), hi_g)))
    return out[0], out[1]


def check_xyz(x, y, z):
    tol = 0.5 / 10 ** COORD_DECIMALS + 1e-6     # commands are rounded to COORD_DECIMALS
    zs = surface_z(x, y)
    z_min = zs - MAX_DEPTH_ALLOWED_MM
    limits = {"x": (x, XY_BOUNDS["x_min"], XY_BOUNDS["x_max"]),
              "y": (y, XY_BOUNDS["y_min"], XY_BOUNDS["y_max"]),
              "z": (z, z_min, max(HOVER_Z, zs + HOVER_GAP_MM))}
    bad = [f"{k}={v:.3f} not in [{lo:.3f}, {hi:.3f}]"
           for k, (v, lo, hi) in limits.items() if not (lo - tol <= v <= hi + tol)]
    if bad:
        raise ValueError(f"Unsafe target x={x:.2f} y={y:.2f} z={z:.2f}: " + "; ".join(bad))


def send_move(swift, x, y, z, speed):
    swift.set_position(x=round(float(x), COORD_DECIMALS), y=round(float(y), COORD_DECIMALS),
                       z=round(float(z), COORD_DECIMALS), speed=f_of(speed), wait=True,
                       timeout=MOVE_TIMEOUT_S, cmd=MOVE_CMD)
    swift.flush_cmd(timeout=MOVE_TIMEOUT_S, wait_stop=True)


def move(swift, x, y, z, speed):
    check_xyz(x, y, z)
    send_move(swift, x, y, z, speed)


def query_position(swift, retries=POS_QUERY_RETRIES, timeout=POS_QUERY_TIMEOUT_S):
    for _ in range(retries):
        try:
            pos = swift.get_position(wait=True, timeout=timeout)
        except Exception:
            pos = None
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            try:
                return tuple(float(p) for p in pos[:3])
            except (TypeError, ValueError):
                pass
        time.sleep(0.1)
    return (float("nan"),) * 3


# --------------------------------------------------------------------------- #
# CSV and resume
# --------------------------------------------------------------------------- #
FIELDS = [
    "pc_time_s", "pc_elapsed_s", "row_type", "point_id", "sample_idx", "t_since_stop_s",
    "target_x", "target_y", "target_z", "depth_cmd_mm", "z_surface",
    "rak_t_ms", "force_g", "force_t_ms", "force_seq", "L1", "L2", "L3", "L4", "L5",
    "raw1", "raw2", "raw3", "raw4", "raw5",
    "x_rep", "y_rep", "z_rep", "x_rep_end", "y_rep_end", "z_rep_end",
]


def write_phase(writer, samples, row_type, meta, t_stop, t0, pos_before, pos_after):
    for s in samples:
        row = dict(s)
        row.update(meta)
        row["row_type"] = row_type
        row["pc_elapsed_s"] = s["pc_time_s"] - t0
        row["t_since_stop_s"] = s["pc_time_s"] - t_stop
        row["x_rep"], row["y_rep"], row["z_rep"] = pos_before
        row["x_rep_end"], row["y_rep_end"], row["z_rep_end"] = pos_after
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def prepare_resume(out_csv, n_baseline, n_hold):
    """Keep complete points of an existing CSV, drop a half-written last point.
    Returns the set of completed point_ids."""
    import pandas as pd
    df = pd.read_csv(out_csv)
    if df.empty:
        return set()
    counts = df.groupby(["point_id", "row_type"]).size().unstack(fill_value=0)
    for col in ("baseline", "hold"):
        if col not in counts.columns:
            counts[col] = 0
    complete = counts.index[(counts["baseline"] >= n_baseline) & (counts["hold"] >= n_hold)]
    complete = set(int(i) for i in complete)
    partial = set(int(i) for i in counts.index) - complete
    if partial:
        shutil.copyfile(out_csv, out_csv + ".bak")
        df[df["point_id"].isin(complete)].to_csv(out_csv, index=False)
        print(f"[resume] removed rows of incomplete point(s) {sorted(partial)} "
              f"(backup: {out_csv}.bak)")
    print(f"[resume] {len(complete)} complete points found in {out_csv}")
    return complete


def run_phase(swift, reader, n, rate_hz):
    pos_before = query_position(swift)
    samples, stats = reader.collect(n, rate_hz)
    pos_after = query_position(swift)
    return samples, stats, pos_before, pos_after


# --------------------------------------------------------------------------- #
# Rate test (Arduino only)
# --------------------------------------------------------------------------- #
def rate_test(n, rate_hz):
    reader = SensorReader(ARDUINO_PORT, ARDUINO_BAUD, SERIAL_TIMEOUT_S)
    try:
        for ln in reader.info():
            print(ln)
        samples, stats = reader.collect(n, rate_hz)
    finally:
        reader.close()
    rt = np.array([s["round_trip_s"] for s in samples]) * 1e3
    seq = np.array([s["force_seq"] for s in samples])
    L = np.array([[s[c] for c in ("L1", "L2", "L3", "L4", "L5")] for s in samples])
    rak = np.array([s["rak_t_ms"] for s in samples])
    print(f"\nRequested rate     : {rate_hz:.2f} Hz")
    print(f"Achieved rate (PC) : {stats['rate_hz']:.2f} Hz   failed: {stats['n_fail']}  late: {stats['n_late']}")
    print(f"Arduino period     : median {np.nanmedian(np.diff(rak)):.1f} ms")
    print(f"Round trip         : median {np.nanmedian(rt):.1f} ms   max {np.nanmax(rt):.1f} ms")
    d_seq = np.diff(seq)
    print(f"Force conversions  : repeated in {np.mean(d_seq == 0):.0%} of rows, "
          f"skipped in {np.mean(d_seq > 1):.0%}")
    print("Median L (uH)      : " + "  ".join(f"L{i + 1} {v:.3f}" for i, v in enumerate(np.nanmedian(L, axis=0))))
    rep = (np.diff(L, axis=0) == 0).mean(axis=0)
    print("Repeated L value   : " + "  ".join(f"L{i + 1} {r:.0%}" for i, r in enumerate(rep)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="3D random indentation DAQ (uArm BLE + LDC1614 + HX711)")
    ap.add_argument("--n-points", type=int, default=N_POINTS_DEFAULT, help="size of the point list")
    ap.add_argument("--start", type=int, default=0, help="first point index to run")
    ap.add_argument("--end", type=int, default=None, help="stop before this index (default: all)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted --out file")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--out", type=str, default=OUTPUT_CSV)
    ap.add_argument("--n-baseline", type=int, default=N_BASELINE_DEFAULT)
    ap.add_argument("--n-hold", type=int, default=N_HOLD_DEFAULT)
    ap.add_argument("--rate", type=float, default=SAMPLE_RATE_HZ, help="sampling rate (Hz)")
    ap.add_argument("--rate-test", action="store_true",
                    help="Arduino only: sample 100 times at --rate and report timing")
    args = ap.parse_args()

    if args.rate_test:
        rate_test(100, args.rate)
        return

    for name, sp in [("TRAVEL_SPEED", TRAVEL_SPEED), ("INDENT_SPEED", INDENT_SPEED)]:
        check_sdk_speed(sp, name)
    if DEPTH_RANGE_MM[1] > MAX_DEPTH_ALLOWED_MM:
        raise ValueError("DEPTH_RANGE_MM exceeds MAX_DEPTH_ALLOWED_MM")

    points = generate_random_points(args.n_points, XY_BOUNDS, DEPTH_RANGE_MM, args.seed)
    end = len(points) if args.end is None else min(args.end, len(points))
    todo = list(range(args.start, end))

    exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    if exists and not args.resume:
        print(f"{args.out} already exists. Use --resume to continue it, or choose another --out.")
        sys.exit(1)
    done = prepare_resume(args.out, args.n_baseline, args.n_hold) if exists else set()
    todo = [i for i in todo if i not in done]
    if not todo:
        print("Nothing to do: every requested point is already complete.")
        return

    print(f"Points {todo[0]}..{todo[-1]} ({len(todo)} to run)  depth {DEPTH_RANGE_MM} mm  "
          f"baseline {args.n_baseline} + hold {args.n_hold} samples at {args.rate} Hz")

    try:
        from uarm_ble import SwiftBLE
    except ImportError as exc:
        print(f"Bluetooth driver not available ({exc}).\n"
              "Put uarm_ble.py in this folder and install bleak: pip install bleak")
        sys.exit(1)

    reader = swift = f = None
    t0 = time.time()
    n_done = 0
    try:
        reader = SensorReader(ARDUINO_PORT, ARDUINO_BAUD, SERIAL_TIMEOUT_S)
        save_info(reader.info(), args.out)

        swift = SwiftBLE(address=UARM_BLE_ADDRESS, name_prefix=UARM_BLE_NAME_PREFIX,
                         scan_timeout=UARM_BLE_SCAN_TIMEOUT_S)
        swift.waiting_ready(timeout=5)
        swift.set_mode(0)
        swift.set_speed_factor(1)
        swift.set_report_position(interval=0)

        cx, cy, cz = query_position(swift)
        if np.isnan(cx):
            raise RuntimeError("Could not read the arm position at startup.")
        if cz < HOVER_Z:
            send_move(swift, cx, cy, HOVER_Z, INDENT_SPEED)    # lift straight up first

        print("Taring load cell -- nothing may touch the sensor ...")
        reader.tare()

        f = open(args.out, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()

        t_loop = time.time()
        skipped = []
        for k, pid in enumerate(todo):
            x, y, depth = points[pid]
            try:
                cx_, cy_ = clamp_xy(x, y)
            except ValueError as exc:
                print(f"Point {pid}: SKIPPED ({exc})")
                skipped.append(pid)
                continue
            if (cx_, cy_) != (x, y):
                print(f"Point {pid}: moved onto the edge ({x:.2f}, {y:.2f}) -> ({cx_:.2f}, {cy_:.2f})")
            x, y = cx_, cy_
            zs = surface_z(x, y)
            hover_z = round(zs + HOVER_GAP_MM, COORD_DECIMALS)
            target_z = round(zs - depth, COORD_DECIMALS)
            meta = {"point_id": pid, "target_x": x, "target_y": y, "target_z": target_z,
                    "depth_cmd_mm": depth, "z_surface": round(zs, 3)}

            # 1) Travel at hover height, settle, baseline
            move(swift, x, y, hover_z, TRAVEL_SPEED)
            t_stop = time.time()
            time.sleep(HOVER_SETTLE_S)
            samples, st_b, p0, p1 = run_phase(swift, reader, args.n_baseline, args.rate)
            write_phase(writer, samples, "baseline", meta, t_stop, t0, p0, p1)

            # 2) Indent, settle, hold
            move(swift, x, y, target_z, INDENT_SPEED)
            t_stop = time.time()
            time.sleep(HOLD_SETTLE_S)
            samples, st_h, p0, p1 = run_phase(swift, reader, args.n_hold, args.rate)
            write_phase(writer, samples, "hold", meta, t_stop, t0, p0, p1)
            f.flush()

            # 3) Retract to hover
            move(swift, x, y, hover_z, INDENT_SPEED)
            n_done += 1

            per_pt = (time.time() - t_loop) / max(n_done, 1)
            eta_h = per_pt * (len(todo) - k - 1) / 3600
            hold_F = np.nanmedian([s["force_g"] for s in samples])
            fails = st_b["n_fail"] + st_h["n_fail"]
            print(f"Point {pid} ({k + 1}/{len(todo)}): x={x:.1f} y={y:.1f} depth={depth:.1f}  "
                  f"F={hold_F:.1f} g  {st_h['rate_hz']:.1f} Hz"
                  f"{f'  failed {fails}' if fails else ''}  |  {per_pt:.1f} s/pt, ETA {eta_h:.1f} h")

        print("All requested points complete."
              + (f" Skipped (outside bounds): {skipped}" if skipped else ""))

    except KeyboardInterrupt:
        print("\nInterrupted by user. Continue later with --resume.")
        print("Lifting the arm (press Ctrl+C again to skip) ...")

    finally:
        if f is not None:
            f.flush()
            f.close()
        if reader is not None:
            reader.close()
        if swift is not None:
            try:
                cx, cy, _ = query_position(swift)
                if not np.isnan(cx):
                    send_move(swift, cx, cy, HOVER_Z + 10.0, INDENT_SPEED)
            except KeyboardInterrupt:
                print("Lift skipped.")
            except Exception as exc:
                print(f"uArm cleanup warning: {exc}")
            swift.disconnect()
        print(f"Data saved to {args.out}  ({n_done} points this session, "
              f"{(time.time() - t0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()