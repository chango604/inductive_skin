"""
uarm_loading_unloading.py
=========================
Repeated loading / unloading with the uArm Swift Pro ball-tip indenter, logging
force (TAL221/HX711) and the 5 LDC1614 coils CONTINUOUSLY at 10 Hz
(Arduino: five_channel_force_daq.ino).

Indentation sites (--site):
    center      the sensor centre (CENTER_X, CENTER_Y): hysteresis and latency
                (1200 cycles in the paper; the first 1000 are analysed)
    L1 .. L5    the centre of one coil: sensitivity of each coil
    all         L1, L2, L3, L4, L5 one after the other
The coil centres are COIL_OFFSETS_MM from the sensor centre: C1 at the centre and the
corner coils at +/-18 mm, the same convention as the 3D random-indentation data.
With one site the data go to --out; with several, each site gets its own file
<out>_<site>.csv (e.g. cycles_L3.csv). Between sites the arm lifts, moves at hover
height, re-tares the load cell and lowers onto the next site.

Surface plane (SURFACE_PLANE, measured with
sparse_tilt_check.py): the arm's real z drifts with reach, so every height is taken
from the local surface
    z_surface(x, y) = Z_SURFACE + dz0 + bx * (x - x0) + by * (y - y0)
Each site is indented from its own surface; the z_surface column of the CSV holds it.
Keep Z_SURFACE and SURFACE_PLANE identical to uarm_random_indentation.py.

Recommended runs:
    python uarm_loading_unloading.py --cycles 1200 --out cycles_1200.csv        # hysteresis (Fig. 3a, S4)
    python uarm_loading_unloading.py --site all --cycles 20 --out cycles        # sensitivity per coil (Fig. 3b)

No acceleration setting is sent to the arm (no set_acceleration); the firmware's
default acceleration is used for every move.

BLUETOOTH VERSION: the uArm is controlled over Bluetooth LE with uarm_ble.py
(SwiftBLE, same methods as the SDK's SwiftAPI), so no uArm USB cable and no uFactory
dongle are needed. The Arduino (sensor + load cell) is still on its USB serial port.
    * pip install bleak pyserial numpy pandas
    * uarm_ble.py must be in the same folder
    * uArm user-defined button DOWN (Bluetooth mode) before power-on
    * No phone / other program connected to the arm (one BLE connection at a time)
    * Test the link first:  python uarm_ble.py

One cycle (the arm never leaves the surface between cycles):
    step 0          baseline window at depth 0 (indenter at the site's surface)
    steps 1..9      loading:   depth 0.5, 1.0, ..., 4.5 mm
    steps 10..18    unloading: depth 4.0, 3.5, ..., 0.0 mm
For every step: move at INDENT_SPEED (same as the random-indentation test) ->
wait until stopped -> WINDOW_SETTLE_S -> query position -> N_WINDOW samples
(the median window, 20 = same as every other test) -> query position.

Continuous logging:
    A background thread sends "SAMPLE" at a fixed 10 Hz for the whole run, including
    while the arm moves. Every row is tagged with the phase it was taken in:
        row_type = "window"  (one of the 20 samples of a step's median window)
                   "move"    (arm moving to the step's depth)
                   "settle"  (arm stopped, waiting before the window)
    Window rows give the hysteresis / sensitivity points (median of the window);
    all rows together give the latency time series.
    The arm is only asked for its position while it is stopped (before and after
    each window); the uArm (Bluetooth) and the Arduino (USB serial) use separate links.

Rows are buffered in memory and written once per cycle (window rows get the
before / after positions of their window).

Resume:
    --resume continues an interrupted --out file: complete cycles are kept, the rows
    of a half-finished cycle are removed (backup <out>.bak), and the run continues
    with the next cycle number. Note that the pause itself is part of the sample's
    loading history (the elastomer recovers while it rests).

Output CSV columns:
    pc_time_s, pc_elapsed_s, cycle, step_idx, step_type, depth_cmd_mm, row_type,
    window_sample_idx, rak_t_ms, force_g, force_t_ms, force_seq, L1..L5, raw1..raw5,
    x_rep, y_rep, z_rep, x_rep_end, y_rep_end, z_rep_end, z_surface, site, site_x, site_y

Output CSV columns (in addition): site, site_x, site_y  (z_surface = that site's surface)

Example:
    python uarm_loading_unloading.py --rate-test
    python uarm_loading_unloading.py --cycles 1200 --out cycles_1200.csv
    python uarm_loading_unloading.py --cycles 1200 --out cycles_1200.csv --resume
    python uarm_loading_unloading.py --site all --cycles 20 --out cycles      # cycles_L1.csv .. cycles_L5.csv
    python uarm_loading_unloading.py --site L3 --cycles 20 --out cycles_L3.csv
"""

import argparse
import csv
import os
import shutil
import sys
import threading
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

CENTER_X = 228.0                 # sensor centre in arm coordinates (mm)
CENTER_Y = 0.0
Z_SURFACE = 99.0                 # reference height: SAME as uarm_random_indentation.py
# Local surface = Z_SURFACE + dz0 + bx*(x - x0) + by*(y - y0). Copied from
# uarm_random_indentation.py (sparse_tilt_check.py result);
# update both together. None = flat Z_SURFACE everywhere.
SURFACE_PLANE = {"x0": 228.0, "y0": 0.0, "dz0": 0.129, "bx": -0.0811, "by": -0.0062}
APPROACH_GAP_MM = 5.0            # start / end height above the surface

# Coil centres relative to the sensor centre (mm): same convention as the 3D data
COIL_OFFSETS_MM = {
    "L1": (0.0, 0.0),            # centre coil (230 uH)
    "L2": (-18.0, 18.0),         # top-left     (-x, +y)
    "L3": (18.0, 18.0),          # top-right    (+x, +y)
    "L4": (-18.0, -18.0),        # bottom-left  (-x, -y)
    "L5": (18.0, -18.0),         # bottom-right (+x, -y)
}


def surface_z(x, y):
    """Local surface height (arm z) at (x, y)."""
    p = SURFACE_PLANE
    if p is None:
        return Z_SURFACE
    return Z_SURFACE + p["dz0"] + p["bx"] * (x - p["x0"]) + p["by"] * (y - p["y0"])


# Indentation sites: name -> (x, y); the surface height comes from surface_z(x, y)
SITES = {"center": (CENTER_X, CENTER_Y)}
SITES.update({k: (CENTER_X + dx, CENTER_Y + dy) for k, (dx, dy) in COIL_OFFSETS_MM.items()})
COIL_SITES = list(COIL_OFFSETS_MM)
# Above every site's surface: safe height for start-up and clean-up
SAFE_Z = max(surface_z(x, y) for x, y in SITES.values()) + APPROACH_GAP_MM
SITE_MAX_DIST_MM = 30.0          # a site further than this from (CENTER_X, CENTER_Y) is refused
SITE_MAX_DZ_MM = 3.0             # a site surface further than this from Z_SURFACE is refused

STEP_MM = 0.5
MAX_DEPTH_MM = 4.5
MAX_DEPTH_ALLOWED_MM = 5.0       # hard safety limit below the local surface

SPEED_FACTOR = 0.1
INDENT_SPEED = 50.0              # F = 5, same as the random-indentation test
TRAVEL_SPEED = 100.0             # F = 10, moves at hover height between sites
MOVE_CMD = "G0"
COORD_DECIMALS = 1
MOVE_TIMEOUT_S = 120

SAMPLE_RATE_HZ = 10.0
N_WINDOW = 20                    # median window (same in every test)
WINDOW_SETTLE_S = 0.5            # arm stopped -> start of the window
WINDOW_TIMEOUT_S = 30.0
N_CYCLES_DEFAULT = 1200

POS_QUERY_RETRIES = 3
POS_QUERY_TIMEOUT_S = 2.0

OUTPUT_CSV = "cycles_1200.csv"


# --------------------------------------------------------------------------- #
# Step plan
# --------------------------------------------------------------------------- #
def cycle_steps(step_mm=STEP_MM, max_depth=MAX_DEPTH_MM):
    """[(step_idx, step_type, depth_mm)] for one cycle: baseline, loading, unloading."""
    n = int(round(max_depth / step_mm))
    load = [round(step_mm * k, 3) for k in range(1, n + 1)]
    unload = [round(step_mm * k, 3) for k in range(n - 1, -1, -1)]
    steps = [(0, "baseline", 0.0)]
    steps += [(i + 1, "load", d) for i, d in enumerate(load)]
    steps += [(n + 1 + i, "unload", d) for i, d in enumerate(unload)]
    return steps


# --------------------------------------------------------------------------- #
# Arduino reader (same protocol as uarm_random_indentation.py)
# --------------------------------------------------------------------------- #
class SensorReader:
    N_COMMAS = 13       # millis,force_g,force_t_ms,force_seq,L1..L5,raw1..raw5 (14 fields)
    KEYS = ["rak_t_ms", "force_g", "force_t_ms", "force_seq",
            "L1", "L2", "L3", "L4", "L5", "raw1", "raw2", "raw3", "raw4", "raw5"]

    def __init__(self, port, baud, timeout):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)
        t_end = time.time() + 2.0
        while time.time() < t_end:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"[Arduino boot] {line}")

    def info(self, timeout_s=3.0):
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
            if "TARE_ACK" in self.ser.readline().decode(errors="ignore"):
                print("[Arduino] load cell tared")
                return True
        print("WARNING: no TARE_ACK received.")
        return False

    def sample_once(self):
        t_send = time.time()
        self.ser.write(b"SAMPLE\n")
        line, t_end = None, time.time() + SAMPLE_REPLY_TIMEOUT_S
        while time.time() < t_end:
            ln = self.ser.readline().decode(errors="ignore").strip()
            if ln and not ln.startswith("#") and ln.count(",") == self.N_COMMAS:
                line = ln
                break
        t_recv = time.time()
        if line is None:
            self.ser.reset_input_buffer()
            s = {k: float("nan") for k in self.KEYS}
        else:
            try:
                s = dict(zip(self.KEYS, [float(p) for p in line.split(",")]))
            except ValueError:
                s = {k: float("nan") for k in self.KEYS}
        s["pc_time_s"] = 0.5 * (t_send + t_recv)
        return s

    def close(self):
        self.ser.close()


def save_info(lines, out_csv):
    path = os.path.splitext(out_csv)[0] + "_arduino_info.txt"
    with open(path, "w") as f:
        f.write(f"# saved {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for ln in lines:
            f.write(ln + "\n")
    print(f"[info] clock settings -> {path}")


# --------------------------------------------------------------------------- #
# Continuous logger thread
# --------------------------------------------------------------------------- #
class ContinuousLogger(threading.Thread):
    """Samples the Arduino at a fixed rate for the whole run and tags every row
    with the current phase. The main thread opens a window with start_window();
    the logger counts N samples into it and then signals window_done."""

    def __init__(self, reader, rate_hz):
        super().__init__(daemon=True)
        self.reader = reader
        self.period = 1.0 / rate_hz
        self.lock = threading.Lock()
        self.tag = {"cycle": 0, "step_idx": -1, "step_type": "", "depth_cmd_mm": np.nan,
                    "row_type": "settle"}
        self.rows = []
        self.win_left = 0
        self.win_idx = 0
        self.window_done = threading.Event()
        self.running = threading.Event()
        self.running.set()
        self.n_fail = 0

    def set_phase(self, **tag):
        with self.lock:
            self.tag.update(tag)

    def start_window(self, n):
        with self.lock:
            self.win_left = n
            self.win_idx = 0
            self.tag["row_type"] = "window"
            self.window_done.clear()

    def pop_rows(self):
        with self.lock:
            rows, self.rows = self.rows, []
        return rows

    def run(self):
        t_next = time.perf_counter()
        while self.running.is_set():
            t_next += self.period
            s = self.reader.sample_once()
            if not np.isfinite(s.get("rak_t_ms", np.nan)):
                self.n_fail += 1
            with self.lock:
                row = dict(s)
                row.update(self.tag)
                if self.tag["row_type"] == "window":
                    row["window_sample_idx"] = self.win_idx
                    self.win_idx += 1
                    self.win_left -= 1
                    if self.win_left <= 0:
                        self.tag["row_type"] = "settle"
                        self.window_done.set()
                else:
                    row["window_sample_idx"] = ""
                self.rows.append(row)
            while True:
                dt = t_next - time.perf_counter()
                if dt <= 0:
                    break
                time.sleep(dt - 0.002 if dt > 0.003 else 0)
            if time.perf_counter() - t_next > self.period:   # a late / lost reply:
                t_next = time.perf_counter()                 # skip the missed ticks, no burst

    def stop(self):
        self.running.clear()
        self.join(timeout=5)


# --------------------------------------------------------------------------- #
# Arm helpers
# --------------------------------------------------------------------------- #
def f_of(speed):
    return round(speed * SPEED_FACTOR, 2)


def send_move(swift, x, y, z, speed, z_surface=Z_SURFACE):
    if not (z_surface - MAX_DEPTH_ALLOWED_MM - 1e-6 <= z <= z_surface + 50):
        raise ValueError(f"Unsafe z target {z:.2f} (surface {z_surface:.2f})")
    swift.set_position(x=round(float(x), COORD_DECIMALS), y=round(float(y), COORD_DECIMALS),
                       z=round(float(z), COORD_DECIMALS), speed=f_of(speed), wait=True,
                       timeout=MOVE_TIMEOUT_S, cmd=MOVE_CMD)
    swift.flush_cmd(timeout=MOVE_TIMEOUT_S, wait_stop=True)


def resolve_site(name):
    """(x, y, z_surface) of a site, checked against the safety limits."""
    if name not in SITES:
        raise ValueError(f"Unknown site '{name}'. Choose from {list(SITES)} or 'all'.")
    x, y = SITES[name]
    zs = round(float(surface_z(x, y)), 3)
    dist = float(np.hypot(x - CENTER_X, y - CENTER_Y))
    if dist > SITE_MAX_DIST_MM:
        raise ValueError(f"Site {name} ({x}, {y}) is {dist:.1f} mm from the centre "
                         f"(limit {SITE_MAX_DIST_MM} mm). Typo?")
    if abs(zs - Z_SURFACE) > SITE_MAX_DZ_MM:
        raise ValueError(f"Site {name} surface {zs} differs from Z_SURFACE by more than "
                         f"{SITE_MAX_DZ_MM} mm: check SURFACE_PLANE.")
    return float(x), float(y), zs


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
FIELDS = ["pc_time_s", "pc_elapsed_s", "cycle", "step_idx", "step_type", "depth_cmd_mm",
          "row_type", "window_sample_idx", "rak_t_ms", "force_g", "force_t_ms", "force_seq",
          "L1", "L2", "L3", "L4", "L5", "raw1", "raw2", "raw3", "raw4", "raw5",
          "x_rep", "y_rep", "z_rep", "x_rep_end", "y_rep_end", "z_rep_end", "z_surface",
          "site", "site_x", "site_y"]


def write_rows(writer, rows, positions, t0, site):
    """site = (name, x, y, z_surface)"""
    for r in sorted(rows, key=lambda r: r["pc_time_s"]):
        r = dict(r)
        r["pc_elapsed_s"] = r["pc_time_s"] - t0
        r["site"], r["site_x"], r["site_y"], r["z_surface"] = site
        if r["row_type"] == "window":
            p0, p1 = positions.get((r["cycle"], r["step_idx"]), ((np.nan,) * 3, (np.nan,) * 3))
            r["x_rep"], r["y_rep"], r["z_rep"] = p0
            r["x_rep_end"], r["y_rep_end"], r["z_rep_end"] = p1
        writer.writerow({k: r.get(k, "") for k in FIELDS})


def prepare_resume(out_csv, n_windows, n_window):
    """Keep complete cycles, drop a half-written one. Returns the last complete cycle."""
    import pandas as pd
    df = pd.read_csv(out_csv)
    if df.empty:
        return 0
    w = df[df["row_type"] == "window"].groupby("cycle").size()
    complete = set(int(c) for c in w.index[w >= n_windows * n_window])
    partial = set(int(c) for c in df["cycle"].dropna().unique()) - complete
    if partial:
        shutil.copyfile(out_csv, out_csv + ".bak")
        df[df["cycle"].isin(complete)].to_csv(out_csv, index=False)
        print(f"[resume] removed rows of incomplete cycle(s) {sorted(partial)} (backup: {out_csv}.bak)")
    last = max(complete) if complete else 0
    print(f"[resume] {len(complete)} complete cycles, continuing after cycle {last}")
    return last


# --------------------------------------------------------------------------- #
# Rate test
# --------------------------------------------------------------------------- #
def rate_test(rate_hz, seconds=10.0):
    reader = SensorReader(ARDUINO_PORT, ARDUINO_BAUD, SERIAL_TIMEOUT_S)
    for ln in reader.info():
        print(ln)
    log = ContinuousLogger(reader, rate_hz)
    log.start()
    time.sleep(seconds)
    log.stop()
    reader.close()
    rows = log.pop_rows()
    rak = np.array([r["rak_t_ms"] for r in rows], float)
    seq = np.array([r["force_seq"] for r in rows], float)
    print(f"\n{len(rows)} samples in {seconds:.0f} s -> {len(rows) / seconds:.2f} Hz   failed {log.n_fail}")
    print(f"Arduino period: median {np.nanmedian(np.diff(rak)):.1f} ms, max {np.nanmax(np.diff(rak)):.1f} ms")
    d = np.diff(seq)
    print(f"Force conversions: repeated in {np.mean(d == 0):.0%} of rows, skipped in {np.mean(d > 1):.0%}")


# --------------------------------------------------------------------------- #
# One site
# --------------------------------------------------------------------------- #
def out_path(out, site, n_sites):
    """--out as given for one site; <out stem>_<site>.csv for several sites."""
    if n_sites == 1:
        return out
    stem = os.path.splitext(out)[0]
    return f"{stem}_{site}.csv"


def run_site(swift, reader, site, out, first_cycle, n_cycles, args, info_lines, t0):
    """Hover above the site, tare, lower onto the surface and run cycles
    first_cycle..n_cycles, appending to out. Returns the number of cycles done."""
    name, sx, sy, zs = site
    steps = cycle_steps()
    z_app = zs + APPROACH_GAP_MM
    save_info(info_lines, out)

    # Lift (if needed), travel at hover height, tare with nothing touching, lower
    cx, cy, cz = query_position(swift)
    if np.isnan(cx):
        raise RuntimeError("Could not read the arm position before moving to the site.")
    if cz < z_app:
        send_move(swift, cx, cy, z_app, INDENT_SPEED, zs)    # straight up first
    send_move(swift, sx, sy, z_app, TRAVEL_SPEED, zs)
    print(f"[{name}] at ({sx:.1f}, {sy:.1f}), surface z = {zs:.2f}. "
          "Taring load cell -- nothing may touch the sensor ...")
    reader.tare()
    send_move(swift, sx, sy, zs, INDENT_SPEED, zs)

    exists = os.path.exists(out) and os.path.getsize(out) > 0
    fields = FIELDS
    if exists:                       # resuming a file written by an older version: keep its columns
        with open(out, newline="") as fh:
            fields = next(csv.reader(fh))
        if fields != FIELDS:
            print(f"[{name}] {out} has the older column set; new columns are not written to it.")
    f = open(out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if not exists:
        writer.writeheader()

    log = ContinuousLogger(reader, args.rate)
    log.set_phase(cycle=first_cycle, step_idx=0, step_type="baseline", depth_cmd_mm=0.0,
                  row_type="settle")             # no stray "cycle 0" rows before the first move
    log.start()
    t_loop = time.time()
    n_done = 0
    try:
        for cycle in range(first_cycle, n_cycles + 1):
            positions = {}
            for step_idx, step_type, depth in steps:
                z = round(zs - depth, COORD_DECIMALS)
                log.set_phase(cycle=cycle, step_idx=step_idx, step_type=step_type,
                              depth_cmd_mm=depth, row_type="move")
                send_move(swift, sx, sy, z, INDENT_SPEED, zs)
                log.set_phase(row_type="settle")
                time.sleep(WINDOW_SETTLE_S)
                p0 = query_position(swift)
                log.start_window(args.n_window)
                if not log.window_done.wait(WINDOW_TIMEOUT_S):
                    print(f"WARNING: [{name}] window cycle {cycle} step {step_idx} timed out")
                p1 = query_position(swift)
                positions[(cycle, step_idx)] = (p0, p1)
            write_rows(writer, log.pop_rows(), positions, t0, site)
            f.flush()
            n_done += 1
            per = (time.time() - t_loop) / n_done
            print(f"[{name}] cycle {cycle}/{n_cycles}  {per:.1f} s/cycle  "
                  f"ETA {per * (n_cycles - cycle) / 3600:.2f} h  failed samples so far {log.n_fail}")
    finally:
        log.stop()
        rest = log.pop_rows()               # rows of an unfinished cycle: removed by --resume
        if rest:
            write_rows(writer, rest, {}, t0, site)
        f.flush()
        f.close()
        print(f"[{name}] {n_done} cycles this session -> {out}")
    send_move(swift, sx, sy, z_app, INDENT_SPEED, zs)        # lift off this site (not logged)
    return n_done


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Loading / unloading cycles at the sensor centre "
                                             "or at the centre of each coil")
    ap.add_argument("--site", nargs="+", default=["center"],
                    help="center, L1..L5, or all (= L1..L5); several allowed")
    ap.add_argument("--cycles", type=int, default=N_CYCLES_DEFAULT, help="total cycles per site")
    ap.add_argument("--out", type=str, default=OUTPUT_CSV,
                    help="output CSV (several sites: <out stem>_<site>.csv)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rate", type=float, default=SAMPLE_RATE_HZ)
    ap.add_argument("--n-window", type=int, default=N_WINDOW)
    ap.add_argument("--rate-test", action="store_true", help="Arduino only: log 10 s, report timing")
    args = ap.parse_args()

    if args.rate_test:
        rate_test(args.rate)
        return

    names = []
    for n in args.site:
        for m in (COIL_SITES if n.lower() == "all" else [n]):
            m = "center" if m.lower() in ("center", "centre", "c") else m.upper()
            if m not in names:
                names.append(m)
    try:
        sites = [(n, *resolve_site(n)) for n in names]
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    steps = cycle_steps()
    print("Cycle steps: " + ", ".join(f"{t[0]}:{t[1]}@{t[2]}" for t in steps))

    # Plan every site before touching the hardware
    plan = []
    for site in sites:
        out = out_path(args.out, site[0], len(sites))
        exists = os.path.exists(out) and os.path.getsize(out) > 0
        if exists and not args.resume:
            print(f"{out} already exists. Use --resume to continue it, or choose another --out.")
            sys.exit(1)
        last = prepare_resume(out, len(steps), args.n_window) if exists else 0
        if last >= args.cycles:
            print(f"[{site[0]}] nothing to do: {args.cycles} cycles already in {out}")
            continue
        plan.append((site, out, last + 1))
        print(f"[{site[0]}] ({site[1]:.1f}, {site[2]:.1f}, surface {site[3]:.2f}) "
              f"cycles {last + 1}..{args.cycles} -> {out}")
    if not plan:
        print("Nothing to do.")
        return

    try:
        from uarm_ble import SwiftBLE
    except ImportError as exc:
        print(f"Bluetooth driver not available ({exc}).\n"
              "Put uarm_ble.py in this folder and install bleak: pip install bleak")
        sys.exit(1)

    reader = swift = None
    t0 = time.time()
    n_total = 0
    try:
        reader = SensorReader(ARDUINO_PORT, ARDUINO_BAUD, SERIAL_TIMEOUT_S)
        info_lines = reader.info()

        swift = SwiftBLE(address=UARM_BLE_ADDRESS, name_prefix=UARM_BLE_NAME_PREFIX,
                         scan_timeout=UARM_BLE_SCAN_TIMEOUT_S)
        swift.waiting_ready(timeout=5)
        swift.set_mode(0)
        swift.set_speed_factor(1)
        swift.set_report_position(interval=0)

        for site, out, first in plan:
            n_total += run_site(swift, reader, site, out, first, args.cycles, args, info_lines, t0)
        print("All requested sites complete.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Continue later with the same command + --resume.")
        print("Lifting the arm (press Ctrl+C again to skip) ...")

    finally:
        if reader is not None:
            reader.close()
        if swift is not None:
            try:
                cx, cy, cz = query_position(swift)
                if not np.isnan(cx):
                    send_move(swift, cx, cy, max(cz, SAFE_Z), INDENT_SPEED, SAFE_Z - APPROACH_GAP_MM)
            except KeyboardInterrupt:
                print("Lift skipped.")
            except Exception as exc:
                print(f"uArm cleanup warning: {exc}")
            swift.disconnect()
        print(f"{n_total} cycles this session, {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
