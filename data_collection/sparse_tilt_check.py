"""
sparse_tilt_check.py
====================
Quick tilt check by indentation: presses a sparse grid of points (default 3 x 3,
+/- 15 mm around the centre) at a few fixed depths and logs force and L1..L5.

If the surface is level, the same commanded depth gives the same force everywhere.
Force grows like F = k * (d - d0)^n. One exponent n is chosen for all points (the one
that makes F^(1/n) most linear in depth); then at each point F^(1/n) vs depth is a
straight line whose zero crossing d0 is the depth where contact starts, i.e. how far
BELOW Z_SURFACE the surface is at that point. A plane fitted through d0 gives the tilt.
Use 3 or more depths so n can be found.

--plane: heights are taken from SURFACE_PLANE in uarm_random_indentation.py (the
correction for the arm's reach-dependent z), so the result is what is LEFT after the
correction. Without --plane the flat Z_SURFACE is used (raw arm behaviour).
Either way the script prints an updated SURFACE_PLANE ready to paste.

Output:
    force at every point and depth (grid as seen from above: top row = +y, left = -x)
    surface height of every point relative to the centre (mm)
    tilt in x and y and the height difference across the grid, plus a verdict
    sparse_tilt_check.csv with every reading

Settings (Bluetooth, Arduino port, Z_SURFACE, speeds) come from uarm_random_indentation.py.
Needs five_channel_force_daq.ino, uarm_ble.py in this folder, nothing touching the
sensor at start-up (the load cell is tared at hover).

    python sparse_tilt_check.py                  # raw: measure the plane
    python sparse_tilt_check.py --plane          # verify the correction (should read LEVEL)
    python sparse_tilt_check.py --center 228.0 0.0 --half 15 --n 3 --depths 1 2 3
"""

import argparse
import csv
import time

import numpy as np

import uarm_random_indentation as uri

DEFAULT_CENTER = (228.0, 0.0)
HALF_MM = 15.0                 # grid half-width (mm)
N_GRID = 3                     # points per side (3 -> 9 points, 5 -> 25 points)
DEPTHS_MM = (1.0, 2.0, 3.0)    # commanded depths below Z_SURFACE
SETTLE_S = 0.5                 # arm stopped -> start reading
N_SAMPLES = 10                 # readings per depth (median)
MAX_HALF_MM = 20.0             # grid larger than this is refused
LEVEL_OK_MM = 0.2              # height spread below this counts as level
OUT_CSV = "sparse_tilt_check.csv"
L_COLS = ["L1", "L2", "L3", "L4", "L5"]


def read(reader, n=N_SAMPLES):
    """Median force and L1..L5 over n samples (n fixed-rate SAMPLE requests)."""
    rows, _ = reader.collect(n, uri.SAMPLE_RATE_HZ)
    F = float(np.nanmedian([r["force_g"] for r in rows]))
    L = [float(np.nanmedian([r[c] for r in rows])) for c in L_COLS]
    return F, L


def local_surface(x, y, use_plane):
    return uri.surface_z(x, y) if use_plane else uri.Z_SURFACE


def indent_point(swift, reader, x, y, depths, use_plane=False):
    """Hover reading, then each depth in turn, then back to hover. Returns rows."""
    zs = local_surface(x, y, use_plane)
    hover = zs + uri.HOVER_GAP_MM
    uri.send_move(swift, x, y, hover, uri.TRAVEL_SPEED)
    time.sleep(SETTLE_S)
    F0, L0 = read(reader)
    rows = []
    for d in depths:
        uri.send_move(swift, x, y, round(zs - d, 1), uri.INDENT_SPEED)
        time.sleep(SETTLE_S)
        F, L = read(reader)
        zr = uri.query_position(swift)[2]
        row = {"x": x, "y": y, "z_surface": round(zs, 3), "depth_cmd_mm": d, "z_rep": zr,
               "F_g": F - F0}
        row.update({f"d{c}": Lk - L0k for c, Lk, L0k in zip(L_COLS, L, L0)})
        rows.append(row)
    uri.send_move(swift, x, y, hover, uri.INDENT_SPEED)
    return rows


MIN_FORCE_G = 2.0              # readings below this are treated as no contact
N_EXPONENTS = np.arange(1.0, 2.51, 0.05)


def _line_fit(rows, n):
    """Fit F^(1/n) = a*d + b on the readings with contact. Returns (d0, sse) or None."""
    d = np.array([r["depth_cmd_mm"] for r in rows])
    F = np.array([r["F_g"] for r in rows])
    m = F > MIN_FORCE_G
    if m.sum() < 2:
        return None
    y = F[m] ** (1.0 / n)
    a, b = np.polyfit(d[m], y, 1)
    if a <= 0:
        return None
    sse = float(np.sum((y - (a * d[m] + b)) ** 2) / np.mean(y) ** 2)
    return -b / a, sse


def contact_depths(results):
    """Common exponent n (best linearity over all points) and the contact depth of
    every point. Returns (n, {(dx, dy): d0})."""
    best_n, best_cost = 1.0, np.inf
    for n in N_EXPONENTS:
        fits = [_line_fit(rr, n) for _, _, rr in results]
        cost = sum(f[1] for f in fits if f is not None)
        if any(f is not None for f in fits) and cost < best_cost:
            best_n, best_cost = n, cost
    d0 = {}
    for dx, dy, rr in results:
        f = _line_fit(rr, best_n)
        d0[(dx, dy)] = f[0] if f is not None else float("nan")
    return best_n, d0


def main():
    ap = argparse.ArgumentParser(description="Sparse indentation tilt check (force)")
    ap.add_argument("--center", nargs=2, type=float, default=DEFAULT_CENTER)
    ap.add_argument("--half", type=float, default=HALF_MM, help="grid half-width (mm)")
    ap.add_argument("--n", type=int, default=N_GRID, help="points per side")
    ap.add_argument("--depths", nargs="+", type=float, default=list(DEPTHS_MM))
    ap.add_argument("--plane", action="store_true",
                    help="use SURFACE_PLANE from uarm_random_indentation.py (verify the correction)")
    args = ap.parse_args()
    if args.plane and getattr(uri, "SURFACE_PLANE", None) is None:
        raise SystemExit("--plane: uarm_random_indentation.py has no SURFACE_PLANE set")

    if args.half > MAX_HALF_MM:
        raise SystemExit(f"--half {args.half} > {MAX_HALF_MM} mm")
    if max(args.depths) > uri.MAX_DEPTH_ALLOWED_MM or min(args.depths) <= 0:
        raise SystemExit(f"depths must be in (0, {uri.MAX_DEPTH_ALLOWED_MM}] mm")
    cx0, cy0 = args.center
    offs = np.linspace(-args.half, args.half, args.n)
    depths = sorted(args.depths)
    grid = [(dx, dy) for dy in offs[::-1] for dx in offs]          # top row first
    print(f"{len(grid)} points, +/- {args.half} mm around ({cx0}, {cy0}), depths {depths} mm, "
          f"Z_SURFACE {uri.Z_SURFACE}, "
          + (f"SURFACE_PLANE {uri.SURFACE_PLANE}" if args.plane else "flat (no plane)"))

    from uarm_ble import SwiftBLE
    reader = swift = None
    hover = uri.Z_SURFACE + uri.HOVER_GAP_MM + 3.0
    results = []                                                  # (dx, dy, rows)
    try:
        reader = uri.SensorReader(uri.ARDUINO_PORT, uri.ARDUINO_BAUD, uri.SERIAL_TIMEOUT_S)
        swift = SwiftBLE(address=uri.UARM_BLE_ADDRESS, name_prefix=uri.UARM_BLE_NAME_PREFIX,
                         scan_timeout=uri.UARM_BLE_SCAN_TIMEOUT_S)
        swift.waiting_ready(timeout=5)
        swift.set_mode(0)
        swift.set_speed_factor(1)
        swift.set_report_position(interval=0)

        x, y, z = uri.query_position(swift)
        if np.isnan(x):
            raise RuntimeError("Could not read the arm position.")
        hover = max(local_surface(cx0 + dx, cy0 + dy, args.plane) for dx in offs for dy in offs) \
            + uri.HOVER_GAP_MM
        if z < hover:
            uri.send_move(swift, x, y, hover, uri.INDENT_SPEED)
        print("Taring load cell -- nothing may touch the sensor ...")
        reader.tare()

        for k, (dx, dy) in enumerate(grid, 1):
            rows = indent_point(swift, reader, cx0 + dx, cy0 + dy, depths, args.plane)
            results.append((dx, dy, rows))
            print(f"  {k}/{len(grid)} ({dx:+5.1f}, {dy:+5.1f})  F = "
                  + "  ".join(f"{r['F_g']:6.1f}" for r in rows) + " g")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if swift is not None:
            try:
                x, y, z = uri.query_position(swift)
                if not np.isnan(x) and z < hover + 10:
                    uri.send_move(swift, x, y, hover + 10, uri.INDENT_SPEED)
            except Exception as exc:
                print(f"uArm cleanup warning: {exc}")
            swift.disconnect()
        if reader is not None:
            reader.close()

    if results:
        save(results)
        report(results, offs, depths, (cx0, cy0), args.plane)


def save(results):
    rows = [dict(r, dx=dx, dy=dy) for dx, dy, rr in results for r in rr]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT_CSV}")


def _grid_print(title, offs, values, fmt):
    print(title)
    for dy in offs[::-1]:
        cells = [fmt(values.get((dx, dy), np.nan)) for dx in offs]
        print(f"  y {dy:+5.1f}  " + "  ".join(cells))


def report(results, offs, depths, center, use_plane):
    for d in depths:
        F = {(dx, dy): next(r["F_g"] for r in rr if r["depth_cmd_mm"] == d)
             for dx, dy, rr in results}
        v = np.array(list(F.values()))
        _grid_print(f"\nForce (g) at {d} mm   [spread {np.nanmax(v) - np.nanmin(v):.1f} g, "
                    f"{100 * (np.nanmax(v) - np.nanmin(v)) / np.nanmean(v):.0f}% of mean]",
                    offs, F, lambda f: f"{f:7.1f}")

    n, d0 = contact_depths(results)
    ref = d0.get((0.0, 0.0), np.nanmedian(list(d0.values())))
    h = {k: ref - v for k, v in d0.items()}          # + = surface higher than at the centre
    _grid_print("\nSurface height vs the centre (mm, + = higher = pressed deeper)",
                offs, h, lambda v: f"{v:+7.2f}" if np.isfinite(v) else "    ---")
    where = "the local SURFACE_PLANE height" if use_plane else "Z_SURFACE"
    print(f"  (force ~ depth^{n:.2f}; at the centre contact starts "
          f"{abs(ref):.2f} mm {'below' if ref >= 0 else 'above'} {where})")

    pts = [(dx, dy, v) for (dx, dy), v in h.items() if np.isfinite(v)]
    if len(pts) < 3:
        print("Too few points with contact to fit a plane.")
        return
    A = np.array([[1, dx, dy] for dx, dy, _ in pts])
    z = np.array([v for *_, v in pts])
    (a, b, c), *_ = np.linalg.lstsq(A, z, rcond=None)
    span = 2 * offs.max()
    spread = z.max() - z.min()
    print(f"\nPlane fit: tilt x {b:+.4f} mm/mm ({np.degrees(np.arctan(b)):+.2f} deg) -> "
          f"{b * span:+.2f} mm across x")
    print(f"           tilt y {c:+.4f} mm/mm ({np.degrees(np.arctan(c)):+.2f} deg) -> "
          f"{c * span:+.2f} mm across y")
    print(f"Height spread over the grid: {spread:.2f} mm  ->  "
          + ("LEVEL (within %.1f mm)" % LEVEL_OK_MM if spread <= LEVEL_OK_MM else
             "NOT level: the higher side is pressed deeper at the same commanded depth"))

    # Surface relative to what was commanded here: contact at (used surface - d0), and
    # d0(dx, dy) = ref - (a + b dx + c dy). Add that to the plane that was used.
    old = uri.SURFACE_PLANE if use_plane else {"x0": center[0], "y0": center[1],
                                               "dz0": 0.0, "bx": 0.0, "by": 0.0}
    cx0, cy0 = center
    x0, y0 = old["x0"], old["y0"]
    corr0 = -ref + a + b * (x0 - cx0) + c * (y0 - cy0)      # correction at (x0, y0)
    new = {"x0": float(x0), "y0": float(y0), "dz0": round(float(old["dz0"] + corr0), 3),
           "bx": round(float(old["bx"] + b), 4), "by": round(float(old["by"] + c), 4)}
    print("\nSURFACE_PLANE for uarm_random_indentation.py (keep Z_SURFACE = "
          f"{uri.Z_SURFACE}):")
    print(f"SURFACE_PLANE = {new}")


if __name__ == "__main__":
    main()