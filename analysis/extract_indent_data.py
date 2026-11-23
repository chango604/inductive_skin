"""
extract_indent_data.py
======================
Reads the CSV written by data_collection/uarm_random_indentation.py (3D random
indentation) and returns one row per valid hold sample.

Pipeline (extract_indent_data):
    1) load_indent_csv            read the CSV by column name
    2) compute_baseline_medians   per-point median of L1..L5 and force over the hover rows
    3) compute_dL_dF              hold rows only: dL_k = L_k - baseline median,
                                  F = force_g - baseline force median (g, 0.1 g)
    4) convert_xyz                x = x_rep - X_OFFSET, y = y_rep - Y_OFFSET,
                                  z = z_rep - z_surface (z < 0 = into the material),
                                  rounded to 0.1 mm; pos_drift_mm = largest |before - after|
    5) flag_invalid_rows          NaN in dL / F / x / y / z, |x| or |y| > XY_LIMIT,
                                  z outside [Z_MIN, Z_MAX), or pos_drift_mm > POS_DRIFT_TOL_MM
    6) drop_sparse_points         drop points with fewer than MIN_HOLD_FRACTION of
                                  their hold samples valid

reduce_per_point() gives one row per indentation (median over the 20 hold samples);
this is the model input. subsample_points() keeps a fixed-seed random subset of points
(5500 of the 5981 valid points in the paper).
"""

import numpy as np
import pandas as pd

L_COLS = ["L1", "L2", "L3", "L4", "L5"]
DL_COLS = ["dL1", "dL2", "dL3", "dL4", "dL5"]
XYZ_COLS = ["x", "y", "z"]

X_OFFSET = 228.0          # arm x of the sensor centre, so the centre is x = 0
Y_OFFSET = 0.0
XY_LIMIT = 20.0           # keep |x| <= 20 and |y| <= 20 mm
XYZ_DECIMALS = 1          # round x, y, z to 0.1 mm
F_DECIMALS = 1            # round F to 0.1 g
Z_MIN = -5.5              # deepest allowed z (the DAQ safety limit is 5 mm below the surface)
Z_MAX = 0.0               # z >= 0: tip at or above the surface -> no indentation
POS_DRIFT_TOL_MM = 0.5    # max |position before - after| during a hold
MIN_HOLD_FRACTION = 0.5   # keep a point only if >= 50 % of its hold samples are valid


def load_indent_csv(filename):
    """Read the DAQ CSV; numeric columns are coerced (bad values -> NaN)."""
    df = pd.read_csv(filename)
    num_cols = ["pc_time_s", "point_id", "sample_idx", "depth_cmd_mm", "z_surface",
                "force_g", *L_COLS,
                "x_rep", "y_rep", "z_rep", "x_rep_end", "y_rep_end", "z_rep_end"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["point_id"]).copy()
    df["point_id"] = df["point_id"].astype(int)
    return df.sort_values(["point_id", "pc_time_s"], kind="stable").reset_index(drop=True)


def compute_baseline_medians(df):
    """Median L1..L5 and force_g of the hover ('baseline') rows of each point."""
    base = df[df["row_type"] == "baseline"]
    return base.groupby("point_id")[L_COLS + ["force_g"]].median()


def compute_dL_dF(df, baseline_medians, f_decimals=F_DECIMALS):
    """Hold rows only, with dL1..dL5 = L - baseline median and
    F = force_g - baseline force median of that point."""
    hold = df[df["row_type"] == "hold"].copy()
    med = baseline_medians.reindex(hold["point_id"].to_numpy())
    for lc, dc in zip(L_COLS, DL_COLS):
        hold[dc] = hold[lc].to_numpy() - med[lc].to_numpy()
    hold["F"] = np.round(hold["force_g"].to_numpy() - med["force_g"].to_numpy(), f_decimals)
    return hold


def convert_xyz(df, x_offset=X_OFFSET, y_offset=Y_OFFSET, decimals=XYZ_DECIMALS):
    """Reported arm position (queried right before the hold) -> sensor coordinates (mm).
    z = z_rep - z_surface, the depth below the local (plane-corrected) surface."""
    df = df.copy()
    df["x"] = np.round(df["x_rep"] - x_offset, decimals)
    df["y"] = np.round(df["y_rep"] - y_offset, decimals)
    df["z"] = np.round(df["z_rep"] - df["z_surface"], decimals)
    drift = np.abs(df[["x_rep_end", "y_rep_end", "z_rep_end"]].to_numpy()
                   - df[["x_rep", "y_rep", "z_rep"]].to_numpy())
    df["pos_drift_mm"] = np.max(drift, axis=1)       # NaN if a position query failed
    return df


def flag_invalid_rows(df, xy_limit=XY_LIMIT, drift_tol=POS_DRIFT_TOL_MM,
                      z_min=Z_MIN, z_max=Z_MAX):
    """Adds a boolean 'valid' column. Returns (df, counts); each invalid row is counted
    under the first reason it fails (NaN, then x/y, then z, then drift)."""
    df = df.copy()
    finite = np.isfinite(df[DL_COLS + ["F"] + XYZ_COLS].to_numpy()).all(axis=1)
    inside = ((df["x"].abs() <= xy_limit) & (df["y"].abs() <= xy_limit)).to_numpy()
    z_ok = ((df["z"] >= z_min) & (df["z"] < z_max)).to_numpy()
    still = (df["pos_drift_mm"] <= drift_tol).to_numpy()        # NaN -> False
    df["valid"] = finite & inside & z_ok & still
    counts = {"nan": int((~finite).sum()),
              "outside_xy": int((finite & ~inside).sum()),
              "outside_z": int((finite & inside & ~z_ok).sum()),
              "moved_during_hold": int((finite & inside & z_ok & ~still).sum())}
    return df, counts


def drop_sparse_points(df, min_fraction=MIN_HOLD_FRACTION):
    """Valid rows of the points with >= min_fraction valid hold samples.
    Returns (df_valid, n_points_dropped)."""
    frac = df.groupby("point_id")["valid"].mean()
    keep_ids = frac.index[frac >= min_fraction]
    out = df[df["valid"] & df["point_id"].isin(keep_ids)].copy()
    return out.drop(columns=["valid"]), int((frac < min_fraction).sum())


def extract_indent_data(filename, verbose=True):
    """Full extraction. Returns one row per valid hold sample with columns
    point_id, sample_idx, depth_cmd_mm, x, y, z, F, dL1..dL5."""
    raw = load_indent_csv(filename)
    baseline_medians = compute_baseline_medians(raw)
    hold = compute_dL_dF(raw, baseline_medians)
    hold = convert_xyz(hold)
    hold, counts = flag_invalid_rows(hold)
    data, n_pts_dropped = drop_sparse_points(hold)
    keep = ["point_id", "sample_idx", "depth_cmd_mm", *XYZ_COLS, "F", *DL_COLS]
    data = data[keep].reset_index(drop=True)

    if verbose:
        print(f"[extract] {filename}: {raw['point_id'].nunique()} points, "
              f"{int((raw['row_type'] == 'hold').sum())} hold rows")
        print(f"[extract] invalid hold rows: NaN {counts['nan']}, "
              f"|x|,|y| > {XY_LIMIT:g}: {counts['outside_xy']}, "
              f"z outside [{Z_MIN:g}, {Z_MAX:g}): {counts['outside_z']}, "
              f"arm moved > {POS_DRIFT_TOL_MM:g} mm: {counts['moved_during_hold']}")
        print(f"[extract] points dropped (< {MIN_HOLD_FRACTION:.0%} valid samples): "
              f"{n_pts_dropped}; valid points: {data['point_id'].nunique()}")
    return data


def reduce_per_point(data):
    """One row per indentation: median of x, y, z, F and dL1..dL5 over the hold."""
    cols = ["depth_cmd_mm", *XYZ_COLS, "F", *DL_COLS]
    return data.groupby("point_id")[cols].median().reset_index()


def subsample_points(data, n_points, seed=42, verbose=True):
    """Keep n_points randomly chosen points (all hold rows of each). Same seed ->
    same subset; None (or more than available) keeps everything."""
    ids = np.sort(data["point_id"].unique())
    if n_points is None or n_points >= len(ids):
        return data
    rng = np.random.default_rng(seed)
    kept = np.sort(rng.choice(ids, size=n_points, replace=False))
    if verbose:
        print(f"[subsample] {len(ids)} valid points -> {len(kept)} (seed {seed})")
    return data[data["point_id"].isin(kept)].reset_index(drop=True)


def load_dataset(filename, n_points=5500, seed=42, verbose=True):
    """The per-point dataset used in the paper: extraction, fixed-seed subset of
    n_points indentations, then one row per indentation."""
    data = extract_indent_data(filename, verbose=verbose)
    data = subsample_points(data, n_points, seed, verbose=verbose)
    return reduce_per_point(data)
