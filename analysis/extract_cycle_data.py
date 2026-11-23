"""
extract_cycle_data.py
=====================
Reads the CSVs written by data_collection/uarm_loading_unloading.py and computes, for
the coil under the indenter (dL1 at the sensor centre, dLk at the centre of coil k;
taken from the CSV's site column):

    windows : one row per (cycle, step) = median of the 20-sample window
              F  = force_g - median force of the cycle's baseline window (g)
              dL = L - median L of the cycle's baseline window (uH)
    metrics : one row per cycle
              sens_uH_per_N   slope of dL = s F + b fitted to the loading windows
              r2              R^2 of that fit
              hyst_pct_fs     max |dL_unload - dL_load| at equal force / dL full scale
                              x 100 (unloading branch interpolated at the loading forces)

extract_cycle_data(): one CSV (cyclic test at the centre, Fig. 3a and S4)
coil_comparison():    one CSV per coil (Fig. 3b): sensitivity of each coil at its own
                      centre and the median loading / unloading curve over the cycles

A cycle counts only if every step has a full window with at least MIN_WINDOW_FRACTION
valid samples; failed cycles were repeated by the DAQ and are dropped here.
1 gf = 9.80665 mN.
"""

import numpy as np
import pandas as pd

L_COLS = ["L1", "L2", "L3", "L4", "L5"]
DL_COLS = ["dL1", "dL2", "dL3", "dL4", "dL5"]
G_TO_N = 9.80665e-3
MIN_WINDOW_FRACTION = 0.5


def load_cycle_csv(filename):
    df = pd.read_csv(filename)
    for c in ["pc_time_s", "cycle", "step_idx", "force_g", *L_COLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["cycle"]).copy()
    df["cycle"] = df["cycle"].astype(int)
    return df.sort_values(["cycle", "pc_time_s"], kind="stable").reset_index(drop=True)


def coil_of(df):
    """dLk for site Lk; dL1 (centre coil) for the centre site."""
    site = str(df["site"].dropna().iloc[0]) if "site" in df.columns else "center"
    s = site.upper()
    return site, (f"d{s}" if s in L_COLS else "dL1")


def good_cycles(raw, min_fraction=MIN_WINDOW_FRACTION):
    """Cycles in which every step has a full window (most common window size) with
    at least min_fraction of its samples finite. Returns (good, n_bad)."""
    w = raw[raw["row_type"] == "window"]
    n_steps = int(w["step_idx"].max()) + 1
    sizes = w.groupby(["cycle", "step_idx"]).size()
    n_window = int(sizes.mode().iloc[0])
    ok_row = np.isfinite(w[L_COLS + ["force_g"]].to_numpy(dtype=float)).all(axis=1)
    n_ok = pd.Series(ok_row, index=w.index).groupby([w["cycle"], w["step_idx"]]).sum()
    good = []
    for cyc in sorted(raw["cycle"].unique()):
        if all(sizes.get((cyc, st), 0) >= n_window
               and n_ok.get((cyc, st), 0) >= min_fraction * n_window
               for st in range(n_steps)):
            good.append(int(cyc))
    return good, raw["cycle"].nunique() - len(good)


def add_dL_F(df):
    """dL1..dL5 and F relative to the median of each cycle's baseline window (step 0)."""
    b = df[(df["row_type"] == "window") & (df["step_idx"] == 0)]
    base = b.groupby("cycle")[L_COLS + ["force_g"]].median()
    m = base.reindex(df["cycle"].to_numpy())
    df = df.copy()
    for lc, dc in zip(L_COLS, DL_COLS):
        df[dc] = df[lc].to_numpy() - m[lc].to_numpy()
    df["F"] = df["force_g"].to_numpy() - m["force_g"].to_numpy()
    return df


def window_medians(df, coil, min_fraction=MIN_WINDOW_FRACTION):
    """One row per (cycle, step): medians over the window's samples."""
    w = df[df["row_type"] == "window"]
    g = w.groupby(["cycle", "step_idx"])
    out = g[["F", *DL_COLS]].median()
    out["step_type"] = g["step_type"].first()
    out["valid"] = g[coil].apply(lambda s: np.isfinite(s).sum()) / g.size() >= min_fraction
    return out.reset_index()


def loading_fit(wc, coil):
    """Slope (uH/g), intercept and R^2 of dL vs F over the loading windows."""
    ld = wc[(wc["step_type"] == "load") & wc["valid"]]
    F, dL = ld["F"].to_numpy(), ld[coil].to_numpy()
    s, b = np.polyfit(F, dL, 1)
    ss_res = np.sum((dL - (s * F + b)) ** 2)
    ss_tot = np.sum((dL - dL.mean()) ** 2)
    return s, b, 1 - ss_res / ss_tot


def hysteresis(wc, coil):
    """Max |unloading - loading| dL at equal force, % of the dL full scale."""
    ld = wc[wc["step_type"].isin(["baseline", "load"]) & wc["valid"]].sort_values("F")
    ul = wc[(wc["step_type"] == "unload") & wc["valid"]].sort_values("F")
    f_lo, f_hi = max(ld["F"].min(), ul["F"].min()), min(ld["F"].max(), ul["F"].max())
    fq = ld["F"][(ld["F"] >= f_lo) & (ld["F"] <= f_hi)].to_numpy()
    dl_ld = np.interp(fq, ld["F"], ld[coil])
    dl_ul = np.interp(fq, ul["F"], ul[coil])
    v = wc.loc[wc["valid"], coil]
    return 100.0 * np.max(np.abs(dl_ul - dl_ld)) / (v.max() - v.min())


def cycle_metrics(windows, coil):
    rows = []
    for cyc, wc in windows.groupby("cycle"):
        s, b, r2 = loading_fit(wc, coil)
        rows.append({"cycle": cyc, "sens_uH_per_N": s / G_TO_N, "intercept_uH": b,
                     "r2": r2, "hyst_pct_fs": hysteresis(wc, coil)})
    return pd.DataFrame(rows).set_index("cycle")


def extract_cycle_data(filename, n_cycles=None, verbose=True):
    """Returns (windows, metrics, coil) for one loading/unloading CSV.
    n_cycles: keep only the first n_cycles successful cycles (None = all). Cycle
    numbers stay the real loading count."""
    raw = load_cycle_csv(filename)
    good, n_bad = good_cycles(raw)
    kept = good if n_cycles is None else good[:n_cycles]
    raw = raw[raw["cycle"].isin(kept)]
    site, coil = coil_of(raw)
    windows = window_medians(add_dL_F(raw), coil)
    metrics = cycle_metrics(windows, coil)
    if verbose:
        print(f"[extract] {filename}: site {site}, coil {coil}; {len(good)} successful "
              f"cycles ({n_bad} failed), using {len(kept)} (cycles {kept[0]}..{kept[-1]})")
    return windows, metrics, coil


def coil_comparison(files, discard_first=1, verbose=True):
    """One CSV per coil site. The first discard_first cycles of every site are left out.
    Returns (summary, curves):
        summary  one row per coil: n_cycles, sensitivity (uH/N, mean and std over
                 cycles), R^2, hysteresis (% FS)
        curves   coil -> median over the kept cycles of each step (F, dL, step_type)
    """
    rows, curves = [], {}
    for fn in files:
        windows, metrics, coil = extract_cycle_data(fn, verbose=verbose)
        keep = sorted(metrics.index)[discard_first:]
        mt = metrics.loc[keep]
        name = f"C{coil[-1]}"
        rows.append({"coil": name, "n_cycles": len(keep),
                     "sens_uH_per_N": mt["sens_uH_per_N"].mean(),
                     "sens_uH_per_N_std": mt["sens_uH_per_N"].std(),
                     "r2": mt["r2"].mean(), "hyst_pct_fs": mt["hyst_pct_fs"].mean()})
        w = windows[windows["cycle"].isin(keep) & windows["valid"]]
        g = w.groupby("step_idx")
        curves[name] = pd.DataFrame({"step_type": g["step_type"].first(),
                                     "F": g["F"].median(), "dL": g[coil].median()}).reset_index()
    return pd.DataFrame(rows).set_index("coil"), curves
