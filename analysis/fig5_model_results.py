"""
fig5_model_results.py  ->  Fig. 5a-f
====================================
Model results on the test set, from the trained MLP models in models/
(nothing is retrained). The script rebuilds the same 5500 indentations and 80/20
split as train_models.py, predicts the test points and draws every panel. All errors
are mean absolute errors (MAE) per output.

    (a) MAE of x and y, z and F per indentation-depth bin
    (b) MAE of x and y versus the minimum indentation depth kept, with the
        percentage of test points kept (right axis); dotted line = MIN_DEPTH_MM
    (c) predicted vs true x, y, z   } test points at least MIN_DEPTH_MM deep
    (d) predicted vs true F         }
    (e) in-plane MAE map: mean of |x error| and |y error| per CELL_MM cell
    (f) force MAE map: |F error| per CELL_MM cell       (e, f: depth >= MIN_DEPTH_MM)

    python analysis/fig5_model_results.py

Writes results/fig5a.pdf .. fig5f.pdf and prints every number shown in the figure.
"""
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np

import extract_indent_data as eid
import indent_models as imd
from paths import INDENT_CSV, MODEL_DIR, data_file, results_dir

N_POINTS = 5500                         # same as train_models.py
SUBSAMPLE_SEED = 42
TEST_SIZE = 0.2
SEED = 42
MODEL_FILES = {"xyz": MODEL_DIR / "indent3d_xyz_MLP.joblib",
               "F": MODEL_DIR / "indent3d_F_MLP.joblib"}

MIN_DEPTH_MM = 2.0                      # (b)-(f)
DEPTH_BINS = (0.0, 1.0, 2.0, 3.0, 4.5)  # (a); the last bin also takes depth > 4.5
THRESHOLDS = np.arange(0.0, 3.01, 0.25) # (b)
CELL_MM = 4.0                           # (e, f): 9 x 9 cells over +/-18 mm
HALF_MM = 18.0                          # half width of the sensing area
COIL_CENTERS = [(0, 0), (-18, 18), (18, 18), (-18, -18), (18, -18)]
COIL_RADIUS = 14.5                      # coil outer radius (mm), red circles in (e, f)

COLORS = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green", "F": "tab:purple"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "pdf.fonttype": 42})


def load_test_predictions():
    """Returns {"depth": (n,), "xy": (n, 2), t: (true, predicted)} for the test
    points, with F in N."""
    pts = eid.load_dataset(data_file(INDENT_CSV), N_POINTS, SUBSAMPLE_SEED)
    _, test_ids = imd.split_points(pts["point_id"], TEST_SIZE, SEED)
    te = np.isin(pts["point_id"], test_ids)
    r = {"depth": -pts.loc[te, "z"].to_numpy(), "xy": pts.loc[te, ["x", "y"]].to_numpy()}
    for name, targets in imd.TARGET_SETS.items():
        X, Y, _, _ = imd.make_dataset(pts[te], targets)
        P = imd.predict(joblib.load(MODEL_FILES[name]), X)
        for j, t in enumerate(targets):
            s = imd.G_TO_N if t == "F" else 1.0
            r[t] = (Y[:, j] * s, P[:, j] * s)
    return r


def mae(r, t, mask=None):
    y, p = r[t]
    e = np.abs(y - p)
    return float(e[mask].mean() if mask is not None else e.mean())


def cell_mean(xy, values, cell=CELL_MM, half=HALF_MM):
    """Mean of values in square cells over +/-half (rows = y); points just outside go
    to the edge cells."""
    edges = np.arange(-half, half + 1e-9, cell)
    x = np.clip(xy[:, 0], -half, half - 1e-9)
    y = np.clip(xy[:, 1], -half, half - 1e-9)
    s, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=values)
    n, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    m[n == 0] = np.nan
    if np.isnan(m).any():
        print(f"  warning: {int(np.isnan(m).sum())} empty cells (shown white)")
    return m.T, edges


def panel_a(r):
    idx = np.digitize(r["depth"], DEPTH_BINS[1:-1])
    labels = [f"{DEPTH_BINS[i]:g}-{DEPTH_BINS[i + 1]:g}" for i in range(len(DEPTH_BINS) - 1)]
    pos = np.arange(len(labels))
    rows = {t: [mae(r, t, idx == i) for i in pos] for t in ("x", "y", "z", "F")}
    print("\n(a) MAE per depth bin")
    for i, lab in enumerate(labels):
        print(f"  {lab:>6s} mm (n={int((idx == i).sum()):4d}): "
              + "  ".join(f"{t} {rows[t][i]:.3f}" for t in rows))
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 1.9))
    w = 0.38
    axes[0].bar(pos - w / 2, rows["x"], w, color=COLORS["x"], label="x")
    axes[0].bar(pos + w / 2, rows["y"], w, color=COLORS["y"], label="y")
    axes[0].set_ylabel("MAE (mm)")
    axes[0].legend(frameon=False)
    axes[1].bar(pos, rows["z"], 0.6, color=COLORS["z"])
    axes[1].set_ylabel("z MAE (mm)")
    axes[2].bar(pos, rows["F"], 0.6, color=COLORS["F"])
    axes[2].set_ylabel("F MAE (N)")
    for ax in axes:
        ax.set_xticks(pos, labels)
        ax.set_xlabel("depth (mm)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def panel_b(r):
    d = r["depth"]
    mx = [mae(r, "x", d >= t) for t in THRESHOLDS]
    my = [mae(r, "y", d >= t) for t in THRESHOLDS]
    kept = [100 * np.mean(d >= t) for t in THRESHOLDS]
    print("\n(b) MAE vs minimum depth")
    for t, a, b, k in zip(THRESHOLDS, mx, my, kept):
        print(f"  >= {t:4.2f} mm: x {a:.3f}  y {b:.3f}  kept {k:5.1f} %")
    fig, ax = plt.subplots(figsize=(2.9, 1.9))
    ax.plot(THRESHOLDS, mx, "o-", ms=3, color=COLORS["x"], label="x MAE")
    ax.plot(THRESHOLDS, my, "s-", ms=3, color=COLORS["y"], label="y MAE")
    ax.axvline(MIN_DEPTH_MM, color="k", ls=":", lw=1)
    ax.set_xlabel("minimum depth (mm)")
    ax.set_ylabel("MAE (mm)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(THRESHOLDS, kept, color="gray", lw=1.2, label="points kept")
    ax2.set_ylabel("points kept (%)", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def _scatter(ax, y, p, name, unit, lims=None):
    ax.scatter(y, p, s=2, alpha=0.4, color="tab:blue", edgecolors="none")
    lo, hi = lims if lims is not None else (min(y.min(), p.min()), max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"true {name} ({unit})")
    ax.set_ylabel(f"predicted {name} ({unit})")
    ax.set_title(f"{name} (MAE = {np.abs(y - p).mean():.{3 if unit == 'N' else 2}f} {unit})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def panel_c(r):
    m = r["depth"] >= MIN_DEPTH_MM
    print(f"\n(c, d) test points >= {MIN_DEPTH_MM:g} mm: n = {int(m.sum())}")
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.2))
    for ax, t in zip(axes, ("x", "y", "z")):
        y, p = r[t]
        _scatter(ax, y[m], p[m], t, "mm")
        print(f"  {t} MAE {np.abs(y[m] - p[m]).mean():.3f} mm")
    fig.tight_layout()
    return fig


def panel_d(r):
    m = r["depth"] >= MIN_DEPTH_MM
    y, p = r["F"]
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    _scatter(ax, y[m], p[m], "F", "N", lims=(0.0, max(y[m].max(), p[m].max())))
    print(f"  F MAE {np.abs(y[m] - p[m]).mean():.4f} N")
    fig.tight_layout()
    return fig


def _map(r, values, label):
    m = r["depth"] >= MIN_DEPTH_MM
    grid, edges = cell_mean(r["xy"][m], values[m])
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    im = ax.imshow(grid, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]],
                   cmap="viridis", vmin=0)
    for cx, cy in COIL_CENTERS:
        ax.add_patch(plt.Circle((cx, cy), COIL_RADIUS, fill=False, color="red", lw=1))
    ax.set_xlim(-HALF_MM, HALF_MM)
    ax.set_ylim(-HALF_MM, HALF_MM)
    ax.set_xticks(np.arange(-15, 16, 5))
    ax.set_yticks(np.arange(-15, 16, 5))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, label=label)
    print(f"  {label}: cell means {np.nanmin(grid):.3f} .. {np.nanmax(grid):.3f}, "
          f"n = {int(m.sum())} points")
    fig.tight_layout()
    return fig


def panel_e(r):
    ex = np.abs(r["x"][0] - r["x"][1])
    ey = np.abs(r["y"][0] - r["y"][1])
    print(f"\n(e, f) maps, test points >= {MIN_DEPTH_MM:g} mm, {CELL_MM:g} mm cells")
    return _map(r, (ex + ey) / 2, "xy MAE (mm)")


def panel_f(r):
    return _map(r, np.abs(r["F"][0] - r["F"][1]), "F MAE (N)")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)   # scikit-learn version notes
    r = load_test_predictions()
    print(f"\nAll test points (n = {len(r['depth'])}): "
          + "  ".join(f"{t} {mae(r, t):.3f}" for t in ("x", "y", "z", "F")))
    out = results_dir()
    for key, fn in zip("abcdef", (panel_a, panel_b, panel_c, panel_d, panel_e, panel_f)):
        fig = fn(r)
        fig.savefig(out / f"fig5{key}.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"\nwrote {out}/fig5a.pdf .. fig5f.pdf")


if __name__ == "__main__":
    main()
