"""
fig4_indentation_data.py  ->  Fig. 4b, Fig. 4c
===============================================
    (b) every indentation of the dataset (n = 5500) at its (x, y), coloured by depth;
        crosses mark the coil centres
    (c) mean dL of each coil over the sensing area in 3 x 3 mm cells, at three depth
        ranges (0-1.5, 1.5-3, 3-4.5 mm); one colour scale per coil

    python analysis/fig4_indentation_data.py

Writes results/fig4b.pdf and results/fig4c.png (600 dpi). Same 5500 indentations as
the models (extract_indent_data.load_dataset).
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

import extract_indent_data as eid
from paths import INDENT_CSV, data_file, results_dir

N_POINTS = 5500
SUBSAMPLE_SEED = 42
DEPTH_BINS = (0.0, 1.5, 3.0, 4.5)       # mm (c); the last bin also takes depth > 4.5
CELL_MM = 3.0                           # cell size (c): 12 x 12 cells over +/-18 mm
HALF = 18.0                             # half-width of the sensing area (mm)
VIEW = 20.5                             # half-width of each plot
COIL_CENTERS = [(0, 0), (-18, 18), (18, 18), (-18, -18), (18, -18)]
COIL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def binned_mean(x, y, v, edges):
    """Mean of v in each grid cell (rows = y); NaN where a cell is empty. Points just
    outside the grid are counted in the outermost cells."""
    x = np.clip(x, edges[0] + 1e-6, edges[-1] - 1e-6)
    y = np.clip(y, edges[0] + 1e-6, edges[-1] - 1e-6)
    s, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=v)
    n, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    m[n == 0] = np.nan
    return m.T


def fig4b(pts):
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(pts["x"], pts["y"], c=-pts["z"], cmap="viridis", s=12)
        fig.colorbar(sc, ax=ax, label="depth -z (mm)", shrink=0.8)
        for k, (cx, cy) in enumerate(COIL_CENTERS):
            ax.plot(cx, cy, marker="x", color=COIL_COLORS[k], ms=9, mew=2)
            ax.annotate(f"C{k + 1}", (cx, cy), textcoords="offset points", xytext=(5, 5),
                        color=COIL_COLORS[k], fontsize=9)
        ax.set_xlim(-21, 21)
        ax.set_ylim(-21, 21)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Indentation points (n = {len(pts)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def fig4c(pts):
    depth = -pts["z"].to_numpy()
    idx = np.digitize(depth, DEPTH_BINS[1:-1])
    labels = [f"{DEPTH_BINS[i]:g}-{DEPTH_BINS[i + 1]:g} mm" for i in range(len(DEPTH_BINS) - 1)]
    edges = np.linspace(-HALF, HALF, int(round(2 * HALF / CELL_MM)) + 1)
    x, y = pts["x"].to_numpy(), pts["y"].to_numpy()

    fig, axes = plt.subplots(3, 5, figsize=(5.6, 3.35), layout="constrained",
                             sharex=True, sharey=True)
    for k in range(5):
        v = pts[f"dL{k + 1}"].to_numpy()
        grids = [binned_mean(x[idx == i], y[idx == i], v[idx == i], edges) for i in range(3)]
        vmax = np.nanmax([np.nanmax(g) for g in grids])
        for i, grid in enumerate(grids):
            ax = axes[i, k]
            im = ax.imshow(grid, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]],
                           cmap="viridis", vmin=0, vmax=vmax)
            ax.plot(*COIL_CENTERS[k], marker="x", color="w", ms=3.5, mew=0.9, clip_on=False,
                    zorder=10, path_effects=[pe.withStroke(linewidth=1.8, foreground="k")])
            ax.set_xlim(-VIEW, VIEW)
            ax.set_ylim(-VIEW, VIEW)
            ax.set_xticks([-18, 0, 18])
            ax.set_yticks([-18, 0, 18])
            ax.tick_params(direction="in", length=2.5, pad=2, labelsize=6)
            if i == 0:
                ax.set_title(f"C{k + 1}", pad=2)
            if k == 0:
                ax.set_ylabel(labels[i], fontsize=6)
        cb = fig.colorbar(im, ax=axes[:, k].tolist(), location="bottom", shrink=0.95,
                          pad=0.02, aspect=12)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.set_label(f"ΔL{k + 1} (µH)", fontsize=6)
    fig.supxlabel("x (mm)", fontsize=6.5)
    fig.supylabel("y (mm), by indentation depth", fontsize=6.5)
    for i, lab in enumerate(labels):
        n = int((idx == i).sum())
        empty = int(np.isnan(binned_mean(x[idx == i], y[idx == i], np.ones(n), edges)).sum())
        print(f"(c) {lab}: {n} indentations, {empty} empty cells")
    return fig


def main():
    pts = eid.load_dataset(data_file(INDENT_CSV), N_POINTS, SUBSAMPLE_SEED)
    out = results_dir()
    fig4b(pts).savefig(out / "fig4b.pdf", bbox_inches="tight")
    fig4c(pts).savefig(out / "fig4c.png", dpi=600)
    print(f"wrote {out}/fig4b.pdf and fig4c.png")


if __name__ == "__main__":
    main()
