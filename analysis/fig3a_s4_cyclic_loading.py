"""
fig3a_s4_cyclic_loading.py  ->  Fig. 3a, Fig. S4a, Fig. S4b
===========================================================
Cyclic loading at the sensor centre (centre coil C1): 1200 cycles were recorded and
the first 1000 successful cycles are analysed.

    Fig. 3a   dL1 vs force for the first and last cycle (loading solid, unloading
              dashed), with the full-scale hysteresis of each
    Fig. S4a  dL1 vs force on loading for the first and last cycle, with linear fits
    Fig. S4b  loading sensitivity and hysteresis of every cycle (dots) with a
              MOVING_AVG-cycle moving average (lines)

    python analysis/fig3a_s4_cyclic_loading.py

Writes results/fig3a.pdf, figS4a.pdf, figS4b.pdf and cyclic_metrics.csv (one row per
cycle), and prints the numbers quoted in the text.
"""
import matplotlib.pyplot as plt
import numpy as np

import extract_cycle_data as ecd
from paths import CYCLES_CSV, data_file, results_dir

N_CYCLES = 1000          # first N successful cycles of the 1200-cycle run
MOVING_AVG = 25          # cycles, Fig. S4b
REPORT_CYCLE = 50        # hysteresis quoted at this cycle
COLORS = ["tab:red", "tab:blue"]
N = ecd.G_TO_N           # g -> N

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def branches(windows, cycle):
    wc = windows[(windows["cycle"] == cycle) & windows["valid"]]
    ld = wc[wc["step_type"].isin(["baseline", "load"])].sort_values("step_idx")
    ul = wc[wc["step_type"] == "unload"].sort_values("step_idx")
    return ld, ul


def fig3a(windows, metrics, coil, cycles):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for col, c in zip(COLORS, cycles):
        ld, ul = branches(windows, c)
        h = metrics.loc[c, "hyst_pct_fs"]
        ax.plot(ld["F"] * N, ld[coil], "o-", ms=3, color=col, label=f"cycle {c}: {h:.1f}% FS")
        ax.plot(np.r_[ld["F"].iloc[-1:], ul["F"]] * N, np.r_[ld[coil].iloc[-1:], ul[coil]],
                "s--", ms=3, color=col, alpha=0.8)
    ax.set_xlabel("Force (N)")
    ax.set_ylabel(r"$\Delta L_1$ ($\mu$H)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def figs4a(windows, metrics, coil, cycles):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for col, c in zip(COLORS, cycles):
        ld, _ = branches(windows, c)
        ld = ld[ld["step_type"] == "load"]
        m = metrics.loc[c]
        F = np.linspace(ld["F"].min(), ld["F"].max(), 50)
        ax.plot(ld["F"] * N, ld[coil], "o", ms=3, color=col)
        ax.plot(F * N, m["sens_uH_per_N"] * N * F + m["intercept_uH"], "-", color=col,
                label=f"cycle {c}: {m['sens_uH_per_N']:.2f} $\\mu$H/N ($R^2$ = {m['r2']:.3f})")
    ax.set_xlabel("Force (N)")
    ax.set_ylabel(r"$\Delta L_1$ ($\mu$H)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def figs4b(metrics):
    x = metrics.index.to_numpy()
    avg = metrics.rolling(MOVING_AVG, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(x, metrics["sens_uH_per_N"], ".", ms=2, color="tab:blue", alpha=0.35)
    ax.plot(x, avg["sens_uH_per_N"], "-", lw=1.5, color="tab:blue", label="sensitivity")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(r"Sensitivity ($\mu$H/N)", color="tab:blue")
    ax.tick_params(axis="y", colors="tab:blue")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, metrics["hyst_pct_fs"], ".", ms=2, color="tab:red", alpha=0.35)
    ax2.plot(x, avg["hyst_pct_fs"], "-", lw=1.5, color="tab:red", label="hysteresis")
    ax2.set_ylabel("Hysteresis (% FS)", color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, avg


def main():
    windows, metrics, coil = ecd.extract_cycle_data(data_file(CYCLES_CSV), n_cycles=N_CYCLES)
    cycles = [metrics.index[0], metrics.index[-1]]
    out = results_dir()
    fig3a(windows, metrics, coil, cycles).savefig(out / "fig3a.pdf", bbox_inches="tight")
    figs4a(windows, metrics, coil, cycles).savefig(out / "figS4a.pdf", bbox_inches="tight")
    fig, avg = figs4b(metrics)
    fig.savefig(out / "figS4b.pdf", bbox_inches="tight")
    metrics.to_csv(out / "cyclic_metrics.csv")

    s = metrics["sens_uH_per_N"]
    for c in cycles:
        m = metrics.loc[c]
        print(f"cycle {c}: sensitivity {m['sens_uH_per_N']:.3f} uH/N (R^2 {m['r2']:.3f}), "
              f"hysteresis {m['hyst_pct_fs']:.1f} % FS")
    print(f"all {len(s)} cycles: sensitivity {s.mean():.3f} +/- {s.std():.3f} uH/N "
          f"(CV {100 * s.std() / s.mean():.1f} %)")
    print(f"cycle {REPORT_CYCLE}: hysteresis {metrics.loc[REPORT_CYCLE, 'hyst_pct_fs']:.1f} % FS "
          f"({MOVING_AVG}-cycle average {avg.loc[REPORT_CYCLE, 'hyst_pct_fs']:.1f} % FS)")
    print(f"wrote {out}/fig3a.pdf, figS4a.pdf, figS4b.pdf, cyclic_metrics.csv")


if __name__ == "__main__":
    main()
