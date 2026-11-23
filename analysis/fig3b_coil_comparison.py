"""
fig3b_coil_comparison.py  ->  Fig. 3b
=====================================
Each coil indented at its own centre for 20 loading/unloading cycles (C1 at the
sensor centre, C2-C5 at +/-18 mm). The first cycle of every coil is discarded.
The curve of each coil is the median, per step, over the remaining cycles; its
sensitivity is the mean over those cycles of the loading-fit slope.

    python analysis/fig3b_coil_comparison.py

Writes results/fig3b.pdf and coil_comparison.csv.
"""
import matplotlib.pyplot as plt
import numpy as np

import extract_cycle_data as ecd
from paths import COIL_CSVS, data_file, results_dir

DISCARD_FIRST = 1
COLORS = {"C1": "tab:red", "C2": "tab:blue", "C3": "tab:green", "C4": "tab:purple",
          "C5": "tab:orange"}
N = ecd.G_TO_N

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def fig3b(summary, curves):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for name, cv in curves.items():
        col = COLORS[name]
        ld = cv[cv["step_type"].isin(["baseline", "load"])].sort_values("step_idx")
        ul = cv[cv["step_type"] == "unload"].sort_values("step_idx")
        s = summary.loc[name, "sens_uH_per_N"]
        ax.plot(ld["F"] * N, ld["dL"], "o-", ms=3, color=col,
                label=f"{name}: {s:.2f} $\\mu$H/N")
        ax.plot(np.r_[ld["F"].iloc[-1:], ul["F"]] * N, np.r_[ld["dL"].iloc[-1:], ul["dL"]],
                "s--", ms=2.5, color=col, alpha=0.7)
    ax.set_xlabel("Force (N)")
    ax.set_ylabel(r"$\Delta L$ ($\mu$H)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def main():
    summary, curves = ecd.coil_comparison([data_file(f) for f in COIL_CSVS], DISCARD_FIRST)
    out = results_dir()
    fig3b(summary, curves).savefig(out / "fig3b.pdf", bbox_inches="tight")
    summary.to_csv(out / "coil_comparison.csv")
    print(summary.round(3))
    print(f"wrote {out}/fig3b.pdf and coil_comparison.csv")


if __name__ == "__main__":
    main()
