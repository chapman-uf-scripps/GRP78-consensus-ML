"""
Figure A only: dose-response curves for the five confirmed actives.
Replicate points (mean ± SD) + 3PL fit. No IC50 in legend.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
OUT_FIG = ROOT / "output" / "main_figures"
OUT_SRC = ROOT / "output" / "source_tables"


DATA = {
    "P2195430": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [95.34884, 91.16279, 95.0],
            [86.74419, 85.46512, 86.62791],
            [60.0, 56.39535, 51.86047],
            [33.48837, 25.11628, 30.11628],
            [7.906977, 9.186047, 6.744186],
            [2.674419, 2.906977, 6.511628],
            [-3.255814, 4.069767, 2.55814],
            [-1.511628, -0.9302326, 3.837209],
        ],
    ),
    "P2195267": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [88.70968, 89.35484, 89.56989],
            [61.07527, 61.72043, 63.22581],
            [34.08602, 34.4086, 34.51613],
            [12.7957, 16.98925, 13.76344],
            [9.462365, 7.204301, 3.225806],
            [1.075269, 3.010753, -0.4301075],
            [-4.516129, 3.978495, 3.870968],
            [0.7526882, -0.9677419, 4.301075],
        ],
    ),
    "Desisobutyl-n-butyl Bortezomib": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [98.02325, 101.3953, 101.5116],
            [76.74419, 80.11628, 78.60465],
            [48.02325, 55.34884, 51.27907],
            [26.86047, 29.76744, 28.37209],
            [14.06977, 15.81395, 13.13953],
            [6.162791, 14.4186, 11.86047],
            [0.4651163, 7.790698, 8.488372],
            [4.534883, 5.813953, 5.813953],
        ],
    ),
    "P2195222": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [86.7033, 87.14286, 81.97802],
            [55.82418, 60.76923, 56.37363],
            [31.75824, 24.61539, 36.48352],
            [10.54945, 7.472528, 16.7033],
            [3.076923, 3.846154, 6.153846],
            [0, 0.5494506, 0.8791209],
            [-2.747253, -0.2197802, 0.6593407],
            [-2.307692, 0.5494506, 2.197802],
        ],
    ),
    "P1157904": dict(
        logC=[-4.30103, -4.90309, -5.50515, -6.10721, -6.70927, -7.31133, -7.91339, -8.51545],
        reps=[
            [99.55882, 97.5, 96.61765],
            [61.91177, 56.32353, 66.91177],
            [23.08824, 26.32353, 24.26471],
            [13.97059, 15.44118, 5.0],
            [9.852942, 4.411765, 11.47059],
            [-2.5, 5.147059, 7.647059],
            [3.529412, 1.323529, 3.235294],
            [3.382353, -12.05882, 4.411765],
        ],
    ),
}

COMPOUND_COLORS = {
    "P2195430": "#159A9C",
    "P2195267": "#0F7C7E",
    "P2195222": "#0A5E60",
    "P1157904": "#3FB5B7",
    "Desisobutyl-n-butyl Bortezomib": "#D99A25",
}

MARKERS = {
    "P2195430": "o",
    "P2195267": "s",
    "P2195222": "^",
    "P1157904": "D",
    "Desisobutyl-n-butyl Bortezomib": "*",
}


def hill_3pl(logC, logIC50, hill):
    return 100.0 / (1.0 + 10.0 ** ((logIC50 - logC) * hill))


def fit(logC_list, mean_pct):
    popt, _ = curve_fit(hill_3pl, logC_list, mean_pct,
                        p0=[-5.5, 1.0],
                        bounds=([-9.0, 0.4], [-3.0, 2.5]),
                        maxfev=20000)
    return popt


def main():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 2.4,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })
    fig, ax = plt.subplots(figsize=(9.5, 6.8))

    x_fit = np.linspace(-9, -3, 400)
    fits = {}
    for name, d in DATA.items():
        logC = np.array(d["logC"])
        reps = np.array(d["reps"])
        mean = reps.mean(axis=1)
        sd   = reps.std(axis=1, ddof=1)
        popt = fit(logC, mean)
        fits[name] = popt
        c    = COMPOUND_COLORS[name]
        mk   = MARKERS[name]

        ax.plot(x_fit, hill_3pl(x_fit, *popt), color=c,
                linewidth=2.6, alpha=0.95, zorder=3)
        ax.errorbar(logC, mean, yerr=sd, fmt=mk, color=c,
                    markersize=6.5, markeredgecolor="black",
                    markeredgewidth=0.9, capsize=4, elinewidth=1.3,
                    ecolor=c, zorder=5)

    # 25 µM marker
    ax.axvline(np.log10(25e-6), color="#5a6573", linestyle=":", linewidth=1.4, zorder=1)
    ax.text(np.log10(25e-6) + 0.05, 5, "25 µM\n(screening conc.)",
            fontsize=14, color="#5a6573", fontweight="bold", family="Arial")

    # Compound legend WITHOUT IC50 values
    handles = []
    for name in fits.keys():
        c  = COMPOUND_COLORS[name]
        mk = MARKERS[name]
        handles.append(plt.Line2D([0], [0], color=c, linewidth=2.8, marker=mk,
                                  markersize=8, markeredgecolor="black",
                                  markeredgewidth=0.9, label=name))
    leg = ax.legend(handles=handles, loc="upper left", fontsize=14,
                    frameon=True, framealpha=0.95, prop={"family": "Arial",
                                                          "size": 14,
                                                          "weight": "bold"})

    ax.set_xlabel("log$_{10}$ [Compound] (M)", fontsize=18, fontweight="bold",
                  family="Arial", labelpad=10)
    ax.set_ylabel("GRP78 inhibition (%)", fontsize=18, fontweight="bold",
                  family="Arial", labelpad=10)
    ax.set_ylim(-10, 110)
    ax.set_xlim(-8.7, -3.5)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(2.6)
        ax.spines[sp].set_color("#111111")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=15, width=2.2, length=7, pad=6)
    # force tick labels to Arial bold
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_family("Arial")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#9aa3ad", alpha=0.55)

    fig.tight_layout()
    out_png = OUT_FIG / "Figure_dose_response_curves.png"
    out_pdf = OUT_FIG / "Figure_dose_response_curves.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
