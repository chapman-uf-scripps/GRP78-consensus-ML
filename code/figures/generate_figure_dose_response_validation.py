"""
Dose-response validation figure using the user's raw 3-replicate FP data.

Two panels:
  (A) Sigmoidal dose-response curves for the five confirmed actives, with
      individual replicate points (mean +/- s.d.) and 4-parameter logistic fit.
      IC50 from the fit annotated; 25 uM marker overlaid.
  (B) Predicted vs measured GRP78 inhibition at 25 uM for the same five
      compounds, with model-predicted consensus-mean alongside the
      experimentally measured percent inhibition at the 25 uM screening
      concentration (from the raw data) including replicate standard
      deviation.
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

# ---------------------------------------------------------------------------
# Raw 3-replicate FP data (log10[M], % inhibition x 3)
# ---------------------------------------------------------------------------
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
        predicted=64.6,
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
        predicted=63.0,
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
        predicted=66.6,
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
        predicted=62.7,
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
        predicted=61.8,
    ),
}

# Compound colors (OTAVA teal, FDA amber for Bortezomib)
COMPOUND_COLORS = {
    "P2195430": "#159A9C",
    "P2195267": "#0F7C7E",
    "P2195222": "#0A5E60",
    "P1157904": "#3FB5B7",
    "Desisobutyl-n-butyl Bortezomib": "#D99A25",
}

# Compound marker styles to disambiguate overlapping curves
MARKERS = {
    "P2195430": "o",
    "P2195267": "s",
    "P2195222": "^",
    "P1157904": "D",
    "Desisobutyl-n-butyl Bortezomib": "*",
}


# 3-parameter logistic (top fixed at 100, bottom fixed at 0)
# Parameters: logIC50 (log10 of IC50 in M), hill
def hill_3pl(logC, logIC50, hill):
    return 100.0 / (1.0 + 10.0 ** ((logIC50 - logC) * hill))


def hill_4pl(logC, bottom, top, logIC50, hill):
    """4PL used only for plotting smooth curves; main IC50 comes from 3PL."""
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((logIC50 - logC) * hill))


def fit_dose_response(logC_list, mean_pct):
    """3-parameter logistic with top=100 and bottom=0 fixed (FP convention)."""
    p0 = [-5.5, 1.0]
    bounds = ([-9.0, 0.4], [-3.0, 2.5])
    try:
        popt, pcov = curve_fit(hill_3pl, logC_list, mean_pct,
                               p0=p0, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        # Return in 4PL format (bottom=0, top=100, logIC50, hill) for plotting
        return np.array([0.0, 100.0, popt[0], popt[1]]), np.array([0.0, 0.0, perr[0], perr[1]])
    except Exception as exc:
        print(f"  fit failed: {exc}")
        return np.array([0.0, 100.0, -5.5, 1.0]), np.array([np.nan]*4)


def measured_at_25uM(name, logC, mean_pct, fit_popt):
    """Return measured percent inhibition at 25 uM:
       - if 25 uM is one of the sampled concentrations (-4.60206), use the
         observed replicate mean directly
       - otherwise predict from the 4PL fit
    """
    target_logC = np.log10(25e-6)   # = -4.60206
    for lc, mp in zip(logC, mean_pct):
        if abs(lc - target_logC) < 1e-3:
            return float(mp), True
    return float(hill_4pl(target_logC, *fit_popt)), False


def sd_at_25uM(name, logC, reps):
    target_logC = np.log10(25e-6)
    # If 25 uM is one of the sampled concentrations, use the observed SD.
    for lc, rep in zip(logC, reps):
        if abs(lc - target_logC) < 1e-3:
            return float(np.std(rep, ddof=1))
    # Otherwise: fit 3PL to each replicate individually, read the predicted
    # value at 25 uM, return the SD across the 3 per-replicate fits.
    reps_arr = np.asarray(reps)               # shape (n_conc, 3)
    pred_at_25 = []
    for r in range(reps_arr.shape[1]):
        y_r = reps_arr[:, r]
        try:
            popt, _ = curve_fit(hill_3pl, logC, y_r,
                                p0=[-5.5, 1.0],
                                bounds=([-9.0, 0.4], [-3.0, 2.5]),
                                maxfev=20000)
            pred_at_25.append(float(hill_3pl(target_logC, *popt)))
        except Exception:
            pass
    if len(pred_at_25) >= 2:
        return float(np.std(pred_at_25, ddof=1))
    return np.nan


def main():
    # ---- summarise replicates and fit ----
    table = []
    fits = {}
    for name, d in DATA.items():
        logC = np.array(d["logC"])
        reps = np.array(d["reps"])
        mean = reps.mean(axis=1)
        sd   = reps.std(axis=1, ddof=1)
        popt, perr = fit_dose_response(logC, mean)
        fits[name] = popt
        ic50_uM = 10 ** popt[2] / 1e-6
        m25, observed = measured_at_25uM(name, logC, mean, popt)
        sd25 = sd_at_25uM(name, logC, reps)
        table.append({
            "compound": name,
            "predicted_pct": d["predicted"],
            "ic50_M": 10 ** popt[2],
            "ic50_uM": ic50_uM,
            "hill": popt[3],
            "top_pct": popt[1],
            "bottom_pct": popt[0],
            "measured_pct_at_25uM": m25,
            "sd_at_25uM": sd25,
            "value_observed_directly": observed,
        })
    fit_df = pd.DataFrame(table)
    fit_df.to_csv(OUT_SRC / "dose_response_fit_summary.csv", index=False)
    print(fit_df.to_string(index=False))

    # =========================================================================
    # PLOT
    # =========================================================================
    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4),
                             gridspec_kw={"wspace": 0.30, "width_ratios": [1.15, 1.0]})

    # ---------------- Panel A: dose-response curves ----------------
    axA = axes[0]

    # smooth x for fitted curves (in log[M])
    x_fit = np.linspace(-9, -3, 400)

    for name, d in DATA.items():
        logC = np.array(d["logC"])
        reps = np.array(d["reps"])
        mean = reps.mean(axis=1)
        sd   = reps.std(axis=1, ddof=1)
        c    = COMPOUND_COLORS[name]
        mk   = MARKERS[name]

        # 4PL fitted curve
        y_fit = hill_4pl(x_fit, *fits[name])
        axA.plot(x_fit, y_fit, color=c, linewidth=2.2, alpha=0.95, zorder=3)

        # error bars + markers
        axA.errorbar(logC, mean, yerr=sd, fmt=mk, color=c,
                     markersize=8, markeredgecolor="black",
                     markeredgewidth=1.0, capsize=4, elinewidth=1.2,
                     ecolor=c, zorder=5)

    # 25 uM dashed vertical
    axA.axvline(np.log10(25e-6), color="#5a6573", linestyle=":", linewidth=1.2, zorder=1)
    axA.text(np.log10(25e-6) + 0.05, 5, "25 µM\n(screening conc.)",
             fontsize=9, color="#5a6573", fontweight="bold")

    # 50% reference horizontal
    axA.axhline(50, color="#9aa3ad", linestyle=":", linewidth=1.0, zorder=1)

    # Compound legend with IC50 inside the box (sorted by potency)
    fit_df_sorted = fit_df.sort_values("ic50_uM").reset_index(drop=True)
    legend_lines = []
    for _, row in fit_df_sorted.iterrows():
        c  = COMPOUND_COLORS[row["compound"]]
        mk = MARKERS[row["compound"]]
        legend_lines.append(plt.Line2D([0], [0], color=c, linewidth=2.5, marker=mk,
                                       markersize=8, markeredgecolor="black",
                                       markeredgewidth=0.8,
                                       label=f"{row['compound']}  (IC$_{{50}}$ = {row['ic50_uM']:.1f} µM)"))
    axA.legend(handles=legend_lines, loc="upper left", fontsize=10,
               frameon=True, framealpha=0.95)

    axA.set_xlabel("log$_{10}$ [Compound] (M)", fontsize=13, fontweight="bold")
    axA.set_ylabel("GRP78 inhibition (%)", fontsize=13, fontweight="bold")
    axA.set_ylim(-10, 110)
    axA.set_xlim(-8.7, -3.5)
    for sp in ("bottom", "left"):
        axA.spines[sp].set_linewidth(2.2)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.tick_params(labelsize=11, width=2.0, length=6)
    axA.grid(True, linestyle=":", linewidth=0.5, color="#9aa3ad", alpha=0.5)
    axA.text(-0.10, 1.04, "A", transform=axA.transAxes,
             fontsize=24, fontweight="bold", va="bottom")

    # ---------------- Panel B: predicted vs measured at 25 uM ----------------
    axB = axes[1]
    # order = potency
    order = fit_df.sort_values("ic50_uM")["compound"].tolist()
    pos = np.arange(len(order))
    bw = 0.36
    pred = [fit_df.set_index("compound").loc[c, "predicted_pct"] for c in order]
    meas = [fit_df.set_index("compound").loc[c, "measured_pct_at_25uM"] for c in order]
    meas_sd = [fit_df.set_index("compound").loc[c, "sd_at_25uM"] for c in order]
    bar_colors = [COMPOUND_COLORS[c] for c in order]

    axB.bar(pos - bw/2, pred, bw, color="#9aa3ad", edgecolor="black", linewidth=1.4,
            label="Predicted (consensus mean)")
    axB.bar(pos + bw/2, meas, bw, color=bar_colors, edgecolor="black", linewidth=1.4,
            yerr=meas_sd, capsize=4, error_kw={"ecolor": "black", "elinewidth": 1.2},
            label="Measured (FP @ 25 µM, n=3)")

    # Value labels
    for xi, p, m, s in zip(pos, pred, meas, meas_sd):
        axB.text(xi - bw/2, p + 1.5, f"{p:.1f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#3c4858")
        axB.text(xi + bw/2, m + s + 1.5, f"{m:.1f}±{s:.1f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="black")

    # Short axis labels
    short = {
        "P2195430": "P2195430",
        "Desisobutyl-n-butyl Bortezomib": "Desisobutyl-n-butyl\nBortezomib",
        "P1157904": "P1157904",
        "P2195267": "P2195267",
        "P2195222": "P2195222",
    }
    axB.set_xticks(pos)
    axB.set_xticklabels([short[c] for c in order], fontsize=10, fontweight="bold")
    axB.set_ylabel("GRP78 inhibition at 25 µM (%)",
                   fontsize=13, fontweight="bold")
    axB.set_ylim(0, 110)
    axB.axhline(38, color="#5a6573", linestyle="--", linewidth=1.0)
    axB.text(len(order) - 0.5, 39.5, "active threshold (38%)",
             fontsize=9, color="#5a6573", ha="right", fontweight="bold")
    for sp in ("bottom", "left"):
        axB.spines[sp].set_linewidth(2.2)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.tick_params(labelsize=11, width=2.0, length=6)
    axB.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#9aa3ad", alpha=0.5)
    axB.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)
    axB.text(-0.10, 1.04, "B", transform=axB.transAxes,
             fontsize=24, fontweight="bold", va="bottom")

    # Pearson r
    r = np.corrcoef(pred, meas)[0, 1]
    axB.text(0.02, 0.95, f"Pearson r = {r:.3f}", transform=axB.transAxes,
             fontsize=11, fontweight="bold", va="top", ha="left",
             bbox=dict(facecolor="white", edgecolor="#3c4858",
                       boxstyle="round,pad=0.3", linewidth=1.2))

    fig.tight_layout()
    out_png = OUT_FIG / "Figure_dose_response_validation.png"
    out_pdf = OUT_FIG / "Figure_dose_response_validation.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
