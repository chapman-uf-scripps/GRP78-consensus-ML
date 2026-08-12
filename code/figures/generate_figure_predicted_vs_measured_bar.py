"""
Figure B: predicted vs measured GRP78 inhibition at 25 uM as paired bars
(no Pearson r, no active-threshold line). Same Arial-bold styling as the
dose-response figure.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
OUT_FIG = ROOT / "output" / "main_figures"
OUT_SRC = ROOT / "output" / "source_tables"

# Use the dose-response fit summary if it exists; otherwise hard-code.
fit_csv = OUT_SRC / "dose_response_fit_summary.csv"
if fit_csv.exists():
    df = pd.read_csv(fit_csv)
    df = df[["compound", "predicted_pct", "measured_pct_at_25uM",
             "sd_at_25uM", "ic50_uM"]].rename(
        columns={"measured_pct_at_25uM": "measured_pct",
                 "sd_at_25uM": "measured_sd"})
else:
    df = pd.DataFrame([
        {"compound": "P2195430",                       "predicted_pct": 64.6, "measured_pct": 86.28, "measured_sd": 0.71, "ic50_uM": 4.3},
        {"compound": "Desisobutyl-n-butyl Bortezomib", "predicted_pct": 66.6, "measured_pct": 78.49, "measured_sd": 1.69, "ic50_uM": 4.7},
        {"compound": "P1157904",                        "predicted_pct": 61.8, "measured_pct": 80.85, "measured_sd": 3.45, "ic50_uM": 7.5},
        {"compound": "P2195267",                        "predicted_pct": 63.0, "measured_pct": 62.01, "measured_sd": 1.10, "ic50_uM": 12.8},
        {"compound": "P2195222",                        "predicted_pct": 62.7, "measured_pct": 57.66, "measured_sd": 2.71, "ic50_uM": 16.1},
    ])

# SD of the 3 model predictions (Extra Trees, CatBoost, GNN) — the "model
# agreement" uncertainty for each compound, looked up from
# consensus_A5_std in the OTAVA / FDA prediction CSVs.
PREDICTED_SD = {
    "P2195430":                       13.78,
    "P2195267":                       13.58,
    "P2195222":                       13.36,
    "P1157904":                       20.62,
    "Desisobutyl-n-butyl Bortezomib": 11.89,
}
df["predicted_sd"] = df["compound"].map(PREDICTED_SD)

# order by potency (most potent left)
df = df.sort_values("ic50_uM").reset_index(drop=True)

COMPOUND_COLORS = {
    "P2195430": "#159A9C",
    "P2195267": "#0F7C7E",
    "P2195222": "#0A5E60",
    "P1157904": "#3FB5B7",
    "Desisobutyl-n-butyl Bortezomib": "#D99A25",
}

PRED_COLOR = "#9aa3ad"

DISPLAY = {
    "P2195430": "P2195430",
    "Desisobutyl-n-butyl Bortezomib": "Desisobutyl-n-butyl\nBortezomib",
    "P1157904": "P1157904",
    "P2195267": "P2195267",
    "P2195222": "P2195222",
}


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

    fig, ax = plt.subplots(figsize=(10.5, 6.6))

    n = len(df)
    x = np.arange(n)
    bw = 0.36
    pred = df["predicted_pct"].values
    meas = df["measured_pct"].values
    meas_sd = df["measured_sd"].fillna(0).values
    bar_colors = [COMPOUND_COLORS[c] for c in df["compound"]]

    pred_sd = df["predicted_sd"].fillna(0).values

    # predicted bars (left of each pair) — no error bar
    ax.bar(x - bw/2, pred, bw, color=PRED_COLOR,
           edgecolor="black", linewidth=1.6,
           label="Predicted")
    # measured bars with error bars (right of each pair)
    ax.bar(x + bw/2, meas, bw, color=bar_colors,
           edgecolor="black", linewidth=1.6,
           yerr=meas_sd, capsize=5,
           error_kw={"ecolor": "black", "elinewidth": 1.6, "capthick": 1.6},
           label="Measured (25 µM)")

    # Value labels above bars
    for xi, p, m, ms in zip(x, pred, meas, meas_sd):
        ax.text(xi - bw/2, p + 2.0, f"{p:.1f}",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#3c4858",
                family="Arial")
        meas_label = f"{m:.1f}" + (f"±{ms:.1f}" if ms > 0.01 else "")
        ax.text(xi + bw/2, m + ms + 2.0, meas_label,
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="black",
                family="Arial")

    # x labels
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[c] for c in df["compound"]],
                       fontsize=14, fontweight="bold", family="Arial")

    ax.set_ylabel("GRP78 inhibition at 25 µM (%)",
                  fontsize=18, fontweight="bold", family="Arial", labelpad=10)
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.7, n - 0.3)

    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(2.6)
        ax.spines[sp].set_color("#111111")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=15, width=2.2, length=7, pad=6)
    ax.tick_params(axis="x", labelsize=14, width=2.2, length=7, pad=6)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_family("Arial")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color="#9aa3ad", alpha=0.55)

    ax.legend(loc="upper right", fontsize=14, frameon=True, framealpha=0.95,
              prop={"family": "Arial", "size": 14, "weight": "bold"})

    fig.tight_layout()
    out_png = OUT_FIG / "Figure_predicted_vs_measured_bar.png"
    out_pdf = OUT_FIG / "Figure_predicted_vs_measured_bar.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
