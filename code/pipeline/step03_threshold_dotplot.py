"""
01c_plot_threshold_selection.py
================================
Dot plot showing A5 activity distribution across all 1,488 training
compounds with the active IFP threshold at 38 annotated.

Output: Output/01b_threshold_sweep/threshold_selection_dotplot.png/.pdf
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_DIR  = Path(__file__).parent
ACTIVITY_PATH = PIPELINE_DIR / "Input" / "training_activity_merged_result.csv"
OUTPUT_DIR    = PIPELINE_DIR / "Output" / "01b_threshold_sweep"
THRESHOLD     = 38.0


def set_plot_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.weight": "bold", "font.size": 16,
        "axes.titlesize": 20, "axes.titleweight": "bold",
        "axes.labelsize": 17, "axes.labelweight": "bold",
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    df = pd.read_csv(ACTIVITY_PATH)
    df["A5"] = pd.to_numeric(df["A5"], errors="coerce")
    df = df.dropna(subset=["A5"]).copy()
    df["A5"] = np.clip(df["A5"], 0.0, 100.0)
    df = df.sort_values("A5").reset_index(drop=True)

    active   = df[df["A5"] >= THRESHOLD]
    inactive = df[df["A5"] <  THRESHOLD]

    n_active   = len(active)
    n_inactive = len(inactive)
    n_total    = len(df)
    pct_active = n_active / n_total * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    # ── Left panel: dot plot sorted by A5 ────────────────────────────────────
    ax = axes[0]
    x_inactive = np.arange(len(inactive))
    x_active   = np.arange(len(inactive), n_total)

    ax.scatter(x_inactive, inactive["A5"].values, s=14, alpha=0.55,
               color="#7A8793", label=f"Inactive A5 < {THRESHOLD:.0f}  (n={n_inactive})", zorder=2)
    ax.scatter(x_active, active["A5"].values, s=14, alpha=0.72,
               color="#159A9C", label=f"Active A5 >= {THRESHOLD:.0f}  (n={n_active}, {pct_active:.0f}%)", zorder=2)

    ax.axhline(THRESHOLD, color="#D95E55", linewidth=2.2, linestyle="--", zorder=3)
    ax.text(n_total * 0.02, THRESHOLD + 2.5,
            f"Threshold = {THRESHOLD:.0f}",
            color="#D95E55", fontsize=14, fontweight="bold")

    ax.fill_between([0, n_total - 1], THRESHOLD, 100,
                    color="#159A9C", alpha=0.06, zorder=1)
    ax.fill_between([0, n_total - 1], 0, THRESHOLD,
                    color="#7A8793", alpha=0.06, zorder=1)

    ax.set_xlabel("Compounds sorted by GRP78 Inhibition")
    ax.set_ylabel("GRP78 Inhibition (%)")
    ax.set_title("Training Compound Activity Distribution\nwith Active IFP Threshold")
    ax.set_xlim(-10, n_total + 10)
    ax.set_ylim(-2, 104)
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    ax.grid(alpha=0.18)

    # ── Right panel: histogram ────────────────────────────────────────────────
    ax2 = axes[1]
    bins = np.linspace(0, 100, 26)
    ax2.hist(inactive["A5"], bins=bins, color="#7A8793", alpha=0.75,
             label=f"Inactive (n={n_inactive})", orientation="horizontal")
    ax2.hist(active["A5"],   bins=bins, color="#159A9C", alpha=0.75,
             label=f"Active (n={n_active})",   orientation="horizontal")
    ax2.axhline(THRESHOLD, color="#D95E55", linewidth=2.2, linestyle="--")
    ax2.set_xlabel("Count")
    ax2.set_ylabel("GRP78 Inhibition (%)")
    ax2.set_title("Distribution")
    ax2.set_ylim(-2, 104)
    ax2.legend(frameon=False, fontsize=12)
    ax2.grid(alpha=0.18)

    # ── Annotation box with key stats ─────────────────────────────────────────
    stats_text = (
        f"Total compounds: {n_total}\n"
        f"Active (>={THRESHOLD:.0f}): {n_active} ({pct_active:.0f}%)\n"
        f"Inactive (<{THRESHOLD:.0f}): {n_inactive} ({100-pct_active:.0f}%)\n"
        f"Sweep best threshold: 40 (dR2=+0.0002)"
    )
    fig.text(0.5, -0.04, stats_text, ha="center", va="top",
             fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#D6DEE6", alpha=0.95))

    fig.tight_layout()
    out = OUTPUT_DIR / "threshold_selection_dotplot.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
