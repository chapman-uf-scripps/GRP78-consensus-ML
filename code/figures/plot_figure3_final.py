"""
Final Figure 3 — feature-class ablation, plotted from the saved metrics CSV.

Uses descriptive feature-class labels directly (no L1/L2/L3/L4 codes).
Highlights the "+ IFP" step as the load-bearing jump on the ladder.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
METRICS = ROOT / "output" / "source_tables" / "figure3_feature_ablation_metrics.csv"
OUT     = ROOT / "output" / "main_figures"

df = pd.read_csv(METRICS)

LEVELS    = ["L1", "L2", "L3", "L4"]
TICK_LABELS = ["Morgan\nonly", "+ RDKit", "+ IFP", "+ docking\n(full)"]
MODELS    = ["ExtraTrees", "CatBoost", "GNN"]
COLORS    = {"ExtraTrees": "#4c72b0", "CatBoost": "#dd8452", "GNN": "#55a868"}

def get_val(lv, m, col):
    sub = df[(df["Level"]==lv)&(df["Model"]==m)]
    return float(sub[col].iloc[0]) if not sub.empty else np.nan

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
width = 0.27
x = np.arange(len(LEVELS))

# Shade the "+ IFP" column to call out the load-bearing step
for ax in axes:
    ax.axvspan(x[2] - 0.5, x[2] + 0.5, color="#fef6e4", alpha=0.55, zorder=0)

# ------------------- Left panel: Test R² -------------------
ax = axes[0]
for i, m in enumerate(MODELS):
    vals = [get_val(lv, m, "Test_R2") for lv in LEVELS]
    bars = ax.bar(x + (i-1)*width, vals, width, label=m, color=COLORS[m],
                  edgecolor="black", linewidth=0.7, zorder=3)
    for b, v in zip(bars, vals):
        if np.isnan(v): continue
        # Annotate above for positive, below for negative
        yt = v + 0.02 if v >= 0 else v - 0.02
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width()/2, yt, f"{v:.3f}",
                ha="center", va=va, fontsize=8.5, zorder=4)

ax.axhline(0, color="black", linewidth=0.7, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(TICK_LABELS, fontsize=10)
ax.set_ylabel("Scaffold-split Test R²", fontsize=11)
ax.set_title("Test R²", fontsize=12, pad=10)
ax.set_ylim(-0.25, 1.15)
ax.grid(axis="y", alpha=0.25, zorder=1)
ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)

# Annotation arrow above the bars showing the L2 -> L3 (IFP) jump
arrow = FancyArrowPatch((x[1], 1.03), (x[2], 1.03),
                         arrowstyle="-|>", mutation_scale=16,
                         color="#b75900", linewidth=1.6, zorder=5)
ax.add_patch(arrow)
ax.text((x[1]+x[2])/2, 1.07,
        "IFP block: dominant jump",
        ha="center", va="bottom", fontsize=9.5, color="#b75900",
        fontweight="bold", zorder=5)

# ------------------- Right panel: Test MAE -------------------
ax = axes[1]
for i, m in enumerate(MODELS):
    vals = [get_val(lv, m, "Test_MAE") for lv in LEVELS]
    bars = ax.bar(x + (i-1)*width, vals, width, label=m, color=COLORS[m],
                  edgecolor="black", linewidth=0.7, zorder=3)
    for b, v in zip(bars, vals):
        if np.isnan(v): continue
        ax.text(b.get_x() + b.get_width()/2, v + 0.2, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, zorder=4)

ax.set_xticks(x); ax.set_xticklabels(TICK_LABELS, fontsize=10)
ax.set_ylabel("Scaffold-split Test MAE  (percent inhibition)", fontsize=11)
ax.set_title("Test MAE", fontsize=12, pad=10)
ax.set_ylim(0, max(df["Test_MAE"]) + 3)
ax.grid(axis="y", alpha=0.25, zorder=1)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

# Subtitle / context line
fig.suptitle(
    "Figure 3.  Feature-class ablation under scaffold split — IFP is the equalizing feature",
    fontsize=12.5, y=1.02, fontweight="bold"
)
fig.tight_layout()

png = OUT / "Figure_3_feature_ablation.png"
pdf = OUT / "Figure_3_feature_ablation.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print("Wrote", png)
print("Wrote", pdf)
