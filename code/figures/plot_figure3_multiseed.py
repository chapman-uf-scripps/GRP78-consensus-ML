"""
Figure 3 with error bars from the multi-seed ablation.

Reads output/source_tables/figure3_feature_ablation_multiseed.csv (mean + std)
and plots grouped bars with std as vertical error bars.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
METRICS = ROOT / "output" / "source_tables" / "figure3_feature_ablation_multiseed.csv"
OUT     = ROOT / "output" / "main_figures"

df = pd.read_csv(METRICS)

LEVELS      = ["L1", "L2", "L3", "L4"]
TICK_LABELS = ["Morgan\nonly", "+ RDKit", "+ IFP", "+ docking\n(full)"]
MODELS      = ["ExtraTrees", "CatBoost", "GNN"]
COLORS      = {"ExtraTrees": "#4c72b0", "CatBoost": "#dd8452", "GNN": "#55a868"}


def get(lv, m, col):
    sub = df[(df["Level"] == lv) & (df["Model"] == m)]
    return float(sub[col].iloc[0]) if not sub.empty else np.nan


fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
width = 0.27
x = np.arange(len(LEVELS))

for ax in axes:
    ax.axvspan(x[2] - 0.5, x[2] + 0.5, color="#fef6e4", alpha=0.55, zorder=0)

# ---------- Left: Test R² ----------
ax = axes[0]
for i, m in enumerate(MODELS):
    means = np.array([get(lv, m, "Test_R2_mean") for lv in LEVELS])
    stds  = np.array([get(lv, m, "Test_R2_std")  for lv in LEVELS])
    xs = x + (i - 1) * width
    ax.bar(xs, means, width, label=m, color=COLORS[m],
           edgecolor="black", linewidth=0.7, zorder=3,
           yerr=stds, capsize=3,
           error_kw=dict(elinewidth=1.0, ecolor="black", zorder=4))
    for xi, mn, sd in zip(xs, means, stds):
        if np.isnan(mn):
            continue
        top = mn + (sd if not np.isnan(sd) else 0)
        yt = top + 0.03 if mn >= 0 else mn - 0.03
        va = "bottom" if mn >= 0 else "top"
        ax.text(xi, yt, f"{mn:.3f}",
                ha="center", va=va, fontsize=8.5, zorder=5)

ax.axhline(0, color="black", linewidth=0.7, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(TICK_LABELS, fontsize=10)
ax.set_ylabel("Scaffold-split Test R²", fontsize=11)
ax.set_title("Test R²", fontsize=12, pad=10)
ax.set_ylim(-0.30, 1.15)
ax.grid(axis="y", alpha=0.25, zorder=1)
ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)

arrow = FancyArrowPatch((x[1], 1.03), (x[2], 1.03),
                         arrowstyle="-|>", mutation_scale=16,
                         color="#b75900", linewidth=1.6, zorder=6)
ax.add_patch(arrow)
ax.text((x[1] + x[2]) / 2, 1.07,
        "IFP block: dominant jump",
        ha="center", va="bottom", fontsize=9.5, color="#b75900",
        fontweight="bold", zorder=6)

# ---------- Right: Test MAE ----------
ax = axes[1]
for i, m in enumerate(MODELS):
    means = np.array([get(lv, m, "Test_MAE_mean") for lv in LEVELS])
    stds  = np.array([get(lv, m, "Test_MAE_std")  for lv in LEVELS])
    xs = x + (i - 1) * width
    ax.bar(xs, means, width, label=m, color=COLORS[m],
           edgecolor="black", linewidth=0.7, zorder=3,
           yerr=stds, capsize=3,
           error_kw=dict(elinewidth=1.0, ecolor="black", zorder=4))
    for xi, mn, sd in zip(xs, means, stds):
        if np.isnan(mn):
            continue
        top = mn + (sd if not np.isnan(sd) else 0)
        ax.text(xi, top + 0.3, f"{mn:.2f}",
                ha="center", va="bottom", fontsize=8.5, zorder=5)

ax.set_xticks(x); ax.set_xticklabels(TICK_LABELS, fontsize=10)
ax.set_ylabel("Scaffold-split Test MAE  (percent inhibition)", fontsize=11)
ax.set_title("Test MAE", fontsize=12, pad=10)
top_mae = df["Test_MAE_mean"].max() + df["Test_MAE_std"].max() + 2
ax.set_ylim(0, top_mae)
ax.grid(axis="y", alpha=0.25, zorder=1)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

fig.suptitle(
    "Figure 3.  Feature-class ablation under scaffold split (mean +/- std across seeds)",
    fontsize=12.5, y=1.02, fontweight="bold"
)
fig.tight_layout()

png = OUT / "Figure_3_feature_ablation.png"
pdf = OUT / "Figure_3_feature_ablation.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print("Wrote", png)
print("Wrote", pdf)
