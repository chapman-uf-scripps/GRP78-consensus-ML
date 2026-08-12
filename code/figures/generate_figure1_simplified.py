"""
Generate a simplified 3-panel main-paper Figure 1.

Goal: clean, high-level overview that opens the paper.
The detailed step-by-step workflow schematic stays as the TOC graphic.

Panels:
  A. Training data         — 1,488 compounds + FP at 25 uM → percent-inhibition table.
  B. Model + interpretation — feature blocks → top-3 → SHAP → residue pharmacophore.
  C. Transferability        — external libraries → consensus → FP + MD validation.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

OUT_DIR = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package\output\main_figures")

PANEL_W = 4.6
PANEL_H = 4.4
TOTAL_W = PANEL_W * 3 + 0.6
TOTAL_H = PANEL_H

fig, axes = plt.subplots(1, 3, figsize=(TOTAL_W, TOTAL_H))
for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

C_BG     = "#f5f6f8"
C_BOX    = "#ffffff"
C_BORDER = "#3b4252"
C_ACCENT = "#4c72b0"
C_HIGHLT = "#dd8452"
C_GREEN  = "#55a868"
C_TEXT   = "#1f2933"


def box(ax, x, y, w, h, label, color=C_BOX, fc=None, fontsize=9, fw="normal"):
    fc = fc or color
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                       linewidth=1.2, edgecolor=C_BORDER, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, fontweight=fw, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=C_BORDER):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        color=color, linewidth=1.4)
    ax.add_patch(a)


# ---- Panel A: Training data ----
ax = axes[0]
ax.set_facecolor(C_BG)
ax.text(5, 9.4, "A.  Training data", ha="center", va="center",
        fontsize=12, fontweight="bold", color=C_TEXT)

box(ax, 1.4, 6.5, 7.2, 1.6, "1,488-compound focused library",
    fc="#e8eef7", fw="bold")
box(ax, 1.4, 4.6, 7.2, 1.5, "Fluorescence-polarization\nsubstrate-displacement assay\nvs. GRP78 SBD,  25 µM single-shot")
box(ax, 1.4, 2.5, 7.2, 1.6, "% GRP78 inhibition,  0 – 100 scale\n(active cutoff = 38, Fig. S1)",
    fc="#fcecdc")
arrow(ax, 5, 6.5, 5, 6.15)
arrow(ax, 5, 4.6, 5, 4.15)

ax.text(5, 1.4, "N = 1,488   uniform readout",
        ha="center", va="center", fontsize=9, style="italic", color=C_TEXT)


# ---- Panel B: Model + interpretation ----
ax = axes[1]
ax.set_facecolor(C_BG)
ax.text(5, 9.4, "B.  Model + interpretation", ha="center", va="center",
        fontsize=12, fontweight="bold", color=C_TEXT)

# Feature blocks (left column)
box(ax, 0.3, 7.6, 3.5, 0.9, "Morgan FP",  fc="#e3eaf3")
box(ax, 0.3, 6.5, 3.5, 0.9, "RDKit",      fc="#e3eaf3")
box(ax, 0.3, 5.4, 3.5, 0.9, "Glide docking", fc="#e3eaf3")
box(ax, 0.3, 4.3, 3.5, 0.9, "Active-filtered IFP", fc="#fcecdc", fw="bold")

# Top-3 models
box(ax, 4.6, 6.7, 5.1, 1.0, "Extra Trees + CatBoost + GNN\n(orthogonal-bias consensus)",
    fc="#e8eef7", fw="bold")
arrow(ax, 3.8, 6.95, 4.6, 7.0)
arrow(ax, 3.8, 5.85, 4.6, 7.0)
arrow(ax, 3.8, 4.75, 4.6, 7.0)

# Interpretation
box(ax, 4.6, 4.9, 5.1, 0.9, "TreeSHAP   +   permutation importance",
    fc="#eef3ec")
arrow(ax, 7.15, 6.7, 7.15, 5.8)

# Pharmacophore output
box(ax, 1.4, 1.9, 7.2, 1.9,
    "Residue-resolved pharmacophore\n"
    "H-bond acceptor at A452  +  contacts at A428 / A429",
    fc="#fcecdc", fw="bold", fontsize=10)
arrow(ax, 7.15, 4.9, 5, 3.8)


# ---- Panel C: Transferability ----
ax = axes[2]
ax.set_facecolor(C_BG)
ax.text(5, 9.4, "C.  Transferability + validation", ha="center", va="center",
        fontsize=12, fontweight="bold", color=C_TEXT)

box(ax, 0.4, 7.4, 4.4, 1.3, "OTAVA library\n(≈ 3.6 k annotated)",
    fc="#e8eef7", fw="bold")
box(ax, 5.2, 7.4, 4.4, 1.3, "Approved-drug-derived\n(124,195 CIDs)",
    fc="#e8eef7", fw="bold")

box(ax, 1.4, 5.3, 7.2, 1.4,
    "Triple-consensus filter  (predicted inhibition ≥ 38)",
    fc="#eef3ec")
arrow(ax, 2.6, 7.4, 4.0, 6.7)
arrow(ax, 7.4, 7.4, 6.0, 6.7)

box(ax, 0.4, 3.0, 4.4, 1.8,
    "FP dose-response\nIC₅₀ values\n4.0 / 4.7 / 7.5 / 12.8 / 12.9 µM",
    fc="#fcecdc", fw="bold")
box(ax, 5.2, 3.0, 4.4, 1.8,
    "100 ns × 3 MD\nA452 / A428 / A429\noccupancy ≥ 50 %",
    fc="#fcecdc", fw="bold")
arrow(ax, 3.5, 5.3, 2.6, 4.8)
arrow(ax, 6.5, 5.3, 7.4, 4.8)

box(ax, 1.4, 0.8, 7.2, 1.4,
    "Pharmacophore = data-anchored + method-robust + MD-confirmed",
    fc="#e3eaf3", fontsize=9.5, fw="bold")
arrow(ax, 2.6, 3.0, 4.0, 2.2)
arrow(ax, 7.4, 3.0, 6.0, 2.2)


fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, wspace=0.05)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_png = OUT_DIR / "Figure_1_simplified_workflow.png"
out_pdf = OUT_DIR / "Figure_1_simplified_workflow.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print("Wrote", out_png)
print("Wrote", out_pdf)
