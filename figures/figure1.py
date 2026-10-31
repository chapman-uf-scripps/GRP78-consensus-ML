"""
Figure 1: workflow of the corrected evaluation (three bands)

Stage: figure1.

Run from the repository root:  python figures/figure1.py [stage ...]   (no stage = all stages in order)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def run_figure1():
    """Figure 1 (corrected workflow): three bands."""
    matplotlib.use("Agg")

    OUT = cfg.FIGURES
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

    BLUE, BLUE_L = "#1F5A99", "#E6EFF9"
    GREY, GREY_L = "#6B6B6B", "#F2F2F2"
    ORG, ORG_L = "#B8601A", "#FBEEDD"
    TXT = "#1A1A1A"


    def box(ax, x, y, w, h, text, fc, ec, ls="-", fs=10.2, lw=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=TXT, linespacing=1.28)


    def arrow(ax, x0, y0, x1, y1, color):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                     lw=1.6, color=color, shrinkA=0, shrinkB=0))


    def band(ax, y, h, color, light, label, sub, ls="-"):
        ax.add_patch(FancyBboxPatch((0.15, y), 15.7, h, boxstyle="round,pad=0.0,rounding_size=0.18",
                                    fc=light, ec=color, lw=1.8, ls=ls, alpha=0.55))
        ax.text(0.45, y + h - 0.32, label, fontsize=13.5, fontweight="bold", color=color, va="center")
        ax.text(0.95, y + h - 0.32, sub, fontsize=10.5, color=color, va="center", style="italic")


    fig, ax = plt.subplots(figsize=(16, 9.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    # ------------------------------------------------------------------ A
    band(ax, 5.05, 4.25, BLUE, BLUE_L, "A", "Corrected evaluation: primary basis of the computational conclusions")
    yA, hA = 6.55, 1.95
    box(ax, 0.45, yA, 2.0, hA, "1,488 compounds\nFP inhibition at 25 µM\n(D2B focused library)", "white", BLUE)
    box(ax, 2.85, yA, 3.35, hA,
        "Features, identical for every\ncompound; no activity input\nMorgan (1,024 bits)  ·  10 RDKit\n31 Glide terms  ·  IFP (top pose)",
        "white", BLUE)
    box(ax, 6.6, yA, 2.75, hA, "Outer loop (evaluation only)\nscaffold-grouped 5-fold\n× 5 seeds (0–4)\n176 scaffold groups", "white", BLUE)
    box(ax, 9.75, yA, 3.0, hA,
        "Inside each outer-training set\n3 inner scaffold-grouped folds:\nfold-local preprocessing,\nfeature filtering,\nBayesian optimization", "white", BLUE, ls="--")
    box(ax, 13.15, yA, 2.45, hA, "Extra Trees · CatBoost\nGNN\nrefit, predict the\nheld-out outer fold", "white", BLUE)
    for x0, x1 in [(2.45, 2.85), (6.2, 6.6), (9.35, 9.75), (12.75, 13.15)]:
        arrow(ax, x0, yA + hA / 2, x1, yA + hA / 2, BLUE)

    box(ax, 0.45, 5.25, 9.1, 0.95,
        "Compared on identical outer folds: mean consensus · ligand-only · docking-only · IFP-only · nearest-active-neighbor\n"
        "Metrics from outer-fold predictions: R², MAE, RMSE, AUROC, precision@k (hit = measured inhibition ≥ 38%)",
        BLUE_L, BLUE, fs=9.8)
    box(ax, 9.95, 5.25, 5.65, 0.95,
        "Diagnostics (Fig. 2): historical IFP masking reproduced in the\nsame cross-validation; deployment on unmasked compounds",
        "white", GREY, ls=":", fs=9.3, lw=1.4)
    arrow(ax, 4.5, yA, 4.5, 6.23, BLUE)

    # ------------------------------------------------------------------ B
    band(ax, 2.55, 2.3, GREY, GREY_L, "B",
         "Historical prospective selection: original pipeline (contained an activity-dependent IFP step)", ls="--")
    yB, hB = 2.75, 1.45
    box(ax, 0.45, yB, 3.2, hB, "17-algorithm survey\n(single scaffold test set);\nExtra Trees, CatBoost, GNN\ncarried forward", "white", GREY, ls="--")
    box(ax, 4.05, yB, 3.6, hB, "Two external libraries ranked\nOTAVA and an\napproved-drug-derived\ncollection", "white", GREY, ls="--")
    box(ax, 8.05, yB, 3.3, hB, "Tested by FP:\ntop 10 OTAVA compounds\n+ 1 approved-drug-derived\ncompound", "white", GREY, ls="--")
    box(ax, 11.75, yB, 3.85, hB, "FP dose–response:\n5 FP-active compounds\n(4 of 10 OTAVA; 1 of 1)\nIC₅₀ 4.3–16.1 µM", "white", GREY)
    for x0, x1 in [(3.65, 4.05), (7.65, 8.05), (11.35, 11.75)]:
        arrow(ax, x0, yB + hB / 2, x1, yB + hB / 2, GREY)

    # ------------------------------------------------------------------ C
    band(ax, 0.1, 2.25, ORG, ORG_L, "C", "Retrospective analyses of the known hits (hits known in advance)")
    yC, hC = 0.3, 1.4
    box(ax, 0.45, yC, 7.2, hC,
        "Re-rank with the corrected ligand-only models (A)\n1,909 OTAVA and 124,195 approved-drug-derived compounds\nsame pool for every scorer; compare with nearest-active-neighbor",
        "white", ORG)
    box(ax, 8.05, yC, 7.55, hC,
        "Docking poses of the five FP-active compounds (B) → MD from docked poses\npose persistence and predicted contacts (S452, T428/V429 region)\nhypotheses only; not binding assays",
        "white", ORG)

    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Figure1_corrected_workflow.{ext}", dpi=300, bbox_inches="tight")
    print("saved", OUT / "Figure1_corrected_workflow.png")


STAGES = {
    "figure1": run_figure1,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    for w in wanted:
        print("=== stage", w, flush=True)
        STAGES[w]()
