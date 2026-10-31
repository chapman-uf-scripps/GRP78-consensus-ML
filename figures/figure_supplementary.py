"""
Supplementary Figures S2 to S5

Stages
  s2_s3  Figures S2 and S3: parity and absolute-error plots and sorted prediction curves of the three models (one file per panel in Output/figures/panels)
  s4     Figure S4: feature-class contributions of the three models under the historical IFP masking (needs analysis/shap_analysis.py class_shapley and analysis/tables.py)
  s5     Figure S5: top 10 OTAVA and top 10 FDA GSRS-derived compounds at selection (needs analysis/tables.py)

Run from the repository root:  python figures/figure_supplementary.py [stage ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error

from figures.style import MODELS, save, setup         # noqa: E402
from main_code import nested_cv as n                  # noqa: E402
from main_code.common import load_training            # noqa: E402

PAN = cfg.FIGURES / "panels"
SEL = cfg.INPUT / "original_selection"
GREY, SLATE, FS = "#6B7480", "#C9D1DB", "ligand_only"
GREEN, RED, GREY2 = "#2A9D58", "#C44E52", "#8E8E9A"


def run_s2_s3():
    setup(**{"axes.titleweight": "bold"})
    P = pd.read_csv(cfg.OUTPUT / "nested_cv_final_bo/outer_predictions_all.csv")
    summ = pd.read_csv(cfg.OUTPUT / "nested_cv_final_bo/metrics_summary_mean_sd.csv")
    act = load_training()
    mols = n.morgan_desc(act.SMILES)[2]
    scaf = pd.Series([n.scaffold(m) for m in mols], index=act.Name.values)
    dominant = set(scaf.index[scaf == scaf.value_counts().index[0]])

    def get(model):                                    # outer-fold predictions of repeat 0 with the dominant-group flag
        g = P[(P.model == model) & (P.feature_set == FS) & (P.repeat == 0)].copy()
        g["dom"] = g.Name.isin(dominant)
        return g

    def trend(ax, x, y, color, label):
        k, b = np.polyfit(x, y, 1)
        xs = np.array([0, 100])
        ax.plot(xs, k * xs + b, color=color, lw=2.2, label=label)

    for i, (m, lab, col) in enumerate(MODELS):
        g = get(m)
        oth, dom = g[~g.dom], g[g.dom]
        r = summ[(summ.model == m) & (summ.feature_set == FS)].iloc[0]
        # S2 A-C, parity
        fig, ax = plt.subplots(figsize=(6.2, 6.6))
        ax.scatter(dom.y, dom.pred, s=16, c=GREY, alpha=0.40, lw=0, label=f"Dominant scaffold group (n = {len(dom)})")
        ax.scatter(oth.y, oth.pred, s=18, c=col, alpha=0.75, lw=0, label=f"Other scaffold groups (n = {len(oth)})")
        ax.plot([0, 100], [0, 100], "k--", lw=1.1, label="Ideal")
        trend(ax, g.y, g.pred, col, "Linear fit (all)")
        ax.text(0.04, 0.965, f"R² = {r.R2_mean:.2f} ± {r.R2_std:.2f}\nMAE = {r.MAE_mean:.1f}", transform=ax.transAxes, va="top",
                fontsize=13, bbox=dict(boxstyle="round", fc="white", ec=SLATE))
        ax.set_xlabel("Observed GRP78 Inhibition (%)", fontsize=13)
        ax.set_ylabel("Predicted GRP78 Inhibition (%)", fontsize=13)
        ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.grid(alpha=0.25)
        save(fig, PAN, f"SI2_{'ABC'[i]}_{m}_parity")
        # S2 D-F, absolute error
        ae = (g.pred - g.y).abs()
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.scatter(dom.y, ae[g.dom], s=16, c=GREY, alpha=0.40, lw=0)
        ax.scatter(oth.y, ae[~g.dom], s=18, c=col, alpha=0.75, lw=0)
        trend(ax, oth.y, ae[~g.dom], col, f"Other groups, trend (MAE = {mean_absolute_error(oth.y, oth.pred):.1f})")
        trend(ax, dom.y, ae[g.dom], GREY, f"Dominant group, trend (MAE = {mean_absolute_error(dom.y, dom.pred):.1f})")
        ax.set_xlabel("Observed GRP78 Inhibition (%)", fontsize=13)
        ax.set_ylabel("Absolute Error", fontsize=13)
        ax.set_xlim(-3, 103); ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=9.5, frameon=True)
        save(fig, PAN, f"SI2_{'DEF'[i]}_{m}_absolute_error")
        # S3 A-C, sorted prediction curves
        g = g.sort_values("y").reset_index(drop=True)
        x = np.arange(len(g))
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.bar(x, g.y, width=1.0, color=SLATE, label="Observed")
        ax.scatter(x, g.pred, s=7, color=col, alpha=0.35, lw=0, label="Predicted (each compound)")
        ax.plot(x, g.pred.rolling(41, center=True, min_periods=10).mean(), color=col, lw=2.6, label="Predicted, rolling mean (41 compounds)")
        ax.set_xlabel("Compounds sorted by observed inhibition (n = 1,488)", fontsize=13)
        ax.set_ylabel("GRP78 Inhibition (%)", fontsize=13)
        ax.grid(alpha=0.25); ax.legend(loc="upper left", frameon=False, fontsize=10)
        save(fig, PAN, f"SI3_{'ABC'[i]}_{m}_prediction_curve")


def run_s4():
    setup()
    t = pd.read_csv(cfg.TABLES / "Supplementary_Table_class_shapley.csv")
    t = t[t["Input"].str.startswith("Active-filtered")]
    models, names = [m[1] for m in MODELS], ["Morgan fingerprint", "RDKit descriptor", "Docking descriptor", "IFP", "Molecular graph"]
    M = np.full((5, 3), np.nan)
    for i, name in enumerate(names):
        r = t[t["Feature class"] == name].iloc[0]
        for j, m in enumerate(models):
            if isinstance(r[m], str):
                M[i, j] = float(r[m].split("±")[0])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cmap = plt.get_cmap("YlGnBu").copy(); cmap.set_bad("#EEF0F3")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=80, aspect="auto")
    for i in range(5):
        for j in range(3):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=11, color="#8A9099")
            else:
                ax.text(j, i, f"{M[i, j]:.1f}%", ha="center", va="center", fontsize=13, color="white" if M[i, j] > 42 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(models, fontsize=12)
    ax.set_yticks(range(5)); ax.set_yticklabels(names, fontsize=12)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Share of total importance (%)", fontsize=12); cb.ax.tick_params(labelsize=11)
    save(fig, cfg.FIGURES, "SI_Figure4_masked_feature_class_summary")


def _barh(ax, labels, values, colors, xlim, title):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6, height=0.78)
    for yi, v in zip(y, values):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01, yi, f"{v:.1f}", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(*xlim); ax.set_xlabel("Consensus predicted inhibition (%)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=13, pad=8)


def run_s5():
    setup()
    outcome = pd.read_csv(cfg.TABLES / "tested_compounds_results.csv").set_index("Compound")["Class"].to_dict()
    ot = pd.read_csv(SEL / "Table_S3_otava_top50_triple_consensus.csv").head(10)
    fd = pd.read_csv(cfg.TABLES / "fda_top50_audit.csv").head(10)
    names = {166157330: "Olgotrelvir sodium", 73951493: "Desisobutyl-n-butyl bortezomib", 156907357: "CID 156907357 (indole-2-carboxamide)",
             59476170: "Bortezomib stereoisomer (CID 59476170)", 387447: "Bortezomib", 165411877: "CID 165411877 *", 166157331: "Olgotrelvir",
             71607434: "CID 71607434 *", 9868571: "CID 9868571", 91827873: "CID 91827873 *"}
    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.62)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _barh(axA, ot["Compound ID"].tolist(), ot["Consensus predicted GRP78 inhibition (mean)"].values,
          [GREEN if outcome.get(c) == "FP-active" else RED for c in ot["Compound ID"]], (50, 73), "Top 10 OTAVA compounds at selection")
    _barh(axB, [names.get(int(c), str(c)) for c in fd["PubChem CID"]], fd["Consensus predicted GRP78 inhibition (mean)"].values,
          [GREEN if int(c) == 73951493 else GREY2 for c in fd["PubChem CID"]], (60, 77), "Top 10 FDA GSRS-derived compounds")
    axA.legend(handles=[Patch(fc=GREEN, ec="black", label="FP-active"), Patch(fc=RED, ec="black", label="Inactive or weak")],
               frameon=False, fontsize=10, loc="lower right")
    axB.legend(handles=[Patch(fc=GREEN, ec="black", label="FP-active"), Patch(fc=GREY2, ec="black", label="Not tested")],
               frameon=False, fontsize=10, loc="lower right")
    for ax, L in ((axA, "A"), (axB, "B")):
        ax.text(-0.05, 1.12, L, transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="right")
    save(fig, cfg.FIGURES, "SI_Figure5_screening_of_external_libraries")


STAGES = {"s2_s3": run_s2_s3, "s4": run_s4, "s5": run_s5}

if __name__ == "__main__":
    for w in (sys.argv[1:] or list(STAGES)):
        print("=== stage", w, flush=True)
        STAGES[w]()
