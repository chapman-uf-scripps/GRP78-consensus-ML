"""
Figure 3 panels

Stages: class_summary (feature-class table and heat map data), panels (the five panels A-E).

Run from the repository root:  python figures/figure3.py [stage ...]   (no stage = all stages in order)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402
from main_code.common import load_training                    # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import fisher_exact
from main_code import nested_cv as n                      # noqa: E402


def run_class_summary():
    """Figure 3: feature-class summary (class-level Shapley importance, % of the total) for Extra Trees, CatBoost and the GNN"""
    matplotlib.use("Agg")

    OUT = cfg.FIGURES
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold"})

    d = pd.read_csv(cfg.OUTPUT / "shap_unmasked/class_shapley_long.csv")
    players = ["morgan", "desc", "dock", "ifp", "graph"]
    names = ["Morgan fingerprint", "RDKit descriptor", "Docking descriptor", "IFP", "Molecular graph"]
    models = [("ExtraTrees", "Extra Trees"), ("CatBoost", "CatBoost"), ("GNN", "GNN")]


    def matrix(setting):
        M = np.full((len(players), len(models)), np.nan)
        S = np.full_like(M, np.nan)
        for j, (m, _) in enumerate(models):
            for i, p in enumerate(players):
                v = d[(d.setting == setting) & (d.model == m) & (d.player == p)].share
                if len(v):
                    M[i, j], S[i, j] = v.mean(), v.std(ddof=1)
        return M, S


    def draw(setting, stem):
        M, S = matrix(setting)
        fig = plt.figure(figsize=(6.4, 5.0))
        gs = GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
        ax, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#EEF0F3")
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=80, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=11, color="#8A9099")
                else:
                    ax.text(j, i, f"{M[i, j]:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold",
                            color="white" if M[i, j] > 42 else "black")
        ax.set_xticks(range(3))
        ax.set_xticklabels([m[1] for m in models], fontsize=12)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(players)))
        ax.set_yticklabels(names, fontsize=12)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Share of total importance (%)", fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return M, S


    M, S = draw("unmasked", "Figure3_feature_class_summary")
    Ma, Sa = draw("active_filtered", "SuppFigure_feature_class_summary_active_filtered")

    rows = []
    for setting, (A, B) in [("Unmasked", (M, S)), ("Active-filtered IFP (historical masking, diagnostic)", (Ma, Sa))]:
        for i, nme in enumerate(names):
            r = {"Input": setting, "Feature class": nme}
            for j, (_, lab) in enumerate(models):
                r[lab] = "n/a" if np.isnan(A[i, j]) else f"{A[i, j]:.1f} ± {B[i, j]:.1f}"
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(cfg.TABLES / "Supplementary_Table_class_shapley.csv", index=False)
    print(T.to_string(index=False))
    print("saved Figure 3 and the supplementary version")


def run_panels():
    """Final Figure 3: the five panels exported individually (PNG + PDF; no titles, no panel letters)."""
    matplotlib.use("Agg")


    PAN = cfg.FIGURES / "panels" / "figure3_final"
    PAN.mkdir(parents=True, exist_ok=True)
    SH = cfg.OUTPUT / "shap_unmasked"
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold"})
    TEAL, GREY = "#1E9AA0", "#6B7480"

    THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
             "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    resname = {}
    for line in open(cfg.RECEPTOR_PDB):
        if line.startswith("ATOM") and line[21] == "A":
            resname[int(line[22:26])] = THREE.get(line[17:20].strip(), "X")
    DESC = {"desc_logp": "MolLogP", "desc_rotb": "Rotatable bonds", "desc_fsp3": "Fraction sp3 carbons", "desc_tpsa": "TPSA",
            "desc_rings": "Ring count", "desc_mw": "Molecular weight", "desc_hbd": "H-bond donors", "desc_hba": "H-bond acceptors",
            "desc_heavy": "Heavy-atom count", "desc_arom": "Aromatic rings"}


    def label(f):
        if f in DESC:
            return DESC[f]
        if f.startswith("morgan_"):
            return f"Morgan bit {int(f.split('_')[1])}"
        if f.startswith("dock_r_i_glide_"):
            return "Glide " + f.replace("dock_r_i_glide_", "").replace("_", " ")
        if f.startswith("dock_r_epik_"):
            return "Epik " + f.replace("dock_r_epik_", "").replace("_", " ")
        if f.startswith("dock_"):
            return f.replace("dock_r_", "").replace("dock_", "").replace("_", " ")
        if f.startswith("ifp_total_"):
            return f"IFP total {f.replace('ifp_total_', '').replace('_residue_any', '')} residues"
        if f.startswith("ifp_A"):
            parts = f.split("_")
            return f"IFP {resname.get(int(parts[1][1:]), 'X')}{int(parts[1][1:])} {parts[2]}"
        return f


    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(PAN / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


    # ---------------------------------------------------------------- A
    d = pd.read_csv(SH / "class_shapley_long.csv")
    d = d[d.setting == "unmasked"]
    players = ["morgan", "desc", "dock", "ifp", "graph"]
    pnames = ["Morgan fingerprint", "RDKit descriptor", "Docking descriptor", "IFP", "Molecular graph"]
    models = [("ExtraTrees", "Extra Trees"), ("CatBoost", "CatBoost"), ("GNN", "GNN")]
    M = np.full((5, 3), np.nan)
    for j, (m, _) in enumerate(models):
        for i, p in enumerate(players):
            v = d[(d.model == m) & (d.player == p)].share
            if len(v):
                M[i, j] = v.mean()
    fig = plt.figure(figsize=(6.4, 5.0))
    gs = GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
    ax, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#EEF0F3")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=80, aspect="auto")
    for i in range(5):
        for j in range(3):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=11, color="#8A9099")
            else:
                ax.text(j, i, f"{M[i, j]:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold", color="white" if M[i, j] > 42 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in models], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks(range(5)); ax.set_yticklabels(pnames, fontsize=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, cax=cax).set_label("Share of total importance (%)", fontsize=11)
    save(fig, "Fig3_A_feature_class_summary")

    # ---------------------------------------------------------------- B
    act = load_training()
    isact = act.A5.values >= 38
    ifp, _ = n.best_pose_features(cfg.IFP_TABLE, act.Name)
    ITEMS = [("S452 contact", "ifp_A452_contact_any"), ("T428 contact", "ifp_A428_contact_any"), ("V429 contact", "ifp_A429_contact_any"),
             ("I426 contact", "ifp_A426_contact_any"), ("F451 contact", "ifp_A451_contact_any"),
             ("S452 acceptor", "ifp_A452_acceptor_any"), ("V429 donor", "ifp_A429_donor_any")]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    xs = np.arange(len(ITEMS))
    fa, fi, ors = [], [], []
    for _, col in ITEMS:
        v = ifp[col].values.astype(float) > 0
        fa.append(100 * v[isact].mean()); fi.append(100 * v[~isact].mean())
        tab = [[int((v & isact).sum()), int((~v & isact).sum())], [int((v & ~isact).sum()), int((~v & ~isact).sum())]]
        ors.append(fisher_exact(tab)[0])
    ax.bar(xs - 0.2, fa, width=0.38, color=TEAL, label=f"Actives (n = {int(isact.sum())})", zorder=2)
    ax.bar(xs + 0.2, fi, width=0.38, color=GREY, alpha=0.75, label=f"Inactives (n = {int((~isact).sum())})", zorder=2)
    for i, (a, b, o) in enumerate(zip(fa, fi, ors)):
        if np.isfinite(o) and abs(a - b) > 4:
            ax.text(i, max(a, b) + 3, f"OR {o:.1f}", ha="center", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([i[0].replace(" ", chr(10)) for i in ITEMS], fontsize=10.5)
    ax.set_ylabel("Compounds with the interaction (%)", fontsize=12)
    ax.set_ylim(0, 118)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    save(fig, "Fig3_B_residue_contact_frequency")


    # ---------------------------------------------------------------- C-E
    def beeswarm(name, out, max_display=15):
        sv = pd.read_csv(SH / f"{name}_shap.csv", index_col=0)
        fv = pd.read_csv(SH / f"{name}_values.csv", index_col=0).reindex(index=sv.index, columns=sv.columns)
        top = sv.abs().mean().sort_values(ascending=False).head(max_display).index
        expl = shap.Explanation(values=sv[top].values, data=fv[top].fillna(0).values, feature_names=[label(c) for c in top])
        fig, ax = plt.subplots(figsize=(5.4, 6.6))
        shap.plots.beeswarm(expl, max_display=max_display, ax=ax, show=False, color_bar=False, plot_size=None)
        ax.set_xlabel("SHAP value (%)", fontsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="x", labelsize=14)
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=shap.plots.colors.red_blue), ax=ax, fraction=0.035, pad=0.02)
        cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=15)
        cb.ax.tick_params(labelsize=14)
        save(fig, out)


    beeswarm("catboost_full", "Fig3_C_SHAP_CatBoost")
    beeswarm("extratrees_full", "Fig3_D_SHAP_ExtraTrees")
    beeswarm("gnn_full", "Fig3_E_SHAP_GNN")
    print("saved", [p.name for p in sorted(PAN.glob("*.png"))])


STAGES = {
    "class_summary": run_class_summary,
    "panels": run_panels,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    for w in wanted:
        print("=== stage", w, flush=True)
        STAGES[w]()
