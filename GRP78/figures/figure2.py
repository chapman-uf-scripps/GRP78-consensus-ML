"""
Figure 2 panels

Stages: panels_ab, panels_cd, panels_de.

Run from the repository root:  python figures/figure2.py [stage ...]   (no stage = all stages in order)
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
import umap
from rdkit import DataStructs
from rdkit.Chem import AllChem
from main_code import nested_cv as n                      # noqa: E402


def run_panels_ab():
    """Figure 2, panels A and B (one PNG/PDF each, no titles, no panel letters)."""
    matplotlib.use("Agg")


    OUT = cfg.FIGURES / "panels"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})

    act = load_training()
    _, _, mols = n.morgan_desc(act.SMILES)
    scaf = np.array([n.scaffold(m) for m in mols])
    X = np.zeros((len(mols), 1024), dtype=bool)
    for i, m in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        arr = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr.astype(bool)
    xy = umap.UMAP(n_neighbors=35, min_dist=0.12, metric="jaccard", random_state=42, low_memory=True).fit_transform(X)

    top = pd.Series(scaf).value_counts().head(10)
    label = {s: f"S{i + 1}" for i, s in enumerate(top.index)}
    coords = pd.DataFrame({"Name": act.Name, "UMAP1": xy[:, 0], "UMAP2": xy[:, 1], "A5": act.A5,
                           "scaffold_group": [label.get(s, "other") for s in scaf]})
    coords.to_csv(cfg.TABLES / "umap_coordinates_training_library.csv", index=False)


    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


    # ---------------------------------------------------------------- A : measured inhibition
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    order = np.argsort(act.A5.values)                      # draw strongest inhibition on top
    sc = ax.scatter(xy[order, 0], xy[order, 1], c=np.clip(act.A5.values[order], 0, 100), cmap="viridis", vmin=0, vmax=100,
                    s=15, alpha=0.88, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Measured GRP78 inhibition (%)", fontsize=12)
    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.grid(alpha=0.2)
    save(fig, "Fig2_A_UMAP_inhibition")

    # ---------------------------------------------------------------- B : ten largest scaffold groups
    palette = ["#6B7480", "#E4572E", "#17BEBB", "#FFC914", "#7B5FB5", "#2E86AB", "#76B041", "#C0397B", "#8C5E3C", "#F18F01"]
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    other = ~np.isin(scaf, list(top.index))
    ax.scatter(xy[other, 0], xy[other, 1], s=13, c="#D5DAE0", alpha=0.75, linewidths=0, zorder=1)
    for i, s in enumerate(top.index):
        m = scaf == s
        ax.scatter(xy[m, 0], xy[m, 1], s=15 if i else 13, c=palette[i], alpha=0.6 if i == 0 else 0.95, linewidths=0, zorder=2 + (i > 0),
                   label=f"S{i + 1} (n = {int(top[s])})")
    ax.scatter([], [], s=30, c="#D5DAE0", label=f"Other groups (n = {int(other.sum())})")
    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9.5, markerscale=1.2, handletextpad=0.3,
              columnspacing=1.0)
    save(fig, "Fig2_B_UMAP_scaffold_groups")
    print("saved UMAP panels; top-10 scaffold group sizes:", top.tolist(), "| other:", int(other.sum()))


def run_panels_cd():
    """Figure 2, new panels C and D (one PNG/PDF each, no titles, no panel letters)."""
    matplotlib.use("Agg")


    OUT = cfg.FIGURES / "panels"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})
    TEAL, GREY = "#1E9AA0", "#6B7480"


    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)



    # ------------------------------------------------------------------ D : nearest-neighbour similarity
    act = load_training()
    _, _, mols = n.morgan_desc(act.SMILES)
    scaf = np.array([n.scaffold(m) for m in mols])
    dom = scaf == pd.Series(scaf).value_counts().index[0]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    sim_all, sim_dom = [], []
    for seed in range(5):
        outer = n.group_folds(scaf, 5, seed)
        for f in range(5):
            te = np.where(outer == f)[0]
            trf = [fps[i] for i in np.where(outer != f)[0]]
            for i in te:
                s = max(DataStructs.BulkTanimotoSimilarity(fps[i], trf))
                sim_all.append(s)
                sim_dom.append(dom[i])
    sim_all, sim_dom = np.array(sim_all), np.array(sim_dom)
    med, p25, p75 = np.median(sim_all), *np.percentile(sim_all, [25, 75])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    bins = np.linspace(0.3, 1.0, 36)
    ax.hist(sim_all[~sim_dom], bins=bins, color=TEAL, alpha=0.8, label=f"Other scaffold groups")
    ax.hist(sim_all[sim_dom], bins=bins, color=GREY, alpha=0.6, label="Dominant scaffold group")
    ax.axvline(med, color="black", ls="--", lw=1.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(med + 0.012, ax.get_ylim()[1] * 0.96, f"median {med:.2f}", ha="left", va="top", fontsize=11)
    ax.set_xlabel("Tanimoto similarity to the nearest training compound", fontsize=12)
    ax.set_ylabel("Held-out compounds (5 repeats)", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    save(fig, "Fig2_C_nearest_neighbour_similarity")
    print(f"dominant-group compounds: median {np.median(sim_all[sim_dom]):.2f}; other groups: median {np.median(sim_all[~sim_dom]):.2f}")
    print(f"median {med:.2f}, IQR {p25:.2f}-{p75:.2f}, >=0.7: {(sim_all >= 0.7).mean():.3f}, <0.5: {(sim_all < 0.5).sum()}")
    print("saved Fig2 C and D")


def run_panels_de():
    """Figure 2, new panels D and E: feature-class ablation under nested scaffold-grouped cross-validation (Bayesian-optimized)."""
    matplotlib.use("Agg")

    OUT = cfg.FIGURES / "panels"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})
    MODELS = [("ExtraTrees", "Extra Trees", "#1E9AA0"), ("CatBoost", "CatBoost", "#D99A25"), ("GNN", "GNN", "#7B5FB5")]

    main = pd.read_csv(cfg.OUTPUT / "nested_cv_final_bo/metrics_summary_mean_sd.csv")
    morgan = pd.concat([pd.read_csv(cfg.OUTPUT / f"{d}/metrics_summary_mean_sd.csv")
                        for d in ["nested_cv_bo_morgan_trees", "nested_cv_bo_morgan_gnn"]])
    morgan = morgan[morgan.model.isin([m[0] for m in MODELS])]
    masked = pd.concat([pd.read_csv(cfg.OUTPUT / f"{d}/metrics_summary_mean_sd.csv")
                        for d in ["nested_cv_masked_diagnostic", "nested_cv_masked_diagnostic_cat", "nested_cv_masked_diagnostic_gnn"]])
    masked = masked[masked.model.isin([m[0] for m in MODELS]) & (masked.feature_set == "ligand+dock+ifp")]

    levels = [("Morgan", morgan, "morgan_only", False), ("+ RDKit", main, "ligand_only", False),
              ("+ Docking", main, "ligand+dock", False), ("+ Docking\n+ IFP", main, "ligand+dock+ifp", False),
              ("Active-filtered\nIFP", masked, "ligand+dock+ifp", True)]


    def get(df, model, fs):
        r = df[(df.model == model) & (df.feature_set == fs)]
        return None if r.empty else r.iloc[0]


    def panel(metric, ylabel, name, ylim=None, fmt="{:.2f}"):
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        w = 0.26
        table = []
        for li, (lab, df, fs, hatched) in enumerate(levels):
            for mi, (m, mlab, col) in enumerate(MODELS):
                r = get(df, m, fs)
                x = li + (mi - 1) * w
                if r is None:
                    ax.text(x, 0.01, "n/a", ha="center", va="bottom", fontsize=8.5, color="#777777", rotation=90)
                    table.append((lab.replace("\n", " "), mlab, np.nan, np.nan))
                    continue
                mean, sd = r[f"{metric}_mean"], r[f"{metric}_std"]
                ax.bar(x, mean, width=w * 0.94, color=col, alpha=0.38 if hatched else 0.92, hatch="///" if hatched else None,
                       edgecolor=col if hatched else "none", lw=1.2, zorder=2)
                ax.errorbar(x, mean, yerr=sd, color="black", capsize=3, lw=1.3, zorder=3)
                table.append((lab.replace("\n", " "), mlab, mean, sd))
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([l[0] for l in levels], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        if ylim:
            ax.set_ylim(*ylim)
        ax.axvline(3.5, color="#999999", ls=":", lw=1.2)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=l) for _, l, c in MODELS], frameon=False, fontsize=10, loc="upper left", ncol=3)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return pd.DataFrame(table, columns=["level", "model", "mean", "sd"])


    tr2 = panel("R2", "Nested CV R² (mean ± SD, 5 repeats)", "Fig2_D_feature_ablation_R2", ylim=(0, 0.95))
    tmae = panel("MAE", "Nested CV MAE (mean ± SD, 5 repeats)", "Fig2_E_feature_ablation_MAE", ylim=(0, 24))
    print(tr2.round(3).pivot(index="level", columns="model", values="mean").loc[[l[0].replace("\n", " ") for l in levels]].to_string())
    tr2.to_csv(cfg.TABLES / "Fig2_D_ablation_values.csv", index=False)
    print("saved Fig2 D and E")


STAGES = {
    "panels_ab": run_panels_ab,
    "panels_cd": run_panels_cd,
    "panels_de": run_panels_de,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    for w in wanted:
        print("=== stage", w, flush=True)
        STAGES[w]()
