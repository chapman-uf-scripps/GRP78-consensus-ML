"""
07_chemical_space_bayesian.py
==============================
UMAP chemical space of the 1,488-compound GRP78 training library.
Produces three panels: measured GRP78 inhibition, A#/B# design series,
and top Murcko scaffolds.

Run from the pipeline directory:
    python 07_chemical_space_bayesian.py

Outputs: Output/07_chemical_space/
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
if not hasattr(np, "in1d") and hasattr(np, "isin"):
    np.in1d = np.isin

import umap

RDLogger.DisableLog("rdApp.*")

_HERE          = Path(__file__).parent
TRAIN_PATH     = _HERE / "Input" / "training_activity_merged_result.csv"
OUT_DIR        = _HERE / "Output" / "07_chemical_space"
RANDOM_STATE   = 42
ACTIVE_THRESHOLD = 38.0       # consistent with pipeline threshold
N_BITS, RADIUS = 1024, 2

COLORS = {
    "ink":   "#17212B",
    "muted": "#627080",
}
SCAFFOLD_PALETTE = [
    "#159A9C", "#7B5FB5", "#D99A25", "#D95E55",
    "#3F6FA8", "#4F9D69", "#C45A9A", "#8A6A4F", "#6E7B8B", "#222222",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def canonical_smiles(smiles):
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else ""


def murcko(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "invalid"
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaf if scaf else "acyclic"


def morgan_matrix(smiles_values):
    matrix = np.zeros((len(smiles_values), N_BITS), dtype=np.float32)
    valid = []
    for idx, smiles in enumerate(smiles_values):
        mol = mol_from_smiles(smiles)
        valid.append(mol is not None)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
        arr = np.zeros((N_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        matrix[idx] = arr
    return matrix, np.array(valid, dtype=bool)


def tanimoto_diversity(smiles_values, sample_size=300):
    mols = [mol_from_smiles(s) for s in smiles_values]
    fps  = [AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)
            for m in mols if m is not None]
    if len(fps) < 2:
        return {"sample_size": len(fps), "mean_pairwise_tanimoto": None,
                "median_nearest_neighbor_tanimoto": None}
    rng     = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(fps), size=min(sample_size, len(fps)), replace=False)
    sample  = [fps[i] for i in indices]
    pair_sims, nn_sims = [], []
    for i, fp in enumerate(sample):
        sims = DataStructs.BulkTanimotoSimilarity(fp, sample)
        sims_other = [s for j, s in enumerate(sims) if j != i]
        pair_sims.extend(sims_other[i + 1:])
        nn_sims.append(max(sims_other) if sims_other else 0.0)
    return {
        "sample_size": len(sample),
        "mean_pairwise_tanimoto": float(np.mean(pair_sims)) if pair_sims else None,
        "median_nearest_neighbor_tanimoto": float(np.median(nn_sims)) if nn_sims else None,
    }


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(TRAIN_PATH)
    df = df[["Name", "SMILES", "A5"]].copy()
    df["compound_id"]    = df["Name"].astype(str)
    df["A5"]             = pd.to_numeric(df["A5"], errors="coerce")
    df["canonical_smiles"] = df["SMILES"].map(canonical_smiles)
    df["scaffold"]       = df["SMILES"].map(murcko)
    df["series"]         = df["compound_id"].str.extract(r"^([A-Z]\d+)C",  expand=False).fillna("Other")
    df["design_family"]  = df["compound_id"].str.extract(r"^([A-Z])",       expand=False).fillna("Other")
    df["series_number"]  = pd.to_numeric(
        df["compound_id"].str.extract(r"^[A-Z](\d+)C", expand=False), errors="coerce"
    )
    return df.reset_index(drop=True)


# ── embeddings ────────────────────────────────────────────────────────────────

def compute_embeddings(x):
    pca_model = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy    = pca_model.fit_transform(x)

    pca50 = PCA(
        n_components=min(50, x.shape[1], x.shape[0] - 1), random_state=RANDOM_STATE
    ).fit_transform(x)
    reducer = umap.UMAP(
        n_neighbors=35, min_dist=0.12, metric="jaccard",
        random_state=RANDOM_STATE, low_memory=True,
    )
    umap_xy = reducer.fit_transform(x.astype(bool))
    tsne_xy = TSNE(
        n_components=2, perplexity=40, init="pca",
        learning_rate="auto", max_iter=1000, random_state=RANDOM_STATE,
    ).fit_transform(pca50)

    return {"PCA": pca_xy, "UMAP": umap_xy, "t-SNE": tsne_xy}, pca_model.explained_variance_ratio_


# ── scaffold enrichment ───────────────────────────────────────────────────────

def scaffold_enrichment(df):
    df2 = df.copy()
    df2["active"]  = df2["A5"] >= ACTIVE_THRESHOLD
    global_rate    = df2["active"].mean()
    table = (
        df2.groupby("scaffold")
        .agg(n=("compound_id","count"), active_n=("active","sum"),
             mean_A5=("A5","mean"), median_A5=("A5","median"))
        .reset_index()
    )
    table["active_rate"]         = table["active_n"] / table["n"]
    table["enrichment_vs_global"] = table["active_rate"] / global_rate if global_rate else np.nan
    table = table.sort_values(["active_n","active_rate","n"], ascending=[False,False,False])
    table.to_csv(OUT_DIR / "Figure_3_training_scaffold_enrichment.csv", index=False)
    return table, float(global_rate)


# ── plot style ────────────────────────────────────────────────────────────────

def setup_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 10.8, "font.weight": "bold",
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 11.5, "axes.labelweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


def label_axes(ax, method):
    ax.set_xlabel(f"{method} 1", fontweight="bold")
    ax.set_ylabel(f"{method} 2", fontweight="bold")
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.grid(color="#E7EBF0", linewidth=0.5, alpha=0.8)


def style_colorbar(cbar, label):
    cbar.set_label(label, fontsize=10.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=9.5)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")


def make_ab_series_colors(series_values):
    cmap = colormaps["tab20"].resampled(max(20, len(series_values)))
    return {s: to_hex(cmap(i % cmap.N)) for i, s in enumerate(series_values)}


# ── panel plotters ────────────────────────────────────────────────────────────

def scatter_activity(ax, xy, df):
    activity = df["A5"].clip(lower=0, upper=100)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=13, c=activity,
                    cmap="viridis", vmin=0, vmax=100, alpha=0.85, linewidths=0)
    ax.set_title("Measured GRP78 Inhibition", fontweight="bold")
    return sc


def scatter_ab_series(ax, xy, df):
    missing = df["series_number"].isna()
    if missing.any():
        ax.scatter(xy[missing, 0], xy[missing, 1], s=9, c="#A9B3BF", alpha=0.35, linewidths=0)
    series_values = sorted(
        df.loc[~missing, "series"].dropna().unique(),
        key=lambda v: (v[0], int(v[1:]))
    )
    series_colors = make_ab_series_colors(series_values)
    marker_map = {"A": "o", "B": "^"}
    for s in series_values:
        family = s[0]
        mask   = (~missing) & (df["series"] == s)
        if not mask.any():
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   s=13 if family == "A" else 17,
                   color=series_colors[s], alpha=0.82, linewidths=0,
                   marker=marker_map[family])
    ax.set_title("A#/B# Design Series", fontweight="bold")
    handles = [
        Line2D([0],[0], marker="o", color="w", label="A# Series", markerfacecolor="#CB181D", markersize=8),
        Line2D([0],[0], marker="^", color="w", label="B# Series", markerfacecolor="#08519C", markersize=8),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              fontsize=10, prop={"weight":"bold","size":10})
    ax.text(0.02, 0.95, "Discrete Colors = Individual A#/B# Series",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold", color=COLORS["muted"])


def scatter_scaffold(ax, xy, df, top_scaffolds):
    other = ~df["scaffold"].isin(top_scaffolds)
    ax.scatter(xy[other, 0], xy[other, 1], s=9, c="#A9B3BF", alpha=0.35, linewidths=0)
    for idx, scaffold in enumerate(top_scaffolds):
        mask = df["scaffold"] == scaffold
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   s=14, c=SCAFFOLD_PALETTE[idx], alpha=0.85, linewidths=0)
    ax.set_title("Top Training Scaffolds", fontweight="bold")


# ── single-panel saves ────────────────────────────────────────────────────────

def save_panel(png_path, plotter, xy, df, top_scaffolds=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=350)
    if plotter == "activity":
        mappable = scatter_activity(ax, xy, df)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.78, pad=0.018)
        style_colorbar(cbar, "GRP78 Inhibition (%)")
    elif plotter == "series":
        scatter_ab_series(ax, xy, df)
    elif plotter == "scaffold":
        scatter_scaffold(ax, xy, df, top_scaffolds)
    label_axes(ax, "UMAP")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ── main composite figure ─────────────────────────────────────────────────────

def plot_figure(df, xy, enrichment_table):
    top_scaffolds = enrichment_table.sort_values("n", ascending=False).head(10)["scaffold"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), dpi=350)
    mappable = scatter_activity(axes[0], xy, df)
    scatter_ab_series(axes[1], xy, df)
    scatter_scaffold(axes[2], xy, df, top_scaffolds)
    for ax in axes:
        label_axes(ax, "UMAP")

    cbar = fig.colorbar(mappable, ax=axes[0], shrink=0.74, pad=0.015)
    style_colorbar(cbar, "GRP78 Inhibition (%)")

    fig.suptitle(
        "Chemical Space of the GRP78 Training Library",
        fontsize=15.5, fontweight="bold", color=COLORS["ink"], y=0.985,
    )
    fig.text(
        0.5, 0.925,
        "UMAP Projection of Morgan Fingerprints for the 1,488-Compound Training Set, "
        "Colored by Measured GRP78 Inhibition (%), A#/B# Design Series, and Top Murcko Scaffolds.",
        ha="center", va="center", fontsize=9.8, fontweight="bold", color=COLORS["muted"],
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.87])

    fig.savefig(OUT_DIR / "Figure_3_training_chemical_space.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_3_training_chemical_space.pdf", bbox_inches="tight")
    plt.close(fig)

    save_panel(OUT_DIR / "Figure_3A_training_activity_umap.png",  "activity",  xy, df)
    save_panel(OUT_DIR / "Figure_3B_training_ab_series_umap.png", "series",    xy, df)
    save_panel(OUT_DIR / "Figure_3C_training_scaffolds_umap.png", "scaffold",  xy, df, top_scaffolds)

    scaffold_key = pd.DataFrame({
        "scaffold_label": [f"S{i+1}" for i in range(len(top_scaffolds))],
        "scaffold": top_scaffolds,
    })
    scaffold_key.to_csv(OUT_DIR / "Figure_3_training_scaffold_color_key.csv", index=False)

    series_values = sorted(df["series"].dropna().unique(), key=lambda v: (v[0], int(v[1:])))
    series_colors = make_ab_series_colors(series_values)
    series_key = (
        df.groupby(["design_family", "series_number", "series"], dropna=False)
        .size().rename("n").reset_index()
    )
    series_key["color"] = series_key["series"].map(series_colors)
    series_key.to_csv(OUT_DIR / "Figure_3_training_ab_series_color_key.csv", index=False)

    return top_scaffolds


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading training data...")
    df = load_data()

    print("Computing Morgan fingerprints...")
    x, valid = morgan_matrix(df["SMILES"].tolist())
    df = df.loc[valid].reset_index(drop=True)
    x  = x[valid]

    print("Computing UMAP / PCA / t-SNE embeddings...")
    embeddings, pca_var = compute_embeddings(x)
    xy = embeddings["UMAP"]

    print("Computing scaffold enrichment...")
    enrich, global_active_rate = scaffold_enrichment(df)

    print("Generating figures...")
    top_scaffolds = plot_figure(df, xy, enrich)

    summary = {
        "training_compounds":          int(len(df)),
        "unique_training_scaffolds":   int(df["scaffold"].nunique()),
        "singleton_training_scaffolds":int((df["scaffold"].value_counts() == 1).sum()),
        "design_family_counts":        df["design_family"].value_counts().to_dict(),
        "series_number_min":           int(df["series_number"].min()),
        "series_number_max":           int(df["series_number"].max()),
        "series_counts":               df["series"].value_counts().to_dict(),
        "active_threshold_grp78_inhibition": ACTIVE_THRESHOLD,
        "active_training_compounds":   int((df["A5"] >= ACTIVE_THRESHOLD).sum()),
        "global_active_rate":          global_active_rate,
        "pca_variance_ratio_first_two":[float(v) for v in pca_var[:2]],
        "diversity":                   tanimoto_diversity(df["SMILES"].tolist()),
        "top_scaffold_enrichment":     enrich.head(10).to_dict(orient="records"),
    }
    (OUT_DIR / "Figure_3_training_chemical_space_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
