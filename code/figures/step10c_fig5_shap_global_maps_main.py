from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
SOURCE_TABLE_DIR = ROOT / "Output" / "08_figures_bayesian" / "10_source_tables"
OUT_DIR = ROOT / "Output" / "08_figures_bayesian" / "12_figure5_shap_global_maps"
SPLIT = "scaffold_split"
TOP_N = 20

MODELS = [
    ("extra_trees", "Extra Trees"),
    ("catboost", "CatBoost"),
]


# Chain-A residue numbering of PDB 5E85 → amino-acid single-letter label
# (single-letter code + position; e.g. A452 → S452 because residue 452 is Ser)
RESIDUE_AA_MAP = {
    "A428": "T428",
    "A429": "V429",
    "A451": "F451",
    "A452": "S452",
    "A453": "T453",
    "A462": "T462",
}


def _apply_residue_map(label):
    """Replace chain-A numeric labels (A###) with amino-acid single-letter labels."""
    # Sort by key length descending in case any prefixes collide (defensive)
    for old, new in sorted(RESIDUE_AA_MAP.items(), key=lambda x: -len(x[0])):
        label = label.replace(old, new)
    return label


def feature_label(feature):
    if feature.startswith("morgan_"):
        return f"MorganBit_{int(feature.split('_', 1)[1])}"
    labels = {
        "desc_logp": "MolLogP",
        "desc_mol_wt": "MolWt",
        "desc_tpsa": "TPSA",
        "desc_hbd": "NumHDonors",
        "desc_hba": "NumHAcceptors",
        "desc_rot_bonds": "RotBonds",
        "desc_rings": "Rings",
        "desc_aromatic_rings": "AromaticRings",
        "desc_fraction_csp3": "FractionCSP3",
    }
    if feature in labels:
        return labels[feature]
    if feature.startswith("ifp_"):
        return _apply_residue_map("IFP " + feature[4:].replace("_", " "))
    if feature.startswith("dock_"):
        return "Dock " + feature[5:].replace("_", " ")
    return _apply_residue_map(feature.replace("_", " "))


def normalize_values(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, 0.5)
    low = np.nanpercentile(values[finite], 1)
    high = np.nanpercentile(values[finite], 99)
    if high <= low:
        return np.full(values.shape, 0.5)
    return np.clip((values - low) / (high - low), 0, 1)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_").lower()


def load_top_features(model_slug):
    shap_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{SPLIT}_shap_values.csv")
    feature_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{SPLIT}_shap_feature_matrix.csv")
    mean_abs = np.mean(np.abs(shap_df.to_numpy(dtype=float)), axis=0)
    ranking = pd.DataFrame({"feature": shap_df.columns, "mean_abs_shap": mean_abs})
    top = ranking.sort_values("mean_abs_shap", ascending=False).head(TOP_N)["feature"].tolist()
    return shap_df, feature_df, ranking, top


def plot_map(ax, model_slug, model_label, cmap, panel_label):
    shap_df, feature_df, ranking, top_features = load_top_features(model_slug)
    rng = np.random.default_rng(1701)
    for row_idx, feature in enumerate(top_features[::-1]):
        shap_values = shap_df[feature].to_numpy(dtype=float)
        feature_values = feature_df[feature].to_numpy(dtype=float)
        y = np.full(shap_values.shape, row_idx, dtype=float)
        y += rng.normal(0, 0.075, size=len(y))
        ax.scatter(
            shap_values,
            y,
            c=normalize_values(feature_values),
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=18,
            alpha=0.92,
            linewidths=0,
        )

    labels = [feature_label(feature) for feature in top_features[::-1]]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.axvline(0, color="#8A8F98", linewidth=1.5)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.42, linestyle=":", linewidth=0.8)
    ax.grid(axis="x", alpha=0.10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=15, fontweight="bold")
    ax.set_title(f"{model_label} | Scaffold split", fontsize=20, fontweight="bold", pad=12)
    ax.text(
        -0.12,
        1.04,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=28,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=13)
    max_abs = np.nanmax(np.abs(shap_df[top_features].to_numpy(dtype=float)))
    xlim = max(5, np.ceil(max_abs / 5.0) * 5.0)
    ax.set_xlim(-xlim, xlim)
    ranking.assign(feature_label=ranking["feature"].map(feature_label)).to_csv(
        OUT_DIR / f"figure5_{model_slug}_scaffold_shap_global_map_ranked_features.csv",
        index=False,
    )


def save_individual(model_slug, model_label, cmap):
    fig, ax = plt.subplots(figsize=(8.0, 9.8))
    plot_map(ax, model_slug, model_label, cmap, "")
    ax.texts[-1].set_visible(False)
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value", fontsize=15, fontweight="bold", labelpad=10)
    colorbar.ax.tick_params(labelsize=13)
    colorbar.outline.set_visible(False)
    fig.suptitle(
        f"{model_label} SHAP Global Feature Contribution Map",
        fontsize=20,
        fontweight="bold",
        y=0.965,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    stem = OUT_DIR / f"figure5_individual_{safe_name(model_label)}_shap_global_map_scaffold"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("blue_magenta", ["#1487E8", "#FF0051"])

    fig, axes = plt.subplots(1, 2, figsize=(18.5, 10.4))
    for panel_idx, (model_slug, model_label) in enumerate(MODELS):
        plot_map(axes[panel_idx], model_slug, model_label, cmap, chr(ord("A") + panel_idx))
        save_individual(model_slug, model_label, cmap)

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.030, pad=0.025)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value", fontsize=17, fontweight="bold", labelpad=12)
    colorbar.ax.tick_params(labelsize=14)
    colorbar.outline.set_visible(False)

    fig.suptitle(
        "Figure 5. SHAP global feature contribution maps",
        fontsize=27,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 0.955, 0.965))
    stem = OUT_DIR / "figure5_tree_model_shap_global_contribution_maps_scaffold"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "panel": "A",
                "model": "Extra Trees",
                "split": "scaffold_split",
                "description": "SHAP beeswarm map for top 20 Extra Trees features.",
            },
            {
                "panel": "B",
                "model": "CatBoost",
                "split": "scaffold_split",
                "description": "SHAP beeswarm map for top 20 CatBoost features.",
            },
        ]
    ).to_csv(OUT_DIR / "figure5_shap_global_maps_manifest.csv", index=False)
    print(f"Saved Figure 5 SHAP maps to: {OUT_DIR}")


if __name__ == "__main__":
    main()
