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
OUT_DIR = ROOT / "Output" / "10_supplementary_figures"
TOP_N = 25

MODELS = [
    ("extra_trees", "Extra Trees"),
    ("catboost", "CatBoost"),
]
SPLITS = [
    ("random_split", "Random Split"),
    ("scaffold_split", "Scaffold Split"),
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
    for old, new in sorted(RESIDUE_AA_MAP.items(), key=lambda x: -len(x[0])):
        label = label.replace(old, new)
    return label


def feature_label(feature):
    if feature.startswith("morgan_"):
        bit = int(feature.split("_", 1)[1])
        return f"MorganBit_{bit}"
    if feature == "desc_logp":
        return "MolLogP"
    if feature == "desc_mol_wt":
        return "MolWt"
    if feature == "desc_tpsa":
        return "TPSA"
    if feature == "desc_hbd":
        return "NumHDonors"
    if feature == "desc_hba":
        return "NumHAcceptors"
    if feature == "desc_rot_bonds":
        return "RotBonds"
    if feature == "desc_rings":
        return "Rings"
    if feature == "desc_aromatic_rings":
        return "AromaticRings"
    if feature.startswith("ifp_"):
        text = feature[4:].replace("_", " ")
        return _apply_residue_map(f"IFP {text}")
    if feature.startswith("dock_"):
        return "Dock " + feature[5:].replace("_", " ")
    return _apply_residue_map(feature.replace("_", " "))


def normalize_feature_values(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, 0.5)
    low = np.nanpercentile(values[finite], 1)
    high = np.nanpercentile(values[finite], 99)
    if high <= low:
        return np.full(values.shape, 0.5)
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0, 1)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def plot_shap_map(model_slug, model_label, split, split_label):
    shap_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{split}_shap_values.csv")
    feature_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{split}_shap_feature_matrix.csv")

    mean_abs = np.mean(np.abs(shap_df.to_numpy(dtype=float)), axis=0)
    ranking = pd.DataFrame({"feature": shap_df.columns, "mean_abs_shap": mean_abs})
    top_features = ranking.sort_values("mean_abs_shap", ascending=False).head(TOP_N)["feature"].tolist()

    cmap = LinearSegmentedColormap.from_list("blue_magenta", ["#1487E8", "#FF0051"])
    rng = np.random.default_rng(42)

    height = max(9.0, 0.38 * len(top_features))
    fig, ax = plt.subplots(figsize=(8.4, height))

    for row_idx, feature in enumerate(top_features[::-1]):
        shap_values = shap_df[feature].to_numpy(dtype=float)
        feature_values = feature_df[feature].to_numpy(dtype=float)
        y = np.full(shap_values.shape, row_idx, dtype=float)
        y += rng.normal(0, 0.075, size=len(y))
        ax.scatter(
            shap_values,
            y,
            c=normalize_feature_values(feature_values),
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=18,
            alpha=0.92,
            linewidths=0,
        )

    labels = [feature_label(feature) for feature in top_features[::-1]]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="#8A8F98", linewidth=1.6)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.42, linestyle=":", linewidth=0.8)
    ax.grid(axis="x", alpha=0.10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{model_label} SHAP Global Feature Contribution Map\n{split_label}",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    max_abs = np.nanmax(np.abs(shap_df[top_features].to_numpy(dtype=float)))
    xlim = np.ceil(max_abs / 5.0) * 5.0
    ax.set_xlim(-xlim, xlim)

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value", fontsize=12, fontweight="bold", labelpad=10)
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    output_stem = OUT_DIR / f"Figure_S10_shap_global_map_{model_slug}_{split}"
    fig.savefig(output_stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    ranking.assign(feature_label=ranking["feature"].map(feature_label)).to_csv(
        output_stem.with_suffix(".csv"), index=False
    )
    return output_stem


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for model_slug, model_label in MODELS:
        for split, split_label in SPLITS:
            stem = plot_shap_map(model_slug, model_label, split, split_label)
            records.append(
                {
                    "si_figure": stem.stem,
                    "caption": f"SHAP beeswarm/global feature contribution map for {model_label} under {split_label}.",
                    "files": f"{stem.name}.png; {stem.name}.pdf; {stem.name}.csv",
                }
            )
    pd.DataFrame(records).to_csv(OUT_DIR / "Figure_S10_shap_global_map_manifest.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
