from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")

PIPELINE_DIR = Path(__file__).resolve().parent
SCREEN_DIR = PIPELINE_DIR / "Output" / "04_screen_otava_best3"
PREDICTIONS_PATH = SCREEN_DIR / "otava_bayesian_top3_extra_catboost_gnn_predictions.csv"
OUT_DIR = PIPELINE_DIR / "Output" / "09_figure5_otava_screening"

ACTIVE_THRESHOLD = 38.0
TOP_N = 10

MODEL_COLUMNS = {
    "Extra Trees": "pred_A5_ExtraTrees_clipped",
    "CatBoost": "pred_A5_CatBoost_clipped",
    "GNN": "pred_A5_GNN_clipped",
}

CONFIRMED_ACTIVE = {
    "P2195430": "~4.0 uM",
    "P2195267": "~4.0 uM",
    "P1157904": "~4.0 uM",
}

INACTIVE_OR_WEAK = {
    "P2195277": ">50 uM",
    "P2195233": ">50 uM",
}

STATUS_COLORS = {
    "Confirmed active": "#2E9D68",
    "Inactive/weak": "#C65A5A",
    "Not tested": "#D9DEE7",
}

MODEL_COLORS = {
    "Extra Trees": "#159A9C",
    "CatBoost": "#D99A25",
    "GNN": "#7B5FB5",
}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "font.size": 15,
            "axes.titlesize": 21,
            "axes.titleweight": "bold",
            "axes.labelsize": 17,
            "axes.labelweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_predictions():
    df = pd.read_csv(PREDICTIONS_PATH)
    df["status"] = df["COMPOUND ID"].map(compound_status)
    df["ic50_label"] = df["COMPOUND ID"].map(compound_ic50_label)
    return df


def compound_status(compound_id):
    compound_id = str(compound_id)
    if compound_id in CONFIRMED_ACTIVE:
        return "Confirmed active"
    if compound_id in INACTIVE_OR_WEAK:
        return "Inactive/weak"
    return "Not tested"


def compound_ic50_label(compound_id):
    compound_id = str(compound_id)
    if compound_id in CONFIRMED_ACTIVE:
        return CONFIRMED_ACTIVE[compound_id]
    if compound_id in INACTIVE_OR_WEAK:
        return INACTIVE_OR_WEAK[compound_id]
    return "Not tested"


def compute_overlap_sets(df):
    sets = {}
    for model, col in MODEL_COLUMNS.items():
        hits = df.loc[df[col] > ACTIVE_THRESHOLD, "COMPOUND ID"].astype(str)
        sets[model] = set(hits)
    return sets


def plot_venn(df):
    sets = compute_overlap_sets(df)
    set_order = ["Extra Trees", "CatBoost", "GNN"]

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    draw_fallback_venn(ax, sets)
    ax.set_title(f"OTAVA predicted hits by top models\nPredicted A5 > {ACTIVE_THRESHOLD:.0f}", pad=18)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "figure5a_otava_top3_model_overlap_predA5_gt38")

    rows = []
    for model in set_order:
        rows.append({"region": model, "count": len(sets[model])})
    rows.extend(
        [
            {"region": "Extra Trees & CatBoost", "count": len(sets["Extra Trees"] & sets["CatBoost"])},
            {"region": "Extra Trees & GNN", "count": len(sets["Extra Trees"] & sets["GNN"])},
            {"region": "CatBoost & GNN", "count": len(sets["CatBoost"] & sets["GNN"])},
            {"region": "All three", "count": len(sets["Extra Trees"] & sets["CatBoost"] & sets["GNN"])},
        ]
    )
    pd.DataFrame(rows).to_csv(OUT_DIR / "figure5a_otava_top3_model_overlap_counts.csv", index=False)


def draw_fallback_venn(ax, sets):
    from matplotlib.patches import Circle

    circles = [
        ("Extra Trees", (0.39, 0.57), MODEL_COLORS["Extra Trees"]),
        ("CatBoost", (0.61, 0.57), MODEL_COLORS["CatBoost"]),
        ("GNN", (0.50, 0.39), MODEL_COLORS["GNN"]),
    ]
    for label, center, color in circles:
        ax.add_patch(Circle(center, 0.31, color=color, alpha=0.42, ec="white", lw=1.5))

    ax.text(
        0.24,
        0.90,
        "Extra Trees",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=MODEL_COLORS["Extra Trees"],
    )
    ax.text(
        0.76,
        0.90,
        "CatBoost",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=MODEL_COLORS["CatBoost"],
    )
    ax.text(
        0.50,
        0.04,
        "GNN",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=MODEL_COLORS["GNN"],
    )

    a, b, c = sets["Extra Trees"], sets["CatBoost"], sets["GNN"]
    counts = {
        "100": len(a - b - c),
        "010": len(b - a - c),
        "001": len(c - a - b),
        "110": len((a & b) - c),
        "101": len((a & c) - b),
        "011": len((b & c) - a),
        "111": len(a & b & c),
    }
    positions = {
        "100": (0.22, 0.62),
        "010": (0.78, 0.62),
        "001": (0.50, 0.18),
        "110": (0.50, 0.75),
        "101": (0.30, 0.39),
        "011": (0.70, 0.39),
        "111": (0.50, 0.51),
    }
    for key, value in counts.items():
        ax.text(*positions[key], str(value), ha="center", va="center", fontsize=20, fontweight="bold")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.00, 0.98)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def plot_top10_ranking(df):
    top = df.sort_values("screening_rank").head(TOP_N).copy()
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    colors = [STATUS_COLORS[status] for status in top["status"]]
    labels = top["COMPOUND ID"].astype(str).tolist()
    bars = ax.barh(labels, top["consensus_A5_mean"], color=colors, edgecolor="white", linewidth=0.8)
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(
            row["consensus_A5_mean"] + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{row['consensus_A5_mean']:.1f}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xlabel("Consensus predicted GRP78 inhibition")
    ax.set_title("Top 10 OTAVA candidates by predicted GRP78 inhibition", pad=16)
    ax.set_xlim(50, 67)
    ax.grid(axis="x", alpha=0.22)
    legend_handles = [Patch(facecolor=color, edgecolor="none", label=status) for status, color in STATUS_COLORS.items()]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save_figure(fig, OUT_DIR / "figure5b_otava_top10_consensus_ranking")

    export_cols = [
        "screening_rank",
        "COMPOUND ID",
        "PLATE NUMBER",
        "RACK POSITION",
        "status",
        "ic50_label",
        "consensus_A5_mean",
        "consensus_A5_median",
        "pred_A5_ExtraTrees_clipped",
        "pred_A5_CatBoost_clipped",
        "pred_A5_GNN_clipped",
        "SMILES",
    ]
    top.iloc[::-1][export_cols].to_csv(OUT_DIR / "figure5b_otava_top10_consensus_ranking.csv", index=False)


def plot_active_structures(df):
    active_df = (
        df[df["COMPOUND ID"].isin(CONFIRMED_ACTIVE)]
        .sort_values("screening_rank")
        .copy()
    )
    mols = []
    legends = []
    for _, row in active_df.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]))
        if mol is None:
            continue
        mols.append(mol)
        legends.append(
            f"{row['COMPOUND ID']}\nRank {int(row['screening_rank'])} | IC50 {CONFIRMED_ACTIVE[row['COMPOUND ID']]}"
        )

    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(430, 330),
        legends=legends,
        useSVG=False,
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title("Experimentally confirmed active OTAVA hits", fontsize=22, fontweight="bold", pad=18)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "figure5c_confirmed_active_otava_structures")

    active_df[
        [
            "screening_rank",
            "COMPOUND ID",
            "PLATE NUMBER",
            "RACK POSITION",
            "ic50_label",
            "consensus_A5_mean",
            "SMILES",
        ]
    ].to_csv(OUT_DIR / "figure5c_confirmed_active_otava_structures.csv", index=False)


def main():
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()
    plot_venn(df)
    plot_top10_ranking(df)
    plot_active_structures(df)


if __name__ == "__main__":
    main()
