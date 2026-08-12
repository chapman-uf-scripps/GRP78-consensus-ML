"""
Figure 7B: UMAP of the 1,488-compound training library with the five confirmed
actives overlaid.

Features: Morgan radius-2, 1024-bit; metric: Jaccard; n_neighbors / min_dist
matched to the existing pipeline UMAP in code/pipeline_core/step07_chemical_space.py.
The training library is plotted as a grey backdrop; the four OTAVA confirmed
actives are teal circles (labeled with COMPOUND ID); the FDA-derived confirmed
active (Desisobutyl-n-butyl Bortezomib) is an amber circle (unlabeled, since
it is the only amber dot).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import umap


ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
OUT  = ROOT / "output" / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
OTAVA_CSV = ROOT / "output" / "source_tables" / "figure5b_otava_top10_consensus_ranking.csv"
FDA_CSV   = ROOT / "output" / "source_tables" / "figure7b_fda_top10_consensus_ranking.csv"

MORGAN_RADIUS = 2
MORGAN_BITS   = 1024
RANDOM_STATE  = 42

OTAVA_C = "#159A9C"
FDA_C   = "#D99A25"


def morgan_row(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS),
        arr)
    return arr


def main():
    # ---- training library (actives only, GRP78 inhibition >= 38%) ----
    train_full = pd.read_csv(TRAIN_CSV, usecols=["Name", "SMILES", "A5"])
    train = train_full[train_full["A5"] >= 38.0].reset_index(drop=True)
    train_fp = np.stack([morgan_row(s) for s in train["SMILES"]])
    print(f"training actives (A5 >= 38%): {train_fp.shape}  "
          f"(out of {len(train_full)} total training compounds)")

    # ---- pull the 5 confirmed actives ----
    otava = pd.read_csv(OTAVA_CSV)
    fda   = pd.read_csv(FDA_CSV)
    otava_active = otava[otava["status"].str.contains("Confirmed active", case=False, na=False)].copy()
    fda_active   = fda[fda["status"].str.contains("Confirmed active", case=False, na=False)].copy()
    print(f"OTAVA confirmed actives: {len(otava_active)}; FDA confirmed actives: {len(fda_active)}")

    hits = []
    hit_fp_rows = []
    for _, row in otava_active.iterrows():
        hits.append({"library": "OTAVA", "id": str(row["COMPOUND ID"]),
                     "smiles": str(row["SMILES"]),
                     "ic50": str(row["ic50_label"])})
        hit_fp_rows.append(morgan_row(str(row["SMILES"])))
    for _, row in fda_active.iterrows():
        hits.append({"library": "FDA-derived", "id": str(row["Compound_CID"]),
                     "smiles": str(row["canonical_smiles"]),
                     "ic50": str(row["ic50_label"])})
        hit_fp_rows.append(morgan_row(str(row["canonical_smiles"])))
    hits_df = pd.DataFrame(hits)
    hit_fp = np.stack(hit_fp_rows)

    # ---- joint UMAP ----
    print("Running UMAP (Morgan radius-2, Jaccard) ...")
    X = np.vstack([train_fp, hit_fp]).astype(bool)
    reducer = umap.UMAP(
        n_neighbors=35, min_dist=0.12, metric="jaccard",
        random_state=RANDOM_STATE, low_memory=True,
    )
    xy = reducer.fit_transform(X)
    n_train = train_fp.shape[0]
    train_xy = xy[:n_train]
    hit_xy   = xy[n_train:]

    # ---- plot ----
    plt.rcParams.update({"font.family": "Arial", "font.weight": "bold"})
    fig, ax = plt.subplots(figsize=(9.0, 7.2))

    ax.scatter(train_xy[:, 0], train_xy[:, 1],
               s=14, c="#C8CED4", edgecolor="none", alpha=0.55,
               zorder=2)

    for i, row in hits_df.iterrows():
        c = OTAVA_C if row["library"] == "OTAVA" else FDA_C
        ax.scatter(hit_xy[i, 0], hit_xy[i, 1],
                   s=320, c=c, edgecolor="black", linewidth=2.0,
                   marker="o", zorder=6)

    # OTAVA labels with per-hit offsets; FDA stays unlabeled.
    LABEL_OFFSETS = {
        "P2195430": (0.30, 0.30),
        "P2195267": (0.30,-0.50),
        "P2195222": (0.30,-0.95),
        "P1157904": (0.30, 0.30),
    }
    for i, row in hits_df.iterrows():
        if row["library"] != "OTAVA":
            continue
        dx, dy = LABEL_OFFSETS.get(row["id"], (0.30, 0.30))
        ax.annotate(row["id"],
                    xy=(hit_xy[i, 0], hit_xy[i, 1]),
                    xytext=(hit_xy[i, 0] + dx, hit_xy[i, 1] + dy),
                    fontsize=15, fontweight="bold", color=OTAVA_C, zorder=7,
                    family="Arial")

    ax.set_xlabel("UMAP-1", fontsize=18, fontweight="bold", labelpad=10, family="Arial")
    ax.set_ylabel("UMAP-2", fontsize=18, fontweight="bold", labelpad=10, family="Arial")
    ax.set_title("Confirmed actives in training-library chemical space\n"
                 "(Morgan radius-2, Jaccard UMAP)",
                 fontsize=19, fontweight="bold", pad=14, family="Arial")

    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(3.0)
        ax.spines[spine].set_color("#111111")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=14, width=2.6, length=8, pad=6)
    ax.tick_params(axis="y", labelsize=14, width=2.6, length=8, pad=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_family("Arial")

    ax.grid(True, linestyle=":", linewidth=0.7, color="#9aa3ad", alpha=0.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               label=f"Training actives (GRP78 inhibition ≥ 38%, n={n_train})",
               markerfacecolor="#C8CED4", markeredgecolor="none",
               markersize=10),
        Line2D([0], [0], marker="o", color="w", label="OTAVA confirmed active",
               markerfacecolor=OTAVA_C, markeredgecolor="black",
               markeredgewidth=1.6, markersize=13),
        Line2D([0], [0], marker="o", color="w", label="FDA-derived confirmed active",
               markerfacecolor=FDA_C, markeredgecolor="black",
               markeredgewidth=1.6, markersize=13),
    ]
    ax.legend(handles=handles, loc="best", fontsize=12,
              frameon=True, framealpha=0.95)

    fig.tight_layout()
    out_png = OUT / "Figure_7B_confirmed_actives_vs_training.png"
    out_pdf = OUT / "Figure_7B_confirmed_actives_vs_training.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")

    # ---- source CSV ----
    coords_rows = []
    for i, m in enumerate(train.itertuples(index=False)):
        coords_rows.append({"library": "Training", "id": m.Name, "smiles": m.SMILES,
                            "A5": float(m.A5),
                            "umap_x": float(train_xy[i, 0]),
                            "umap_y": float(train_xy[i, 1])})
    for i, row in hits_df.iterrows():
        coords_rows.append({"library": row["library"], "id": row["id"],
                            "smiles": row["smiles"], "A5": np.nan,
                            "umap_x": float(hit_xy[i, 0]),
                            "umap_y": float(hit_xy[i, 1])})
    src = ROOT / "output" / "source_tables" / "figure7b_confirmed_actives_vs_training.csv"
    pd.DataFrame(coords_rows).to_csv(src, index=False)
    print(f"Wrote {src}")


if __name__ == "__main__":
    main()
