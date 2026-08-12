"""
Compare OTAVA and FDA-derived top-10 picks in the training library's chemical
space.

Builds:
  output/top_hits_chemical_space/
    top_hits_umap_coords.csv          UMAP coords for every plotted compound
    top_hits_umap.png / .pdf          UMAP scatter, training=grey,
                                      OTAVA top-10 in teal, FDA top-10 in orange,
                                      confirmed actives starred and labeled
    top_hits_tanimoto_otava_vs_fda.csv  20x20 Tanimoto matrix
    top_hits_summary.csv              one row per top-hit with library, ID,
                                      consensus mean, FP outcome, and the
                                      single nearest neighbor in the other
                                      library plus its Tanimoto similarity
"""

from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
RDLogger.DisableLog("rdApp.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import umap


ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
OUT  = ROOT / "output" / "top_hits_chemical_space"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
OTAVA_CSV = ROOT / "output" / "source_tables" / "figure5b_otava_top10_consensus_ranking.csv"
FDA_CSV   = ROOT / "output" / "source_tables" / "figure7b_fda_top10_consensus_ranking.csv"

MORGAN_RADIUS = 2
MORGAN_BITS   = 1024
RANDOM_STATE  = 42


def morgan_bitvec(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


def morgan_row(smi):
    bv = morgan_bitvec(smi)
    if bv is None:
        return None
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def load_training():
    df = pd.read_csv(TRAIN_CSV)
    morgan_cols = [c for c in df.columns if c.startswith("morgan_")]
    # The training table only stored 380 surviving bits; expand back to a
    # 1024-bit dense matrix so we can stack against freshly computed external
    # library fingerprints.
    keep_idx = [int(c.split("_")[1]) for c in morgan_cols]
    dense = np.zeros((len(df), MORGAN_BITS), dtype=np.uint8)
    dense[:, keep_idx] = df[morgan_cols].astype(np.uint8).values
    return df[["Name", "SMILES", "A5"]].copy(), dense


def load_external(csv_path, library_name):
    df = pd.read_csv(csv_path)
    smi_col = "canonical_smiles" if "canonical_smiles" in df.columns else "SMILES"
    id_col  = "Compound_CID" if "Compound_CID" in df.columns else "COMPOUND ID"
    name_col = "Name" if "Name" in df.columns else None
    df["_smi"] = df[smi_col].astype(str)
    df["_label"] = df[id_col].astype(str)
    df["_display"] = df[name_col].astype(str) if name_col else df["_label"]
    fps = np.stack([morgan_row(s) for s in df["_smi"]])
    df = df.assign(library=library_name)
    return df, fps


def main():
    print("Loading training fingerprints ...")
    train_meta, train_fp = load_training()
    print(f"  training: {train_fp.shape}")

    print("Loading OTAVA top-10 ...")
    otava_df, otava_fp = load_external(OTAVA_CSV, "OTAVA")
    print(f"  OTAVA top-10: {otava_fp.shape}")

    print("Loading FDA top-10 ...")
    fda_df, fda_fp = load_external(FDA_CSV, "FDA-derived")
    print(f"  FDA top-10:   {fda_fp.shape}")

    # ---- joint UMAP ----
    print("Running joint UMAP (Morgan radius-2, Jaccard) ...")
    X = np.vstack([train_fp, otava_fp, fda_fp]).astype(bool)
    reducer = umap.UMAP(
        n_neighbors=35, min_dist=0.12, metric="jaccard",
        random_state=RANDOM_STATE, low_memory=True,
    )
    xy = reducer.fit_transform(X)

    n_train = train_fp.shape[0]
    n_otava = otava_fp.shape[0]
    train_xy = xy[:n_train]
    otava_xy = xy[n_train:n_train + n_otava]
    fda_xy   = xy[n_train + n_otava:]

    # ---- coords CSV ----
    rows = []
    for i, (xy_, m) in enumerate(zip(train_xy, train_meta.itertuples(index=False))):
        rows.append({"library": "Training", "id": m.Name, "smiles": m.SMILES,
                     "consensus_mean": np.nan, "status": "training",
                     "umap_x": xy_[0], "umap_y": xy_[1]})
    for i, row in otava_df.iterrows():
        rows.append({"library": "OTAVA", "id": row["_label"], "smiles": row["_smi"],
                     "consensus_mean": row.get("consensus_A5_mean", np.nan),
                     "status": row.get("status", ""),
                     "umap_x": otava_xy[i][0], "umap_y": otava_xy[i][1]})
    for i, row in fda_df.iterrows():
        rows.append({"library": "FDA-derived", "id": row["_label"],
                     "smiles": row["_smi"],
                     "consensus_mean": row.get("consensus_A5_mean", np.nan),
                     "status": row.get("status", ""),
                     "umap_x": fda_xy[i][0], "umap_y": fda_xy[i][1]})
    coords = pd.DataFrame(rows)
    coords.to_csv(OUT / "top_hits_umap_coords.csv", index=False)
    print(f"Wrote {OUT / 'top_hits_umap_coords.csv'}")

    # ---- plot ----
    print("Plotting ...")
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    # training background
    ax.scatter(train_xy[:, 0], train_xy[:, 1], s=10, c="#C8CED4",
               edgecolor="none", alpha=0.55, label=f"Training library (n={n_train})")

    OTAVA_COLOR  = "#159A9C"  # teal
    FDA_COLOR    = "#D99A25"  # amber
    STAR_OTAVA   = "#0A6E70"
    STAR_FDA     = "#A16916"

    otava_status = otava_df.get("status", pd.Series([""] * len(otava_df))).fillna("").astype(str)
    fda_status   = fda_df.get("status",   pd.Series([""] * len(fda_df))).fillna("").astype(str)

    otava_active_mask = otava_status.str.contains("Confirmed active", case=False, na=False).values
    fda_active_mask   = fda_status.str.contains("Confirmed active", case=False, na=False).values

    # OTAVA non-actives
    if (~otava_active_mask).any():
        ax.scatter(otava_xy[~otava_active_mask, 0], otava_xy[~otava_active_mask, 1],
                   s=110, c=OTAVA_COLOR, edgecolor="white", linewidth=1.2,
                   marker="o", label=f"OTAVA top-10 (inactive/weak)", zorder=4)
    # OTAVA confirmed actives
    if otava_active_mask.any():
        ax.scatter(otava_xy[otava_active_mask, 0], otava_xy[otava_active_mask, 1],
                   s=320, c=OTAVA_COLOR, edgecolor=STAR_OTAVA, linewidth=2.0,
                   marker="*", label="OTAVA confirmed active", zorder=6)

    # FDA non-actives (currently "Untested" for 9 of 10)
    if (~fda_active_mask).any():
        ax.scatter(fda_xy[~fda_active_mask, 0], fda_xy[~fda_active_mask, 1],
                   s=110, c=FDA_COLOR, edgecolor="white", linewidth=1.2,
                   marker="o", label="FDA-derived top-10 (untested)", zorder=4)
    # FDA confirmed actives
    if fda_active_mask.any():
        ax.scatter(fda_xy[fda_active_mask, 0], fda_xy[fda_active_mask, 1],
                   s=320, c=FDA_COLOR, edgecolor=STAR_FDA, linewidth=2.0,
                   marker="*", label="FDA-derived confirmed active", zorder=6)

    # Labels for confirmed actives only (keep plot readable)
    LABEL_OFFSET = (0.20, 0.20)
    for i, row in otava_df.iterrows():
        if not otava_active_mask[i]:
            continue
        ax.annotate(str(row["_label"]),
                    xy=(otava_xy[i][0], otava_xy[i][1]),
                    xytext=(otava_xy[i][0] + LABEL_OFFSET[0], otava_xy[i][1] + LABEL_OFFSET[1]),
                    fontsize=11, fontweight="bold", color=STAR_OTAVA, zorder=7)
    for i, row in fda_df.iterrows():
        if not fda_active_mask[i]:
            continue
        nm = str(row["_display"])
        if len(nm) > 28:
            nm = nm[:26] + "…"
        ax.annotate(nm,
                    xy=(fda_xy[i][0], fda_xy[i][1]),
                    xytext=(fda_xy[i][0] + LABEL_OFFSET[0], fda_xy[i][1] + LABEL_OFFSET[1]),
                    fontsize=11, fontweight="bold", color=STAR_FDA, zorder=7)

    ax.set_xlabel("UMAP-1", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP-2", fontsize=13, fontweight="bold")
    ax.set_title("Top OTAVA and FDA-derived hits in training chemical space\n"
                 "(Morgan radius-2, Jaccard metric)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#CDD5DF", alpha=0.6)
    ax.legend(loc="best", fontsize=10, frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "top_hits_umap.png", dpi=300)
    fig.savefig(OUT / "top_hits_umap.pdf")
    plt.close(fig)
    print(f"Wrote {OUT / 'top_hits_umap.png'}")

    # ---- OTAVA vs FDA Tanimoto matrix ----
    print("Computing Tanimoto similarity between the two top-10 sets ...")
    otava_bv = [morgan_bitvec(s) for s in otava_df["_smi"]]
    fda_bv   = [morgan_bitvec(s) for s in fda_df["_smi"]]
    tan = np.zeros((len(otava_bv), len(fda_bv)))
    for i, a in enumerate(otava_bv):
        for j, b in enumerate(fda_bv):
            tan[i, j] = DataStructs.TanimotoSimilarity(a, b)
    tan_df = pd.DataFrame(
        tan,
        index=[str(x) for x in otava_df["_label"]],
        columns=[str(x) for x in fda_df["_label"]],
    )
    tan_df.to_csv(OUT / "top_hits_tanimoto_otava_vs_fda.csv")
    print(f"Wrote {OUT / 'top_hits_tanimoto_otava_vs_fda.csv'}")

    # ---- per-hit summary with nearest-neighbor in the OTHER library ----
    summary_rows = []
    for i, row in otava_df.iterrows():
        j = int(np.argmax(tan[i, :]))
        summary_rows.append({
            "library": "OTAVA",
            "id": row["_label"],
            "display_name": str(row["_display"])[:60],
            "consensus_mean": row.get("consensus_A5_mean", np.nan),
            "status": row.get("status", ""),
            "ic50": row.get("ic50_label", ""),
            "nearest_in_other_library_id": str(fda_df["_label"].iloc[j]),
            "nearest_tanimoto": round(float(tan[i, j]), 3),
        })
    for j, row in fda_df.iterrows():
        i = int(np.argmax(tan[:, j]))
        summary_rows.append({
            "library": "FDA-derived",
            "id": row["_label"],
            "display_name": str(row["_display"])[:60],
            "consensus_mean": row.get("consensus_A5_mean", np.nan),
            "status": row.get("status", ""),
            "ic50": row.get("ic50_label", ""),
            "nearest_in_other_library_id": str(otava_df["_label"].iloc[i]),
            "nearest_tanimoto": round(float(tan[i, j]), 3),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "top_hits_summary.csv", index=False)
    print(f"Wrote {OUT / 'top_hits_summary.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
