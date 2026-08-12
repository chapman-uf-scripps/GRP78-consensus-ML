"""
Chemical-space visualization of the triple-consensus predicted-active sets
overlaid on the training library, for both external libraries.

Note: on Windows consoles (cp1252) print() of Unicode chars like the
ge-sign fails; this is handled by reconfiguring stdout below.

Two-panel figure:
  Panel A : training library (1488, grey) + OTAVA triple-consensus predicted
            actives (≥38% by all three models) coloured by consensus_A5_mean
            + 4 OTAVA confirmed actives (large stars, labelled).
  Panel B : training library (1488, grey) + FDA-GSRS triple-consensus
            predicted actives coloured by consensus_A5_mean + 1 FDA confirmed
            active (large star, labelled).

Both panels share the same joint UMAP coordinates (UMAP fitted on the union
of training + OTAVA-consensus + FDA-consensus), so the training cloud sits
in the same place in both panels and the two libraries are directly
comparable.

Output:
  output/main_figures/Figure_external_libraries_chemspace.png/.pdf
  output/source_tables/figure_external_libraries_chemspace_coords.csv
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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
OUT_DIR = ROOT / "output" / "main_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = ROOT / "output" / "source_tables"

TRAIN_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
OTAVA_PRED = ROOT / "output" / "full_pipeline_output" / "04_screen_otava_best3" / "otava_bayesian_top3_extra_catboost_gnn_predictions.csv"
FDA_PRED = ROOT / "output" / "full_pipeline_output" / "11c_screen_fda_kevin_docking_corrected_20260605" / "fda_bayesian_top3_extra_catboost_gnn_predictions.csv"
OTAVA_TOP10 = ROOT / "output" / "source_tables" / "figure5b_otava_top10_consensus_ranking.csv"

MORGAN_RADIUS = 2
MORGAN_BITS   = 1024
RANDOM_STATE  = 42
ACTIVE_THRESHOLD = 38.0

OTAVA_CONFIRMED_IDS = ["P2195430", "P2195267", "P2195222", "P1157904"]
FDA_CONFIRMED_CID   = 73951493  # Desisobutyl-n-butyl Bortezomib


def morgan_row(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS),
        arr)
    return arr


def safe_morgan_stack(smiles_iter):
    rows, idx = [], []
    for i, s in enumerate(smiles_iter):
        arr = morgan_row(s)
        if arr is not None:
            rows.append(arr)
            idx.append(i)
    return np.stack(rows), np.array(idx)


def canonical_smiles(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def find_overlap(set_a_smiles, set_b_smiles, name_a, name_b):
    """Return (n_overlap, overlap_canonical_smiles list)."""
    canon_a = set(s for s in (canonical_smiles(x) for x in set_a_smiles) if s)
    canon_b = set(s for s in (canonical_smiles(x) for x in set_b_smiles) if s)
    overlap = canon_a & canon_b
    print(f"  Overlap {name_a} ∩ {name_b}: {len(overlap)} compounds "
          f"(unique-canonical: {len(canon_a)} vs {len(canon_b)})")
    return overlap


def main():
    # ---------- training set ----------
    print("Loading training library …")
    train = pd.read_csv(TRAIN_CSV, usecols=["Name", "SMILES", "A5"])
    train_fp, _ = safe_morgan_stack(train["SMILES"])
    print(f"  training: {len(train)} compounds")

    # ---------- OTAVA predictions ----------
    print("Loading OTAVA predictions …")
    otava = pd.read_csv(OTAVA_PRED)
    # triple-consensus ≥38% (all three models predict ≥38)
    otava["models_gt_38"] = (
        (otava["pred_A5_ExtraTrees_clipped"] >= ACTIVE_THRESHOLD).astype(int)
        + (otava["pred_A5_CatBoost_clipped"]   >= ACTIVE_THRESHOLD).astype(int)
        + (otava["pred_A5_GNN_clipped"]        >= ACTIVE_THRESHOLD).astype(int)
    )
    otava_consensus = otava[otava["models_gt_38"] == 3].copy()
    print(f"  OTAVA triple-consensus ≥38%: {len(otava_consensus)} compounds")

    # ---------- FDA predictions ----------
    print("Loading FDA-GSRS predictions …")
    fda = pd.read_csv(FDA_PRED)
    fda_consensus = fda[fda["models_pred_gt_38"] == 3].copy()
    print(f"  FDA-GSRS triple-consensus ≥38%: {len(fda_consensus)} compounds")

    # ---------- fingerprints for OTAVA-consensus and FDA-consensus ----------
    print("Computing Morgan fingerprints …")
    otava_smiles = otava_consensus["SMILES"].astype(str).tolist()
    otava_fp, otava_keep = safe_morgan_stack(otava_smiles)
    otava_consensus = otava_consensus.iloc[otava_keep].reset_index(drop=True)
    print(f"  OTAVA-consensus valid SMILES: {len(otava_consensus)}")

    fda_smiles = fda_consensus["canonical_smiles"].astype(str).tolist()
    fda_fp, fda_keep = safe_morgan_stack(fda_smiles)
    fda_consensus = fda_consensus.iloc[fda_keep].reset_index(drop=True)
    print(f"  FDA-consensus valid SMILES: {len(fda_consensus)}")

    # ---------- check for overlap between training and screening libraries ----------
    print("\nChecking SMILES overlap between training library and screening libraries…")
    overlap_train_otava = find_overlap(
        train["SMILES"], otava_consensus["SMILES"], "Training", "OTAVA-consensus")
    overlap_train_fda = find_overlap(
        train["SMILES"], fda_consensus["canonical_smiles"], "Training", "FDA-consensus")
    if overlap_train_otava:
        print(f"    OTAVA overlap SMILES (first 3): {list(overlap_train_otava)[:3]}")
    if overlap_train_fda:
        print(f"    FDA overlap SMILES (first 3): {list(overlap_train_fda)[:3]}")
    # also check full (un-filtered) OTAVA / FDA against training
    overlap_train_otava_full = find_overlap(
        train["SMILES"], otava["SMILES"], "Training", "OTAVA-full")
    overlap_train_fda_full = find_overlap(
        train["SMILES"], fda["canonical_smiles"], "Training", "FDA-full")
    print()

    # ---------- joint UMAP ----------
    print("Running joint UMAP (training + OTAVA-consensus + FDA-consensus) …")
    X = np.vstack([train_fp, otava_fp, fda_fp]).astype(bool)
    reducer = umap.UMAP(
        n_neighbors=35, min_dist=0.12, metric="jaccard",
        random_state=RANDOM_STATE, low_memory=True,
    )
    xy = reducer.fit_transform(X)

    n_train  = train_fp.shape[0]
    n_otava  = otava_fp.shape[0]
    train_xy = xy[:n_train]
    otava_xy = xy[n_train:n_train + n_otava]
    fda_xy   = xy[n_train + n_otava:]

    # ---------- confirmed-active masks ----------
    otava_conf_mask = otava_consensus["COMPOUND ID"].astype(str).isin(OTAVA_CONFIRMED_IDS).values
    fda_conf_mask   = (fda_consensus["Compound_CID"] == FDA_CONFIRMED_CID).values
    print(f"  OTAVA confirmed found in consensus set: {otava_conf_mask.sum()}/{len(OTAVA_CONFIRMED_IDS)}")
    print(f"  FDA confirmed found in consensus set:   {fda_conf_mask.sum()}/1")

    # ---------- common plot bounds ----------
    x_min, x_max = xy[:, 0].min() - 0.5, xy[:, 0].max() + 0.5
    y_min, y_max = xy[:, 1].min() - 0.5, xy[:, 1].max() + 0.5

    # ---------- plot ----------
    print("Plotting …")
    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2),
                             gridspec_kw={"wspace": 0.20, "right": 0.90})

    CMAP_TRAIN = "Reds"      # measured GRP78 inhibition (training)
    CMAP_EXT   = "viridis"   # predicted GRP78 inhibition (screening)
    train_A5   = train["A5"].values

    LABEL_OFFSETS_OTAVA = {
        "P2195430": ( 0.40,  0.50),
        "P2195267": ( 0.40, -0.40),
        "P2195222": ( 0.40, -0.90),
        "P1157904": (-1.40,  0.50),
    }
    LABEL_OFFSETS_FDA = {
        "73951493": (-3.0, 0.55),  # nudge Bortezomib label to the LEFT (it sits near right edge)
    }

    def plot_panel(ax, ext_xy, ext_consensus_col, conf_mask, conf_label_col,
                   confirmed_color, lib_name, panel_label,
                   label_offsets, label_text_map=None):
        # external consensus — coloured by PREDICTED inhibition (viridis), in back
        sc = ax.scatter(ext_xy[:, 0], ext_xy[:, 1],
                        s=42, c=ext_consensus_col, cmap=CMAP_EXT,
                        edgecolor="white", linewidth=0.4, alpha=0.85, zorder=2,
                        vmin=38, vmax=ext_consensus_col.max(),
                        label=f"{lib_name} predicted (consensus ≥ 38%)")

        # training library on TOP — coloured by MEASURED inhibition (red colormap)
        # smaller dots so screening behind is partially visible where there's no
        # training compound at the same point
        sc_train = ax.scatter(train_xy[:, 0], train_xy[:, 1],
                              s=10, c=train_A5, cmap=CMAP_TRAIN,
                              edgecolor="none", alpha=0.85, zorder=4,
                              vmin=0, vmax=100,
                              label=f"Training library (n = {n_train})")

        # confirmed actives — stars
        if conf_mask.any():
            ax.scatter(ext_xy[conf_mask, 0], ext_xy[conf_mask, 1],
                       s=520, c=confirmed_color, edgecolor="black", linewidth=2.0,
                       marker="*", zorder=7,
                       label=f"{lib_name} confirmed active")

            # label confirmed with per-hit offsets to avoid overlap
            for i in np.where(conf_mask)[0]:
                raw_label = str(conf_label_col[i])
                disp = label_text_map.get(raw_label, raw_label) if label_text_map else raw_label
                dx, dy = label_offsets.get(raw_label, (0.30, 0.30))
                ax.annotate(disp,
                            xy=(ext_xy[i, 0], ext_xy[i, 1]),
                            xytext=(ext_xy[i, 0] + dx, ext_xy[i, 1] + dy),
                            fontsize=12, fontweight="bold", color="black", zorder=8)

        # cosmetics
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("UMAP-1", fontsize=15, fontweight="bold")
        ax.set_ylabel("UMAP-2", fontsize=15, fontweight="bold")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_linewidth(2.5)
            ax.spines[spine].set_color("#111111")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=11, width=2.0, length=6)
        ax.grid(True, linestyle=":", linewidth=0.5, color="#9aa3ad", alpha=0.5)
        ax.text(-0.04, 1.02, panel_label, transform=ax.transAxes,
                fontsize=24, fontweight="bold", va="bottom", ha="left")
        return sc, sc_train

    # Panel A — OTAVA
    sc_a, sc_train_a = plot_panel(
        axes[0],
        otava_xy,
        otava_consensus["consensus_A5_mean"].values,
        otava_conf_mask,
        otava_consensus["COMPOUND ID"].astype(str).values,
        confirmed_color="#159A9C",
        lib_name="OTAVA",
        panel_label="A",
        label_offsets=LABEL_OFFSETS_OTAVA,
    )

    # Panel B — FDA-GSRS
    sc_b, sc_train_b = plot_panel(
        axes[1],
        fda_xy,
        fda_consensus["consensus_A5_mean"].values,
        fda_conf_mask,
        fda_consensus["Compound_CID"].astype(str).values,  # use CID as key for offset map
        confirmed_color="#D99A25",
        lib_name="FDA GSRS",
        panel_label="B",
        label_offsets=LABEL_OFFSETS_FDA,
        label_text_map={"73951493": "Desisobutyl-n-butyl\nBortezomib"},
    )

    # two side-by-side colourbars on the right
    cbar_pred = fig.colorbar(sc_b, ax=axes, orientation="vertical",
                             fraction=0.024, pad=0.02)
    cbar_pred.set_label("Predicted GRP78 inhibition (screening libraries, %)",
                        fontsize=11, fontweight="bold")
    cbar_pred.ax.tick_params(labelsize=10)

    cbar_train = fig.colorbar(sc_train_b, ax=axes, orientation="vertical",
                              fraction=0.024, pad=0.06)
    cbar_train.set_label("Measured GRP78 inhibition (training library, %)",
                         fontsize=11, fontweight="bold")
    cbar_train.ax.tick_params(labelsize=10)

    # legend block in the lower-left of each panel for clarity
    for ax in axes:
        ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.95)

    out_png = OUT_DIR / "Figure_external_libraries_chemspace.png"
    out_pdf = OUT_DIR / "Figure_external_libraries_chemspace.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")

    # ---------- source CSV ----------
    rows = []
    for i, r in enumerate(train.itertuples(index=False)):
        rows.append({"library": "Training", "id": r.Name, "smiles": r.SMILES,
                     "A5": float(r.A5),
                     "consensus_mean": np.nan, "is_confirmed": 0,
                     "umap_x": float(train_xy[i, 0]), "umap_y": float(train_xy[i, 1])})
    for i, r in otava_consensus.reset_index(drop=True).iterrows():
        rows.append({"library": "OTAVA", "id": str(r["COMPOUND ID"]),
                     "smiles": str(r.get("SMILES", "")),
                     "A5": np.nan,
                     "consensus_mean": float(r["consensus_A5_mean"]),
                     "is_confirmed": int(bool(otava_conf_mask[i])),
                     "umap_x": float(otava_xy[i, 0]), "umap_y": float(otava_xy[i, 1])})
    for i, r in fda_consensus.reset_index(drop=True).iterrows():
        rows.append({"library": "FDA-GSRS",
                     "id": str(r.get("Compound_CID", "")),
                     "smiles": str(r.get("canonical_smiles", "")),
                     "A5": np.nan,
                     "consensus_mean": float(r["consensus_A5_mean"]),
                     "is_confirmed": int(bool(fda_conf_mask[i])),
                     "umap_x": float(fda_xy[i, 0]), "umap_y": float(fda_xy[i, 1])})
    src_csv = SRC_DIR / "figure_external_libraries_chemspace_coords.csv"
    pd.DataFrame(rows).to_csv(src_csv, index=False)
    print(f"Wrote {src_csv}")


if __name__ == "__main__":
    main()
