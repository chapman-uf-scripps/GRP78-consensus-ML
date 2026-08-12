"""
Quantify cluster-level structural similarity between the training library
and the two screened libraries' triple-consensus predicted-active sets.

For each screened predicted-active compound (OTAVA 369 and FDA-GSRS 534),
compute:
  (i)  the Tanimoto similarity to its single NEAREST training compound, and
  (ii) the measured GRP78 inhibition (A5) of that nearest training compound.

Report distribution statistics and stratification (how many sit near a
training-ACTIVE vs a training-INACTIVE).

Outputs:
  output/source_tables/cluster_similarity_train_vs_screened.csv
  output/main_figures/Figure_cluster_similarity_train_vs_screened.png/.pdf
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


ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
OUT_FIG = ROOT / "output" / "main_figures"
OUT_SRC = ROOT / "output" / "source_tables"

TRAIN_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
OTAVA_PRED = ROOT / "output" / "full_pipeline_output" / "04_screen_otava_best3" / "otava_bayesian_top3_extra_catboost_gnn_predictions.csv"
FDA_PRED = ROOT / "output" / "full_pipeline_output" / "11c_screen_fda_kevin_docking_corrected_20260605" / "fda_bayesian_top3_extra_catboost_gnn_predictions.csv"

MORGAN_RADIUS = 2
MORGAN_BITS   = 1024
ACTIVE_T      = 38.0


def morgan_bv(smi):
    mol = Chem.MolFromSmiles(str(smi))
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


def main():
    print("Loading training library …")
    train = pd.read_csv(TRAIN_CSV, usecols=["Name", "SMILES", "A5"])
    train_bv = [morgan_bv(s) for s in train["SMILES"]]
    train_A5 = train["A5"].values
    train_is_active = (train_A5 >= ACTIVE_T).astype(int)
    print(f"  training: n={len(train)}  active(>=38%)={int(train_is_active.sum())}")

    print("Loading OTAVA-consensus 369 …")
    otava = pd.read_csv(OTAVA_PRED)
    otava["models_gt_38"] = (
        (otava["pred_A5_ExtraTrees_clipped"] >= ACTIVE_T).astype(int)
        + (otava["pred_A5_CatBoost_clipped"]   >= ACTIVE_T).astype(int)
        + (otava["pred_A5_GNN_clipped"]        >= ACTIVE_T).astype(int)
    )
    otava_c = otava[otava["models_gt_38"] == 3].reset_index(drop=True)
    otava_bv = [morgan_bv(s) for s in otava_c["SMILES"]]
    print(f"  OTAVA-consensus: n={len(otava_c)}")

    print("Loading FDA-consensus 534 …")
    fda = pd.read_csv(FDA_PRED)
    fda_c = fda[fda["models_pred_gt_38"] == 3].reset_index(drop=True)
    fda_bv = [morgan_bv(s) for s in fda_c["canonical_smiles"]]
    print(f"  FDA-consensus: n={len(fda_c)}")

    # ---- compute nearest-training-neighbor for each screened compound ----
    def nearest_train(query_bv_list, source_name):
        rows = []
        for i, bv in enumerate(query_bv_list):
            if bv is None:
                continue
            sims = np.array([DataStructs.TanimotoSimilarity(bv, t) for t in train_bv])
            j = int(np.argmax(sims))
            rows.append({
                "library": source_name,
                "ext_index": i,
                "nearest_train_idx": j,
                "nearest_train_id": train["Name"].iloc[j],
                "nearest_train_A5": float(train_A5[j]),
                "nearest_train_is_active": int(train_is_active[j]),
                "nearest_tanimoto": float(sims[j]),
            })
        return pd.DataFrame(rows)

    print("Computing nearest training neighbor for each predicted compound …")
    otava_nn = nearest_train(otava_bv, "OTAVA")
    fda_nn   = nearest_train(fda_bv,   "FDA-GSRS")

    summary_rows = []
    for name, df in (("OTAVA", otava_nn), ("FDA-GSRS", fda_nn)):
        T = df["nearest_tanimoto"].values
        a = df["nearest_train_is_active"].values
        summary_rows.append({
            "library": name,
            "n": len(df),
            "tanimoto_mean":   round(T.mean(), 3),
            "tanimoto_median": round(np.median(T), 3),
            "tanimoto_p25":    round(np.percentile(T, 25), 3),
            "tanimoto_p75":    round(np.percentile(T, 75), 3),
            "tanimoto_min":    round(T.min(), 3),
            "tanimoto_max":    round(T.max(), 3),
            "n_T_ge_0.6":      int((T >= 0.6).sum()),
            "n_T_0.4_to_0.6":  int(((T >= 0.4) & (T < 0.6)).sum()),
            "n_T_0.3_to_0.4":  int(((T >= 0.3) & (T < 0.4)).sum()),
            "n_T_lt_0.3":      int((T < 0.3).sum()),
            "%_near_active_train(T>=0.4)": round(
                100 * ((T >= 0.4) & (a == 1)).sum() / len(df), 1),
            "%_near_inactive_train(T>=0.4)": round(
                100 * ((T >= 0.4) & (a == 0)).sum() / len(df), 1),
            "%_T_lt_0.3(novel-ish)": round(
                100 * (T < 0.3).sum() / len(df), 1),
        })
    summary = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    # Save raw data + summary
    src = OUT_SRC / "cluster_similarity_train_vs_screened.csv"
    pd.concat([otava_nn, fda_nn], ignore_index=True).to_csv(src, index=False)
    summary.to_csv(OUT_SRC / "cluster_similarity_train_vs_screened_summary.csv", index=False)
    print(f"\nWrote {src}")
    print(f"Wrote {OUT_SRC / 'cluster_similarity_train_vs_screened_summary.csv'}")

    # ---- plot: histogram of nearest-training Tanimoto for both libraries ----
    print("\nPlotting …")
    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5),
                             gridspec_kw={"wspace": 0.25})

    OTAVA_COLOR = "#159A9C"
    FDA_COLOR   = "#D99A25"

    # (A) histogram of nearest Tanimoto, OTAVA vs FDA on same axes
    bins = np.linspace(0, 1.0, 41)
    axes[0].hist(otava_nn["nearest_tanimoto"], bins=bins, alpha=0.65,
                 color=OTAVA_COLOR, edgecolor="white",
                 label=f"OTAVA-consensus (n = {len(otava_nn)})")
    axes[0].hist(fda_nn["nearest_tanimoto"], bins=bins, alpha=0.65,
                 color=FDA_COLOR, edgecolor="white",
                 label=f"FDA GSRS-consensus (n = {len(fda_nn)})")
    axes[0].axvline(0.3, color="#5a6573", linestyle="--", linewidth=1.0)
    axes[0].axvline(0.4, color="#5a6573", linestyle="--", linewidth=1.0)
    axes[0].text(0.30, axes[0].get_ylim()[1] * 0.97, "novel",
                 fontsize=10, ha="right", va="top", color="#5a6573", fontweight="bold")
    axes[0].text(0.55, axes[0].get_ylim()[1] * 0.97, "neighborhood-adjacent",
                 fontsize=10, ha="center", va="top", color="#5a6573", fontweight="bold")
    axes[0].set_xlabel("Tanimoto similarity to nearest training compound",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of predicted hits", fontsize=13, fontweight="bold")
    axes[0].text(-0.10, 1.03, "A", transform=axes[0].transAxes,
                 fontsize=22, fontweight="bold", va="bottom")
    for sp in ("bottom", "left"):
        axes[0].spines[sp].set_linewidth(2.0)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(labelsize=11, width=1.6, length=6)
    axes[0].legend(loc="upper right", fontsize=10, frameon=True)
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(True, linestyle=":", linewidth=0.5, color="#9aa3ad", alpha=0.5)

    # (B) for each predicted hit: scatter of (nearest Tanimoto, nearest A5)
    axes[1].axhline(ACTIVE_T, color="#5a6573", linestyle="--", linewidth=1.2)
    axes[1].text(0.02, ACTIVE_T + 2, "training-active threshold (38%)",
                 fontsize=9, color="#5a6573", fontweight="bold")
    axes[1].scatter(otava_nn["nearest_tanimoto"], otava_nn["nearest_train_A5"],
                    s=22, c=OTAVA_COLOR, edgecolor="white", linewidth=0.4,
                    alpha=0.75, label=f"OTAVA-consensus")
    axes[1].scatter(fda_nn["nearest_tanimoto"], fda_nn["nearest_train_A5"],
                    s=22, c=FDA_COLOR, edgecolor="white", linewidth=0.4,
                    alpha=0.75, label=f"FDA GSRS-consensus")
    axes[1].set_xlabel("Tanimoto to nearest training compound",
                       fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Nearest training compound's GRP78 inhibition (%)",
                       fontsize=13, fontweight="bold")
    axes[1].text(-0.10, 1.03, "B", transform=axes[1].transAxes,
                 fontsize=22, fontweight="bold", va="bottom")
    for sp in ("bottom", "left"):
        axes[1].spines[sp].set_linewidth(2.0)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(labelsize=11, width=1.6, length=6)
    axes[1].legend(loc="lower right", fontsize=10, frameon=True)
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(True, linestyle=":", linewidth=0.5, color="#9aa3ad", alpha=0.5)

    fig.tight_layout()
    out_png = OUT_FIG / "Figure_cluster_similarity_train_vs_screened.png"
    out_pdf = OUT_FIG / "Figure_cluster_similarity_train_vs_screened.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
