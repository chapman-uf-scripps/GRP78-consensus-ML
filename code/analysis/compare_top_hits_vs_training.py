"""
Compare OTAVA and FDA-derived top-10 picks against the 1,488-compound
training library: nearest-neighbor Tanimoto, nearest training compound's
measured GRP78 inhibition, and per-library distributions.

Outputs (in output/top_hits_chemical_space/):
  vs_training_per_hit.csv             one row per top hit, with top-3 training
                                      neighbors (id, A5, Tanimoto)
  vs_training_distribution.png/.pdf   two-panel figure:
                                        (left)  KDE+strip of max Tanimoto to
                                                training, OTAVA vs FDA hits
                                        (right) nearest-training-A5 vs
                                                nearest-Tanimoto, colored by
                                                library and starred for the
                                                5 confirmed actives
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


ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
OUT  = ROOT / "output" / "top_hits_chemical_space"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
OTAVA_CSV = ROOT / "output" / "source_tables" / "figure5b_otava_top10_consensus_ranking.csv"
FDA_CSV   = ROOT / "output" / "source_tables" / "figure7b_fda_top10_consensus_ranking.csv"

MORGAN_RADIUS = 2
MORGAN_BITS   = 1024
ACTIVE_T      = 38.0


def morgan_bv(smi):
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


def main():
    print("Loading training library ...")
    train = pd.read_csv(TRAIN_CSV, usecols=["Name", "SMILES", "A5"])
    train_bv = [morgan_bv(s) for s in train["SMILES"]]
    train["is_active"] = (train["A5"] >= ACTIVE_T).astype(int)
    print(f"  training: n={len(train)}  active(>=38%)={int(train['is_active'].sum())}")

    print("Loading OTAVA top-10 ...")
    otava = pd.read_csv(OTAVA_CSV)
    otava["_smi"]   = otava["SMILES"].astype(str)
    otava["_label"] = otava["COMPOUND ID"].astype(str)
    otava["_display"] = otava["_label"]
    otava_bv = [morgan_bv(s) for s in otava["_smi"]]

    print("Loading FDA top-10 ...")
    fda = pd.read_csv(FDA_CSV)
    smi_col = "canonical_smiles" if "canonical_smiles" in fda.columns else "SMILES"
    fda["_smi"]     = fda[smi_col].astype(str)
    fda["_label"]   = fda["Compound_CID"].astype(str)
    fda["_display"] = fda["Name"].astype(str)
    fda_bv = [morgan_bv(s) for s in fda["_smi"]]

    # ---- per-hit nearest-training-neighbor ----
    def neighbors_table(df, bv_list, lib_name):
        rows = []
        for i, bv in enumerate(bv_list):
            sims = np.array([DataStructs.TanimotoSimilarity(bv, t) for t in train_bv])
            top3 = np.argsort(-sims)[:3]
            r = {
                "library":         lib_name,
                "id":              df["_label"].iloc[i],
                "display":         str(df["_display"].iloc[i])[:60],
                "consensus_mean":  df.get("consensus_A5_mean",
                                          pd.Series([np.nan]*len(df))).iloc[i],
                "status":          str(df.get("status", pd.Series([""]*len(df))).iloc[i]),
                "ic50":            str(df.get("ic50_label", pd.Series([""]*len(df))).iloc[i]),
                "nearest_train_id":          train["Name"].iloc[top3[0]],
                "nearest_train_tanimoto":    round(float(sims[top3[0]]), 3),
                "nearest_train_A5":          round(float(train["A5"].iloc[top3[0]]), 1),
                "nearest_train_was_active":  int(train["is_active"].iloc[top3[0]]),
                "neighbor2_id":              train["Name"].iloc[top3[1]],
                "neighbor2_tanimoto":        round(float(sims[top3[1]]), 3),
                "neighbor2_A5":              round(float(train["A5"].iloc[top3[1]]), 1),
                "neighbor3_id":              train["Name"].iloc[top3[2]],
                "neighbor3_tanimoto":        round(float(sims[top3[2]]), 3),
                "neighbor3_A5":              round(float(train["A5"].iloc[top3[2]]), 1),
                # bonus: max similarity to an ACTIVE training compound only
                "max_tanimoto_to_active_train": round(
                    float(sims[train["is_active"].values == 1].max()), 3),
            }
            rows.append(r)
        return pd.DataFrame(rows)

    per_hit = pd.concat([
        neighbors_table(otava, otava_bv, "OTAVA"),
        neighbors_table(fda,   fda_bv,   "FDA-derived"),
    ], ignore_index=True)
    per_hit.to_csv(OUT / "vs_training_per_hit.csv", index=False)
    print(f"Wrote {OUT / 'vs_training_per_hit.csv'}")
    print(per_hit[["library", "id", "consensus_mean", "status",
                   "nearest_train_id", "nearest_train_tanimoto",
                   "nearest_train_A5", "max_tanimoto_to_active_train"]].to_string(index=False))

    # ---- two-panel figure ----
    confirmed = per_hit["status"].str.contains("Confirmed active", case=False, na=False)
    OTAVA_C, FDA_C = "#159A9C", "#D99A25"
    color = np.where(per_hit["library"] == "OTAVA", OTAVA_C, FDA_C)
    marker_size = np.where(confirmed, 320, 110)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.7], "wspace": 0.32})

    # --- left: distribution of max-Tanimoto-to-training-active ---
    for lib, c in (("OTAVA", OTAVA_C), ("FDA-derived", FDA_C)):
        vals = per_hit.loc[per_hit["library"] == lib, "max_tanimoto_to_active_train"].values
        # strip plot
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(vals))
        x_base = 0 if lib == "OTAVA" else 1
        is_active_hit = per_hit.loc[per_hit["library"] == lib, "status"].str.contains(
            "Confirmed active", case=False, na=False).values
        axL.scatter(np.full(len(vals), x_base) + jitter, vals,
                    s=np.where(is_active_hit, 260, 110),
                    c=c,
                    edgecolor=np.where(is_active_hit, "black", "white"),
                    linewidth=np.where(is_active_hit, 1.6, 1.0),
                    marker="o", alpha=0.95, zorder=3)
        # mean line
        axL.hlines(vals.mean(), x_base - 0.3, x_base + 0.3,
                   color="black", linewidth=2.2, zorder=4)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["OTAVA\ntop-10", "FDA-derived\ntop-10"],
                        fontsize=12, fontweight="bold")
    axL.set_ylabel("Max Tanimoto to an active training compound\n(Morgan radius-2)",
                   fontsize=12, fontweight="bold")
    axL.set_title("How close is each top hit to the training actives?",
                  fontsize=13, fontweight="bold")
    axL.grid(True, axis="y", linestyle=":", linewidth=0.6, color="#CDD5DF", alpha=0.7)
    axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)
    axL.set_ylim(0, 1.0)

    # --- right: nearest-training-A5 vs nearest-Tanimoto ---
    x = per_hit["nearest_train_tanimoto"].values
    y = per_hit["nearest_train_A5"].values
    axR.axhline(ACTIVE_T, color="#9aa3ad", linestyle="--", linewidth=1.2, zorder=1)
    axR.text(axR.get_xlim()[1] if axR.get_xlim()[1] > 0 else 1.0,
             ACTIVE_T + 1, "active threshold (38%)", fontsize=9, color="#5a6573",
             ha="right", va="bottom")
    for lib, c in (("OTAVA", OTAVA_C), ("FDA-derived", FDA_C)):
        mask = (per_hit["library"] == lib).values
        m_act = mask & confirmed.values
        m_not = mask & ~confirmed.values
        if m_not.any():
            axR.scatter(x[m_not], y[m_not], s=110, c=c, edgecolor="white",
                        linewidth=1.0, marker="o", zorder=3, label=f"{lib} (untested / inactive)")
        if m_act.any():
            axR.scatter(x[m_act], y[m_act], s=320, c=c, edgecolor="black",
                        linewidth=1.6, marker="*", zorder=5, label=f"{lib} confirmed active")
    # label confirmed actives — per-hit offsets to avoid overlap
    LABEL_OFFSETS = {
        "P2195430": (0.012,  1.2),
        "P2195267": (0.012,  2.5),
        "P2195222": (0.012, -3.0),
        "P1157904": (0.012,  1.2),
        "73951493": (-0.040, 2.2),   # Desisobutyl-n-butyl Bortezomib
    }
    for i, row in per_hit[confirmed].iterrows():
        nm = str(row["id"])[:18] if row["library"] == "OTAVA" else str(row["display"])[:22]
        dx, dy = LABEL_OFFSETS.get(str(row["id"]), (0.012, 1.2))
        axR.annotate(nm,
                     xy=(row["nearest_train_tanimoto"], row["nearest_train_A5"]),
                     xytext=(row["nearest_train_tanimoto"] + dx,
                             row["nearest_train_A5"] + dy),
                     fontsize=10, fontweight="bold",
                     color=OTAVA_C if row["library"] == "OTAVA" else FDA_C,
                     zorder=6)
    axR.set_xlabel("Tanimoto to nearest training compound", fontsize=12, fontweight="bold")
    axR.set_ylabel("Nearest training compound's GRP78 inhibition (%)",
                   fontsize=12, fontweight="bold")
    axR.set_title("Does the nearest training neighbor predict the hit?",
                  fontsize=13, fontweight="bold")
    axR.grid(True, linestyle=":", linewidth=0.6, color="#CDD5DF", alpha=0.7)
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)
    axR.legend(loc="best", fontsize=9, frameon=True, framealpha=0.95)

    axL.set_xlim(-0.6, 1.6)
    fig.suptitle("Top OTAVA and FDA-derived hits vs the 1,488-compound training library",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "vs_training_distribution.png", dpi=300)
    fig.savefig(OUT / "vs_training_distribution.pdf")
    plt.close(fig)
    print(f"Wrote {OUT / 'vs_training_distribution.png'}")


if __name__ == "__main__":
    main()
