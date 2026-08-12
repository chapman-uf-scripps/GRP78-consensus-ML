"""
Generate SI Table S3 (OTAVA top-50) and SI Table S4 (FDA-derived top-50).

Each table contains the top 50 compounds from its library that passed the
triple-consensus filter (all three models predicted GRP78 inhibition >= 38%),
sorted by mean consensus prediction (descending).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
SI   = ROOT / "output" / "supplementary_figures"
SI.mkdir(parents=True, exist_ok=True)
SRC  = ROOT / "output" / "source_tables"

# ---------------- Table S3: OTAVA top-50 ----------------
otava_pred_path = ROOT / "output" / "supplementary_figures" / "Table_S1_full_otava_top3_model_predictions.csv"
if not otava_pred_path.exists():
    # Fall back to the all-predictions table in the screening output
    otava_pred_path = ROOT / "output" / "full_pipeline_output" / "04_screen_otava_best3" \
                           / "otava_bayesian_top3_extra_catboost_gnn_predictions.csv"
otava = pd.read_csv(otava_pred_path)

# Identify the triple-consensus column (>= 38 in all three)
if "models_pred_gt_38" in otava.columns:
    otava_tc = otava[otava["models_pred_gt_38"] == 3].copy()
else:
    cols = ("pred_A5_ExtraTrees_clipped", "pred_A5_CatBoost_clipped", "pred_A5_GNN_clipped")
    otava_tc = otava[(otava[cols[0]] >= 38) & (otava[cols[1]] >= 38) & (otava[cols[2]] >= 38)].copy()

otava_tc = otava_tc.sort_values("consensus_A5_mean", ascending=False).head(50).reset_index(drop=True)
otava_tc.insert(0, "rank", range(1, len(otava_tc) + 1))

# Trim to the columns useful for an SI table
id_cols = [c for c in ["COMPOUND ID", "Name", "Compound_CID"] if c in otava_tc.columns]
plate_cols = [c for c in ["PLATE NUMBER", "RACK POSITION"] if c in otava_tc.columns]
keep = ["rank"] + id_cols + plate_cols + [
    "consensus_A5_mean",
    "pred_A5_ExtraTrees_clipped",
    "pred_A5_CatBoost_clipped",
    "pred_A5_GNN_clipped",
    "SMILES",
]
keep = [c for c in keep if c in otava_tc.columns]
otava_tc = otava_tc[keep].rename(columns={
    "consensus_A5_mean":          "Consensus predicted GRP78 inhibition (mean)",
    "pred_A5_ExtraTrees_clipped": "Predicted GRP78 inhibition (Extra Trees)",
    "pred_A5_CatBoost_clipped":   "Predicted GRP78 inhibition (CatBoost)",
    "pred_A5_GNN_clipped":        "Predicted GRP78 inhibition (GNN)",
    "COMPOUND ID":                "Compound ID",
    "PLATE NUMBER":               "Plate",
    "RACK POSITION":              "Rack Position",
})
for col in [c for c in otava_tc.columns if "Predicted" in c or "Consensus" in c]:
    otava_tc[col] = otava_tc[col].round(2)

out_s3 = SI / "Table_S3_otava_top50_triple_consensus.csv"
otava_tc.to_csv(out_s3, index=False)
print(f"Wrote {out_s3}  (rows = {len(otava_tc)})")

# ---------------- Table S4: FDA-derived top-50 ----------------
fda_pred_path = (
    ROOT / "output" / "full_pipeline_output"
         / "11c_screen_fda_kevin_docking_corrected_20260605"
         / "fda_bayesian_top3_extra_catboost_gnn_predictions.csv"
)
fda = pd.read_csv(fda_pred_path)
fda_tc = fda[fda["models_pred_gt_38"] == 3].copy()
fda_tc = fda_tc.sort_values("consensus_A5_mean", ascending=False).head(50).reset_index(drop=True)
fda_tc.insert(0, "rank", range(1, len(fda_tc) + 1))

keep_fda = ["rank", "Compound_CID", "Name",
            "consensus_A5_mean",
            "pred_A5_ExtraTrees_clipped",
            "pred_A5_CatBoost_clipped",
            "pred_A5_GNN_clipped",
            "canonical_smiles"]
keep_fda = [c for c in keep_fda if c in fda_tc.columns]
fda_tc = fda_tc[keep_fda].rename(columns={
    "Compound_CID":               "PubChem CID",
    "consensus_A5_mean":          "Consensus predicted GRP78 inhibition (mean)",
    "pred_A5_ExtraTrees_clipped": "Predicted GRP78 inhibition (Extra Trees)",
    "pred_A5_CatBoost_clipped":   "Predicted GRP78 inhibition (CatBoost)",
    "pred_A5_GNN_clipped":        "Predicted GRP78 inhibition (GNN)",
    "canonical_smiles":           "SMILES",
})
for col in [c for c in fda_tc.columns if "Predicted" in c or "Consensus" in c]:
    fda_tc[col] = fda_tc[col].round(2)

out_s4 = SI / "Table_S4_fda_top50_triple_consensus.csv"
fda_tc.to_csv(out_s4, index=False)
print(f"Wrote {out_s4}  (rows = {len(fda_tc)})")
