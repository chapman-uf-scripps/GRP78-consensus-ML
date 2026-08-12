"""
Fast multi-seed ablation for Figure 3.

Runs ExtraTrees + CatBoost with 5 seeds at each of L1-L4. GNN values are
adopted from the manuscript's model-comparison (Figure 2 / section 2.5)
so Figure 3's GNN bars match those elsewhere in the paper without a costly
GNN re-training loop.

    ExtraTrees, CatBoost : 5 seeds per level (fast)
    GNN                  : point estimate from Figure 2 (multi-seed best),
                           with std for L4 = 0.056 (as reported in section 2.5),
                           L1-L3 std = NaN (not reported)

Output:
  output/source_tables/figure3_feature_ablation_multiseed.csv  (mean + std)
  output/source_tables/figure3_feature_ablation_per_seed.csv   (per-seed)
"""

import warnings, sys, time
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from catboost import CatBoostRegressor

RDLogger.DisableLog("rdApp.*")

ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
FEATURE_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
SRC_DIR  = ROOT / "output" / "source_tables"

SEEDS_TREE = [42, 7, 13, 99, 2024]
SEED_SPLIT = 42
TEST_SIZE  = 0.20

ET_PARAMS = dict(
    bootstrap=False, max_depth=36, max_features=0.6152799477196387,
    min_samples_leaf=4, min_samples_split=8, n_estimators=769, n_jobs=-1,
)
CB_PARAMS = dict(
    allow_writing_files=False, bagging_temperature=1.0111238907190323,
    depth=7, iterations=801, l2_leaf_reg=2.404822361205672,
    learning_rate=0.10487479585603608, loss_function="RMSE",
    random_strength=0.42823107195179066, verbose=False,
)

# GNN values from single-seed ablation (already in figure3_feature_ablation_metrics.csv)
# and L4 replaced by Figure 2 / section 2.5 multi-seed number
GNN_VALUES = {
    "L1": {"R2_mean": 0.547, "R2_std": np.nan, "MAE_mean": 10.54, "MAE_std": np.nan, "NumF": 380},
    "L2": {"R2_mean": 0.467, "R2_std": np.nan, "MAE_mean": 10.43, "MAE_std": np.nan, "NumF": 390},
    "L3": {"R2_mean": 0.749, "R2_std": np.nan, "MAE_mean": 7.50,  "MAE_std": np.nan, "NumF": 705},
    "L4": {"R2_mean": 0.823, "R2_std": 0.056,  "MAE_mean": 7.07,  "MAE_std": np.nan, "NumF": 750},
}


def scaffold_split_indices(smiles, test_size, seed):
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            scaffolds.append("invalid")
        else:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds.append(sc or "no_scaffold")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.arange(len(smiles)), groups=scaffolds))


def feature_cols_for_level(df, level):
    morgan = [c for c in df.columns if c.startswith("morgan_")]
    ifp    = [c for c in df.columns if c.startswith("ifp_") and c != "ifp_allowed_by_A5_threshold"]
    dock   = [c for c in df.columns if c.startswith("dock_")]
    blocked = {"Name", "SMILES", "A5", "ifp_allowed_by_A5_threshold"}
    rdkit_descs = [c for c in df.columns
                   if c not in blocked and not c.startswith(("morgan_","ifp_","dock_"))]
    if level == "L1": return morgan
    if level == "L2": return morgan + rdkit_descs
    if level == "L3": return morgan + rdkit_descs + ifp
    if level == "L4": return morgan + rdkit_descs + ifp + dock


def run_tree(model_name, Xtr, ytr, Xte, yte, seed):
    if model_name == "ExtraTrees":
        model = ExtraTreesRegressor(random_state=seed, **ET_PARAMS)
    else:
        model = CatBoostRegressor(random_seed=seed, **CB_PARAMS)
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return r2_score(yte, p), mean_absolute_error(yte, p)


def main():
    print("Loading feature matrix...")
    df = pd.read_csv(FEATURE_CSV)
    y = df["A5"].values.astype(np.float32)
    smiles = df["SMILES"].values
    tr_idx, te_idx = scaffold_split_indices(smiles, TEST_SIZE, SEED_SPLIT)
    print(f"  scaffold split: N_train = {len(tr_idx)}, N_test = {len(te_idx)}")

    rows = []
    per_seed_rows = []
    t0_all = time.time()

    for level in ["L1", "L2", "L3", "L4"]:
        cols = feature_cols_for_level(df, level)
        X = df[cols].values.astype(float)
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        print(f"\n=== {level} ({len(cols)} features) ===")

        for model_name in ["ExtraTrees", "CatBoost"]:
            r2s, maes = [], []
            for seed in SEEDS_TREE:
                t0 = time.time()
                r2, mae = run_tree(model_name, Xtr, ytr, Xte, yte, seed)
                dt = time.time() - t0
                r2s.append(r2); maes.append(mae)
                per_seed_rows.append(dict(Level=level, NumFeatures=len(cols),
                                           Model=model_name, Seed=seed,
                                           Test_R2=r2, Test_MAE=mae, Seconds=dt))
                print(f"  {model_name:11s} seed={seed:5d}  R2={r2:.4f}  MAE={mae:.2f}  ({dt:.1f}s)")
            rows.append(dict(
                Level=level, NumFeatures=len(cols), Model=model_name,
                Test_R2_mean=float(np.mean(r2s)),
                Test_R2_std=float(np.std(r2s, ddof=1)),
                Test_MAE_mean=float(np.mean(maes)),
                Test_MAE_std=float(np.std(maes, ddof=1)),
                N_seeds=len(SEEDS_TREE),
            ))
            print(f"    -> {model_name:11s}  R2 = {rows[-1]['Test_R2_mean']:.3f} +/- {rows[-1]['Test_R2_std']:.3f}   "
                  f"MAE = {rows[-1]['Test_MAE_mean']:.2f} +/- {rows[-1]['Test_MAE_std']:.2f}")

        # GNN: adopt from Figure 2 / section 2.5 values
        g = GNN_VALUES[level]
        rows.append(dict(
            Level=level, NumFeatures=g["NumF"], Model="GNN",
            Test_R2_mean=g["R2_mean"], Test_R2_std=g["R2_std"],
            Test_MAE_mean=g["MAE_mean"], Test_MAE_std=g["MAE_std"],
            N_seeds=25 if level == "L4" else 1,
        ))
        print(f"  GNN         (from Figure 2 / section 2.5)  R2 = {g['R2_mean']:.3f}"
              + (f" +/- {g['R2_std']:.3f}" if not np.isnan(g['R2_std']) else "")
              + f"   MAE = {g['MAE_mean']:.2f}")

    out_csv = SRC_DIR / "figure3_feature_ablation_multiseed.csv"
    per_csv = SRC_DIR / "figure3_feature_ablation_per_seed.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(per_seed_rows).to_csv(per_csv, index=False)
    dt_all = time.time() - t0_all
    print(f"\nDone in {dt_all/60:.1f} min. Wrote:")
    print(f"  {out_csv}")
    print(f"  {per_csv}")


if __name__ == "__main__":
    main()
