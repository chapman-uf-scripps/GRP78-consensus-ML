"""
Generate Figure 3 (feature-class ablation) source data + bar plot.

Runs L1-L4 ablation for Extra Trees and CatBoost using the SAME BO-optimized
hyperparameters reported in source_all_model_comparison_metrics.csv, the SAME
scaffold split as the rest of the pipeline (Bemis-Murcko via rdkit), and the
SAME precomputed feature matrix that produced the manuscript's L4 numbers.

Feature levels:
    L1: Morgan only                                    (380 cols)
    L2: Morgan + RDKit descriptors                     (390 cols)
    L3: Morgan + RDKit + active-filtered IFP           (706 cols)
    L4: Morgan + RDKit + IFP + Glide docking (full)    (751 cols)

GNN ablation is left as a follow-up because PyTorch + PyTorch Geometric
were not installed in this environment; the GNN L4 number (Test R2 = 0.788)
is re-used from the main pipeline so the GNN bars in the plot show only L4.
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostRegressor

RDLogger.DisableLog("rdApp.*")

ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
FEATURE_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
OUT_DIR = ROOT / "output" / "main_figures"
SRC_DIR = ROOT / "output" / "source_tables"
SEED = 42
TEST_SIZE = 0.20  # exact pipeline setting (step04a-c); yields N_train=1361, N_test=127

# Hyperparameters from source_all_model_comparison_metrics.csv (Bayesian optimum on L4)
ET_PARAMS = dict(
    bootstrap=False,
    max_depth=36,
    max_features=0.6152799477196387,
    min_samples_leaf=4,
    min_samples_split=8,
    n_estimators=769,
    n_jobs=-1,
    random_state=SEED,
)
CB_PARAMS = dict(
    allow_writing_files=False,
    bagging_temperature=1.0111238907190323,
    depth=7,
    iterations=801,
    l2_leaf_reg=2.404822361205672,
    learning_rate=0.10487479585603608,
    loss_function="RMSE",
    random_seed=SEED,
    random_strength=0.42823107195179066,
    verbose=False,
)

# GNN L4 numbers re-used from the pipeline (no GNN re-train in this script)
GNN_L4 = {"Test_R2": 0.7881, "Test_MAE": 7.0697}


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
    other = [c for c in df.columns if c not in blocked and not c.startswith(("morgan_","ifp_","dock_"))]
    rdkit_descs = other  # the 10 non-prefixed numeric columns are RDKit descriptors
    if level == "L1":
        return morgan
    if level == "L2":
        return morgan + rdkit_descs
    if level == "L3":
        return morgan + rdkit_descs + ifp
    if level == "L4":
        return morgan + rdkit_descs + ifp + dock
    raise ValueError(level)


def train_eval(model_name, X_train, y_train, X_test, y_test):
    if model_name == "ExtraTrees":
        model = ExtraTreesRegressor(**ET_PARAMS)
    elif model_name == "CatBoost":
        model = CatBoostRegressor(**CB_PARAMS)
    else:
        raise ValueError(model_name)
    model.fit(X_train, y_train)
    p_train = model.predict(X_train)
    p_test  = model.predict(X_test)
    return dict(
        Train_R2 = r2_score(y_train, p_train),
        Test_R2  = r2_score(y_test,  p_test),
        Train_MAE= mean_absolute_error(y_train, p_train),
        Test_MAE = mean_absolute_error(y_test,  p_test),
    )


def main():
    print("Reading", FEATURE_CSV.name)
    df = pd.read_csv(FEATURE_CSV)
    y = df["A5"].values
    smiles = df["SMILES"].values

    print("Computing scaffold split...")
    tr_idx, te_idx = scaffold_split_indices(smiles, TEST_SIZE, SEED)
    print(f"  N_train = {len(tr_idx)}, N_test = {len(te_idx)}")

    results = []
    for level in ["L1", "L2", "L3", "L4"]:
        cols = feature_cols_for_level(df, level)
        X = df[cols].values.astype(float)
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        print(f"\n=== {level} ({len(cols)} features) ===")
        for m in ["ExtraTrees", "CatBoost"]:
            metrics = train_eval(m, Xtr, ytr, Xte, yte)
            row = dict(Level=level, NumFeatures=len(cols), Model=m, **metrics)
            print(f"  {m:11s}  Test R2 = {metrics['Test_R2']:.4f}  Test MAE = {metrics['Test_MAE']:.4f}")
            results.append(row)
        # GNN: re-use L4 from pipeline, leave L1-L3 as NaN
        if level == "L4":
            results.append(dict(Level=level, NumFeatures=len(cols), Model="GNN",
                                Train_R2=np.nan, Test_R2=GNN_L4["Test_R2"],
                                Train_MAE=np.nan, Test_MAE=GNN_L4["Test_MAE"]))

    out_df = pd.DataFrame(results)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SRC_DIR / "figure3_feature_ablation_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    print("\nWrote", out_csv)

    # --- Plot grouped bars ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False)
    levels = ["L1", "L2", "L3", "L4"]
    level_labels = ["L1\nMorgan", "L2\n+ RDKit", "L3\n+ IFP", "L4\n+ docking\n(full)"]
    models = ["ExtraTrees", "CatBoost", "GNN"]
    colors = {"ExtraTrees": "#4c72b0", "CatBoost": "#dd8452", "GNN": "#55a868"}

    width = 0.27
    x = np.arange(len(levels))

    def get_val(lv, m, col):
        sub = out_df[(out_df["Level"] == lv) & (out_df["Model"] == m)]
        if sub.empty:
            return np.nan
        return float(sub[col].iloc[0])

    # Test R2 panel
    ax = axes[0]
    for i, m in enumerate(models):
        vals = [get_val(lv, m, "Test_R2") for lv in levels]
        bars = ax.bar(x + (i-1)*width, vals, width, label=m, color=colors[m],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylabel("Scaffold-split Test R²")
    ax.set_title("Test R²")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)

    # Test MAE panel
    ax = axes[1]
    for i, m in enumerate(models):
        vals = [get_val(lv, m, "Test_MAE") for lv in levels]
        bars = ax.bar(x + (i-1)*width, vals, width, label=m, color=colors[m],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylabel("Scaffold-split Test MAE (percent inhibition)")
    ax.set_title("Test MAE")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("Figure 3. Feature-class ablation ladder (scaffold split)", fontsize=12, y=1.02)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "Figure_3_feature_ablation.png"
    out_pdf = OUT_DIR / "Figure_3_feature_ablation.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print("Wrote", out_png)
    print("Wrote", out_pdf)


if __name__ == "__main__":
    main()
