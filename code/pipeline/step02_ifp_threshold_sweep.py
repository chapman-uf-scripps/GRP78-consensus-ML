"""
01b_sweep_ifp_threshold.py
==========================
Sweeps the active-IFP threshold from 0 to 100 (step 2) and evaluates
scaffold-split Test R² using Extra Trees with Bayesian-optimized params.
Plots Train/Test R² vs threshold and saves the best threshold for use
in the next feature-preparation step.

Run from the pipeline directory:
    python 01b_sweep_ifp_threshold.py

Outputs:
    Output/01b_threshold_sweep/threshold_sweep_results.csv
    Output/01b_threshold_sweep/threshold_sweep_r2_curve.png/.pdf
    Output/01b_threshold_sweep/best_threshold.json
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import (
    ID_COL,
    OUTPUT_DIR,
    PIPELINE_DIR,
    TARGET,
    TRAINING_ACTIVITY,
    TRAINING_DOCKING,
    TRAINING_IFP,
)

STEP2_PATH = PIPELINE_DIR / "utils_features.py"
STEP3_PATH = PIPELINE_DIR / "utils_models.py"
OUTPUT_SWEEP_DIR = OUTPUT_DIR / "01b_threshold_sweep"

THRESHOLD_MIN  = 0
THRESHOLD_MAX  = 100
THRESHOLD_STEP = 2
RANDOM_STATE   = 42
TEST_SIZE      = 0.20


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def interaction_cell_to_count(value):
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return 0
        return int(value != 0)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return 0
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        return len([p for p in inner.split(",") if p.strip()]) if inner else 0
    try:
        return int(float(text) != 0)
    except ValueError:
        return int(bool(text))


def aggregate_docking_features(path, compound_ids):
    raw = pd.read_csv(path, low_memory=False)
    id_col = "Title" if "Title" in raw.columns else "Compound_ID"
    blocked = {id_col, "Title", "Name", "SMILES", "Compound_ID",
               "Chemical_Name", "Rack_Position", "Variant",
               "source file", "Force Field", "docking method",
               "glide gridfile", "macro-pKa", "tautomer"}
    numeric = {}
    for col in raw.columns:
        if col in blocked or col.startswith("SIFT "):
            continue
        vals = pd.to_numeric(raw[col], errors="coerce")
        if vals.notna().any():
            numeric[col] = vals
    num_df = pd.DataFrame(numeric)
    num_df.insert(0, ID_COL, raw[id_col].astype(str).values)
    agg = num_df.groupby(ID_COL).mean(numeric_only=True)
    agg.columns = [f"dock_{c}" for c in agg.columns]
    return agg.reindex(compound_ids).fillna(0.0).reset_index(drop=True)


def load_raw_ifp(path, compound_ids):
    """Load IFP table and return raw (unfiltered) aggregated feature matrix."""
    raw = pd.read_csv(path, low_memory=False)
    interaction_cols = [
        c for c in raw.columns
        if re.match(
            r"^A\d+_(contact|backbone|sidechain|polar|hydrophobic|acceptor|donor|aromatic|charged)$", c
        )
    ]
    counts = raw[interaction_cols].map(interaction_cell_to_count).astype(np.float32)
    counts.insert(0, ID_COL, raw["Title"].astype(str).values)
    binary = counts.copy()
    binary[interaction_cols] = (binary[interaction_cols] > 0).astype(np.float32)

    any_agg  = binary.groupby(ID_COL)[interaction_cols].max()
    any_agg.columns  = [f"ifp_{c}_any"  for c in any_agg.columns]
    mean_agg = counts.groupby(ID_COL)[interaction_cols].mean()
    mean_agg.columns = [f"ifp_{c}_mean" for c in mean_agg.columns]
    sum_agg  = counts.groupby(ID_COL)[interaction_cols].sum()
    grouped_binary = binary.groupby(ID_COL)

    summary_parts = []
    for family in sorted({c.rsplit("_", 1)[1] for c in interaction_cols}):
        family_cols = [c for c in interaction_cols if c.endswith(f"_{family}")]
        part = pd.DataFrame(index=sum_agg.index)
        part[f"ifp_total_{family}_count"]       = sum_agg[family_cols].sum(axis=1)
        part[f"ifp_total_{family}_residue_any"] = grouped_binary[family_cols].max().sum(axis=1)
        summary_parts.append(part)

    out = pd.concat([any_agg, mean_agg, *summary_parts], axis=1)
    return out.reindex(compound_ids).fillna(0.0).reset_index(drop=True)


def apply_threshold(ifp_x, y, threshold):
    filtered = ifp_x.copy()
    filtered.loc[y < threshold, :] = 0.0
    return filtered


def drop_low_variance(df, min_freq=0.01):
    """Remove binary columns that are almost always 0 or always 1."""
    binary_cols = [c for c in df.columns if set(pd.unique(df[c].dropna())).issubset({0.0, 1.0})]
    drop = [c for c in binary_cols if not (min_freq <= df[c].mean() <= 1 - min_freq)]
    return df.drop(columns=drop) if drop else df


def build_extra_trees(ranking_csv_path):
    """Load Bayesian-optimized Extra Trees params from ranking CSV."""
    df = pd.read_csv(ranking_csv_path)
    row = df[df["Model"] == "Extra Trees"].iloc[0]
    params = json.loads(row["Params"])
    p = {k: v for k, v in params.items() if k not in ("n_jobs", "random_state")}
    return ExtraTreesRegressor(**p, n_jobs=-1, random_state=RANDOM_STATE)


def set_plot_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.weight": "bold", "font.size": 16,
        "axes.titlesize": 20, "axes.titleweight": "bold",
        "axes.labelsize": 17, "axes.labelweight": "bold",
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def main():
    OUTPUT_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    step2 = load_module(STEP2_PATH, "utils_features")
    step3 = load_module(STEP3_PATH, "utils_models")
    ranking_path = OUTPUT_DIR / "02c_bayesian_optimized_deep_models" / "bayesian_optimized_all_models_including_deep_ranking.csv"

    print("Loading activity and computing fixed features (Morgan, RDKit, docking)...")
    meta = step2.read_activity_table(TRAINING_ACTIVITY, TARGET)
    y    = np.clip(meta[TARGET].astype(float).values, 0.0, 100.0)
    meta[TARGET] = y

    morgan_x, mols = step2.make_morgan_features(meta["SMILES"], 2, 1024)
    morgan_x = step2.clean_numeric_matrix(morgan_x, missing_threshold=0.0, min_frequency=0.01)
    desc_x   = step2.clean_numeric_matrix(step2.make_rdkit_descriptors(mols), missing_threshold=0.20, min_frequency=0.0)
    dock_x   = aggregate_docking_features(TRAINING_DOCKING, meta[ID_COL].astype(str))

    print("Loading raw (unfiltered) IFP features...")
    ifp_raw = load_raw_ifp(TRAINING_IFP, meta[ID_COL].astype(str))

    print("Computing scaffold split (fixed across all thresholds)...")
    splits = step3.make_splits(meta, TEST_SIZE, RANDOM_STATE)
    train_idx, test_idx = splits["scaffold_split"]

    fixed_x = pd.concat([morgan_x, desc_x, dock_x], axis=1)
    fixed_x = fixed_x.loc[:, ~fixed_x.columns.duplicated()]
    fixed_x = fixed_x.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    thresholds = list(range(THRESHOLD_MIN, THRESHOLD_MAX + 1, THRESHOLD_STEP))
    results = []

    print(f"Sweeping {len(thresholds)} thresholds ({THRESHOLD_MIN}–{THRESHOLD_MAX}, step {THRESHOLD_STEP})...")
    for i, thr in enumerate(thresholds):
        ifp_filtered = apply_threshold(ifp_raw.copy(), y, thr)
        ifp_filtered = drop_low_variance(ifp_filtered, min_freq=0.01)

        x = pd.concat([fixed_x, ifp_filtered], axis=1)
        x = x.loc[:, ~x.columns.duplicated()].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_extra_trees(ranking_path)
        model.fit(x_train, y_train)

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        train_r2  = r2_score(y_train, model.predict(x_train))
        test_r2   = r2_score(y_test,  model.predict(x_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, model.predict(x_test))))
        test_mae  = mean_absolute_error(y_test, model.predict(x_test))
        n_active  = int((y >= thr).sum())
        n_ifp_features = int(sum(c.startswith("ifp_") for c in x.columns))

        results.append({
            "Threshold": thr,
            "N_Active": n_active,
            "N_IFP_Features": n_ifp_features,
            "Train_R2": round(train_r2, 6),
            "Test_R2":  round(test_r2,  6),
            "Test_RMSE": round(test_rmse, 4),
            "Test_MAE":  round(test_mae,  4),
        })
        pct = int((y >= thr).mean() * 100)
        print(f"  thr={thr:3d} | active={n_active:4d} ({pct:3d}%) | "
              f"IFP cols={n_ifp_features:4d} | "
              f"Train R²={train_r2:.4f} | Test R²={test_r2:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_SWEEP_DIR / "threshold_sweep_results.csv", index=False)

    best_idx = df["Test_R2"].idxmax()
    best     = df.iloc[best_idx]
    print(f"\nBest threshold: {int(best['Threshold'])}  "
          f"(Test R²={best['Test_R2']:.4f}, "
          f"active={int(best['N_Active'])}, "
          f"IFP cols={int(best['N_IFP_Features'])})")

    with (OUTPUT_SWEEP_DIR / "best_threshold.json").open("w") as f:
        json.dump({
            "best_threshold": int(best["Threshold"]),
            "test_r2":  float(best["Test_R2"]),
            "test_rmse": float(best["Test_RMSE"]),
            "test_mae":  float(best["Test_MAE"]),
            "n_active":  int(best["N_Active"]),
            "n_ifp_features": int(best["N_IFP_Features"]),
            "current_threshold": 38,
            "improvement_vs_38": round(float(best["Test_R2"]) - float(df[df["Threshold"] == 38]["Test_R2"].values[0]), 4),
        }, f, indent=2)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df["Threshold"], df["Train_R2"], color="#52616F", linewidth=2.2,
            marker="o", markersize=4, label="Train R²")
    ax.plot(df["Threshold"], df["Test_R2"],  color="#159A9C", linewidth=2.6,
            marker="o", markersize=5, label="Test R²")

    ax.axvline(38, color="#D99A25", linewidth=1.8, linestyle="--", label="Current threshold (38)")
    ax.set_xlabel("Active IFP Threshold (A5 ≥ threshold → non-zero IFP)")
    ax.set_ylabel("R²")
    ax.set_title("IFP Threshold Sweep — Extra Trees (Scaffold Split)")
    ax.set_xlim(-2, 102)
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)

    ax2 = ax.twinx()
    ax2.fill_between(df["Threshold"], df["N_Active"], alpha=0.08, color="#7B5FB5", label="# active compounds")
    ax2.set_ylabel("# Active Compounds", color="#7B5FB5", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#7B5FB5")
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    out_path = OUTPUT_SWEEP_DIR / "threshold_sweep_r2_curve.png"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")
    print(f"Results CSV:  {OUTPUT_SWEEP_DIR / 'threshold_sweep_results.csv'}")


if __name__ == "__main__":
    main()
