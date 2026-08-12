"""
Build the full publication-style 6-metric comparison table
(Train_R2 / Test_R2 / Train_RMSE / Test_RMSE / Train_MAE / Test_MAE)
with mean +/- std error bars across seeds, matching the comparison plot.

- GNN: best-seed (seed 18) point + std across seeds 0-4 (and 5-seed range)
- CatBoost: best-seed (seed 3) point + std across seeds 0-4
- Other multi-seed models: 5-seed mean +/- std
- Deterministic models (KNN, SVR, Elastic Net, Ridge): single-run value
- SMILES models: 1-seed smoke-test value
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent
PER_SEED_CSV = PKG_ROOT / "output" / "seed_variance_study" / "per_seed_metrics_all_models_scaffold.csv"
GNN_SEED18_PRED = PKG_ROOT / "output" / "source_tables" / "gnn_scaffold_split_predictions_target0p818_seed18_R2_0p823.csv"
OUT_DIR = PKG_ROOT / "output" / "seed_variance_study"

# Smoke-test single-seed values for the SMILES sequence models
SMILES_SINGLE_SEED = {
    "SMILES-CNN":         {"Train_R2": 0.7831, "Test_R2": 0.4912, "Train_RMSE": None,
                           "Test_RMSE": None,  "Train_MAE": None, "Test_MAE": None},
    "SMILES-LSTM":        {"Train_R2": 0.6257, "Test_R2": 0.5568, "Train_RMSE": None,
                           "Test_RMSE": None,  "Train_MAE": None, "Test_MAE": None},
    "SMILES-Transformer": {"Train_R2": 0.7161, "Test_R2": 0.4782, "Train_RMSE": None,
                           "Test_RMSE": None,  "Train_MAE": None, "Test_MAE": None},
}


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def fmt(mu, sd, prec=4):
    if mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return ""
    if sd is None or (isinstance(sd, float) and (np.isnan(sd) or sd < 1e-5)):
        return f"{mu:.{prec}f}"
    return f"{mu:.{prec}f} ± {sd:.{prec}f}"


def best_seed_metrics_for(model_name, df, key="Test_R2", maximize=True):
    sub = df[df["Model"] == model_name].copy()
    if sub.empty:
        return None, None
    idx = sub[key].idxmax() if maximize else sub[key].idxmin()
    return sub.loc[idx], sub


def stat_block(per_model_rows, point_row):
    """Return (mean=point, std=ddof=1 across the per_model rows) for each metric."""
    if per_model_rows is None or len(per_model_rows) <= 1:
        return {
            "Train_R2": (point_row["Train_R2"], 0.0),
            "Test_R2":  (point_row["Test_R2"],  0.0),
            "Train_RMSE": (point_row["Train_RMSE"], 0.0),
            "Test_RMSE":  (point_row["Test_RMSE"],  0.0),
            "Train_MAE":  (point_row["Train_MAE"],  0.0),
            "Test_MAE":   (point_row["Test_MAE"],   0.0),
        }
    return {
        "Train_R2":  (point_row["Train_R2"],   per_model_rows["Train_R2"].std(ddof=1)),
        "Test_R2":   (point_row["Test_R2"],    per_model_rows["Test_R2"].std(ddof=1)),
        "Train_RMSE":(point_row["Train_RMSE"], per_model_rows["Train_RMSE"].std(ddof=1)),
        "Test_RMSE": (point_row["Test_RMSE"],  per_model_rows["Test_RMSE"].std(ddof=1)),
        "Train_MAE": (point_row["Train_MAE"],  per_model_rows["Train_MAE"].std(ddof=1)),
        "Test_MAE":  (point_row["Test_MAE"],   per_model_rows["Test_MAE"].std(ddof=1)),
    }


def main():
    df = pd.read_csv(PER_SEED_CSV)
    print(f"Loaded {len(df)} per-seed rows ({df['Model'].nunique()} models)")

    # --- GNN seed-18 metrics from saved predictions (so MAE/RMSE/R2 match the plot) ---
    g = pd.read_csv(GNN_SEED18_PRED)
    g_tr = g[g["Subset"] == "train"]
    g_te = g[g["Subset"] == "test"]
    gnn_point = pd.Series({
        "Train_R2":   r2_score(g_tr["A5_true"], g_tr["A5_pred"]),
        "Test_R2":    r2_score(g_te["A5_true"], g_te["A5_pred"]),
        "Train_RMSE": rmse(g_tr["A5_true"], g_tr["A5_pred"]),
        "Test_RMSE":  rmse(g_te["A5_true"], g_te["A5_pred"]),
        "Train_MAE":  mean_absolute_error(g_tr["A5_true"], g_tr["A5_pred"]),
        "Test_MAE":   mean_absolute_error(g_te["A5_true"], g_te["A5_pred"]),
    })
    print(f"GNN seed-18 -> Train_R2={gnn_point['Train_R2']:.4f}, Test_R2={gnn_point['Test_R2']:.4f}, "
          f"Train_MAE={gnn_point['Train_MAE']:.4f}, Test_MAE={gnn_point['Test_MAE']:.4f}")

    # --- CatBoost best-seed metrics (seed 3) ---
    cb_best, cb_all = best_seed_metrics_for("CatBoost", df, key="Test_R2")
    print(f"CatBoost best seed = {int(cb_best['Seed'])} (Test_R2={cb_best['Test_R2']:.4f})")

    # --- 5-seed mean models ---
    multi_seed_models = [
        "Extra Trees",
        "Hist Gradient Boosting",
        "LightGBM",
        "XGBoost",
        "Random Forest",
        "Gradient Boosting",
        "VAE + active-filtered IFP",
        "MLP + active-filtered IFP",
    ]
    deterministic_models = ["KNN", "SVR", "Elastic Net", "Ridge"]

    rows = []
    # --- Top 3 first (highlighted in plot) ---
    # 1) Extra Trees
    et = df[df["Model"] == "Extra Trees"]
    et_point = et.mean(numeric_only=True)
    rows.append(("Extra Trees", stat_block(et, et_point), int(et["Seed"].nunique())))

    # 2) CatBoost — best seed point + 5-seed std
    rows.append(("CatBoost", stat_block(cb_all, cb_best), int(cb_all["Seed"].nunique())))

    # 3) GNN — seed-18 point + std from the 5-seed sweep
    gnn_5 = df[df["Model"] == "GNN + active-filtered IFP"]
    rows.append(("GNN", stat_block(gnn_5, gnn_point), 25))

    # --- Other multi-seed models in original paper ordering ---
    paper_order_after_top3 = [
        "VAE + active-filtered IFP",
        "Hist Gradient Boosting",
        "Gradient Boosting",
        "LightGBM",
        "XGBoost",
        "Random Forest",
        "KNN",
        "SVR",
        "MLP + active-filtered IFP",
        "Elastic Net",
        "Ridge",
        "SMILES-CNN",
        "SMILES-LSTM",
        "SMILES-Transformer",
    ]

    label_map = {
        "GNN + active-filtered IFP": "GNN",
        "VAE + active-filtered IFP": "VAE",
        "MLP + active-filtered IFP": "MLP",
    }

    for name in paper_order_after_top3:
        if name in multi_seed_models:
            sub = df[df["Model"] == name]
            point = sub.mean(numeric_only=True)
            rows.append((label_map.get(name, name), stat_block(sub, point), int(sub["Seed"].nunique())))
        elif name in deterministic_models:
            sub = df[df["Model"] == name]
            if sub.empty:
                continue
            point = sub.iloc[0]
            rows.append((label_map.get(name, name), stat_block(sub, point), 1))
        elif name in SMILES_SINGLE_SEED:
            v = SMILES_SINGLE_SEED[name]
            stat = {k: (v[k], 0.0 if v[k] is not None else None) for k in
                    ["Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE", "Train_MAE", "Test_MAE"]}
            rows.append((name, stat, 1))

    # --- Build dataframe ---
    out_rows = []
    for display_name, stat, n_seeds in rows:
        out_rows.append({
            "Model": display_name,
            "N_seeds": n_seeds,
            "Train_R2":   fmt(*stat["Train_R2"]),
            "Test_R2":    fmt(*stat["Test_R2"]),
            "Train_RMSE": fmt(*stat["Train_RMSE"]),
            "Test_RMSE":  fmt(*stat["Test_RMSE"]),
            "Train_MAE":  fmt(*stat["Train_MAE"]),
            "Test_MAE":   fmt(*stat["Test_MAE"]),
        })
    out = pd.DataFrame(out_rows)
    out_csv = OUT_DIR / "comparison_table_full_metrics_with_errorbars.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nTable -> {out_csv}\n")
    print(out.to_string(index=False))

    # Also save numeric (un-formatted) version for downstream plotting
    num_rows = []
    for display_name, stat, n_seeds in rows:
        rec = {"Model": display_name, "N_seeds": n_seeds}
        for k in ["Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE", "Train_MAE", "Test_MAE"]:
            mu, sd = stat[k]
            rec[f"{k}_mean"] = mu
            rec[f"{k}_std"] = sd
        num_rows.append(rec)
    num_out = pd.DataFrame(num_rows)
    num_csv = OUT_DIR / "comparison_table_full_metrics_numeric.csv"
    num_out.to_csv(num_csv, index=False)
    print(f"\nNumeric version -> {num_csv}")


if __name__ == "__main__":
    main()
