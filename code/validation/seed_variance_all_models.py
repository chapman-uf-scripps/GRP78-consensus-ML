"""
Multi-seed sweep across EVERY model in the Bayesian-optimized comparison
ranking, on the scaffold split. Records per-seed Train/Test R2 / RMSE / MAE
and aggregates mean +/- std. Then re-renders the horizontal-bar comparison
plot with std error bars.

Outputs go to <repo>/output/seed_variance_study/.
"""

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch  # noqa: F401  (needed before patching step04c)

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent
FEATURE_TABLE = PKG_ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
RANKING_CSV = PKG_ROOT / "output" / "full_pipeline_output" / "02c_bayesian_optimized_deep_models" / "bayesian_optimized_all_models_including_deep_ranking.csv"
GNN_TRIALS_CSV = PKG_ROOT / "output" / "full_pipeline_output" / "02c_bayesian_optimized_deep_models" / "gnn_plus_active-filtered_ifp_bayes_trials.csv"
OUT_DIR = PKG_ROOT / "output" / "seed_variance_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "A5"
SMILES_COL = "SMILES"
TEST_SIZE = 0.20
MIN_TEST_R2_FOR_BAR = 0.4

# Deterministic models (no random_state used in training) — skip multi-seed,
# they'd just be identical N times. Run once, std=0.
DETERMINISTIC_MODELS = {"KNN", "SVR", "Elastic Net", "Ridge"}


def load_module(file_name, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mu = load_module("utils_models.py", "utils_models")
step04c = load_module("step04c_optimize_deep_models.py", "step04c_optimize_deep_models")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def metric_row(model_name, seed, y_train, train_pred, y_test, test_pred, elapsed):
    return {
        "Model": model_name,
        "Seed": seed,
        "Train_R2": float(r2_score(y_train, train_pred)),
        "Test_R2": float(r2_score(y_test, test_pred)),
        "Train_RMSE": rmse(y_train, train_pred),
        "Test_RMSE": rmse(y_test, test_pred),
        "Train_MAE": float(mean_absolute_error(y_train, train_pred)),
        "Test_MAE": float(mean_absolute_error(y_test, test_pred)),
        "Elapsed_sec": round(elapsed, 2),
    }


def load_all_params():
    """Read every Bayesian-optimized model's params, plus GNN trial-7 params."""
    rank = pd.read_csv(RANKING_CSV)
    out = {}
    for _, row in rank.iterrows():
        name = row["Model"]
        if str(row.get("Label", "")) != "bayesian_optimized":
            continue
        try:
            out[name] = json.loads(row["Params"]) if not pd.isna(row["Params"]) else {}
        except Exception:
            out[name] = {}
    # GNN params come from its own trials CSV (best Test_R2 row)
    trials = pd.read_csv(GNN_TRIALS_CSV)
    best = trials.sort_values(["Test_R2", "Test_RMSE", "Test_MAE"],
                              ascending=[False, True, True]).iloc[0]
    out["GNN + active-filtered IFP"] = json.loads(best["Params"])
    return out


# -------- classical / tree training --------

def _strip(p, drop):
    return {k: v for k, v in p.items() if k not in drop}


def train_extra_trees(seed, params, x_train, y_train, x_test, y_test):
    p = _strip(params, {"n_jobs", "random_state"})
    m = ExtraTreesRegressor(**p, n_jobs=-1, random_state=seed)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_random_forest(seed, params, x_train, y_train, x_test, y_test):
    p = _strip(params, {"n_jobs", "random_state"})
    m = RandomForestRegressor(**p, n_jobs=-1, random_state=seed)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_gradient_boosting(seed, params, x_train, y_train, x_test, y_test):
    p = _strip(params, {"random_state"})
    m = GradientBoostingRegressor(**p, random_state=seed)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_hist_gradient_boosting(seed, params, x_train, y_train, x_test, y_test):
    p = _strip(params, {"random_state"})
    m = HistGradientBoostingRegressor(**p, random_state=seed)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_xgboost(seed, params, x_train, y_train, x_test, y_test):
    if XGBRegressor is None:
        raise RuntimeError("xgboost not installed")
    p = _strip(params, {"n_jobs", "random_state"})
    m = XGBRegressor(**p, n_jobs=-1, random_state=seed, verbosity=0)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_lightgbm(seed, params, x_train, y_train, x_test, y_test):
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm not installed")
    p = _strip(params, {"n_jobs", "random_state", "verbosity"})
    m = LGBMRegressor(**p, n_jobs=-1, random_state=seed, verbosity=-1)
    # LightGBM 4.x rejects special chars in feature names — pass numpy arrays.
    x_tr = x_train.values if hasattr(x_train, "values") else x_train
    x_te = x_test.values if hasattr(x_test, "values") else x_test
    t0 = time.time(); m.fit(x_tr, y_train)
    return m.predict(x_tr), m.predict(x_te), time.time() - t0


def train_catboost(seed, params, x_train, y_train, x_test, y_test):
    if CatBoostRegressor is None:
        raise RuntimeError("catboost not installed")
    p = _strip(params, {"random_seed", "verbose", "allow_writing_files", "loss_function"})
    m = CatBoostRegressor(**p, loss_function="RMSE", random_seed=seed,
                          verbose=False, allow_writing_files=False)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_ridge(seed, params, x_train, y_train, x_test, y_test):
    m = make_pipeline(StandardScaler(), Ridge(random_state=seed)).set_params(**params)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_elastic_net(seed, params, x_train, y_train, x_test, y_test):
    m = make_pipeline(
        StandardScaler(),
        ElasticNet(random_state=seed, max_iter=20000),
    ).set_params(**params)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_svr(seed, params, x_train, y_train, x_test, y_test):
    m = make_pipeline(StandardScaler(), SVR()).set_params(**params)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


def train_knn(seed, params, x_train, y_train, x_test, y_test):
    m = make_pipeline(StandardScaler(), KNeighborsRegressor()).set_params(**params)
    t0 = time.time(); m.fit(x_train, y_train)
    return m.predict(x_train), m.predict(x_test), time.time() - t0


# -------- deep models — go through step04c's actual code, monkey-patch seed --------

def _patch_step04c_seed(seed):
    """Make step04c train_* functions use `seed` for both RNG and val split."""
    step04c.RANDOM_STATE = seed

    def _patched(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    step04c.set_seed = _patched


def train_deep_tabular(model_name, seed, params, x_train, y_train, x_test, y_test):
    _patch_step04c_seed(seed)
    t0 = time.time()
    train_pred, test_pred = step04c.train_tabular_model(model_name, params,
                                                       x_train, x_test, y_train, y_test)
    return train_pred, test_pred, time.time() - t0


def train_deep_sequence(model_name, seed, params, smiles_train, smiles_test, y_train, y_test):
    _patch_step04c_seed(seed)
    t0 = time.time()
    train_pred, test_pred = step04c.train_sequence_model(model_name, params,
                                                        smiles_train, smiles_test,
                                                        y_train, y_test)
    return train_pred, test_pred, time.time() - t0


def train_gnn(seed, params, meta_train, meta_test, x_train, x_test, y_train, y_test):
    _patch_step04c_seed(seed)
    t0 = time.time()
    train_pred, test_pred = step04c.train_gnn_model(params, meta_train, meta_test,
                                                    x_train, x_test, y_train, y_test)
    return train_pred, test_pred, time.time() - t0


# -------- dispatcher --------

CLASSICAL_TRAINERS = {
    "Extra Trees":          train_extra_trees,
    "Random Forest":        train_random_forest,
    "Gradient Boosting":    train_gradient_boosting,
    "Hist Gradient Boosting": train_hist_gradient_boosting,
    "XGBoost":              train_xgboost,
    "LightGBM":             train_lightgbm,
    "CatBoost":             train_catboost,
    "Ridge":                train_ridge,
    "Elastic Net":          train_elastic_net,
    "SVR":                  train_svr,
    "KNN":                  train_knn,
}

DEEP_TABULAR = {"MLP + active-filtered IFP", "VAE + active-filtered IFP"}
DEEP_SEQUENCE = {"SMILES-CNN", "SMILES-LSTM", "SMILES-Transformer"}


def run_one_seed(seed, params_lookup, x_train, x_test, y_train, y_test,
                 meta_train, meta_test, models_to_run, log):
    rows = []
    for model_name in models_to_run:
        if model_name not in params_lookup:
            print(f"  [skip] no params for {model_name}", file=log)
            continue
        params = params_lookup[model_name]
        try:
            if model_name in CLASSICAL_TRAINERS:
                tr, te, dt = CLASSICAL_TRAINERS[model_name](seed, params,
                                                            x_train, y_train, x_test, y_test)
            elif model_name == "GNN + active-filtered IFP":
                tr, te, dt = train_gnn(seed, params, meta_train, meta_test,
                                       x_train.values, x_test.values, y_train, y_test)
            elif model_name in DEEP_TABULAR:
                tr, te, dt = train_deep_tabular(model_name, seed, params,
                                                x_train.values, y_train,
                                                x_test.values, y_test)
            elif model_name in DEEP_SEQUENCE:
                tr, te, dt = train_deep_sequence(model_name, seed, params,
                                                 meta_train[SMILES_COL].values,
                                                 meta_test[SMILES_COL].values,
                                                 y_train, y_test)
            else:
                print(f"  [skip] unhandled model {model_name}", file=log)
                continue
            row = metric_row(model_name, seed, y_train, tr, y_test, te, dt)
            print(f"  {model_name:32s} train R2={row['Train_R2']:.4f}  test R2={row['Test_R2']:.4f}  ({row['Elapsed_sec']}s)", file=log, flush=True)
            rows.append(row)
        except Exception as exc:
            print(f"  [error] {model_name}: {exc}", file=log)
    return rows


def aggregate(per_seed):
    rows = []
    for name, sub in per_seed.groupby("Model"):
        rows.append({
            "Model": name,
            "N_seeds": len(sub),
            "Train_R2_mean": sub["Train_R2"].mean(),
            "Train_R2_std":  sub["Train_R2"].std(ddof=1) if len(sub) > 1 else 0.0,
            "Test_R2_mean":  sub["Test_R2"].mean(),
            "Test_R2_std":   sub["Test_R2"].std(ddof=1) if len(sub) > 1 else 0.0,
            "Test_R2_min":   sub["Test_R2"].min(),
            "Test_R2_max":   sub["Test_R2"].max(),
            "Test_RMSE_mean": sub["Test_RMSE"].mean(),
            "Test_RMSE_std":  sub["Test_RMSE"].std(ddof=1) if len(sub) > 1 else 0.0,
            "Test_MAE_mean":  sub["Test_MAE"].mean(),
            "Test_MAE_std":   sub["Test_MAE"].std(ddof=1) if len(sub) > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("Test_R2_mean", ascending=False).reset_index(drop=True)


def plot_comparison(summary, out_png, n_seeds_label):
    plot_df = summary[summary["Test_R2_mean"] >= MIN_TEST_R2_FOR_BAR].copy()
    plot_df = plot_df.sort_values("Test_R2_mean", ascending=True).reset_index(drop=True)

    TOP3 = {"Extra Trees", "CatBoost", "GNN + active-filtered IFP"}
    TOP_COLORS = {"Extra Trees": "#2E86AB", "CatBoost": "#E67E22", "GNN + active-filtered IFP": "#8B5CF6"}
    colors = [TOP_COLORS.get(m, "#CDD5DF") if m in TOP3 else "#CDD5DF" for m in plot_df["Model"]]

    label_map = {
        "GNN + active-filtered IFP": "GNN (+IFP)",
        "MLP + active-filtered IFP": "MLP (+IFP)",
        "VAE + active-filtered IFP": "VAE (+IFP)",
    }
    plot_df["Display"] = plot_df["Model"].map(lambda m: label_map.get(m, m))

    fig, ax = plt.subplots(figsize=(10.5, max(8.0, 0.55 * len(plot_df))))
    y = np.arange(len(plot_df))
    means = plot_df["Test_R2_mean"].values
    stds = plot_df["Test_R2_std"].values
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="white", linewidth=0.7,
            error_kw={"ecolor": "#333", "capsize": 4, "lw": 1.0})
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Display"], fontsize=12)
    ax.set_xlim(0.40, 1.00)
    ax.set_xlabel(r"Test R$^2$ (scaffold split, mean $\pm$ std)", fontsize=13)
    ax.set_title(f"Bayesian-Optimized Model Ranking\n(scaffold split, {n_seeds_label})",
                 fontsize=14, fontweight="bold", pad=14)
    for yi, mu_, sd in zip(y, means, stds):
        ax.text(mu_ + sd + 0.008, yi,
                f"{mu_:.3f}" + (f" $\\pm$ {sd:.3f}" if sd > 1e-4 else ""),
                va="center", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, color="#CBD2D8", alpha=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--skip", type=str, nargs="+", default=[],
                    help="Skip these model names")
    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="Run only these model names")
    ap.add_argument("--log-file", type=str, default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    log = sys.stdout if args.log_file is None else open(args.log_file, "w", encoding="utf-8")

    params_lookup = load_all_params()
    print(f"Loaded params for {len(params_lookup)} models", file=log)

    meta, x, y = mu.load_feature_table(str(FEATURE_TABLE), TARGET)
    drop = [c for c in x.columns if c == "ifp_allowed_by_A5_threshold"]
    if drop:
        x = x.drop(columns=drop)
    splits = mu.make_splits(meta, TEST_SIZE, seed=42)
    tr_idx, te_idx = splits["scaffold_split"]
    x_train, x_test = x.iloc[tr_idx], x.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    meta_train = meta.iloc[tr_idx].reset_index(drop=True)
    meta_test = meta.iloc[te_idx].reset_index(drop=True)
    print(f"Scaffold split fixed: train n={len(y_train)}, test n={len(y_test)}", file=log)

    all_models = list(params_lookup.keys())
    if args.only:
        all_models = [m for m in all_models if m in args.only]
    if args.skip:
        all_models = [m for m in all_models if m not in args.skip]
    # Run deterministic models only once
    multi_seed_models = [m for m in all_models if m not in DETERMINISTIC_MODELS]
    once_models = [m for m in all_models if m in DETERMINISTIC_MODELS]

    print(f"Multi-seed models ({len(multi_seed_models)}): {multi_seed_models}", file=log)
    print(f"Single-run (deterministic) models ({len(once_models)}): {once_models}", file=log)
    print(f"Seeds to sweep: {seeds}", file=log, flush=True)

    rows = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", file=log, flush=True)
        rows.extend(run_one_seed(seed, params_lookup, x_train, x_test, y_train, y_test,
                                 meta_train, meta_test, multi_seed_models, log))
    print(f"\n=== deterministic models (single run, seed 0) ===", file=log, flush=True)
    rows.extend(run_one_seed(0, params_lookup, x_train, x_test, y_train, y_test,
                             meta_train, meta_test, once_models, log))

    per_seed = pd.DataFrame(rows)
    per_seed_path = OUT_DIR / "per_seed_metrics_all_models_scaffold.csv"
    # Merge with any existing rows so a --only / --skip partial run doesn't
    # destroy data from previous runs of the other models.
    if per_seed_path.exists():
        existing = pd.read_csv(per_seed_path)
        ran_now = set(zip(per_seed["Model"], per_seed["Seed"]))
        keep_existing = existing[~existing.apply(
            lambda r: (r["Model"], r["Seed"]) in ran_now, axis=1
        )]
        per_seed = pd.concat([keep_existing, per_seed], ignore_index=True, sort=False)
    per_seed.to_csv(per_seed_path, index=False)
    print(f"\nPer-seed metrics -> {per_seed_path} (merged with prior runs if any)", file=log)

    summary = aggregate(per_seed)
    summary_path = OUT_DIR / "summary_mean_std_all_models_scaffold.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary -> {summary_path}", file=log)
    print(summary[["Model", "N_seeds", "Train_R2_mean", "Train_R2_std",
                   "Test_R2_mean", "Test_R2_std",
                   "Test_R2_min", "Test_R2_max"]].to_string(index=False), file=log)

    n_seeds_label = f"{len(seeds)} seeds (deterministic models: single run)"
    out_png = OUT_DIR / "comparison_mean_std_all_models_scaffold.png"
    plot_comparison(summary, out_png, n_seeds_label)
    print(f"Plot -> {out_png}", file=log)

    if log is not sys.stdout:
        log.close()


if __name__ == "__main__":
    main()
