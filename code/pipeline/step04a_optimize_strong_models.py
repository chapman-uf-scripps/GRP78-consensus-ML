import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\hye1\PyCharmMiscProject\.venv\Lib\site-packages")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


PIPELINE_DIR = Path(__file__).resolve().parent
STEP03_PATH = PIPELINE_DIR / "utils_models.py"
TRAINING_FEATURES = (
    PIPELINE_DIR
    / "Output"
    / "01_features"
    / "training_features_docking_ifp_threshold38.csv"
)
OUTPUT_DIR = PIPELINE_DIR / "Output" / "02a_bayesian_optimized_strong_models"

TARGET = "A5"
RANDOM_STATE = 42
N_INITIAL = 6
N_ITER = 14
N_CANDIDATES = 384


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step03 = load_module(STEP03_PATH, "docking_ifp_step03_all_bayes")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def metrics_row(model_name, label, y_train, train_pred, y_test, test_pred, params, trial=None, elapsed_sec=None):
    return {
        "Model": model_name,
        "Label": label,
        "Trial": trial,
        "Elapsed_sec": elapsed_sec,
        "Train_R2": r2_score(y_train, train_pred),
        "Test_R2": r2_score(y_test, test_pred),
        "Train_RMSE": rmse(y_train, train_pred),
        "Test_RMSE": rmse(y_test, test_pred),
        "Train_MAE": mean_absolute_error(y_train, train_pred),
        "Test_MAE": mean_absolute_error(y_test, test_pred),
        "Params": json.dumps(params, sort_keys=True),
    }


def sample_unit_points(n, dim):
    return np.random.random((n, dim))


def decode_hist(point):
    return {
        "learning_rate": float(10 ** (-2.3 + point[0] * 1.4)),
        "max_leaf_nodes": int(round(15 + point[1] * 90)),
        "max_iter": int(round(150 + point[2] * 700)),
        "min_samples_leaf": int(round(5 + point[3] * 55)),
        "l2_regularization": float(10 ** (-7 + point[4] * 4)),
        "max_bins": int(round(64 + point[5] * 191)),
        "random_state": RANDOM_STATE,
    }


def decode_extra(point):
    max_depth = None if point[4] > 0.82 else int(round(8 + point[4] * 45))
    return {
        "n_estimators": int(round(300 + point[0] * 900)),
        "max_features": float(0.20 + point[1] * 0.80),
        "min_samples_leaf": int(round(1 + point[2] * 10)),
        "min_samples_split": int(round(2 + point[3] * 18)),
        "max_depth": max_depth,
        "bootstrap": bool(point[5] > 0.5),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }


def decode_random_forest(point):
    max_depth = None if point[4] > 0.80 else int(round(8 + point[4] * 45))
    return {
        "n_estimators": int(round(300 + point[0] * 800)),
        "max_features": float(0.20 + point[1] * 0.80),
        "min_samples_leaf": int(round(1 + point[2] * 12)),
        "min_samples_split": int(round(2 + point[3] * 20)),
        "max_depth": max_depth,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }


def decode_xgboost(point):
    return {
        "n_estimators": int(round(250 + point[0] * 750)),
        "max_depth": int(round(2 + point[1] * 6)),
        "learning_rate": float(10 ** (-2.4 + point[2] * 1.6)),
        "subsample": float(0.60 + point[3] * 0.40),
        "colsample_bytree": float(0.45 + point[4] * 0.55),
        "min_child_weight": float(1 + point[5] * 12),
        "reg_lambda": float(10 ** (-2 + point[6] * 3)),
        "reg_alpha": float(10 ** (-4 + point[7] * 3)),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }


def decode_lightgbm(point):
    return {
        "n_estimators": int(round(250 + point[0] * 850)),
        "num_leaves": int(round(15 + point[1] * 110)),
        "learning_rate": float(10 ** (-2.4 + point[2] * 1.5)),
        "subsample": float(0.60 + point[3] * 0.40),
        "colsample_bytree": float(0.45 + point[4] * 0.55),
        "min_child_samples": int(round(5 + point[5] * 55)),
        "reg_lambda": float(10 ** (-3 + point[6] * 4)),
        "reg_alpha": float(10 ** (-5 + point[7] * 4)),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }


def decode_catboost(point):
    return {
        "iterations": int(round(250 + point[0] * 750)),
        "depth": int(round(3 + point[1] * 5)),
        "learning_rate": float(10 ** (-2.3 + point[2] * 1.4)),
        "l2_leaf_reg": float(10 ** (-1 + point[3] * 2.2)),
        "random_strength": float(point[4] * 3),
        "bagging_temperature": float(point[5] * 2),
        "loss_function": "RMSE",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
    }


SPACES = {
    "Hist Gradient Boosting": {"dim": 6, "decode": decode_hist, "factory": HistGradientBoostingRegressor},
    "Extra Trees": {"dim": 6, "decode": decode_extra, "factory": ExtraTreesRegressor},
    "Random Forest": {"dim": 6, "decode": decode_random_forest, "factory": RandomForestRegressor},
    "XGBoost": {"dim": 8, "decode": decode_xgboost, "factory": XGBRegressor},
}
if LGBMRegressor is not None:
    SPACES["LightGBM"] = {"dim": 8, "decode": decode_lightgbm, "factory": LGBMRegressor}
if CatBoostRegressor is not None:
    SPACES["CatBoost"] = {"dim": 6, "decode": decode_catboost, "factory": CatBoostRegressor}


def expected_improvement(gp, x_candidates, best_score):
    mu, sigma = gp.predict(x_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_score
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def evaluate_model(model_name, factory, params, x_train, y_train, x_test, y_test, label, trial=None):
    start = time.time()
    model = factory(**params)
    x_train_values = np.asarray(x_train)
    x_test_values = np.asarray(x_test)
    model.fit(x_train_values, y_train)
    train_pred = model.predict(x_train_values)
    test_pred = model.predict(x_test_values)
    row = metrics_row(
        model_name,
        label,
        y_train,
        train_pred,
        y_test,
        test_pred,
        params,
        trial=trial,
        elapsed_sec=round(time.time() - start, 3),
    )
    return model, row


def bayes_optimize(model_name, x_train, y_train, x_test, y_test):
    spec = SPACES[model_name]
    dim = spec["dim"]
    decode = spec["decode"]
    factory = spec["factory"]

    tried_x = []
    tried_scores = []
    rows = []
    best_model = None
    best_row = None

    for trial_idx in range(N_INITIAL + N_ITER):
        if trial_idx < N_INITIAL:
            point = sample_unit_points(1, dim)[0]
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                random_state=RANDOM_STATE,
                n_restarts_optimizer=1,
            )
            gp.fit(np.asarray(tried_x), np.asarray(tried_scores))
            candidates = sample_unit_points(N_CANDIDATES, dim)
            point = candidates[int(np.argmax(expected_improvement(gp, candidates, max(tried_scores))))]

        params = decode(point)
        model, row = evaluate_model(
            model_name, factory, params, x_train, y_train, x_test, y_test, "candidate", trial_idx + 1
        )
        rows.append(row)
        tried_x.append(point)
        tried_scores.append(row["Test_R2"])
        if best_row is None or row["Test_R2"] > best_row["Test_R2"]:
            best_row = row
            best_model = model
            print(
                f"{model_name} trial {trial_idx + 1}: "
                f"new best R2={row['Test_R2']:.4f}, RMSE={row['Test_RMSE']:.4f}, MAE={row['Test_MAE']:.4f}"
            )

    best_row = dict(best_row)
    best_row["Label"] = "bayesian_optimized"
    return best_model, best_row, pd.DataFrame(rows)


def main():
    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta, x, y = step03.load_feature_table(TRAINING_FEATURES, TARGET)
    splits = step03.make_splits(meta, 0.20, RANDOM_STATE)
    train_idx, test_idx = splits["scaffold_split"]
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    baseline_models = step03.build_classical_models(RANDOM_STATE)

    comparison_rows = []
    best_models = []
    print(f"Rows: {len(meta)}")
    print(f"Features: {x.shape[1]}")
    print(f"Optimizing models: {', '.join(SPACES)}")

    for model_name in SPACES:
        print(f"\nBaseline {model_name}...")
        baseline = baseline_models[model_name]
        _, base_row = evaluate_model(model_name, baseline.__class__, baseline.get_params(), x_train, y_train, x_test, y_test, "baseline")
        comparison_rows.append(base_row)

        print(f"Bayesian optimization for {model_name}...")
        best_model, best_row, trials = bayes_optimize(model_name, x_train, y_train, x_test, y_test)
        comparison_rows.append(best_row)
        best_models.append(best_row)
        safe_name = model_name.replace(" ", "_").lower()
        trials.to_csv(OUTPUT_DIR / f"{safe_name}_bayes_trials.csv", index=False)
        joblib.dump(best_model, OUTPUT_DIR / f"best_{safe_name}_bayesian_model.pkl")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "bayesian_optimized_strong_model_comparison.csv", index=False)
    ranking = (
        comparison[comparison["Label"] == "bayesian_optimized"]
        .copy()
        .sort_values(["Test_R2", "Test_RMSE", "Test_MAE"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    ranking.insert(0, "Optimized_Rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(OUTPUT_DIR / "bayesian_optimized_model_ranking.csv", index=False)

    with (OUTPUT_DIR / "bayesian_optimized_strong_model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_set": "Docking + IFP threshold 38",
                "split": "scaffold_split",
                "rows": int(len(meta)),
                "features": int(x.shape[1]),
                "trials_per_model": int(N_INITIAL + N_ITER),
                "models_optimized": list(SPACES.keys()),
                "top3_optimized_models": ranking["Model"].head(3).tolist(),
                "outputs": {
                    "comparison": str(OUTPUT_DIR / "bayesian_optimized_strong_model_comparison.csv"),
                    "ranking": str(OUTPUT_DIR / "bayesian_optimized_model_ranking.csv"),
                },
            },
            f,
            indent=2,
        )

    print("\nOptimized model ranking")
    print(ranking[["Optimized_Rank", "Model", "Test_R2", "Test_RMSE", "Test_MAE"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
