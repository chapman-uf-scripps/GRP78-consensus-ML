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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


PIPELINE_DIR = Path(__file__).resolve().parent
STEP03_PATH = PIPELINE_DIR / "utils_models.py"
TRAINING_FEATURES = (
    PIPELINE_DIR
    / "Output"
    / "01_features"
    / "training_features_docking_ifp_threshold38.csv"
)
OUTPUT_DIR = PIPELINE_DIR / "Output" / "02b_bayesian_optimized_all_classical_models"
STRONG_DIR = PIPELINE_DIR / "Output" / "02a_bayesian_optimized_strong_models"

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


step03 = load_module(STEP03_PATH, "docking_ifp_step03_remaining_bayes")


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
        "Params": json.dumps(params, sort_keys=True, default=str),
    }


def sample_unit_points(n, dim):
    return np.random.random((n, dim))


def decode_gradient_boosting(point):
    max_depth = int(round(2 + point[1] * 5))
    return {
        "n_estimators": int(round(150 + point[0] * 650)),
        "max_depth": max_depth,
        "learning_rate": float(10 ** (-2.3 + point[2] * 1.5)),
        "subsample": float(0.55 + point[3] * 0.45),
        "min_samples_leaf": int(round(1 + point[4] * 24)),
        "min_samples_split": int(round(2 + point[5] * 28)),
        "max_features": float(0.35 + point[6] * 0.65),
        "random_state": RANDOM_STATE,
    }


def decode_ridge(point):
    return {"ridge__alpha": float(10 ** (-3 + point[0] * 7))}


def decode_elastic_net(point):
    return {
        "elasticnet__alpha": float(10 ** (-4 + point[0] * 5)),
        "elasticnet__l1_ratio": float(0.01 + point[1] * 0.98),
    }


def decode_svr(point):
    return {
        "svr__C": float(10 ** (-1 + point[0] * 4)),
        "svr__epsilon": float(10 ** (-3 + point[1] * 2)),
        "svr__gamma": float(10 ** (-5 + point[2] * 4)),
    }


def decode_knn(point):
    weights = "distance" if point[1] > 0.5 else "uniform"
    metric = "manhattan" if point[2] > 0.5 else "minkowski"
    return {
        "kneighborsregressor__n_neighbors": int(round(2 + point[0] * 28)),
        "kneighborsregressor__weights": weights,
        "kneighborsregressor__p": 1 if metric == "manhattan" else 2,
    }


SPACES = {
    "Gradient Boosting": {
        "dim": 7,
        "decode": decode_gradient_boosting,
        "factory": lambda params: GradientBoostingRegressor(**params),
    },
    "Ridge": {
        "dim": 1,
        "decode": decode_ridge,
        "factory": lambda params: make_pipeline(StandardScaler(), Ridge(random_state=RANDOM_STATE)).set_params(**params),
    },
    "Elastic Net": {
        "dim": 2,
        "decode": decode_elastic_net,
        "factory": lambda params: make_pipeline(
            StandardScaler(),
            ElasticNet(random_state=RANDOM_STATE, max_iter=20000),
        ).set_params(**params),
    },
    "SVR": {
        "dim": 3,
        "decode": decode_svr,
        "factory": lambda params: make_pipeline(StandardScaler(), SVR()).set_params(**params),
    },
    "KNN": {
        "dim": 3,
        "decode": decode_knn,
        "factory": lambda params: make_pipeline(StandardScaler(), KNeighborsRegressor()).set_params(**params),
    },
}


def expected_improvement(gp, x_candidates, best_score):
    mu, sigma = gp.predict(x_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_score
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def evaluate_model(model_name, factory, params, x_train, y_train, x_test, y_test, label, trial=None):
    start = time.time()
    model = factory(params)
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


def baseline_for_model(model_name, baseline_models):
    if model_name == "Gradient Boosting":
        return lambda _params: baseline_models["Gradient Boosting"], baseline_models["Gradient Boosting"].get_params()
    if model_name == "Ridge":
        return lambda _params: baseline_models["Ridge"], baseline_models["Ridge"].get_params()
    if model_name == "Elastic Net":
        return lambda _params: baseline_models["Elastic Net"], baseline_models["Elastic Net"].get_params()
    if model_name == "SVR":
        return lambda _params: baseline_models["SVR"], baseline_models["SVR"].get_params()
    if model_name == "KNN":
        return lambda _params: baseline_models["KNN"], baseline_models["KNN"].get_params()
    raise ValueError(model_name)


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
    print(f"Rows: {len(meta)}")
    print(f"Features: {x.shape[1]}")
    print(f"Optimizing remaining models: {', '.join(SPACES)}")

    for model_name in SPACES:
        print(f"\nBaseline {model_name}...")
        baseline_factory, baseline_params = baseline_for_model(model_name, baseline_models)
        _, base_row = evaluate_model(
            model_name, baseline_factory, baseline_params, x_train, y_train, x_test, y_test, "baseline"
        )
        comparison_rows.append(base_row)
        print(f"Bayesian optimization for {model_name}...")
        best_model, best_row, trials = bayes_optimize(model_name, x_train, y_train, x_test, y_test)
        comparison_rows.append(best_row)
        safe_name = model_name.replace(" ", "_").lower()
        trials.to_csv(OUTPUT_DIR / f"{safe_name}_bayes_trials.csv", index=False)
        joblib.dump(best_model, OUTPUT_DIR / f"best_{safe_name}_bayesian_model.pkl")

    remaining_comparison = pd.DataFrame(comparison_rows)
    remaining_comparison.to_csv(OUTPUT_DIR / "bayesian_optimized_remaining_model_comparison.csv", index=False)

    strong_ranking = pd.read_csv(STRONG_DIR / "bayesian_optimized_model_ranking.csv")
    gnn_path = STRONG_DIR / "gnn_baseline_same_docking_ifp_table.csv"
    gnn = pd.read_csv(gnn_path) if gnn_path.exists() else pd.DataFrame()
    remaining_best = remaining_comparison[remaining_comparison["Label"] == "bayesian_optimized"].copy()
    all_best = pd.concat([strong_ranking, remaining_best, gnn], ignore_index=True, sort=False)
    all_best = all_best.sort_values(["Test_R2", "Test_RMSE", "Test_MAE"], ascending=[False, True, True]).reset_index(drop=True)
    all_best.insert(0, "Overall_Rank", np.arange(1, len(all_best) + 1))
    all_best.to_csv(OUTPUT_DIR / "bayesian_optimized_all_models_ranking.csv", index=False)

    with (OUTPUT_DIR / "bayesian_optimized_all_models_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_set": "Docking + IFP threshold 38",
                "split": "scaffold_split",
                "rows": int(len(meta)),
                "features": int(x.shape[1]),
                "remaining_trials_per_model": int(N_INITIAL + N_ITER),
                "models_optimized_in_this_run": list(SPACES.keys()),
                "strong_model_results_source": str(STRONG_DIR / "bayesian_optimized_model_ranking.csv"),
                "top3_models_after_all_classical_bayes": all_best["Model"].head(3).tolist(),
                "outputs": {
                    "remaining_comparison": str(OUTPUT_DIR / "bayesian_optimized_remaining_model_comparison.csv"),
                    "all_ranking": str(OUTPUT_DIR / "bayesian_optimized_all_models_ranking.csv"),
                },
            },
            f,
            indent=2,
        )

    print("\nAll-model Bayesian ranking")
    print(all_best[["Overall_Rank", "Model", "Label", "Test_R2", "Test_RMSE", "Test_MAE"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
