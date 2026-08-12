import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

PIPELINE_DIR = Path(__file__).parent
STEP11_PATH = PIPELINE_DIR / "utils_models.py"
FEATURE_TABLE_PATH = PIPELINE_DIR / "Output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
MODEL_COMPARISON_PATH = PIPELINE_DIR / "Output" / "02c_bayesian_optimized_deep_models" / "bayesian_optimized_all_models_including_deep_ranking.csv"
OUTPUT_DIR = PIPELINE_DIR / "Output" / "08_figures_bayesian"
SOURCE_TABLE_DIR = OUTPUT_DIR / "10_source_tables"

TARGET = "A5"
RANDOM_STATE = 42
TEST_SIZE = 0.20
SHAP_TOP_N = 15
DEPENDENCE_TOP_N = 6
SHAP_MAX_SAMPLES = 220

TOP3 = ["Extra Trees", "CatBoost", "GNN + active-filtered IFP"]
MODEL_LABELS = {
    "Extra Trees": "Extra Trees",
    "CatBoost": "CatBoost",
    "VAE + active-filtered IFP": "VAE",
    "MLP + active-filtered IFP": "MLP",
    "GNN + active-filtered IFP": "GNN",
    "XGBoost": "XGBoost",
    "Gradient Boosting": "Gradient Boosting",
    "Hist Gradient Boosting": "Hist Gradient Boosting",
    "LightGBM": "LightGBM",
    "Random Forest": "Random Forest",
    "SMILES-CNN": "CNN",
    "SMILES-LSTM": "LSTM",
    "SMILES-Transformer": "Transformer",
    "KNN": "KNN",
    "SVR": "SVR",
    "Ridge": "Ridge",
    "Elastic Net": "Elastic Net",
}
MODEL_COLORS = {
    "Extra Trees": "#159A9C",
    "CatBoost": "#D99A25",
    "GNN + active-filtered IFP": "#7B5FB5",
}
SPLIT_LABELS = {
    "random_split": "Random Split",
    "scaffold_split": "Scaffold Split",
}
VALIDATION_SPLIT_LABELS = {
    "random_split": "RANDOM Split",
    "scaffold_split": "SCAFFOLD Split",
}


def validation_model_label(model):
    return MODEL_LABELS[model].upper()


def validation_title(model, split):
    return f"{validation_model_label(model)} {VALIDATION_SPLIT_LABELS[split]} Validation"


def style_validation_text(ax):
    ax.title.set_fontsize(28)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(22)
    ax.yaxis.label.set_fontsize(22)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(18)
        tick_label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(17)
            text.set_fontweight("bold")


def load_step11():
    spec = importlib.util.spec_from_file_location("utils_models", STEP11_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_name(value):
    return (
        str(value)
        .replace(" + ", "_plus_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .lower()
    )


def save_figure(fig, output_path):
    for ax in fig.axes:
        ax.title.set_fontweight("bold")
        ax.title.set_fontsize(max(ax.title.get_fontsize(), 24))
        ax.xaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_fontsize(), 20))
        ax.yaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_fontsize(), 20))
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(max(tick_label.get_fontsize(), 17))
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontweight("bold")
                text.set_fontsize(max(text.get_fontsize(), 15))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_path(folder, filename):
    path = OUTPUT_DIR / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 15,
            "figure.titlesize": 26,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axes(ax):
    ax.grid(alpha=0.22)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(17)
        tick_label.set_fontweight("bold")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_row(model, split, train_n, test_n, y_train, train_pred, y_test, test_pred):
    return {
        "Model": model,
        "Split": split,
        "Train_N": int(train_n),
        "Test_N": int(test_n),
        "Train_R2": float(r2_score(y_train, train_pred)),
        "Test_R2": float(r2_score(y_test, test_pred)),
        "Train_RMSE": rmse(y_train, train_pred),
        "Test_RMSE": rmse(y_test, test_pred),
        "Train_MAE": float(mean_absolute_error(y_train, train_pred)),
        "Test_MAE": float(mean_absolute_error(y_test, test_pred)),
    }


def prediction_frame(meta, y, pred, model, split, subset):
    out = meta.copy().reset_index(drop=True)
    out["Model"] = model
    out["Split"] = split
    out["Subset"] = subset
    out["A5_true"] = y
    out["A5_pred"] = pred
    out["Residual"] = out["A5_true"] - out["A5_pred"]
    out["Abs_Error"] = out["Residual"].abs()
    return out


def fit_line(x, y):
    model = LinearRegression()
    model.fit(np.asarray(x).reshape(-1, 1), np.asarray(y))
    return model


def top_shap_indices(shap_values, top_n):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return np.argsort(mean_abs)[::-1][:top_n]


def feature_group(feature_name):
    if feature_name.startswith("morgan_"):
        return "Morgan fingerprint"
    if feature_name.startswith("desc_"):
        return "RDKit descriptor"
    if feature_name.startswith("dock_"):
        return "Docking score/pose"
    if feature_name.startswith("ifp_"):
        return "Active-filtered IFP"
    return "Other"


# Chain-A residue numbering of PDB 5E85 → amino-acid single-letter label
# (single-letter code + position; e.g. A452 → S452 because residue 452 is Ser).
# Applied to any figure label that names a chain-A residue.
_RESIDUE_AA_MAP = {
    "A428": "T428",
    "A429": "V429",
    "A451": "F451",
    "A452": "S452",
    "A453": "T453",
    "A462": "T462",
}


def short_feature_name(feature_name):
    label = (
        feature_name.replace("morgan_", "morgan ")
        .replace("desc_", "")
        .replace("dock_", "Dock ")
        .replace("ifp_", "IFP ")
        .replace("interaction_", "")
    )
    label = label.replace("_", " ")
    for old, new in sorted(_RESIDUE_AA_MAP.items(), key=lambda x: -len(x[0])):
        label = label.replace(old, new)
    return label


def sample_for_shap(x_test, seed):
    if len(x_test) <= SHAP_MAX_SAMPLES:
        return x_test.copy()
    return x_test.sample(n=SHAP_MAX_SAMPLES, random_state=seed).copy()


def _load_bayesian_params():
    """Return {model_name: params_dict} from the master ranking CSV."""
    df = pd.read_csv(MODEL_COMPARISON_PATH)
    result = {}
    for _, row in df.iterrows():
        model_name = str(row["Model"])
        params_str = row.get("Params", None)
        if params_str and not pd.isna(params_str):
            try:
                result[model_name] = json.loads(params_str)
            except Exception:
                result[model_name] = {}
        else:
            result[model_name] = {}
    return result


def _build_extra_trees_bayesian(params, seed):
    from sklearn.ensemble import ExtraTreesRegressor
    p = {k: v for k, v in params.items() if k not in ("n_jobs", "random_state")}
    return ExtraTreesRegressor(**p, n_jobs=-1, random_state=seed)


def _build_catboost_bayesian(params, seed):
    from catboost import CatBoostRegressor
    p = {k: v for k, v in params.items()
         if k not in ("random_seed", "verbose", "allow_writing_files", "loss_function")}
    return CatBoostRegressor(
        **p,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def train_top3_predictions():
    step11 = load_step11()
    meta, x, y = step11.load_feature_table(FEATURE_TABLE_PATH, TARGET)
    label_derived_columns = [col for col in x.columns if col == "ifp_allowed_by_A5_threshold"]
    if label_derived_columns:
        print(f"Excluding label-derived helper columns: {label_derived_columns}")
        x = x.drop(columns=label_derived_columns)
    splits = step11.make_splits(meta, TEST_SIZE, RANDOM_STATE)

    bayesian_params = _load_bayesian_params()
    rows = []
    pred_frames = []
    trained = {}

    for split_name, (train_idx, test_idx) in splits.items():
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        meta_train = meta.iloc[train_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        print(f"Training top 3 for {split_name}: Extra Trees (Bayesian params)")
        et_params = bayesian_params.get("Extra Trees", {})
        extra_trees = _build_extra_trees_bayesian(et_params, RANDOM_STATE)
        extra_trees.fit(x_train, y_train)
        train_pred = extra_trees.predict(x_train)
        test_pred = extra_trees.predict(x_test)
        rows.append(metric_row("Extra Trees", split_name, len(train_idx), len(test_idx), y_train, train_pred, y_test, test_pred))
        pred_frames.append(prediction_frame(meta_train, y_train, train_pred, "Extra Trees", split_name, "train"))
        pred_frames.append(prediction_frame(meta_test, y_test, test_pred, "Extra Trees", split_name, "test"))
        trained[(split_name, "Extra Trees")] = {
            "model": extra_trees,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print(f"Training top 3 for {split_name}: CatBoost (Bayesian params)")
        cb_params = bayesian_params.get("CatBoost", {})
        catboost = _build_catboost_bayesian(cb_params, RANDOM_STATE)
        catboost.fit(x_train, y_train)
        train_pred = catboost.predict(x_train)
        test_pred = catboost.predict(x_test)
        rows.append(metric_row("CatBoost", split_name, len(train_idx), len(test_idx), y_train, train_pred, y_test, test_pred))
        pred_frames.append(prediction_frame(meta_train, y_train, train_pred, "CatBoost", split_name, "train"))
        pred_frames.append(prediction_frame(meta_test, y_test, test_pred, "CatBoost", split_name, "test"))
        trained[(split_name, "CatBoost")] = {
            "model": catboost,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print(f"Training top 3 for {split_name}: GNN (Bayesian params)")
        gnn_result = step11.train_gnn(meta_train, meta_test, x_train.values, x_test.values, y_train, y_test, RANDOM_STATE)
        train_pred, test_pred, gnn_model = gnn_result
        rows.append(metric_row("GNN + active-filtered IFP", split_name, len(train_idx), len(test_idx), y_train, train_pred, y_test, test_pred))
        pred_frames.append(prediction_frame(meta_train, y_train, train_pred, "GNN + active-filtered IFP", split_name, "train"))
        pred_frames.append(prediction_frame(meta_test, y_test, test_pred, "GNN + active-filtered IFP", split_name, "test"))
        trained[(split_name, "GNN + active-filtered IFP")] = {
            "model": gnn_model,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "meta_test": meta_test,
        }

    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True), trained


def compute_catboost_shap(trained, split):
    item = trained[(split, "CatBoost")]
    model = item["model"]
    x_shap = sample_for_shap(item["x_test"], RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    return model, x_shap, shap_values
