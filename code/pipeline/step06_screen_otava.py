import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\hye1\PyCharmMiscProject\.venv\Lib\site-packages")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as GraphDataLoader


PIPELINE_DIR = Path(__file__).resolve().parent
STEP16_PATH = PIPELINE_DIR / "docking_ifp_screening_utils.py"
STEP21_PATH = PIPELINE_DIR / "02c_bayesian_optimize_deep_models.py"
TRAINING_FEATURES = (
    PIPELINE_DIR
    / "Output"
    / "01_features"
    / "training_features_docking_ifp_threshold38.csv"
)
STRONG_COMPARISON = (
    PIPELINE_DIR
    / "Output"
    / "02a_bayesian_optimized_strong_models"
    / "bayesian_optimized_strong_model_comparison.csv"
)
DEEP_GNN_TRIALS = (
    PIPELINE_DIR
    / "Output"
    / "02c_bayesian_optimized_deep_models"
    / "gnn_plus_active-filtered_ifp_bayes_trials.csv"
)
SELECTED_BEST3 = PIPELINE_DIR / "Output" / "03_selected_best3" / "selected_best3_models.json"
OUTPUT_DIR = PIPELINE_DIR / "Output" / "04_screen_otava_best3"

TARGET = "A5"
SEED = 42
TOP_N = 200
PATIENCE = 8
MAX_EPOCHS = 60


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step16 = load_module(STEP16_PATH, "step16_docking_ifp_for_bayes_top3")
deep = load_module(STEP21_PATH, "deep_bayes_helpers_for_screen")


def load_tree_params(model_name):
    comparison = pd.read_csv(STRONG_COMPARISON)
    row = comparison[(comparison["Model"] == model_name) & (comparison["Label"] == "bayesian_optimized")]
    if row.empty:
        raise ValueError(f"No Bayesian optimized params found for {model_name}")
    return json.loads(row.iloc[0]["Params"])


def load_gnn_params():
    trials = pd.read_csv(DEEP_GNN_TRIALS)
    row = trials.sort_values(["Test_R2", "Test_RMSE", "Test_MAE"], ascending=[False, True, True]).iloc[0]
    return json.loads(row["Params"]), row.to_dict()


def train_optimized_gnn(meta, x_train, y_train, params):
    deep.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32)
    fit_idx, val_idx = train_test_split(np.arange(len(meta)), test_size=0.15, random_state=SEED, shuffle=True)
    train_dataset = step16.step11.GraphTabularDataset(meta.iloc[fit_idx], x_scaled[fit_idx], y_train[fit_idx])
    val_dataset = step16.step11.GraphTabularDataset(meta.iloc[val_idx], x_scaled[val_idx], y_train[val_idx])
    train_loader = GraphDataLoader(train_dataset, batch_size=int(params["batch_size"]), shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=int(params["batch_size"]), shuffle=False)
    sample = train_dataset[0]
    model = step16.step11.GNNTabularRegressor(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        tabular_dim=sample.tabular.shape[-1],
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    best_state = None
    best_val_rmse = float("inf")
    stale = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.tabular)
            loss = F.mse_loss(pred.view(-1), batch.y.view(-1))
            loss.backward()
            optimizer.step()
        val_true, val_pred = step16.step11.predict_gnn(model, val_loader, device)
        val_rmse = step16.step11.rmse(val_true, val_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_val_rmse


def train_models(x_train, y_train, meta):
    extra_params = load_tree_params("Extra Trees")
    cat_params = load_tree_params("CatBoost")
    gnn_params, gnn_trial = load_gnn_params()
    extra = ExtraTreesRegressor(**extra_params)
    cat = CatBoostRegressor(**cat_params)
    extra.fit(x_train, y_train)
    cat.fit(x_train, y_train)
    gnn, scaler, gnn_val_rmse = train_optimized_gnn(meta, x_train.values, y_train, gnn_params)
    return extra, cat, gnn, scaler, extra_params, cat_params, gnn_params, gnn_trial, gnn_val_rmse


def add_predictions(valid, x_external, extra, cat, gnn, gnn_scaler):
    out = valid.copy()
    out["pred_A5_ExtraTrees"] = extra.predict(x_external)
    out["pred_A5_CatBoost"] = cat.predict(x_external)
    out["pred_A5_GNN"] = step16.predict_gnn_external(gnn, gnn_scaler, valid, x_external.values)
    pred_cols = ["pred_A5_ExtraTrees", "pred_A5_CatBoost", "pred_A5_GNN"]
    for col in pred_cols:
        out[f"{col}_clipped"] = out[col].clip(0, 100)
    clipped_cols = [f"{col}_clipped" for col in pred_cols]
    out["consensus_A5_mean"] = out[clipped_cols].mean(axis=1)
    out["consensus_A5_median"] = out[clipped_cols].median(axis=1)
    out["consensus_A5_std"] = out[clipped_cols].std(axis=1)
    out["models_pred_ge_50"] = (out[clipped_cols] >= 50).sum(axis=1)
    out["models_pred_ge_60"] = (out[clipped_cols] >= 60).sum(axis=1)
    out["models_pred_ge_70"] = (out[clipped_cols] >= 70).sum(axis=1)
    out = out.sort_values(["consensus_A5_mean", "models_pred_ge_60"], ascending=[False, False]).reset_index(drop=True)
    out["screening_rank"] = np.arange(1, len(out) + 1)
    return out


def write_overlap_files(pred):
    keep = [
        "source_id",
        "COMPOUND ID",
        "PLATE NUMBER",
        "RACK POSITION",
        "SMILES",
        "consensus_A5_mean",
        "consensus_A5_median",
        "consensus_A5_std",
        "pred_A5_ExtraTrees_clipped",
        "pred_A5_CatBoost_clipped",
        "pred_A5_GNN_clipped",
        "has_otava_docking",
        "has_otava_ifp",
        "otava_docking_pose_count",
        "otava_ifp_pose_count",
        "screening_rank",
    ]
    keep = [col for col in keep if col in pred.columns]
    outputs = {}
    counts = {}
    for cutoff in [50, 60, 70]:
        frame = pred[pred[f"models_pred_ge_{cutoff}"].eq(3)].copy()
        csv_path = OUTPUT_DIR / f"otava_bayesian_top3_extra_catboost_gnn_overlap_predA5_ge{cutoff}_with_plate_rack.csv"
        xlsx_path = OUTPUT_DIR / f"otava_bayesian_top3_extra_catboost_gnn_overlap_predA5_ge{cutoff}_with_plate_rack.xlsx"
        frame[keep].to_csv(csv_path, index=False)
        frame[keep].to_excel(xlsx_path, index=False)
        outputs[f"overlap_ge{cutoff}_csv"] = str(csv_path)
        outputs[f"overlap_ge{cutoff}_xlsx"] = str(xlsx_path)
        counts[f"overlap_pred_ge_{cutoff}_count"] = int(len(frame))
    return outputs, counts


def main():
    step16.step12.set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SELECTED_BEST3.exists():
        selected = json.loads(SELECTED_BEST3.read_text(encoding="utf-8"))["selected_models"]
        supported = ["Extra Trees", "CatBoost", "GNN + active-filtered IFP"]
        if selected != supported:
            raise ValueError(
                "This screening script supports the selected top 3 "
                f"{supported}, but selected_best3_models.json contains {selected}."
            )
    meta, x_train, y_train = step16.step11.load_feature_table(TRAINING_FEATURES, TARGET)
    feature_cols = list(x_train.columns)
    print(f"Training rows: {len(meta)}")
    print(f"Feature count: {len(feature_cols)}")
    print("Training final Bayesian top 3: Extra Trees, CatBoost, GNN...")
    extra, cat, gnn, gnn_scaler, extra_params, cat_params, gnn_params, gnn_trial, gnn_val_rmse = train_models(
        x_train, y_train, meta
    )

    library = step16.step12.read_library(step16.OTAVA_LIBRARY, "OTAVA")
    valid = library[library["valid_smiles"]].reset_index(drop=True)
    invalid = library[~library["valid_smiles"]].copy()
    x_external, valid, shared_dock_cols, shared_ifp_cols, dock_raw, ifp_raw = step16.build_otava_feature_matrix(
        valid, feature_cols
    )
    pred = add_predictions(valid, x_external, extra, cat, gnn, gnn_scaler)

    all_path = OUTPUT_DIR / "otava_bayesian_top3_extra_catboost_gnn_predictions.csv"
    top_csv = OUTPUT_DIR / f"otava_bayesian_top3_extra_catboost_gnn_top{TOP_N}_hits.csv"
    top_xlsx = OUTPUT_DIR / f"otava_bayesian_top3_extra_catboost_gnn_top{TOP_N}_hits.xlsx"
    coverage_path = OUTPUT_DIR / "otava_bayesian_top3_extra_catboost_gnn_coverage.csv"
    pred.to_csv(all_path, index=False)
    pred.head(TOP_N).to_csv(top_csv, index=False)
    pred.head(TOP_N).to_excel(top_xlsx, index=False)
    valid.to_csv(coverage_path, index=False)
    overlap_outputs, overlap_counts = write_overlap_files(pred)
    if not invalid.empty:
        invalid.to_csv(OUTPUT_DIR / "otava_invalid_smiles.csv", index=False)

    joblib.dump(extra, OUTPUT_DIR / "final_bayesian_extra_trees_model.pkl")
    joblib.dump(cat, OUTPUT_DIR / "final_bayesian_catboost_model.pkl")
    joblib.dump(gnn_scaler, OUTPUT_DIR / "final_bayesian_gnn_scaler.pkl")
    torch.save(
        {
            "state_dict": gnn.state_dict(),
            "feature_cols": feature_cols,
            "params": gnn_params,
            "feature_mode": "Morgan + RDKit + docking + IFP; Bayesian top-3 screen.",
        },
        OUTPUT_DIR / "final_bayesian_gnn_model.pt",
    )

    summary = {
        "library": "OTAVA",
        "models": ["Bayesian Extra Trees", "Bayesian CatBoost", "Bayesian GNN"],
        "feature_mode": "Docking + IFP threshold 38; same feature alignment as OTAVA docking/IFP screen.",
        "training_rows": int(len(meta)),
        "feature_count": int(len(feature_cols)),
        "feature_families": {
            "morgan": int(sum(c.startswith("morgan_") for c in feature_cols)),
            "rdkit": int(sum(c.startswith("desc_") for c in feature_cols)),
            "docking": int(sum(c.startswith("dock_") for c in feature_cols)),
            "ifp": int(sum(c.startswith("ifp_") for c in feature_cols)),
        },
        "optimized_params": {
            "Extra Trees": extra_params,
            "CatBoost": cat_params,
            "GNN": gnn_params,
        },
        "gnn_best_bayesian_trial": gnn_trial,
        "gnn_validation_rmse_during_final_training": float(gnn_val_rmse),
        "valid_unique_compounds": int(len(valid)),
        "invalid_smiles": int(len(invalid)),
        "library_compounds_with_docking": int(valid["has_otava_docking"].sum()),
        "library_compounds_with_ifp": int(valid["has_otava_ifp"].sum()),
        "shared_docking_feature_columns_used": int(len(shared_dock_cols)),
        "shared_ifp_feature_columns_used": int(len(shared_ifp_cols)),
        **overlap_counts,
        "outputs": {
            "all_predictions": str(all_path),
            "top_hits_csv": str(top_csv),
            "top_hits_xlsx": str(top_xlsx),
            "coverage": str(coverage_path),
            **overlap_outputs,
        },
    }
    with (OUTPUT_DIR / "bayesian_top3_extra_catboost_gnn_screening_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTop 25 OTAVA hits: Bayesian Extra Trees + CatBoost + GNN")
    print(
        pred[
            [
                "source_id",
                "PLATE NUMBER",
                "RACK POSITION",
                "consensus_A5_mean",
                "consensus_A5_std",
                "models_pred_ge_60",
                "pred_A5_ExtraTrees_clipped",
                "pred_A5_CatBoost_clipped",
                "pred_A5_GNN_clipped",
                "SMILES",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
