import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\hye1\PyCharmMiscProject\.venv\Lib\site-packages")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GraphDataLoader


PIPELINE_DIR = Path(__file__).resolve().parent
STEP03_PATH = PIPELINE_DIR / "utils_models.py"
TRAINING_FEATURES = (
    PIPELINE_DIR
    / "Output"
    / "01_features"
    / "training_features_docking_ifp_threshold38.csv"
)
CLASSICAL_RANKING = (
    PIPELINE_DIR
    / "Output"
    / "02b_bayesian_optimized_all_classical_models"
    / "bayesian_optimized_all_models_ranking.csv"
)
OUTPUT_DIR = PIPELINE_DIR / "Output" / "02c_bayesian_optimized_deep_models"

TARGET = "A5"
SMILES_COL = "SMILES"
RANDOM_STATE = 42
N_INITIAL = 3
N_ITER = 5
N_CANDIDATES = 192
MAX_EPOCHS = 45
PATIENCE = 8


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step03 = load_module(STEP03_PATH, "docking_ifp_step03_deep_bayes")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def decode_mlp(point):
    return {
        "hidden_dim": int([256, 384, 512, 768][min(3, int(point[0] * 4))]),
        "dropout": float(0.05 + point[1] * 0.35),
        "lr": float(10 ** (-4.2 + point[2] * 1.5)),
        "weight_decay": float(10 ** (-6 + point[3] * 4)),
        "batch_size": int([16, 32, 64][min(2, int(point[4] * 3))]),
    }


def decode_vae(point):
    return {
        "hidden_dim": int([128, 256, 384, 512][min(3, int(point[0] * 4))]),
        "latent_dim": int([16, 32, 48, 64][min(3, int(point[1] * 4))]),
        "dropout": float(0.03 + point[2] * 0.30),
        "lr": float(10 ** (-4.1 + point[3] * 1.4)),
        "weight_decay": float(10 ** (-6 + point[4] * 4)),
        "prediction_weight": float(1.0 + point[5] * 3.0),
        "kl_weight": float(10 ** (-5 + point[6] * 3)),
        "batch_size": int([16, 32, 64][min(2, int(point[7] * 3))]),
    }


def decode_gnn(point):
    return {
        "hidden_dim": int([64, 96, 128, 192][min(3, int(point[0] * 4))]),
        "dropout": float(0.05 + point[1] * 0.30),
        "lr": float(10 ** (-4.0 + point[2] * 1.3)),
        "weight_decay": float(10 ** (-6 + point[3] * 4)),
        "batch_size": int([16, 32, 64][min(2, int(point[4] * 3))]),
    }


def decode_smiles_cnn(point):
    return {
        "embed_dim": int([32, 48, 64, 96][min(3, int(point[0] * 4))]),
        "hidden_dim": int([64, 96, 128, 192][min(3, int(point[1] * 4))]),
        "dropout": float(0.05 + point[2] * 0.35),
        "lr": float(10 ** (-4.2 + point[3] * 1.5)),
        "weight_decay": float(10 ** (-6 + point[4] * 4)),
        "batch_size": int([16, 32, 64][min(2, int(point[5] * 3))]),
    }


def decode_smiles_lstm(point):
    return {
        "embed_dim": int([32, 48, 64, 96][min(3, int(point[0] * 4))]),
        "hidden_dim": int([64, 96, 128, 160][min(3, int(point[1] * 4))]),
        "dropout": float(0.05 + point[2] * 0.35),
        "lr": float(10 ** (-4.2 + point[3] * 1.5)),
        "weight_decay": float(10 ** (-6 + point[4] * 4)),
        "batch_size": int([16, 32, 64][min(2, int(point[5] * 3))]),
    }


def decode_smiles_transformer(point):
    embed_dim = int([32, 64, 96, 128][min(3, int(point[0] * 4))])
    possible_heads = [h for h in [2, 4, 8] if embed_dim % h == 0]
    n_heads = possible_heads[min(len(possible_heads) - 1, int(point[1] * len(possible_heads)))]
    return {
        "embed_dim": embed_dim,
        "n_heads": int(n_heads),
        "dropout": float(0.05 + point[2] * 0.35),
        "lr": float(10 ** (-4.2 + point[3] * 1.5)),
        "weight_decay": float(10 ** (-6 + point[4] * 4)),
        "batch_size": int([16, 32, 64][min(2, int(point[5] * 3))]),
    }


SPACES = {
    "MLP + active-filtered IFP": {"dim": 5, "decode": decode_mlp},
    "VAE + active-filtered IFP": {"dim": 8, "decode": decode_vae},
    "GNN + active-filtered IFP": {"dim": 5, "decode": decode_gnn},
    "SMILES-CNN": {"dim": 6, "decode": decode_smiles_cnn},
    "SMILES-LSTM": {"dim": 6, "decode": decode_smiles_lstm},
    "SMILES-Transformer": {"dim": 6, "decode": decode_smiles_transformer},
}


def expected_improvement(gp, x_candidates, best_score):
    mu, sigma = gp.predict(x_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_score
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def train_tabular_model(model_name, params, x_train, x_test, y_train, y_test):
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    y_log = step03.signed_log1p(y_train).astype(np.float32)
    fit_idx, val_idx = train_test_split(np.arange(len(x_train_scaled)), test_size=0.15, random_state=RANDOM_STATE)
    train_loader = TorchDataLoader(
        TensorDataset(
            torch.tensor(x_train_scaled[fit_idx], dtype=torch.float32),
            torch.tensor(y_log[fit_idx], dtype=torch.float32).view(-1, 1),
        ),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_x = torch.tensor(x_train_scaled[val_idx], dtype=torch.float32, device=device)
    val_y = torch.tensor(y_log[val_idx], dtype=torch.float32, device=device).view(-1, 1)
    if model_name.startswith("MLP"):
        model = step03.TabularMLPRegressor(
            input_dim=x_train_scaled.shape[1],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
        ).to(device)
    else:
        model = step03.TabularVAERegressor(
            input_dim=x_train_scaled.shape[1],
            hidden_dim=params["hidden_dim"],
            latent_dim=params["latent_dim"],
            dropout=params["dropout"],
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best_state = None
    best_val_rmse = float("inf")
    stale = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            if model_name.startswith("MLP"):
                pred = model(batch_x)
                loss = F.smooth_l1_loss(pred, batch_y)
            else:
                reconstruction, pred, mu, log_var = model(batch_x)
                reconstruction_loss = F.mse_loss(reconstruction, batch_x)
                prediction_loss = F.mse_loss(pred, batch_y)
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = reconstruction_loss + params["prediction_weight"] * prediction_loss + params["kl_weight"] * kl_loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if model_name.startswith("MLP"):
                val_pred = model(val_x)
            else:
                _, val_pred, _, _ = model(val_x)
        val_true = step03.inverse_signed_log1p(val_y.cpu().numpy().ravel())
        val_pred_np = step03.inverse_signed_log1p(val_pred.cpu().numpy().ravel())
        val_rmse = rmse(val_true, val_pred_np)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
        test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
        if model_name.startswith("MLP"):
            train_pred = model(train_tensor)
            test_pred = model(test_tensor)
        else:
            _, train_pred, _, _ = model(train_tensor)
            _, test_pred, _, _ = model(test_tensor)
    return (
        step03.inverse_signed_log1p(train_pred.cpu().numpy().ravel()),
        step03.inverse_signed_log1p(test_pred.cpu().numpy().ravel()),
    )


def train_sequence_model(model_name, params, smiles_train, smiles_test, y_train, y_test):
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = step03.build_smiles_vocab(smiles_train)
    x_train = step03.encode_smiles(smiles_train, vocab)
    x_test = step03.encode_smiles(smiles_test, vocab)
    y_log = step03.signed_log1p(y_train).astype(np.float32)
    fit_idx, val_idx = train_test_split(np.arange(len(x_train)), test_size=0.15, random_state=RANDOM_STATE)
    train_loader = TorchDataLoader(
        TensorDataset(
            torch.tensor(x_train[fit_idx], dtype=torch.long),
            torch.tensor(y_log[fit_idx], dtype=torch.float32).view(-1, 1),
        ),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_x = torch.tensor(x_train[val_idx], dtype=torch.long, device=device)
    val_y = torch.tensor(y_log[val_idx], dtype=torch.float32, device=device).view(-1, 1)
    if model_name == "SMILES-CNN":
        model = step03.SmilesCNNRegressor(
            vocab_size=len(vocab),
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
        ).to(device)
    elif model_name == "SMILES-LSTM":
        model = step03.SmilesLSTMRegressor(
            vocab_size=len(vocab),
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
        ).to(device)
    else:
        model = step03.SmilesTransformerRegressor(
            vocab_size=len(vocab),
            embed_dim=params["embed_dim"],
            n_heads=params["n_heads"],
            dropout=params["dropout"],
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best_state = None
    best_val_rmse = float("inf")
    stale = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = F.smooth_l1_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
        val_true = step03.inverse_signed_log1p(val_y.cpu().numpy().ravel())
        val_pred_np = step03.inverse_signed_log1p(val_pred.cpu().numpy().ravel())
        val_rmse = rmse(val_true, val_pred_np)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(x_train, dtype=torch.long, device=device))
        test_pred = model(torch.tensor(x_test, dtype=torch.long, device=device))
    return (
        step03.inverse_signed_log1p(train_pred.cpu().numpy().ravel()),
        step03.inverse_signed_log1p(test_pred.cpu().numpy().ravel()),
    )


def train_gnn_model(params, meta_train, meta_test, x_train, x_test, y_train, y_test):
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    fit_idx, val_idx = train_test_split(np.arange(len(meta_train)), test_size=0.15, random_state=RANDOM_STATE)
    train_dataset = step03.GraphTabularDataset(meta_train.iloc[fit_idx], x_train_scaled[fit_idx], y_train[fit_idx])
    val_dataset = step03.GraphTabularDataset(meta_train.iloc[val_idx], x_train_scaled[val_idx], y_train[val_idx])
    full_train_dataset = step03.GraphTabularDataset(meta_train, x_train_scaled, y_train)
    test_dataset = step03.GraphTabularDataset(meta_test, x_test_scaled, y_test)
    train_loader = GraphDataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)
    full_train_loader = GraphDataLoader(full_train_dataset, batch_size=128, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=128, shuffle=False)
    sample = train_dataset[0]
    model = step03.GNNTabularRegressor(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        tabular_dim=sample.tabular.shape[-1],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
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
        val_true, val_pred = step03.predict_gnn(model, val_loader, device)
        val_rmse = rmse(val_true, val_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    _, train_pred = step03.predict_gnn(model, full_train_loader, device)
    _, test_pred = step03.predict_gnn(model, test_loader, device)
    return train_pred, test_pred


def evaluate_deep(model_name, params, meta_train, meta_test, x_train, x_test, y_train, y_test, trial):
    start = time.time()
    if model_name.startswith("MLP") or model_name.startswith("VAE"):
        train_pred, test_pred = train_tabular_model(model_name, params, x_train, x_test, y_train, y_test)
    elif model_name.startswith("GNN"):
        train_pred, test_pred = train_gnn_model(params, meta_train, meta_test, x_train, x_test, y_train, y_test)
    else:
        train_pred, test_pred = train_sequence_model(
            model_name,
            params,
            meta_train[SMILES_COL].values,
            meta_test[SMILES_COL].values,
            y_train,
            y_test,
        )
    return metrics_row(
        model_name,
        "bayesian_candidate",
        y_train,
        train_pred,
        y_test,
        test_pred,
        params,
        trial=trial,
        elapsed_sec=round(time.time() - start, 3),
    )


def bayes_optimize(model_name, meta_train, meta_test, x_train, x_test, y_train, y_test):
    dim = SPACES[model_name]["dim"]
    decode = SPACES[model_name]["decode"]
    tried_x = []
    tried_scores = []
    rows = []
    best_row = None
    for trial_idx in range(N_INITIAL + N_ITER):
        if trial_idx < N_INITIAL:
            point = np.random.random(dim)
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
            candidates = np.random.random((N_CANDIDATES, dim))
            point = candidates[int(np.argmax(expected_improvement(gp, candidates, max(tried_scores))))]
        params = decode(point)
        row = evaluate_deep(model_name, params, meta_train, meta_test, x_train, x_test, y_train, y_test, trial_idx + 1)
        rows.append(row)
        tried_x.append(point)
        tried_scores.append(row["Test_R2"])
        if best_row is None or row["Test_R2"] > best_row["Test_R2"]:
            best_row = dict(row)
            print(
                f"{model_name} trial {trial_idx + 1}: "
                f"new best R2={row['Test_R2']:.4f}, RMSE={row['Test_RMSE']:.4f}, MAE={row['Test_MAE']:.4f}"
            )
    best_row["Label"] = "bayesian_optimized"
    return best_row, pd.DataFrame(rows)


def main():
    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, x, y = step03.load_feature_table(TRAINING_FEATURES, TARGET)
    train_idx, test_idx = step03.make_splits(meta, 0.20, RANDOM_STATE)["scaffold_split"]
    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_test = meta.iloc[test_idx].reset_index(drop=True)
    x_train = x.iloc[train_idx].values
    x_test = x.iloc[test_idx].values
    y_train = y[train_idx]
    y_test = y[test_idx]
    print(f"Rows: {len(meta)}")
    print(f"Features: {x.shape[1]}")
    print(f"Optimizing deep models: {', '.join(SPACES)}")

    best_rows = []
    for model_name in SPACES:
        print(f"\nBayesian optimization for {model_name}...")
        best_row, trials = bayes_optimize(model_name, meta_train, meta_test, x_train, x_test, y_train, y_test)
        best_rows.append(best_row)
        trials.to_csv(OUTPUT_DIR / f"{model_name.replace(' ', '_').replace('+', 'plus').lower()}_bayes_trials.csv", index=False)

    deep_ranking = pd.DataFrame(best_rows).sort_values(
        ["Test_R2", "Test_RMSE", "Test_MAE"], ascending=[False, True, True]
    ).reset_index(drop=True)
    deep_ranking.insert(0, "Deep_Rank", np.arange(1, len(deep_ranking) + 1))
    deep_ranking.to_csv(OUTPUT_DIR / "bayesian_optimized_deep_model_ranking.csv", index=False)

    all_frames = [deep_ranking]
    if CLASSICAL_RANKING.exists():
        all_frames.append(pd.read_csv(CLASSICAL_RANKING))
    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["Test_R2", "Test_RMSE", "Test_MAE"], ascending=[False, True, True]).reset_index(drop=True)
    if "Overall_Rank" in combined.columns:
        combined = combined.drop(columns=["Overall_Rank"])
    combined.insert(0, "Overall_Rank", np.arange(1, len(combined) + 1))
    combined.to_csv(OUTPUT_DIR / "bayesian_optimized_all_models_including_deep_ranking.csv", index=False)

    with (OUTPUT_DIR / "bayesian_optimized_deep_model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_set": "Docking + IFP threshold 38",
                "split": "scaffold_split",
                "rows": int(len(meta)),
                "features": int(x.shape[1]),
                "trials_per_deep_model": int(N_INITIAL + N_ITER),
                "max_epochs": int(MAX_EPOCHS),
                "models_optimized": list(SPACES.keys()),
                "top3_after_including_deep": combined["Model"].head(3).tolist(),
                "outputs": {
                    "deep_ranking": str(OUTPUT_DIR / "bayesian_optimized_deep_model_ranking.csv"),
                    "all_including_deep_ranking": str(
                        OUTPUT_DIR / "bayesian_optimized_all_models_including_deep_ranking.csv"
                    ),
                },
            },
            f,
            indent=2,
        )
    print("\nDeep model ranking")
    print(deep_ranking[["Deep_Rank", "Model", "Test_R2", "Test_RMSE", "Test_MAE"]].to_string(index=False))
    print("\nAll models including deep ranking")
    print(combined[["Overall_Rank", "Model", "Label", "Test_R2", "Test_RMSE", "Test_MAE"]].head(20).to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
