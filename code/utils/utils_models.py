import argparse
import json
import math
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import TensorDataset
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader as GraphDataLoader
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError:
    torch = None
    nn = None
    F = None
    TorchDataLoader = None
    TensorDataset = None
    Data = None
    Dataset = object
    GraphDataLoader = None
    GATv2Conv = None
    global_mean_pool = None


RDLogger.DisableLog("rdApp.*")

ID_COL = "Name"
SMILES_COL = "SMILES"
TARGET = "A5"
RANDOM_STATE = 42
GNN_EPOCHS = 60
VAE_EPOCHS = 80
MLP_EPOCHS = 80
SEQUENCE_EPOCHS = 60
BATCH_SIZE = 32
VAE_LATENT_DIM = 32
SMILES_MAX_LEN = 180

ATOM_SYMBOLS = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple regression models using Morgan + RDKit + docking summary + "
            "active-filtered interaction fingerprint features."
        )
    )
    _here = Path(__file__).parent
    parser.add_argument(
        "--features",
        type=Path,
        default=_here / "Output" / "02_features" / "active_filtered_compound_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_here / "Output" / "03_model_benchmark",
    )
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-deep", action="store_true", help="Skip VAE and GNN models.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def signed_log1p(values):
    if torch is not None and isinstance(values, torch.Tensor):
        return torch.sign(values) * torch.log1p(torch.abs(values))
    values = np.asarray(values)
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_signed_log1p(values):
    if torch is not None and isinstance(values, torch.Tensor):
        return torch.sign(values) * torch.expm1(torch.abs(values))
    values = np.asarray(values)
    return np.sign(values) * np.expm1(np.abs(values))


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def metric_row(model_name, split_name, y_train, train_pred, y_test, test_pred):
    return {
        "Model": model_name,
        "Split": split_name,
        "Train_N": int(len(y_train)),
        "Test_N": int(len(y_test)),
        "Train_R2": r2_score(y_train, train_pred),
        "Test_R2": r2_score(y_test, test_pred),
        "Train_RMSE": rmse(y_train, train_pred),
        "Test_RMSE": rmse(y_test, test_pred),
        "Train_MAE": mean_absolute_error(y_train, train_pred),
        "Test_MAE": mean_absolute_error(y_test, test_pred),
    }


def load_feature_table(path, target):
    df = pd.read_csv(path)
    required = {ID_COL, SMILES_COL, target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    feature_cols = [
        col
        for col in df.columns
        if (
            col.startswith("morgan_")
            or col.startswith("desc_")
            or col.startswith("dock_")
            or col.startswith("ifp_")
        )
    ]
    feature_cols = [col for col in feature_cols if col != "ifp_allowed_by_A5_threshold"]
    if not feature_cols:
        raise ValueError("No Morgan/RDKit/docking/IFP feature columns were found.")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[target].astype(float).values
    y = np.clip(y, 0.0, 100.0)  # clamp inhibition to physically meaningful range
    meta = df[[ID_COL, SMILES_COL, target]].copy()
    meta[target] = y  # keep meta consistent with clipped values
    return meta, x, y


def scaffold_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid_smiles"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or "no_scaffold"


def make_splits(meta, test_size, seed):
    random_train, random_test = train_test_split(
        np.arange(len(meta)),
        test_size=test_size,
        random_state=seed,
    )
    scaffolds = meta[SMILES_COL].map(scaffold_from_smiles)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    scaffold_train, scaffold_test = next(splitter.split(np.arange(len(meta)), groups=scaffolds))
    return {
        "random_split": (random_train, random_test),
        "scaffold_split": (scaffold_train, scaffold_test),
    }


def build_classical_models(seed):
    models = {
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            min_child_weight=3,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=seed),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Elastic Net": make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=seed),
        ),
        "SVR": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    }
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=700,
            learning_rate=0.035,
            depth=5,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    return models


def one_hot_with_unknown(value, choices):
    encoded = [int(value == choice) for choice in choices]
    encoded.append(int(value not in choices))
    return encoded


def atom_features(atom):
    features = []
    features.extend(one_hot_with_unknown(atom.GetSymbol(), ATOM_SYMBOLS))
    features.extend(one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend(one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS))
    features.append(atom.GetAtomicNum() / 100.0)
    features.append(atom.GetTotalDegree() / 6.0)
    features.append(atom.GetFormalCharge())
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))
    return features


def bond_features(bond):
    features = one_hot_with_unknown(bond.GetBondType(), BOND_TYPES)
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    features.append(float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE))
    return features


class TabularVAERegressor(nn.Module):
    def __init__(self, input_dim, latent_dim=VAE_LATENT_DIM, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
        )
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return self.mu(encoded), self.log_var(encoded)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        prediction = self.regressor(z)
        return reconstruction, prediction, mu, log_var


class TabularMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_tabular_mlp(x_train, x_test, y_train, y_test, seed):
    if torch is None:
        return None
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    y_train_log = signed_log1p(y_train).astype(np.float32)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_scaled,
        y_train_log,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
    )
    train_loader = TorchDataLoader(
        TensorDataset(
            torch.tensor(x_fit, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    model = TabularMLPRegressor(input_dim=x_train_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_state = None
    best_val_rmse = float("inf")
    stale_epochs = 0

    for _ in range(MLP_EPOCHS):
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
        val_true = inverse_signed_log1p(np.array(val_y.cpu().tolist(), dtype=np.float32).ravel())
        val_pred = inverse_signed_log1p(np.array(val_pred.cpu().tolist(), dtype=np.float32).ravel())
        val_rmse = rmse(val_true, val_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 15:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(x_train_scaled, dtype=torch.float32, device=device))
        test_pred = model(torch.tensor(x_test_scaled, dtype=torch.float32, device=device))
    train_pred = inverse_signed_log1p(np.array(train_pred.cpu().tolist(), dtype=np.float32).ravel())
    test_pred = inverse_signed_log1p(np.array(test_pred.cpu().tolist(), dtype=np.float32).ravel())
    return train_pred, test_pred, model


def train_vae(x_train, x_test, y_train, y_test, seed):
    if torch is None:
        return None
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    y_train_log = signed_log1p(y_train).astype(np.float32)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_scaled,
        y_train_log,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
    )
    train_loader = TorchDataLoader(
        TensorDataset(
            torch.tensor(x_fit, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    model = TabularVAERegressor(input_dim=x_train_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = None
    best_val_rmse = float("inf")
    stale_epochs = 0

    for _ in range(VAE_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            reconstruction, prediction, mu, log_var = model(batch_x)
            reconstruction_loss = F.mse_loss(reconstruction, batch_x)
            prediction_loss = F.mse_loss(prediction, batch_y)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + 2.0 * prediction_loss + 1e-3 * kl_loss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, val_pred, _, _ = model(val_x)
        val_true = inverse_signed_log1p(np.array(val_y.cpu().tolist(), dtype=np.float32).ravel())
        val_pred = inverse_signed_log1p(np.array(val_pred.cpu().tolist(), dtype=np.float32).ravel())
        val_rmse = rmse(val_true, val_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 15:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, train_pred, _, _ = model(torch.tensor(x_train_scaled, dtype=torch.float32, device=device))
        _, test_pred, _, _ = model(torch.tensor(x_test_scaled, dtype=torch.float32, device=device))
    train_pred = inverse_signed_log1p(np.array(train_pred.cpu().tolist(), dtype=np.float32).ravel())
    test_pred = inverse_signed_log1p(np.array(test_pred.cpu().tolist(), dtype=np.float32).ravel())
    return train_pred, test_pred, model


def build_smiles_vocab(smiles_values):
    chars = sorted({ch for smiles in smiles_values for ch in str(smiles)})
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_smiles(smiles_values, vocab, max_len=SMILES_MAX_LEN):
    encoded = np.zeros((len(smiles_values), max_len), dtype=np.int64)
    for row_idx, smiles in enumerate(smiles_values):
        text = str(smiles)[:max_len]
        for col_idx, ch in enumerate(text):
            encoded[row_idx, col_idx] = vocab.get(ch, vocab["<unk>"])
    return encoded


class SmilesCNNRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, dropout=0.20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=7, padding=3)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        x = self.embedding(tokens).transpose(1, 2)
        pooled = [
            F.adaptive_max_pool1d(F.relu(self.conv3(x)), 1).squeeze(-1),
            F.adaptive_max_pool1d(F.relu(self.conv5(x)), 1).squeeze(-1),
            F.adaptive_max_pool1d(F.relu(self.conv7(x)), 1).squeeze(-1),
        ]
        return self.head(self.dropout(torch.cat(pooled, dim=1)))


class SmilesLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, dropout=0.20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        mask = tokens.ne(0)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        x = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.head(self.dropout(h))


class SmilesTransformerRegressor(nn.Module):
    def __init__(self, vocab_size, max_len=SMILES_MAX_LEN, embed_dim=64, n_heads=4, dropout=0.20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        mask = tokens.eq(0)
        x = self.embedding(tokens) + self.position(positions)
        x = self.encoder(x, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def train_sequence_model(model_name, model_class, smiles_train, smiles_test, y_train, y_test, seed):
    if torch is None:
        return None
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_smiles_vocab(smiles_train)
    x_train = encode_smiles(smiles_train, vocab)
    x_test = encode_smiles(smiles_test, vocab)
    y_train_log = signed_log1p(y_train).astype(np.float32)

    train_idx, val_idx = train_test_split(
        np.arange(len(x_train)),
        test_size=0.15,
        random_state=seed,
        shuffle=True,
    )
    train_loader = TorchDataLoader(
        TensorDataset(
            torch.tensor(x_train[train_idx], dtype=torch.long),
            torch.tensor(y_train_log[train_idx], dtype=torch.float32).view(-1, 1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_x = torch.tensor(x_train[val_idx], dtype=torch.long, device=device)
    val_y = torch.tensor(y_train_log[val_idx], dtype=torch.float32, device=device).view(-1, 1)

    model = model_class(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    best_state = None
    best_val_rmse = float("inf")
    stale_epochs = 0

    for _ in range(SEQUENCE_EPOCHS):
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
        val_true = inverse_signed_log1p(np.array(val_y.cpu().tolist(), dtype=np.float32).ravel())
        val_pred = inverse_signed_log1p(np.array(val_pred.cpu().tolist(), dtype=np.float32).ravel())
        val_rmse = rmse(val_true, val_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 12:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(x_train, dtype=torch.long, device=device))
        test_pred = model(torch.tensor(x_test, dtype=torch.long, device=device))
    train_pred = inverse_signed_log1p(np.array(train_pred.cpu().tolist(), dtype=np.float32).ravel())
    test_pred = inverse_signed_log1p(np.array(test_pred.cpu().tolist(), dtype=np.float32).ravel())
    model.vocab = vocab
    model.model_name = model_name
    return train_pred, test_pred, model


class GraphTabularDataset(Dataset):
    def __init__(self, meta, x_tabular, y):
        super().__init__()
        self.rows = []
        for idx, row in meta.reset_index(drop=True).iterrows():
            mol = Chem.MolFromSmiles(str(row[SMILES_COL]))
            if mol is None:
                continue
            atom_x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                features = bond_features(bond)
                edge_index.extend([[start, end], [end, start]])
                edge_attr.extend([features, features])
            if not edge_index:
                edge_index = [[0, 0]]
                edge_attr = [[0.0] * (len(BOND_TYPES) + 1 + 3)]

            graph = Data(
                x=atom_x,
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                tabular=torch.tensor(x_tabular[idx], dtype=torch.float).view(1, -1),
                y=torch.tensor([signed_log1p(y[idx])], dtype=torch.float),
            )
            self.rows.append(graph)

    def len(self):
        return len(self.rows)

    def get(self, idx):
        return self.rows[idx]


class GNNTabularRegressor(nn.Module):
    def __init__(self, node_dim, edge_dim, tabular_dim, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, tabular):
        tabular = tabular.view(-1, self.tabular_dim)
        x = self.node_encoder(x)
        residual = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(F.relu(x + residual))
        x = self.dropout(x)
        residual = x
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(F.relu(x + residual))
        x = self.dropout(x)
        graph_embedding = global_mean_pool(x, batch)
        tabular_embedding = self.tabular_encoder(tabular)
        return self.head(torch.cat([graph_embedding, tabular_embedding], dim=1))


def predict_gnn(model, loader, device):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.tabular)
            targets.extend(inverse_signed_log1p(np.array(batch.y.cpu().tolist(), dtype=np.float32).ravel()).tolist())
            predictions.extend(inverse_signed_log1p(np.array(pred.cpu().tolist(), dtype=np.float32).ravel()).tolist())
    return np.asarray(targets), np.asarray(predictions)


def train_gnn(meta_train, meta_test, x_train, x_test, y_train, y_test, seed):
    if torch is None or GraphDataLoader is None:
        return None
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    rel_idx = np.arange(len(meta_train))
    fit_idx, val_idx = train_test_split(rel_idx, test_size=0.15, random_state=seed, shuffle=True)
    train_dataset = GraphTabularDataset(meta_train.iloc[fit_idx], x_train_scaled[fit_idx], y_train[fit_idx])
    val_dataset = GraphTabularDataset(meta_train.iloc[val_idx], x_train_scaled[val_idx], y_train[val_idx])
    full_train_dataset = GraphTabularDataset(meta_train, x_train_scaled, y_train)
    test_dataset = GraphTabularDataset(meta_test, x_test_scaled, y_test)

    train_loader = GraphDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    full_train_loader = GraphDataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    sample = train_dataset[0]

    model = GNNTabularRegressor(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        tabular_dim=sample.tabular.shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = None
    best_val_rmse = float("inf")
    stale_epochs = 0

    for _ in range(GNN_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.tabular)
            loss = F.mse_loss(pred.view(-1), batch.y.view(-1))
            loss.backward()
            optimizer.step()

        val_true, val_pred = predict_gnn(model, val_loader, device)
        val_rmse = rmse(val_true, val_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 15:
            break

    model.load_state_dict(best_state)
    model.scaler = scaler
    train_true, train_pred = predict_gnn(model, full_train_loader, device)
    test_true, test_pred = predict_gnn(model, test_loader, device)
    return train_pred, test_pred, model


def evaluate_models(meta, x, y, splits, output_dir, seed, skip_deep=False):
    rows = []
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    for split_name, (train_idx, test_idx) in splits.items():
        print(f"\n=== {split_name} ===")
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        meta_train = meta.iloc[train_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        for name, model in build_classical_models(seed).items():
            print(f"Training {name}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            rows.append(metric_row(name, split_name, y_train, train_pred, y_test, test_pred))
            pred_df = meta_test[[ID_COL, SMILES_COL]].copy()
            pred_df["A5_true"] = y_test
            pred_df["A5_pred"] = test_pred
            pred_df.to_csv(predictions_dir / f"{name.replace(' ', '_')}__{split_name}.csv", index=False)
            joblib.dump(model, output_dir / f"{name.replace(' ', '_')}__{split_name}.pkl")

        if not skip_deep and torch is not None:
            print("Training MLP + active-filtered IFP...")
            mlp_result = train_tabular_mlp(x_train.values, x_test.values, y_train, y_test, seed)
            if mlp_result is not None:
                train_pred, test_pred, model = mlp_result
                rows.append(metric_row("MLP + active-filtered IFP", split_name, y_train, train_pred, y_test, test_pred))
                torch.save(model.state_dict(), output_dir / f"MLP_active_filtered_IFP__{split_name}.pt")

            print("Training VAE + active-filtered IFP...")
            vae_result = train_vae(x_train.values, x_test.values, y_train, y_test, seed)
            if vae_result is not None:
                train_pred, test_pred, model = vae_result
                rows.append(metric_row("VAE + active-filtered IFP", split_name, y_train, train_pred, y_test, test_pred))
                torch.save(model.state_dict(), output_dir / f"VAE_active_filtered_IFP__{split_name}.pt")

            print("Training GNN + active-filtered IFP...")
            gnn_result = train_gnn(meta_train, meta_test, x_train.values, x_test.values, y_train, y_test, seed)
            if gnn_result is not None:
                train_pred, test_pred, model = gnn_result
                rows.append(metric_row("GNN + active-filtered IFP", split_name, y_train, train_pred, y_test, test_pred))
                torch.save(model.state_dict(), output_dir / f"GNN_active_filtered_IFP__{split_name}.pt")

            sequence_models = {
                "SMILES-CNN": SmilesCNNRegressor,
                "SMILES-LSTM": SmilesLSTMRegressor,
                "SMILES-Transformer": SmilesTransformerRegressor,
            }
            for seq_name, seq_class in sequence_models.items():
                print(f"Training {seq_name}...")
                seq_result = train_sequence_model(
                    seq_name,
                    seq_class,
                    meta_train[SMILES_COL].values,
                    meta_test[SMILES_COL].values,
                    y_train,
                    y_test,
                    seed,
                )
                if seq_result is not None:
                    train_pred, test_pred, model = seq_result
                    rows.append(metric_row(seq_name, split_name, y_train, train_pred, y_test, test_pred))
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "vocab": model.vocab,
                            "max_len": SMILES_MAX_LEN,
                        },
                        output_dir / f"{seq_name.replace(' ', '_')}__{split_name}.pt",
                    )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta, x, y = load_feature_table(args.features, args.target)
    splits = make_splits(meta, args.test_size, args.seed)
    metrics = evaluate_models(meta, x, y, splits, args.output_dir, args.seed, skip_deep=args.skip_deep)
    metrics = metrics.sort_values(["Split", "Test_R2"], ascending=[True, False]).reset_index(drop=True)
    metrics.to_csv(args.output_dir / "model_comparison_active_filtered_ifp.csv", index=False)

    summary = {
        "rows": int(len(meta)),
        "feature_set": "Morgan + RDKit + docking summary + active-filtered IFP",
        "feature_count": int(x.shape[1]),
        "best_by_split": metrics.sort_values("Test_R2", ascending=False)
        .groupby("Split")
        .head(1)
        .to_dict(orient="records"),
        "outputs": {
            "metrics": str(args.output_dir / "model_comparison_active_filtered_ifp.csv"),
            "predictions_dir": str(args.output_dir / "predictions"),
        },
    }
    with (args.output_dir / "model_comparison_active_filtered_ifp_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n==============================")
    print("MODEL COMPARISON: Morgan + RDKit + docking summary + active-filtered IFP")
    print("==============================")
    print(metrics[["Model", "Split", "Test_R2", "Test_RMSE", "Test_MAE"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
