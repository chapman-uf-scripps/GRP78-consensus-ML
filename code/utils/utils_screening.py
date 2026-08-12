import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


RDLogger.DisableLog("rdApp.*")

ID_COL = "Name"
SMILES_COL = "SMILES"
TARGET = "A5"
RANDOM_STATE = 42
BATCH_SIZE = 32
MLP_EPOCHS = 90
VAE_EPOCHS = 90
VAE_LATENT_DIM = 32
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train final VAE, MLP, and LightGBM models on the active-filtered feature "
            "table, then screen OTAVA and FDA libraries."
        )
    )
    _here = Path(__file__).parent
    parser.add_argument(
        "--training-features",
        type=Path,
        default=_here / "Output" / "02_features" / "active_filtered_compound_features.csv",
    )
    parser.add_argument(
        "--otava",
        type=Path,
        default=_here / "Input" / "Libraries" / "otava_library.xlsx",
    )
    parser.add_argument(
        "--fda",
        type=Path,
        default=_here / "Input" / "Libraries" / "FDA.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_here / "Output" / "04_screening_otava_d",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument(
        "--exclude-docking-ifp",
        action="store_true",
        help=(
            "Use only Morgan + RDKit columns. Default keeps docking/IFP columns in the "
            "model but fills them as zero for external compounds without docking/IFP."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def signed_log1p(values):
    if isinstance(values, torch.Tensor):
        return torch.sign(values) * torch.log1p(torch.abs(values))
    values = np.asarray(values)
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_signed_log1p(values):
    if isinstance(values, torch.Tensor):
        return torch.sign(values) * torch.expm1(torch.abs(values))
    values = np.asarray(values)
    return np.sign(values) * np.expm1(np.abs(values))


def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def canonical_smiles(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def make_morgan_features(smiles, radius=FINGERPRINT_RADIUS, n_bits=FINGERPRINT_BITS):
    mols = [mol_from_smiles(s) for s in smiles]
    matrix = np.zeros((len(mols), n_bits), dtype=np.int8)
    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, matrix[idx])
    cols = [f"morgan_{idx:04d}" for idx in range(n_bits)]
    return pd.DataFrame(matrix, columns=cols), mols


def make_rdkit_descriptors(mols):
    rows = []
    for mol in mols:
        if mol is None:
            rows.append(
                {
                    "desc_mol_wt": np.nan,
                    "desc_logp": np.nan,
                    "desc_tpsa": np.nan,
                    "desc_hbd": np.nan,
                    "desc_hba": np.nan,
                    "desc_rot_bonds": np.nan,
                    "desc_heavy_atoms": np.nan,
                    "desc_rings": np.nan,
                    "desc_aromatic_rings": np.nan,
                    "desc_fraction_csp3": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "desc_mol_wt": Descriptors.MolWt(mol),
                    "desc_logp": Descriptors.MolLogP(mol),
                    "desc_tpsa": Descriptors.TPSA(mol),
                    "desc_hbd": Descriptors.NumHDonors(mol),
                    "desc_hba": Descriptors.NumHAcceptors(mol),
                    "desc_rot_bonds": Descriptors.NumRotatableBonds(mol),
                    "desc_heavy_atoms": Descriptors.HeavyAtomCount(mol),
                    "desc_rings": Descriptors.RingCount(mol),
                    "desc_aromatic_rings": Descriptors.NumAromaticRings(mol),
                    "desc_fraction_csp3": Descriptors.FractionCSP3(mol),
                }
            )
    return pd.DataFrame(rows)


def feature_columns(df, exclude_docking_ifp=False):
    prefixes = ("morgan_", "desc_") if exclude_docking_ifp else ("morgan_", "desc_", "dock_", "ifp_")
    cols = [col for col in df.columns if col.startswith(prefixes)]
    # This column encodes the measured A5 threshold and must not be used prospectively.
    cols = [col for col in cols if col != "ifp_allowed_by_A5_threshold"]
    return cols


def load_training_table(path, exclude_docking_ifp=False):
    df = pd.read_csv(path)
    required = {ID_COL, SMILES_COL, TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Training feature table is missing required columns: {missing}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    cols = feature_columns(df, exclude_docking_ifp=exclude_docking_ifp)
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = np.clip(df[TARGET].astype(float).values, 0.0, 100.0)  # clamp to physically meaningful range
    meta = df[[ID_COL, SMILES_COL, TARGET]].copy()
    meta[TARGET] = y  # keep meta consistent with clipped values
    return meta, x, y, cols


def read_library(path, library_name):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if SMILES_COL not in df.columns:
        matches = [col for col in df.columns if col.lower() in {"smiles", "smile"}]
        if not matches:
            raise ValueError(f"{path} does not contain a SMILES column.")
        df = df.rename(columns={matches[0]: SMILES_COL})

    id_candidates = [
        "COMPOUND ID",
        "ID",
        "Name",
        "Compound_ID",
        "InChIKey",
        "INN",
        "CAS_RN",
    ]
    id_col = next((col for col in id_candidates if col in df.columns), None)
    out = df.copy()
    out["library"] = library_name
    out["source_id"] = out[id_col].astype(str) if id_col else [f"{library_name}_{idx+1}" for idx in range(len(out))]
    out[SMILES_COL] = out[SMILES_COL].astype(str)
    out["canonical_smiles"] = out[SMILES_COL].map(canonical_smiles)
    out["valid_smiles"] = out["canonical_smiles"].astype(bool)
    out = out.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
    return out


def build_external_feature_matrix(library_df, training_columns):
    morgan_x, mols = make_morgan_features(library_df[SMILES_COL])
    desc_x = make_rdkit_descriptors(mols)
    generated = pd.concat([morgan_x, desc_x], axis=1)

    x = pd.DataFrame(0.0, index=np.arange(len(library_df)), columns=training_columns)
    shared_cols = [col for col in generated.columns if col in x.columns]
    x.loc[:, shared_cols] = generated[shared_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x


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


def predict_torch_model(model, scaler, x):
    device = next(model.parameters()).device
    x_scaled = scaler.transform(x).astype(np.float32)
    loader = DataLoader(torch.tensor(x_scaled, dtype=torch.float32), batch_size=256, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            if isinstance(model, TabularVAERegressor):
                _, pred, _, _ = model(batch_x)
            else:
                pred = model(batch_x)
            preds.extend(inverse_signed_log1p(pred.cpu().numpy().ravel()).tolist())
    return np.asarray(preds)


def train_mlp(x, y, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    y_log = signed_log1p(y).astype(np.float32)
    fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.15, random_state=seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_scaled[fit_idx], dtype=torch.float32),
            torch.tensor(y_log[fit_idx], dtype=torch.float32).view(-1, 1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_x = torch.tensor(x_scaled[val_idx], dtype=torch.float32, device=device)
    val_y = torch.tensor(y_log[val_idx], dtype=torch.float32, device=device).view(-1, 1)
    model = TabularMLPRegressor(input_dim=x_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    for _ in range(MLP_EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = F.smooth_l1_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
        val_true = inverse_signed_log1p(val_y.cpu().numpy().ravel())
        val_pred = inverse_signed_log1p(val_pred.cpu().numpy().ravel())
        val_rmse = math.sqrt(np.mean((val_true - val_pred) ** 2))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 15:
            break

    model.load_state_dict(best_state)
    return model, scaler


def train_vae(x, y, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    y_log = signed_log1p(y).astype(np.float32)
    fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.15, random_state=seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_scaled[fit_idx], dtype=torch.float32),
            torch.tensor(y_log[fit_idx], dtype=torch.float32).view(-1, 1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_x = torch.tensor(x_scaled[val_idx], dtype=torch.float32, device=device)
    val_y = torch.tensor(y_log[val_idx], dtype=torch.float32, device=device).view(-1, 1)
    model = TabularVAERegressor(input_dim=x_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    for _ in range(VAE_EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
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
        val_true = inverse_signed_log1p(val_y.cpu().numpy().ravel())
        val_pred = inverse_signed_log1p(val_pred.cpu().numpy().ravel())
        val_rmse = math.sqrt(np.mean((val_true - val_pred) ** 2))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 15:
            break

    model.load_state_dict(best_state)
    return model, scaler


def train_lightgbm(x, y, seed):
    model = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x, y)
    return model


def add_predictions(library_df, x_external, models):
    out = library_df.copy()
    out["pred_A5_VAE"] = predict_torch_model(models["vae"][0], models["vae"][1], x_external)
    out["pred_A5_MLP"] = predict_torch_model(models["mlp"][0], models["mlp"][1], x_external)
    out["pred_A5_LightGBM"] = models["lightgbm"].predict(x_external)
    pred_cols = ["pred_A5_VAE", "pred_A5_MLP", "pred_A5_LightGBM"]
    for col in pred_cols:
        out[f"{col}_clipped"] = out[col].clip(0, 100)
    clipped_cols = [f"{col}_clipped" for col in pred_cols]
    out["consensus_A5_mean"] = out[clipped_cols].mean(axis=1)
    out["consensus_A5_median"] = out[clipped_cols].median(axis=1)
    out["consensus_A5_std"] = out[clipped_cols].std(axis=1)
    out["models_pred_ge_50"] = (out[clipped_cols] >= 50).sum(axis=1)
    out["models_pred_ge_60"] = (out[clipped_cols] >= 60).sum(axis=1)
    out["models_pred_ge_70"] = (out[clipped_cols] >= 70).sum(axis=1)
    out["screening_rank"] = out["consensus_A5_mean"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["consensus_A5_mean", "models_pred_ge_60"], ascending=[False, False])


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta, x_train, y_train, feature_cols = load_training_table(
        args.training_features,
        exclude_docking_ifp=args.exclude_docking_ifp,
    )
    print(f"Training rows: {len(meta)}")
    print(f"Feature count: {len(feature_cols)}")
    print("Training final LightGBM...")
    lightgbm_model = train_lightgbm(x_train, y_train, args.seed)
    print("Training final MLP...")
    mlp_model, mlp_scaler = train_mlp(x_train.values, y_train, args.seed)
    print("Training final VAE...")
    vae_model, vae_scaler = train_vae(x_train.values, y_train, args.seed)

    models = {
        "lightgbm": lightgbm_model,
        "mlp": (mlp_model, mlp_scaler),
        "vae": (vae_model, vae_scaler),
    }

    libraries = [
        read_library(args.otava, "OTAVA"),
        read_library(args.fda, "FDA"),
    ]
    results = []
    for library in libraries:
        name = library["library"].iloc[0]
        valid = library[library["valid_smiles"]].reset_index(drop=True)
        invalid = library[~library["valid_smiles"]].copy()
        print(f"Screening {name}: {len(valid)} valid, {len(invalid)} invalid after canonical deduplication")
        x_external = build_external_feature_matrix(valid, feature_cols)
        pred = add_predictions(valid, x_external.values, models)
        pred.to_csv(args.output_dir / f"{name.lower()}_top3_model_predictions.csv", index=False)
        pred.head(args.top_n).to_csv(args.output_dir / f"{name.lower()}_top{args.top_n}_hits.csv", index=False)
        if not invalid.empty:
            invalid.to_csv(args.output_dir / f"{name.lower()}_invalid_smiles.csv", index=False)
        results.append(pred)

    combined = pd.concat(results, axis=0, ignore_index=True)
    combined = combined.sort_values(["consensus_A5_mean", "models_pred_ge_60"], ascending=[False, False])
    combined.to_csv(args.output_dir / "combined_otava_fda_top3_model_predictions.csv", index=False)
    combined.head(args.top_n).to_csv(args.output_dir / f"combined_top{args.top_n}_hits.csv", index=False)

    joblib.dump(lightgbm_model, args.output_dir / "final_lightgbm_model.pkl")
    torch.save({"state_dict": mlp_model.state_dict(), "feature_cols": feature_cols}, args.output_dir / "final_mlp_model.pt")
    torch.save({"state_dict": vae_model.state_dict(), "feature_cols": feature_cols}, args.output_dir / "final_vae_model.pt")
    joblib.dump(mlp_scaler, args.output_dir / "final_mlp_scaler.pkl")
    joblib.dump(vae_scaler, args.output_dir / "final_vae_scaler.pkl")

    summary = {
        "training_rows": int(len(meta)),
        "feature_count": int(len(feature_cols)),
        "feature_mode": "Morgan + RDKit only" if args.exclude_docking_ifp else "Morgan + RDKit + docking/IFP columns filled as zero for external libraries",
        "excluded_columns": ["ifp_allowed_by_A5_threshold"],
        "libraries": {
            "OTAVA": int((combined["library"] == "OTAVA").sum()),
            "FDA": int((combined["library"] == "FDA").sum()),
        },
        "outputs": {
            "combined_predictions": str(args.output_dir / "combined_otava_fda_top3_model_predictions.csv"),
            "combined_top_hits": str(args.output_dir / f"combined_top{args.top_n}_hits.csv"),
            "otava_predictions": str(args.output_dir / "otava_top3_model_predictions.csv"),
            "fda_predictions": str(args.output_dir / "fda_top3_model_predictions.csv"),
        },
    }
    with (args.output_dir / "screening_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTop 25 consensus hits")
    print(
        combined[
            [
                "library",
                "source_id",
                "consensus_A5_mean",
                "consensus_A5_std",
                "models_pred_ge_60",
                "pred_A5_VAE_clipped",
                "pred_A5_MLP_clipped",
                "pred_A5_LightGBM_clipped",
                SMILES_COL,
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
