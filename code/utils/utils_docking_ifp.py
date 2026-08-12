import importlib.util
import json
import re
import sys
from pathlib import Path


VENV_SITE_PACKAGES = Path(r"C:\Users\hye1\PyCharmMiscProject\.venv\Lib\site-packages")
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GraphDataLoader


PIPELINE_DIR = Path(__file__).resolve().parent
STEP2_PATH = PIPELINE_DIR / "utils_features.py"
STEP11_PATH = PIPELINE_DIR / "utils_models.py"
STEP12_PATH = PIPELINE_DIR / "utils_screening.py"
TRAINING_ACTIVITY = PIPELINE_DIR / "Input" / "training_activity_merged_result.csv"
TRAINING_DOCKING = PIPELINE_DIR / "Input" / "training_docking_result.csv"
TRAINING_IFP = PIPELINE_DIR / "Input" / "training_interaction_fingerprints.csv"
OTAVA_LIBRARY = PIPELINE_DIR / "Input" / "Libraries" / "otava_library.xlsx"
OTAVA_DOCKING = PIPELINE_DIR / "Input" / "External" / "OTAVA_docking_result.csv"
OTAVA_IFP = PIPELINE_DIR / "Input" / "External" / "Otava_IFP.csv"
OUTPUT_DIR = PIPELINE_DIR / "Output" / "01_features"

ID_COL = "Name"
SMILES_COL = "SMILES"
TARGET = "A5"
BATCH_SIZE = 32
GNN_EPOCHS = 60
TOP_N = 200
SEED = 42
ACTIVE_THRESHOLD = 38.0


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step2 = load_module(STEP2_PATH, "utils_features")
step11 = load_module(STEP11_PATH, "utils_models")
step12 = load_module(STEP12_PATH, "step12_screening")


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
        if inner == "":
            return 0
        return len([part for part in inner.split(",") if part.strip()])
    try:
        return int(float(text) != 0)
    except ValueError:
        return int(bool(text))


def aggregate_docking_features(path, compound_ids):
    raw = pd.read_csv(path, low_memory=False)
    id_col = "Title" if "Title" in raw.columns else "Compound_ID"
    if id_col not in raw.columns:
        raise ValueError(f"{path} must contain Title or Compound_ID.")
    blocked = {
        id_col,
        "Title",
        "Name",
        "SMILES",
        "Compound_ID",
        "Chemical_Name",
        "Rack_Position",
        "Variant",
        "source file",
        "Force Field",
        "docking method",
        "glide gridfile",
        "macro-pKa",
        "tautomer",
    }
    numeric = {}
    for col in raw.columns:
        if col in blocked or col.startswith("SIFT "):
            continue
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.notna().any():
            numeric[col] = values
    numeric_df = pd.DataFrame(numeric)
    numeric_df.insert(0, ID_COL, raw[id_col].astype(str).values)
    agg = numeric_df.groupby(ID_COL).mean(numeric_only=True)
    agg.columns = [f"dock_{col}" for col in agg.columns]
    return agg.reindex(compound_ids).fillna(0.0).reset_index(drop=True), raw


def make_external_ifp_features(ifp_path, compound_ids):
    raw = pd.read_csv(ifp_path, low_memory=False)
    if "Title" not in raw.columns:
        raise ValueError("IFP table must contain a Title column.")

    interaction_cols = [
        col
        for col in raw.columns
        if re.match(
            r"^A\d+_(contact|backbone|sidechain|polar|hydrophobic|acceptor|donor|aromatic|charged)$",
            col,
        )
    ]
    if not interaction_cols:
        raise ValueError("No residue-level IFP columns were found.")

    counts = raw[interaction_cols].map(interaction_cell_to_count).astype(np.float32)
    counts.insert(0, ID_COL, raw["Title"].astype(str).values)

    binary = counts.copy()
    binary[interaction_cols] = (binary[interaction_cols] > 0).astype(np.float32)

    any_agg = binary.groupby(ID_COL)[interaction_cols].max()
    any_agg.columns = [f"ifp_{col}_any" for col in any_agg.columns]
    mean_agg = counts.groupby(ID_COL)[interaction_cols].mean()
    mean_agg.columns = [f"ifp_{col}_mean" for col in mean_agg.columns]
    sum_agg = counts.groupby(ID_COL)[interaction_cols].sum()
    grouped_binary = binary.groupby(ID_COL)
    summary_parts = []
    for family in sorted({col.rsplit("_", 1)[1] for col in interaction_cols}):
        family_cols = [col for col in interaction_cols if col.endswith(f"_{family}")]
        part = pd.DataFrame(index=sum_agg.index)
        part[f"ifp_total_{family}_count"] = sum_agg[family_cols].sum(axis=1)
        part[f"ifp_total_{family}_residue_any"] = grouped_binary[family_cols].max().sum(axis=1)
        summary_parts.append(part)

    out = pd.concat([any_agg, mean_agg, *summary_parts], axis=1)
    return out.reindex(compound_ids).fillna(0.0).reset_index(drop=True), raw


def build_training_table():
    meta = step2.read_activity_table(TRAINING_ACTIVITY, TARGET)
    y = np.clip(meta[TARGET].astype(float).values, 0.0, 100.0)
    meta[TARGET] = y

    morgan_x, mols = step2.make_morgan_features(meta[SMILES_COL], 2, 1024)
    morgan_x = step2.clean_numeric_matrix(morgan_x, missing_threshold=0.0, min_frequency=0.01)
    desc_x = step2.clean_numeric_matrix(step2.make_rdkit_descriptors(mols), missing_threshold=0.20, min_frequency=0.0)
    dock_x, train_docking_raw = aggregate_docking_features(TRAINING_DOCKING, meta[ID_COL].astype(str))
    ifp_x = step2.make_interaction_features(TRAINING_IFP, meta[ID_COL], min_frequency=0.01)
    active_ifp_x, inactive_mask = step2.apply_active_ifp_filter(ifp_x, y, ACTIVE_THRESHOLD)

    x = pd.concat([morgan_x, desc_x, dock_x, active_ifp_x], axis=1)
    x = x.loc[:, ~x.columns.duplicated()]
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    feature_table = pd.concat(
        [
            meta[[ID_COL, SMILES_COL, TARGET]],
            pd.DataFrame({"ifp_allowed_by_A5_threshold": (~inactive_mask).astype(int)}),
            x,
        ],
        axis=1,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(OUTPUT_DIR / "training_features_docking_ifp_threshold38.csv", index=False)
    return meta, x, y, list(x.columns), train_docking_raw


class ExternalGraphTabularDataset(Dataset):
    def __init__(self, library_df, x_tabular):
        super().__init__()
        self.library_df = library_df.reset_index(drop=True)
        self.x_tabular = np.asarray(x_tabular, dtype=np.float32)

    def len(self):
        return len(self.library_df)

    def get(self, idx):
        row = self.library_df.iloc[idx]
        mol = Chem.MolFromSmiles(str(row[SMILES_COL]))
        if mol is None:
            mol = Chem.MolFromSmiles("C")
        atom_x = torch.tensor([step11.atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            features = step11.bond_features(bond)
            edge_index.extend([[start, end], [end, start]])
            edge_attr.extend([features, features])
        if not edge_index:
            edge_index = [[0, 0]]
            edge_attr = [[0.0] * (len(step11.BOND_TYPES) + 1 + 3)]
        return Data(
            x=atom_x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            tabular=torch.tensor(self.x_tabular[idx], dtype=torch.float).view(1, -1),
        )


def train_tree_models(x_train, y_train):
    models = step11.build_classical_models(SEED)
    hist_gradient_boosting = models["Hist Gradient Boosting"]
    extra_trees = models["Extra Trees"]
    hist_gradient_boosting.fit(x_train, y_train)
    extra_trees.fit(x_train, y_train)
    return hist_gradient_boosting, extra_trees


def train_final_gnn(meta, x_train, y_train):
    step11.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    rel_idx = np.arange(len(meta))
    fit_idx, val_idx = train_test_split(rel_idx, test_size=0.15, random_state=SEED, shuffle=True)
    train_dataset = step11.GraphTabularDataset(meta.iloc[fit_idx], x_train_scaled[fit_idx], y_train[fit_idx])
    val_dataset = step11.GraphTabularDataset(meta.iloc[val_idx], x_train_scaled[val_idx], y_train[val_idx])
    train_loader = GraphDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    sample = train_dataset[0]
    model = step11.GNNTabularRegressor(
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
        val_true, val_pred = step11.predict_gnn(model, val_loader, device)
        val_rmse = step11.rmse(val_true, val_pred)
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
    return model, scaler, best_val_rmse


def build_otava_feature_matrix(valid, feature_cols):
    x_external = step12.build_external_feature_matrix(valid, feature_cols)
    dock_features, dock_raw = aggregate_docking_features(OTAVA_DOCKING, valid["source_id"].astype(str))
    ifp_features, ifp_raw = make_external_ifp_features(OTAVA_IFP, valid["source_id"].astype(str))

    shared_dock_cols = [col for col in dock_features.columns if col in x_external.columns]
    shared_ifp_cols = [col for col in ifp_features.columns if col in x_external.columns]
    if shared_dock_cols:
        x_external.loc[:, shared_dock_cols] = dock_features[shared_dock_cols].values
    if shared_ifp_cols:
        x_external.loc[:, shared_ifp_cols] = ifp_features[shared_ifp_cols].values

    dock_ids = set(dock_raw["Title"].astype(str))
    ifp_ids = set(ifp_raw["Title"].astype(str))
    valid["has_otava_docking"] = valid["source_id"].astype(str).isin(dock_ids)
    valid["has_otava_ifp"] = valid["source_id"].astype(str).isin(ifp_ids)
    valid["otava_docking_pose_count"] = valid["source_id"].astype(str).map(dock_raw["Title"].astype(str).value_counts()).fillna(0).astype(int)
    valid["otava_ifp_pose_count"] = valid["source_id"].astype(str).map(ifp_raw["Title"].astype(str).value_counts()).fillna(0).astype(int)
    return x_external.apply(pd.to_numeric, errors="coerce").fillna(0.0), valid, shared_dock_cols, shared_ifp_cols, dock_raw, ifp_raw


def predict_gnn_external(model, scaler, library_df, x_external):
    device = next(model.parameters()).device
    x_scaled = scaler.transform(x_external).astype(np.float32)
    dataset = ExternalGraphTabularDataset(library_df, x_scaled)
    loader = GraphDataLoader(dataset, batch_size=256, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.tabular)
            predictions.extend(step11.inverse_signed_log1p(pred.cpu().numpy().ravel()).tolist())
    return np.asarray(predictions)


def add_top3_predictions(valid, x_external, gnn_model, gnn_scaler, hist_gradient_boosting, extra_trees):
    out = valid.copy()
    out["pred_A5_GNN"] = predict_gnn_external(gnn_model, gnn_scaler, valid, x_external.values)
    out["pred_A5_HistGradientBoosting"] = hist_gradient_boosting.predict(x_external)
    out["pred_A5_ExtraTrees"] = extra_trees.predict(x_external)
    pred_cols = ["pred_A5_GNN", "pred_A5_HistGradientBoosting", "pred_A5_ExtraTrees"]
    for col in pred_cols:
        out[f"{col}_clipped"] = out[col].clip(0, 100)
    clipped_cols = [f"{col}_clipped" for col in pred_cols]
    out["consensus_A5_mean"] = out[clipped_cols].mean(axis=1)
    out["consensus_A5_median"] = out[clipped_cols].median(axis=1)
    out["consensus_A5_std"] = out[clipped_cols].std(axis=1)
    out["models_pred_ge_50"] = (out[clipped_cols] >= 50).sum(axis=1)
    out["models_pred_ge_60"] = (out[clipped_cols] >= 60).sum(axis=1)
    out["models_pred_ge_70"] = (out[clipped_cols] >= 70).sum(axis=1)
    out = out.sort_values(["consensus_A5_mean", "models_pred_ge_60"], ascending=[False, False])
    out["screening_rank"] = np.arange(1, len(out) + 1)
    return out


def main():
    step12.set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, x_train, y_train, feature_cols, train_docking_raw = build_training_table()
    print(f"Training rows: {len(meta)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Training feature families: Morgan={sum(c.startswith('morgan_') for c in feature_cols)}, "
          f"RDKit={sum(c.startswith('desc_') for c in feature_cols)}, "
          f"Docking={sum(c.startswith('dock_') for c in feature_cols)}, "
          f"IFP={sum(c.startswith('ifp_') for c in feature_cols)}")
    print("Training final Hist Gradient Boosting and Extra Trees...")
    hist_gradient_boosting, extra_trees = train_tree_models(x_train, y_train)
    print("Training final GNN...")
    gnn_model, gnn_scaler, gnn_val_rmse = train_final_gnn(meta, x_train.values, y_train)

    library = step12.read_library(OTAVA_LIBRARY, "OTAVA")
    valid = library[library["valid_smiles"]].reset_index(drop=True)
    invalid = library[~library["valid_smiles"]].copy()
    x_external, valid, shared_dock_cols, shared_ifp_cols, dock_raw, ifp_raw = build_otava_feature_matrix(valid, feature_cols)
    pred = add_top3_predictions(valid, x_external, gnn_model, gnn_scaler, hist_gradient_boosting, extra_trees)

    all_path = OUTPUT_DIR / "otava_with_docking_ifp_top3_model_predictions.csv"
    top_csv = OUTPUT_DIR / f"otava_with_docking_ifp_top{TOP_N}_hits.csv"
    top_xlsx = OUTPUT_DIR / f"otava_with_docking_ifp_top{TOP_N}_hits.xlsx"
    merged_path = OUTPUT_DIR / "otava_library_merged_with_docking_ifp_coverage.csv"
    pred.to_csv(all_path, index=False)
    pred.head(TOP_N).to_csv(top_csv, index=False)
    pred.head(TOP_N).to_excel(top_xlsx, index=False)
    valid.to_csv(merged_path, index=False)

    joblib.dump(hist_gradient_boosting, OUTPUT_DIR / "final_hist_gradient_boosting_model.pkl")
    joblib.dump(extra_trees, OUTPUT_DIR / "final_extra_trees_model.pkl")
    joblib.dump(gnn_scaler, OUTPUT_DIR / "final_gnn_scaler.pkl")
    torch.save(
        {
            "state_dict": gnn_model.state_dict(),
            "feature_cols": feature_cols,
            "feature_mode": "Morgan + RDKit + docking + IFP; missing OTAVA docking/IFP rows are zero-filled.",
            "ifp_active_threshold": ACTIVE_THRESHOLD,
        },
        OUTPUT_DIR / "final_gnn_model.pt",
    )

    lib_ids = set(valid["source_id"].astype(str))
    dock_ids = set(dock_raw["Title"].astype(str))
    ifp_ids = set(ifp_raw["Title"].astype(str))
    summary = {
        "library": "OTAVA",
        "models": ["GNN", "Hist Gradient Boosting", "Extra Trees"],
        "feature_alignment": "Model was retrained with the same ligand + docking + IFP feature families used for OTAVA screening.",
        "training_rows": int(len(meta)),
        "feature_count": int(len(feature_cols)),
        "feature_families": {
            "morgan": int(sum(c.startswith("morgan_") for c in feature_cols)),
            "rdkit": int(sum(c.startswith("desc_") for c in feature_cols)),
            "docking": int(sum(c.startswith("dock_") for c in feature_cols)),
            "ifp": int(sum(c.startswith("ifp_") for c in feature_cols)),
        },
        "training_docking_rows": int(len(train_docking_raw)),
        "training_docking_unique_titles": int(train_docking_raw["Title"].astype(str).nunique()),
        "training_docking_titles_with_score": int(pd.to_numeric(train_docking_raw["docking score"], errors="coerce").notna().groupby(train_docking_raw["Title"].astype(str)).max().sum()),
        "shared_docking_feature_columns_used": int(len(shared_dock_cols)),
        "shared_ifp_feature_columns_used": int(len(shared_ifp_cols)),
        "valid_unique_compounds": int(len(valid)),
        "invalid_smiles": int(len(invalid)),
        "otava_docking_rows": int(len(dock_raw)),
        "otava_docking_unique_titles": int(dock_raw["Title"].astype(str).nunique()),
        "otava_ifp_rows": int(len(ifp_raw)),
        "otava_ifp_unique_titles": int(ifp_raw["Title"].astype(str).nunique()),
        "library_compounds_with_docking": int(valid["has_otava_docking"].sum()),
        "library_compounds_missing_docking": int((~valid["has_otava_docking"]).sum()),
        "library_compounds_with_ifp": int(valid["has_otava_ifp"].sum()),
        "library_compounds_missing_ifp": int((~valid["has_otava_ifp"]).sum()),
        "missing_docking_compound_ids": sorted(lib_ids - dock_ids),
        "missing_ifp_compound_ids": sorted(lib_ids - ifp_ids),
        "gnn_validation_rmse_during_training": float(gnn_val_rmse),
        "outputs": {
            "all_predictions": str(all_path),
            "top_hits_csv": str(top_csv),
            "top_hits_xlsx": str(top_xlsx),
            "merged_coverage": str(merged_path),
            "training_features": str(OUTPUT_DIR / "training_features_docking_ifp_threshold38.csv"),
        },
    }
    with (OUTPUT_DIR / "otava_with_docking_ifp_screening_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nTop 25 OTAVA hits with docking + IFP")
    print(pred[[
        "source_id", "has_otava_docking", "has_otava_ifp", "otava_docking_pose_count", "otava_ifp_pose_count",
        "consensus_A5_mean", "consensus_A5_std", "models_pred_ge_60",
        "pred_A5_GNN_clipped", "pred_A5_HistGradientBoosting_clipped", "pred_A5_ExtraTrees_clipped", SMILES_COL
    ]].head(25).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
