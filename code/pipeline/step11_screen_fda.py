import importlib.util
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\hye1\PyCharmMiscProject\.venv\Lib\site-packages")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor
from rdkit import Chem
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GraphDataLoader


PIPELINE_DIR = Path(__file__).resolve().parent
KEVIN_DIR = Path(r"C:\Users\hye1\Desktop\Kevin Docking")
FDA_LIBRARY = KEVIN_DIR / "FDA.csv"
FDA_DOCKING = KEVIN_DIR / "FDA_docking_result.csv"
FDA_IFP = KEVIN_DIR / "FDA_ifp.csv"
TRAINING_FEATURES = PIPELINE_DIR / "Output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
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
OUTPUT_DIR = PIPELINE_DIR / "Output" / "11c_screen_fda_kevin_docking_corrected_20260605"

TARGET = "A5"
ID_COL = "Name"
SMILES_COL = "SMILES"
SEED = 42
TOP_N = 200
PATIENCE = 8
MAX_EPOCHS = 60
ACTIVE_THRESHOLD = 38.0


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


features_utils = load_module(PIPELINE_DIR / "utils_features.py", "grp78_fda_features")
models_utils = load_module(PIPELINE_DIR / "utils_models.py", "grp78_fda_models")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def normalize_cid(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def read_fda_library():
    df = pd.read_csv(FDA_LIBRARY, low_memory=False)
    required = {"Compound_CID", SMILES_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"FDA library missing columns: {missing}")
    out = df.copy()
    out["Compound_CID"] = out["Compound_CID"].map(normalize_cid)
    out["source_id"] = out["Compound_CID"]
    out["library"] = "FDA"
    out[SMILES_COL] = out[SMILES_COL].astype(str)
    out["canonical_smiles"] = out[SMILES_COL].map(canonical_smiles)
    out["valid_smiles"] = out["canonical_smiles"].astype(bool)
    out = out.drop_duplicates(subset=["Compound_CID"], keep="first")
    out = out.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
    return out


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
    if "Compound_CID" not in raw.columns:
        raise ValueError("FDA docking table must contain Compound_CID.")
    raw = raw.copy()
    raw["Compound_CID"] = raw["Compound_CID"].map(normalize_cid)
    blocked = {
        "Compound_CID",
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
        "Data_Source",
        "IUPAC_Name",
        "Molecular_Formula",
    }
    numeric = {}
    for col in raw.columns:
        if col in blocked or col.startswith("SIFT "):
            continue
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.notna().any():
            numeric[col] = values
    numeric_df = pd.DataFrame(numeric)
    numeric_df.insert(0, "Compound_CID", raw["Compound_CID"].values)
    agg = numeric_df.groupby("Compound_CID").mean(numeric_only=True)
    agg.columns = [f"dock_{col}" for col in agg.columns]
    return agg.reindex(compound_ids).fillna(0.0).reset_index(drop=True), raw


def infer_ifp_id_series(ifp_raw, dock_raw):
    if "Compound_CID" in ifp_raw.columns:
        return ifp_raw["Compound_CID"].map(normalize_cid)
    if "Title" not in ifp_raw.columns:
        raise ValueError("FDA IFP table must contain Title or Compound_CID.")
    title_to_cid = {}
    for col in ["Title", "Compound_ID"]:
        if col in dock_raw.columns:
            pairs = dock_raw[[col, "Compound_CID"]].dropna().copy()
            pairs[col] = pairs[col].astype(str)
            pairs["Compound_CID"] = pairs["Compound_CID"].map(normalize_cid)
            title_to_cid.update(dict(zip(pairs[col], pairs["Compound_CID"])))
    mapped = ifp_raw["Title"].astype(str).map(title_to_cid).fillna("")
    direct_title = ifp_raw["Title"].astype(str).map(normalize_cid)
    return mapped.where(mapped.astype(bool), direct_title)


def aggregate_ifp_features(path, compound_ids, dock_raw):
    raw = pd.read_csv(path, low_memory=False)
    raw = raw.copy()
    raw["Compound_CID"] = infer_ifp_id_series(raw, dock_raw)
    interaction_cols = [
        col
        for col in raw.columns
        if re.match(
            r"^A\d+_(contact|backbone|sidechain|polar|hydrophobic|acceptor|donor|aromatic|charged)$",
            col,
        )
    ]
    if not interaction_cols:
        raise ValueError("No residue-level FDA IFP columns were found.")
    counts = raw[interaction_cols].map(interaction_cell_to_count).astype(np.float32)
    counts.insert(0, "Compound_CID", raw["Compound_CID"].values)
    binary = counts.copy()
    binary[interaction_cols] = (binary[interaction_cols] > 0).astype(np.float32)
    any_agg = binary.groupby("Compound_CID")[interaction_cols].max()
    any_agg.columns = [f"ifp_{col}_any" for col in any_agg.columns]
    mean_agg = counts.groupby("Compound_CID")[interaction_cols].mean()
    mean_agg.columns = [f"ifp_{col}_mean" for col in mean_agg.columns]
    sum_agg = counts.groupby("Compound_CID")[interaction_cols].sum()
    grouped_binary = binary.groupby("Compound_CID")
    summary_parts = []
    for family in sorted({col.rsplit("_", 1)[1] for col in interaction_cols}):
        family_cols = [col for col in interaction_cols if col.endswith(f"_{family}")]
        part = pd.DataFrame(index=sum_agg.index)
        part[f"ifp_total_{family}_count"] = sum_agg[family_cols].sum(axis=1)
        part[f"ifp_total_{family}_residue_any"] = grouped_binary[family_cols].max().sum(axis=1)
        summary_parts.append(part)
    out = pd.concat([any_agg, mean_agg, *summary_parts], axis=1)
    return out.reindex(compound_ids).fillna(0.0).reset_index(drop=True), raw


def build_fda_feature_matrix(valid, feature_cols):
    morgan_x, mols = features_utils.make_morgan_features(valid[SMILES_COL], 2, 1024)
    desc_x = features_utils.make_rdkit_descriptors(mols)
    generated = pd.concat([morgan_x, desc_x], axis=1)
    x_external = pd.DataFrame(0.0, index=np.arange(len(valid)), columns=feature_cols)
    shared_ligand_cols = [col for col in generated.columns if col in x_external.columns]
    x_external.loc[:, shared_ligand_cols] = (
        generated[shared_ligand_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    )
    compound_ids = valid["Compound_CID"].astype(str)
    dock_features, dock_raw = aggregate_docking_features(FDA_DOCKING, compound_ids)
    ifp_features, ifp_raw = aggregate_ifp_features(FDA_IFP, compound_ids, dock_raw)
    shared_dock_cols = [col for col in dock_features.columns if col in x_external.columns]
    shared_ifp_cols = [col for col in ifp_features.columns if col in x_external.columns]
    if shared_dock_cols:
        x_external.loc[:, shared_dock_cols] = dock_features[shared_dock_cols].values
    if shared_ifp_cols:
        x_external.loc[:, shared_ifp_cols] = ifp_features[shared_ifp_cols].values
    dock_counts = dock_raw["Compound_CID"].astype(str).value_counts()
    ifp_counts = ifp_raw["Compound_CID"].astype(str).value_counts()
    valid = valid.copy()
    valid["has_fda_docking"] = compound_ids.isin(set(dock_counts.index))
    valid["has_fda_ifp"] = compound_ids.isin(set(ifp_counts.index))
    valid["fda_docking_pose_count"] = compound_ids.map(dock_counts).fillna(0).astype(int)
    valid["fda_ifp_pose_count"] = compound_ids.map(ifp_counts).fillna(0).astype(int)
    return (
        x_external.apply(pd.to_numeric, errors="coerce").fillna(0.0),
        valid,
        shared_ligand_cols,
        shared_dock_cols,
        shared_ifp_cols,
        dock_raw,
        ifp_raw,
    )


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
        atom_x = torch.tensor([models_utils.atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            features = models_utils.bond_features(bond)
            edge_index.extend([[start, end], [end, start]])
            edge_attr.extend([features, features])
        if not edge_index:
            edge_index = [[0, 0]]
            edge_attr = [[0.0] * (len(models_utils.BOND_TYPES) + 1 + 3)]
        return Data(
            x=atom_x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            tabular=torch.tensor(self.x_tabular[idx], dtype=torch.float).view(1, -1),
        )


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
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32)
    fit_idx, val_idx = train_test_split(np.arange(len(meta)), test_size=0.15, random_state=SEED, shuffle=True)
    train_dataset = models_utils.GraphTabularDataset(meta.iloc[fit_idx], x_scaled[fit_idx], y_train[fit_idx])
    val_dataset = models_utils.GraphTabularDataset(meta.iloc[val_idx], x_scaled[val_idx], y_train[val_idx])
    train_loader = GraphDataLoader(train_dataset, batch_size=int(params["batch_size"]), shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=int(params["batch_size"]), shuffle=False)
    sample = train_dataset[0]
    model = models_utils.GNNTabularRegressor(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        tabular_dim=sample.tabular.shape[-1],
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
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
        val_true, val_pred = models_utils.predict_gnn(model, val_loader, device)
        val_rmse = models_utils.rmse(val_true, val_pred)
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
            predictions.extend(models_utils.inverse_signed_log1p(pred.cpu().numpy().ravel()).tolist())
    return np.asarray(predictions)


def add_predictions(valid, x_external, extra, cat, gnn, gnn_scaler):
    out = valid.copy()
    out["pred_A5_ExtraTrees"] = extra.predict(x_external)
    out["pred_A5_CatBoost"] = cat.predict(x_external)
    out["pred_A5_GNN"] = predict_gnn_external(gnn, gnn_scaler, valid, x_external.values)
    pred_cols = ["pred_A5_ExtraTrees", "pred_A5_CatBoost", "pred_A5_GNN"]
    for col in pred_cols:
        out[f"{col}_clipped"] = out[col].clip(0, 100)
    clipped_cols = [f"{col}_clipped" for col in pred_cols]
    out["consensus_A5_mean"] = out[clipped_cols].mean(axis=1)
    out["consensus_A5_median"] = out[clipped_cols].median(axis=1)
    out["consensus_A5_std"] = out[clipped_cols].std(axis=1)
    out["models_pred_gt_38"] = (out[clipped_cols] > 38).sum(axis=1)
    out["models_pred_ge_50"] = (out[clipped_cols] >= 50).sum(axis=1)
    out["models_pred_ge_60"] = (out[clipped_cols] >= 60).sum(axis=1)
    out["models_pred_ge_70"] = (out[clipped_cols] >= 70).sum(axis=1)
    out = out.sort_values(["models_pred_gt_38", "consensus_A5_mean"], ascending=[False, False]).reset_index(drop=True)
    out["screening_rank"] = np.arange(1, len(out) + 1)
    return out


def write_overlap_files(pred):
    keep = [
        "Compound_CID",
        "Name",
        "IUPAC_Name",
        "SMILES",
        "consensus_A5_mean",
        "consensus_A5_median",
        "consensus_A5_std",
        "models_pred_gt_38",
        "pred_A5_ExtraTrees_clipped",
        "pred_A5_CatBoost_clipped",
        "pred_A5_GNN_clipped",
        "has_fda_docking",
        "has_fda_ifp",
        "fda_docking_pose_count",
        "fda_ifp_pose_count",
        "screening_rank",
    ]
    keep = [col for col in keep if col in pred.columns]
    outputs = {}
    counts = {}
    for cutoff in [38, 50, 60, 70]:
        col = "models_pred_gt_38" if cutoff == 38 else f"models_pred_ge_{cutoff}"
        frame = pred[pred[col].eq(3)].copy()
        csv_path = OUTPUT_DIR / f"fda_bayesian_top3_overlap_predA5_{'gt' if cutoff == 38 else 'ge'}{cutoff}.csv"
        xlsx_path = OUTPUT_DIR / f"fda_bayesian_top3_overlap_predA5_{'gt' if cutoff == 38 else 'ge'}{cutoff}.xlsx"
        frame[keep].to_csv(csv_path, index=False)
        frame[keep].to_excel(xlsx_path, index=False)
        outputs[f"overlap_{cutoff}_csv"] = str(csv_path)
        outputs[f"overlap_{cutoff}_xlsx"] = str(xlsx_path)
        counts[f"overlap_pred_{cutoff}_count"] = int(len(frame))
    return outputs, counts


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, x_train, y_train = models_utils.load_feature_table(TRAINING_FEATURES, TARGET)
    feature_cols = list(x_train.columns)
    print(f"Training rows: {len(meta)}")
    print(f"Feature count: {len(feature_cols)}")
    print("Training final Bayesian top 3: Extra Trees, CatBoost, GNN...")
    extra, cat, gnn, gnn_scaler, extra_params, cat_params, gnn_params, gnn_trial, gnn_val_rmse = train_models(
        x_train, y_train, meta
    )
    library = read_fda_library()
    valid = library[library["valid_smiles"]].reset_index(drop=True)
    invalid = library[~library["valid_smiles"]].copy()
    x_external, valid, shared_ligand_cols, shared_dock_cols, shared_ifp_cols, dock_raw, ifp_raw = build_fda_feature_matrix(
        valid, feature_cols
    )
    pred = add_predictions(valid, x_external, extra, cat, gnn, gnn_scaler)

    all_path = OUTPUT_DIR / "fda_bayesian_top3_extra_catboost_gnn_predictions.csv"
    top_csv = OUTPUT_DIR / f"fda_bayesian_top3_extra_catboost_gnn_top{TOP_N}_hits.csv"
    top_xlsx = OUTPUT_DIR / f"fda_bayesian_top3_extra_catboost_gnn_top{TOP_N}_hits.xlsx"
    coverage_path = OUTPUT_DIR / "fda_kevin_docking_ifp_coverage.csv"
    pred.to_csv(all_path, index=False)
    pred.head(TOP_N).to_csv(top_csv, index=False)
    pred.head(TOP_N).to_excel(top_xlsx, index=False)
    valid.to_csv(coverage_path, index=False)
    if not invalid.empty:
        invalid.to_csv(OUTPUT_DIR / "fda_invalid_smiles.csv", index=False)
    overlap_outputs, overlap_counts = write_overlap_files(pred)

    joblib.dump(extra, OUTPUT_DIR / "final_bayesian_extra_trees_model.pkl")
    joblib.dump(cat, OUTPUT_DIR / "final_bayesian_catboost_model.pkl")
    joblib.dump(gnn_scaler, OUTPUT_DIR / "final_bayesian_gnn_scaler.pkl")
    torch.save(
        {
            "state_dict": gnn.state_dict(),
            "feature_cols": feature_cols,
            "params": gnn_params,
            "feature_mode": "FDA Kevin Docking screen using Morgan + RDKit + docking + IFP aligned to training columns.",
        },
        OUTPUT_DIR / "final_bayesian_gnn_model.pt",
    )

    summary = {
        "library": "FDA",
        "input_folder": str(KEVIN_DIR),
        "merge_key": "Compound_CID",
        "models": ["Bayesian Extra Trees", "Bayesian CatBoost", "Bayesian GNN"],
        "training_rows": int(len(meta)),
        "feature_count": int(len(feature_cols)),
        "valid_unique_compounds": int(len(valid)),
        "invalid_smiles": int(len(invalid)),
        "fda_library_rows_after_dedup": int(len(library)),
        "fda_docking_rows": int(len(dock_raw)),
        "fda_docking_unique_cids": int(dock_raw["Compound_CID"].astype(str).nunique()),
        "fda_ifp_rows": int(len(ifp_raw)),
        "fda_ifp_unique_cids": int(ifp_raw["Compound_CID"].astype(str).nunique()),
        "library_compounds_with_docking": int(valid["has_fda_docking"].sum()),
        "library_compounds_with_ifp": int(valid["has_fda_ifp"].sum()),
        "shared_ligand_feature_columns_used": int(len(shared_ligand_cols)),
        "shared_docking_feature_columns_used": int(len(shared_dock_cols)),
        "shared_ifp_feature_columns_used": int(len(shared_ifp_cols)),
        "gnn_validation_rmse_during_final_training": float(gnn_val_rmse),
        **overlap_counts,
        "optimized_params": {
            "Extra Trees": extra_params,
            "CatBoost": cat_params,
            "GNN": gnn_params,
        },
        "outputs": {
            "all_predictions": str(all_path),
            "top_hits_csv": str(top_csv),
            "top_hits_xlsx": str(top_xlsx),
            "coverage": str(coverage_path),
            **overlap_outputs,
        },
    }
    with (OUTPUT_DIR / "fda_bayesian_top3_screening_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTop 25 FDA hits")
    keep = [
        "screening_rank",
        "Compound_CID",
        "Name",
        "consensus_A5_mean",
        "consensus_A5_std",
        "models_pred_gt_38",
        "pred_A5_ExtraTrees_clipped",
        "pred_A5_CatBoost_clipped",
        "pred_A5_GNN_clipped",
        "has_fda_docking",
        "has_fda_ifp",
        SMILES_COL,
    ]
    print(pred[[col for col in keep if col in pred.columns]].head(25).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
