import argparse
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBRegressor


RDLogger.DisableLog("rdApp.*")

ID_COL = "Name"
SMILES_COL = "SMILES"
TARGET = "A5"
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train an XGBoost regression model where docking interaction fingerprints "
            "are used only for compounds with measured inhibition above a threshold."
        )
    )
    _here = Path(__file__).parent
    parser.add_argument(
        "--docking-features",
        type=Path,
        default=_here / "Output" / "01_enumerated_compounds" / "merged_result.csv",
    )
    parser.add_argument(
        "--interaction-fingerprints",
        type=Path,
        default=_here / "Input" / "interaction_fingerprints.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_here / "Output" / "02_features",
    )
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--active-threshold", type=float, default=38.0)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--min-ifp-frequency", type=float, default=0.01)
    return parser.parse_args()


def make_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_activity_table(path, target):
    df = pd.read_csv(path)
    required = {ID_COL, SMILES_COL, target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str)
    df[SMILES_COL] = df[SMILES_COL].astype(str)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).drop_duplicates(subset=[ID_COL], keep="first")
    return df.reset_index(drop=True)


def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def make_morgan_features(smiles, radius, n_bits):
    mols = [mol_from_smiles(s) for s in smiles]
    matrix = np.zeros((len(mols), n_bits), dtype=np.int8)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, matrix[i])
    cols = [f"morgan_{i:04d}" for i in range(n_bits)]
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


def make_interaction_features(path, compound_ids, min_frequency):
    raw = pd.read_csv(path, low_memory=False)
    if "Title" not in raw.columns:
        raise ValueError("Interaction fingerprint table must contain a Title column.")

    interaction_cols = [
        col
        for col in raw.columns
        if re.match(
            r"^A\d+_(contact|backbone|sidechain|polar|hydrophobic|acceptor|donor|aromatic|charged)$",
            col,
        )
    ]
    if not interaction_cols:
        raise ValueError("No residue-level interaction fingerprint columns were found.")

    counts = raw[interaction_cols].map(interaction_cell_to_count).astype(np.float32)
    counts.insert(0, ID_COL, raw["Title"].astype(str).values)

    binary = counts.copy()
    binary[interaction_cols] = (binary[interaction_cols] > 0).astype(np.float32)

    any_agg = binary.groupby(ID_COL)[interaction_cols].max()
    any_agg.columns = [f"ifp_{col}_any" for col in any_agg.columns]

    mean_agg = counts.groupby(ID_COL)[interaction_cols].mean()
    mean_agg.columns = [f"ifp_{col}_mean" for col in mean_agg.columns]

    sum_agg = counts.groupby(ID_COL)[interaction_cols].sum()
    families = sorted({col.rsplit("_", 1)[1] for col in interaction_cols})
    summary_parts = []
    for family in families:
        family_cols = [col for col in interaction_cols if col.endswith(f"_{family}")]
        part = pd.DataFrame(index=sum_agg.index)
        part[f"ifp_total_{family}_count"] = sum_agg[family_cols].sum(axis=1)
        part[f"ifp_total_{family}_residue_any"] = binary.groupby(ID_COL)[family_cols].max().sum(axis=1)
        summary_parts.append(part)

    out = pd.concat([any_agg, mean_agg, *summary_parts], axis=1)
    out = out.reindex(compound_ids).reset_index(drop=True)
    out = clean_numeric_matrix(out, missing_threshold=0.50, min_frequency=min_frequency)
    return out


def make_docking_features(df, target):
    blocked = {
        ID_COL,
        SMILES_COL,
        target,
        "SMILES_from_docking",
        "best_pose_pdb_file",
        "best_pose_variant",
    }
    features = {}
    for col in df.columns:
        if col in blocked:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            features[f"dock_{col}"] = values
    return clean_numeric_matrix(pd.DataFrame(features), missing_threshold=0.50, min_frequency=0.0)


def clean_numeric_matrix(df, missing_threshold=0.40, min_frequency=0.01):
    out = df.copy()
    if out.empty:
        return out
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, out.isna().mean() <= missing_threshold]
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    out = out.loc[:, out.nunique(dropna=False) > 1]

    binary_cols = [
        col
        for col in out.columns
        if set(pd.unique(out[col].dropna())).issubset({0, 1})
    ]
    drop = []
    for col in binary_cols:
        freq = out[col].mean()
        if freq < min_frequency or freq > 1 - min_frequency:
            drop.append(col)
    if drop:
        out = out.drop(columns=drop)
    return out


def apply_active_ifp_filter(ifp_x, y, threshold):
    filtered = ifp_x.copy()
    inactive_mask = y < threshold
    filtered.loc[inactive_mask, :] = 0.0
    return filtered, inactive_mask


def scaffold_from_smiles(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "invalid_smiles"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or "no_scaffold"


def group_split_indices(groups, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(groups))
    train_idx, test_idx = next(splitter.split(idx, groups=groups))
    return train_idx, test_idx


def make_splits(meta, test_size, seed):
    random_train, random_test = train_test_split(
        np.arange(len(meta)),
        test_size=test_size,
        random_state=seed,
    )
    scaffolds = meta[SMILES_COL].map(scaffold_from_smiles)
    scaffold_train, scaffold_test = group_split_indices(scaffolds, test_size, seed)
    return {
        "random_split": (random_train, random_test),
        "scaffold_split": (scaffold_train, scaffold_test),
    }


def make_model(seed):
    return XGBRegressor(
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
    )


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_feature_sets(feature_sets, y, splits, meta, output_dir, seed):
    rows = []
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    for feature_name, x in feature_sets.items():
        if x.empty:
            continue
        for split_name, (train_idx, test_idx) in splits.items():
            model = make_model(seed)
            model.fit(x.iloc[train_idx], y[train_idx])
            train_pred = model.predict(x.iloc[train_idx])
            test_pred = model.predict(x.iloc[test_idx])
            rows.append(
                {
                    "feature_set": feature_name,
                    "split": split_name,
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "train_rmse": rmse(y[train_idx], train_pred),
                    "test_rmse": rmse(y[test_idx], test_pred),
                    "train_mae": mean_absolute_error(y[train_idx], train_pred),
                    "test_mae": mean_absolute_error(y[test_idx], test_pred),
                    "train_r2": r2_score(y[train_idx], train_pred),
                    "test_r2": r2_score(y[test_idx], test_pred),
                }
            )

            pred_df = meta.iloc[test_idx][[ID_COL, SMILES_COL]].copy()
            pred_df["A5_true"] = y[test_idx]
            pred_df["A5_pred"] = test_pred
            pred_df["residual"] = pred_df["A5_true"] - pred_df["A5_pred"]
            pred_df.to_csv(predictions_dir / f"{feature_name}__{split_name}.csv", index=False)
    return pd.DataFrame(rows)


def feature_family(feature):
    if feature.startswith("morgan_"):
        return "morgan"
    if feature.startswith("desc_"):
        return "rdkit_descriptor"
    if feature.startswith("dock_"):
        return "docking_summary"
    if feature.startswith("ifp_"):
        return "active_filtered_interaction"
    return "other"


def main():
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)

    meta = read_activity_table(args.docking_features, args.target)
    y = np.clip(meta[args.target].astype(float).values, 0.0, 100.0)  # clamp to physically meaningful range
    meta[args.target] = y  # keep meta consistent with clipped values

    morgan_x, mols = make_morgan_features(
        meta[SMILES_COL],
        args.fingerprint_radius,
        args.fingerprint_bits,
    )
    morgan_x = clean_numeric_matrix(morgan_x, missing_threshold=0.0, min_frequency=0.01)
    desc_x = clean_numeric_matrix(make_rdkit_descriptors(mols), missing_threshold=0.20, min_frequency=0.0)
    docking_x = make_docking_features(meta, args.target)
    ifp_x = make_interaction_features(args.interaction_fingerprints, meta[ID_COL], args.min_ifp_frequency)
    active_ifp_x, inactive_mask = apply_active_ifp_filter(ifp_x, y, args.active_threshold)

    feature_sets = {
        "morgan_rdkit_baseline": pd.concat([morgan_x, desc_x], axis=1),
        "active_filtered_ifp_only": active_ifp_x,
        "morgan_rdkit_plus_active_filtered_ifp": pd.concat([morgan_x, desc_x, active_ifp_x], axis=1),
        "morgan_rdkit_plus_docking_plus_active_filtered_ifp": pd.concat(
            [morgan_x, desc_x, docking_x, active_ifp_x],
            axis=1,
        ),
    }

    all_features = pd.concat(
        [
            meta[[ID_COL, SMILES_COL, args.target]],
            pd.DataFrame({"ifp_allowed_by_A5_threshold": (~inactive_mask).astype(int)}),
            feature_sets["morgan_rdkit_plus_docking_plus_active_filtered_ifp"],
        ],
        axis=1,
    )
    all_features.to_csv(output_dir / "active_filtered_compound_features.csv", index=False)

    splits = make_splits(meta, args.test_size, args.seed)
    metrics = evaluate_feature_sets(feature_sets, y, splits, meta, output_dir, args.seed)
    metrics.to_csv(output_dir / "active_filtered_ifp_xgboost_metrics.csv", index=False)

    final_features = feature_sets["morgan_rdkit_plus_docking_plus_active_filtered_ifp"]
    final_model = make_model(args.seed)
    final_model.fit(final_features, y)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": final_features.columns.tolist(),
            "target": args.target,
            "active_threshold": args.active_threshold,
            "note": (
                "Interaction fingerprint columns were set to zero for training compounds "
                "with measured A5 below the active threshold."
            ),
        },
        output_dir / "xgboost_active_filtered_ifp_A5_model.pkl",
    )

    importance = pd.DataFrame(
        {
            "feature": final_features.columns,
            "xgb_importance": final_model.feature_importances_,
        }
    )
    importance["family"] = importance["feature"].map(feature_family)
    importance = importance.sort_values("xgb_importance", ascending=False)
    importance.to_csv(output_dir / "active_filtered_ifp_feature_importance.csv", index=False)

    summary = {
        "rows": int(len(meta)),
        "active_threshold": args.active_threshold,
        "compounds_with_ifp_allowed": int((~inactive_mask).sum()),
        "compounds_with_ifp_zeroed": int(inactive_mask.sum()),
        "feature_counts": {name: int(x.shape[1]) for name, x in feature_sets.items()},
        "metrics": metrics.to_dict(orient="records"),
        "outputs": {
            "features": str(output_dir / "active_filtered_compound_features.csv"),
            "metrics": str(output_dir / "active_filtered_ifp_xgboost_metrics.csv"),
            "model": str(output_dir / "xgboost_active_filtered_ifp_A5_model.pkl"),
            "feature_importance": str(output_dir / "active_filtered_ifp_feature_importance.csv"),
        },
    }
    with (output_dir / "active_filtered_ifp_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
