import importlib.util
import json
import re
import sys

from config import (
    ACTIVE_THRESHOLD,
    ID_COL,
    OUTPUT_DIR,
    PIPELINE_DIR,
    TARGET,
    TRAINING_ACTIVITY,
    TRAINING_DOCKING,
    TRAINING_IFP,
    VENV_SITE_PACKAGES,
)


if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import numpy as np
import pandas as pd


STEP2_PATH = PIPELINE_DIR / "utils_features.py"
FEATURE_OUTPUT_DIR = OUTPUT_DIR / "01_features"


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step2 = load_module(STEP2_PATH, "utils_features")


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


def main():
    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = step2.read_activity_table(TRAINING_ACTIVITY, TARGET)
    y = np.clip(meta[TARGET].astype(float).values, 0.0, 100.0)
    meta[TARGET] = y

    morgan_x, mols = step2.make_morgan_features(meta["SMILES"], 2, 1024)
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
            meta[[ID_COL, "SMILES", TARGET]],
            pd.DataFrame({"ifp_allowed_by_A5_threshold": (~inactive_mask).astype(int)}),
            x,
        ],
        axis=1,
    )
    feature_path = FEATURE_OUTPUT_DIR / "training_features_docking_ifp_threshold38.csv"
    feature_table.to_csv(feature_path, index=False)
    summary = {
        "rows": int(len(meta)),
        "active_threshold": ACTIVE_THRESHOLD,
        "feature_count": int(x.shape[1]),
        "feature_families": {
            "morgan": int(sum(c.startswith("morgan_") for c in x.columns)),
            "rdkit": int(sum(c.startswith("desc_") for c in x.columns)),
            "docking": int(sum(c.startswith("dock_") for c in x.columns)),
            "ifp": int(sum(c.startswith("ifp_") for c in x.columns)),
        },
        "training_docking_rows": int(len(train_docking_raw)),
        "outputs": {"training_features": str(feature_path)},
    }
    with (FEATURE_OUTPUT_DIR / "feature_preparation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
