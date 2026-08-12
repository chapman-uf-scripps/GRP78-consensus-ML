"""
Compare IFP weighting schemes head-to-head on the GRP78 training data.

Schemes (all share the same Morgan + RDKit + docking baseline):
  A. no_ifp                — baseline (no IFP at all)
  B. raw_ifp               — full IFP, no filtering
  C. active_masked         — current method: zero IFP rows for inactives
  D. enrichment_weighted   — every IFP column scaled by log-enrichment in actives
  E. mi_weighted           — every IFP column scaled by mutual information with label
  F. mi_top128             — keep only the top-128 IFP columns by MI

Evaluation:
  - ExtraTrees regressor (fast, the same family as one of the consensus models)
  - Scaffold split (Bemis–Murcko), test_size=0.20
  - 5 seeds → mean ± SD

Outputs:
  - output/source_tables/ifp_scheme_comparison.csv
  - print summary table
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import importlib.util
from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
PIPELINE_DIR = ROOT / "code" / "pipeline_core"
OUT = ROOT / "output" / "source_tables"
OUT.mkdir(parents=True, exist_ok=True)

ACTIVE_THRESHOLD = 38.0
TEST_SIZE = 0.20
SEEDS = [42, 7, 13, 99, 2024]
TOP_K_FOR_F = 128


def load_utils():
    spec = importlib.util.spec_from_file_location("utils_features", PIPELINE_DIR / "utils_features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PIPELINE_DIR))
    spec.loader.exec_module(mod)
    return mod


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def scaffold_split(smiles_series, test_size, seed):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def scaf(s):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return "invalid"
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        return sc or "none"

    groups = smiles_series.map(scaf).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(groups))
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    return train_idx, test_idx


def build_base_features(utils, meta):
    morgan_x, mols = utils.make_morgan_features(meta["SMILES"], 2, 1024)
    morgan_x = utils.clean_numeric_matrix(morgan_x, missing_threshold=0.0, min_frequency=0.01)
    desc_x = utils.clean_numeric_matrix(utils.make_rdkit_descriptors(mols),
                                        missing_threshold=0.20, min_frequency=0.0)
    return morgan_x.reset_index(drop=True), desc_x.reset_index(drop=True)


def build_docking_features(utils, docking_path, compound_ids):
    raw = pd.read_csv(docking_path, low_memory=False)
    id_col = "Title" if "Title" in raw.columns else "Compound_ID"
    blocked = {id_col, "Title", "Name", "SMILES", "Compound_ID", "Chemical_Name",
               "Rack_Position", "Variant", "source file", "Force Field",
               "docking method", "glide gridfile", "macro-pKa", "tautomer"}
    numeric = {}
    for col in raw.columns:
        if col in blocked or col.startswith("SIFT "):
            continue
        v = pd.to_numeric(raw[col], errors="coerce")
        if v.notna().any():
            numeric[col] = v
    df = pd.DataFrame(numeric)
    df.insert(0, "Name", raw[id_col].astype(str).values)
    agg = df.groupby("Name").mean(numeric_only=True)
    agg.columns = [f"dock_{c}" for c in agg.columns]
    return agg.reindex(compound_ids).fillna(0.0).reset_index(drop=True)


def build_ifp_raw(utils, ifp_path, compound_ids):
    """Return the cleaned (unfiltered) IFP frame and the binary version."""
    ifp = utils.make_interaction_features(ifp_path, compound_ids, min_frequency=0.01)
    ifp = ifp.reset_index(drop=True)
    binary = (ifp > 0).astype(np.float32)
    return ifp, binary


def enrichment_weights(ifp_binary, y_train, threshold, smoothing=1.0):
    """log((P(bit=1|active)+s) / (P(bit=1|inactive)+s)) per column."""
    active_mask = y_train >= threshold
    if active_mask.sum() == 0 or (~active_mask).sum() == 0:
        return np.ones(ifp_binary.shape[1])
    p_act = (ifp_binary[active_mask].sum(axis=0) + smoothing) / (active_mask.sum() + 2 * smoothing)
    p_ina = (ifp_binary[~active_mask].sum(axis=0) + smoothing) / ((~active_mask).sum() + 2 * smoothing)
    w = np.log(p_act / p_ina)
    return w.values


def mi_weights(ifp_train, y_train, seed):
    mi = mutual_info_regression(ifp_train, y_train, random_state=seed)
    return mi


def evaluate(x_train, y_train, x_test, y_test, seed):
    model = ExtraTreesRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)
    return {
        "test_r2": r2_score(y_test, pred_test),
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_rmse": rmse(y_test, pred_test),
    }


def main():
    utils = load_utils()
    activity_path = ROOT / "input" / "Input" / "training_activity_merged_result.csv"
    ifp_path = ROOT / "input" / "Input" / "training_interaction_fingerprints.csv"
    docking_path = ROOT / "input" / "Input" / "training_docking_result.csv"

    meta = utils.read_activity_table(activity_path, "A5")
    y = np.clip(meta["A5"].astype(float).values, 0.0, 100.0)
    meta["A5"] = y

    print(f"Loaded {len(meta)} compounds. Actives (A5>={ACTIVE_THRESHOLD}): {(y>=ACTIVE_THRESHOLD).sum()}")

    morgan_x, desc_x = build_base_features(utils, meta)
    dock_x = build_docking_features(utils, docking_path, meta["Name"].astype(str))
    base_x = pd.concat([morgan_x, desc_x, dock_x], axis=1)
    print(f"Base feature count (morgan+rdkit+docking): {base_x.shape[1]}")

    ifp_x, ifp_binary = build_ifp_raw(utils, ifp_path, meta["Name"])
    print(f"IFP feature count: {ifp_x.shape[1]}")

    schemes = ["A_no_ifp", "B_raw_ifp", "C_active_masked",
               "D_enrichment_weighted", "E_mi_weighted", "F_mi_top128",
               "G_published_setup_pre_masked"]

    all_rows = []
    for seed in SEEDS:
        print(f"\n==== seed={seed} ====")
        train_idx, test_idx = scaffold_split(meta["SMILES"], TEST_SIZE, seed)
        y_train, y_test = y[train_idx], y[test_idx]

        base_tr = base_x.iloc[train_idx].reset_index(drop=True)
        base_te = base_x.iloc[test_idx].reset_index(drop=True)
        ifp_tr = ifp_x.iloc[train_idx].reset_index(drop=True)
        ifp_te = ifp_x.iloc[test_idx].reset_index(drop=True)
        ifp_bin_tr = ifp_binary.iloc[train_idx].reset_index(drop=True)

        for scheme in schemes:
            if scheme == "A_no_ifp":
                x_tr, x_te = base_tr, base_te

            elif scheme == "B_raw_ifp":
                x_tr = pd.concat([base_tr, ifp_tr], axis=1)
                x_te = pd.concat([base_te, ifp_te], axis=1)

            elif scheme == "C_active_masked":
                ifp_tr_masked = ifp_tr.copy()
                ifp_tr_masked.loc[y_train < ACTIVE_THRESHOLD, :] = 0.0
                x_tr = pd.concat([base_tr, ifp_tr_masked], axis=1)
                # At test time IFP is shown raw — same convention as the current pipeline
                x_te = pd.concat([base_te, ifp_te], axis=1)

            elif scheme == "D_enrichment_weighted":
                w = enrichment_weights(ifp_bin_tr, y_train, ACTIVE_THRESHOLD)
                ifp_tr_w = ifp_tr.values * w
                ifp_te_w = ifp_te.values * w
                x_tr = pd.concat([base_tr,
                                  pd.DataFrame(ifp_tr_w, columns=ifp_tr.columns)], axis=1)
                x_te = pd.concat([base_te,
                                  pd.DataFrame(ifp_te_w, columns=ifp_te.columns)], axis=1)

            elif scheme == "E_mi_weighted":
                w = mi_weights(ifp_tr.values, y_train, seed)
                # normalize MI to mean 1 so the magnitude is comparable to raw IFP
                if w.sum() > 0:
                    w = w / w.mean()
                ifp_tr_w = ifp_tr.values * w
                ifp_te_w = ifp_te.values * w
                x_tr = pd.concat([base_tr,
                                  pd.DataFrame(ifp_tr_w, columns=ifp_tr.columns)], axis=1)
                x_te = pd.concat([base_te,
                                  pd.DataFrame(ifp_te_w, columns=ifp_te.columns)], axis=1)

            elif scheme == "F_mi_top128":
                w = mi_weights(ifp_tr.values, y_train, seed)
                k = min(TOP_K_FOR_F, len(w))
                keep = np.argsort(w)[::-1][:k]
                keep_cols = ifp_tr.columns[keep]
                x_tr = pd.concat([base_tr, ifp_tr[keep_cols].reset_index(drop=True)], axis=1)
                x_te = pd.concat([base_te, ifp_te[keep_cols].reset_index(drop=True)], axis=1)

            elif scheme == "G_published_setup_pre_masked":
                # Reproduce the existing pipeline: IFP rows for ANY compound with
                # true A5 < threshold are pre-masked, INCLUDING test compounds.
                # This is what training_features_docking_ifp_threshold38.csv does.
                ifp_tr_masked = ifp_tr.copy()
                ifp_te_masked = ifp_te.copy()
                ifp_tr_masked.loc[y_train < ACTIVE_THRESHOLD, :] = 0.0
                ifp_te_masked.loc[y_test  < ACTIVE_THRESHOLD, :] = 0.0
                x_tr = pd.concat([base_tr, ifp_tr_masked], axis=1)
                x_te = pd.concat([base_te, ifp_te_masked], axis=1)

            metrics = evaluate(x_tr.values, y_train, x_te.values, y_test, seed)
            metrics.update({"scheme": scheme, "seed": seed,
                            "n_features": x_tr.shape[1],
                            "n_train": len(train_idx), "n_test": len(test_idx)})
            all_rows.append(metrics)
            print(f"  {scheme:25s}  n_feat={x_tr.shape[1]:4d}  "
                  f"R2={metrics['test_r2']:.4f}  MAE={metrics['test_mae']:.3f}  "
                  f"RMSE={metrics['test_rmse']:.3f}")

    res = pd.DataFrame(all_rows)
    summary = (res.groupby("scheme")
                 .agg(R2_mean=("test_r2", "mean"), R2_std=("test_r2", "std"),
                      MAE_mean=("test_mae", "mean"), MAE_std=("test_mae", "std"),
                      RMSE_mean=("test_rmse", "mean"), RMSE_std=("test_rmse", "std"),
                      n_features=("n_features", "mean"))
                 .reset_index()
                 .sort_values("R2_mean", ascending=False))

    out_long = OUT / "ifp_scheme_comparison_long.csv"
    out_short = OUT / "ifp_scheme_comparison_summary.csv"
    res.to_csv(out_long, index=False)
    summary.to_csv(out_short, index=False)

    print("\n=========== SUMMARY (mean over 5 seeds) ===========")
    print(summary.to_string(index=False))
    print(f"\nWrote {out_long}")
    print(f"Wrote {out_short}")


if __name__ == "__main__":
    main()
