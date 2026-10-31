"""Loaders shared by the analysis and figure scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402

import pandas as pd
from main_code import nested_cv as n                  # noqa: E402


def load_training():
    """Training library with a measured inhibition (column A5), one row per compound."""
    return pd.read_csv(cfg.TRAINING).dropna(subset=["A5"]).drop_duplicates("Name").reset_index(drop=True)


def load_features(act):
    """Morgan and RDKit blocks, the molecules, and the best-pose interaction-fingerprint and docking blocks of the training library."""
    mg, ds, mols = n.morgan_desc(act.SMILES)
    ifp, dock = n.best_pose_features(cfg.IFP_TABLE, act.Name)
    return mg, ds, mols, ifp, dock
