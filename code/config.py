"""
Global configuration for the consensus machine-learning pipeline.

All paths are resolved *relative* to the repository root, so the code
runs unchanged when the repository is cloned to any location. Do not
hard-code absolute paths anywhere in this project.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
# code/config.py lives at CODE_ROOT/config.py, so:
CODE_ROOT = Path(__file__).resolve().parent          # code/
REPO_ROOT = CODE_ROOT.parent                          # repository root
INPUT_DIR = REPO_ROOT / "input" / "Input"             # training / external CSVs
OUTPUT_DIR = REPO_ROOT / "output"                     # generated artifacts

# Backwards-compat alias
PIPELINE_DIR = CODE_ROOT

# ---------------------------------------------------------------------------
# Training data (proprietary — not included in the public repository)
# ---------------------------------------------------------------------------
TRAINING_ACTIVITY = INPUT_DIR / "training_activity_merged_result.csv"
TRAINING_DOCKING = INPUT_DIR / "training_docking_result.csv"
TRAINING_IFP = INPUT_DIR / "training_interaction_fingerprints.csv"

# ---------------------------------------------------------------------------
# External libraries (OTAVA + FDA-derived collection)
# ---------------------------------------------------------------------------
OTAVA_LIBRARY = INPUT_DIR / "Libraries" / "otava_library.xlsx"
OTAVA_DOCKING = INPUT_DIR / "External" / "OTAVA_docking_result.csv"
OTAVA_IFP     = INPUT_DIR / "External" / "Otava_IFP.csv"

# ---------------------------------------------------------------------------
# Pipeline constants (identical to those used to produce the paper)
# ---------------------------------------------------------------------------
ACTIVE_THRESHOLD = 38.0    # % inhibition cutoff for IFP active-filtering
TARGET = "A5"              # activity column name in the training CSV
ID_COL = "Name"            # compound identifier column
SMILES_COL = "SMILES"      # canonical SMILES column
SEED = 42                  # global random seed for reproducibility
TOP_N = 200                # number of top-ranked hits retained per screen


def require_training_data():
    """Fail fast with a helpful error if proprietary training data is missing."""
    missing = [p for p in (TRAINING_ACTIVITY, TRAINING_DOCKING, TRAINING_IFP)
               if not p.exists()]
    if missing:
        names = ", ".join(str(p.name) for p in missing)
        raise FileNotFoundError(
            f"Required training-data file(s) not found: {names}\n\n"
            "The 1,488-compound focused training library is proprietary to the\n"
            "Chapman Laboratory (University of Florida) and is not distributed\n"
            "with this repository. See the Data Availability section of the\n"
            "accompanying manuscript for access instructions.\n\n"
            "The published model checkpoints can be applied to any external\n"
            "library WITHOUT access to the training data."
        )
