"""Paths and constants. Every path can be overridden with an environment variable of the same name (for example GRP78_OUTPUT)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _p(name, default):
    return Path(os.environ.get(name, default))


INPUT = _p("GRP78_INPUT", REPO / "Input")
OUTPUT = _p("GRP78_OUTPUT", REPO / "Output")
TABLES = _p("GRP78_TABLES", OUTPUT / "tables")
FIGURES = _p("GRP78_FIGURES", OUTPUT / "figures")

# Proprietary training library (chemical structures): NOT distributed, see Input/README.md
TRAINING = _p("GRP78_TRAINING", INPUT / "training_library" / "training_activity_merged_result.csv")
IFP_TABLE = _p("GRP78_IFP", INPUT / "training_library" / "interaction_fingerprints.csv")

RECEPTOR_PDB = _p("GRP78_PDB", INPUT / "5E85.pdb")                              # public PDB entry
EXTERNAL_LIBRARIES = _p("GRP78_EXTERNAL", INPUT / "external_libraries")         # OTAVA and approved-drug-derived libraries
MD_SIMULATIONS = _p("GRP78_MD", INPUT / "md_simulations")                       # MD_Analysis_Summary_1/2/3.csv

HIT_CUTOFF = 38.0           # measured inhibition (%) at 25 uM that defines a hit
SEEDS = (0, 1, 2, 3, 4)
for _d in (TABLES, FIGURES):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
