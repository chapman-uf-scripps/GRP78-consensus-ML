## Overview

This repository implements a target-agnostic consensus machine-learning
framework for prospective inhibitor discovery from focused-library
biochemical screens. Four orthogonal feature blocks — Morgan fingerprints,
RDKit descriptors, Glide docking descriptors, and active-filtered per-residue
interaction fingerprints — feed a Bayesian-optimized consensus of three model
families with non-overlapping inductive priors (Extra Trees, CatBoost, and a
graph-tabular graph neural network), interpreted by TreeSHAP and permutation
importance.

The framework is demonstrated end-to-end on the GRP78 substrate-binding
domain (SBD) starting from a 1,488-compound focused fluorescence-polarization
screen, and produces five experimentally confirmed inhibitors with FP IC₅₀
values of 4.3–16.1 µM, including a peptidoboronate scaffold not previously
reported against GRP78 SBD.

---

## Repository structure

```
GRP78-consensus-ML/
├── README.md                     ← this file
├── LICENSE                       ← MIT
├── CITATION.cff                  ← citation metadata (GitHub renders it)
├── requirements.txt              ← pinned Python dependencies
├── .gitignore                    ← excludes proprietary training data
└── code/
    ├── config.py                 ← global paths and constants
    ├── run_pipeline.py           ← top-level pipeline driver
    ├── generate_publication_outputs.py   ← end-to-end figure/table driver
    │
    ├── pipeline/                 ← core numbered pipeline (steps 01–11)
    ├── utils/                    ← shared helper modules
    ├── figures/                  ← publication figure generation
    ├── ablation/                 ← feature-class ablation studies
    ├── validation/               ← seed-variance and robustness studies
    └── analysis/                 ← post-hoc analysis
```

Detailed per-script summaries live inside each subdirectory.

---

## Installation

```bash
git clone https://github.com/chapman-uf-scripps/GRP78-consensus-ML.git
cd GRP78-consensus-ML
pip install -r requirements.txt
```

Python 3.10 or later is required.

---

## Data availability

**The 1,488-compound focused training library is proprietary to the Chapman
Laboratory at the University of Florida and is not distributed with this
repository.** Aggregate library statistics (chemotype count, Bemis–Murcko
scaffold distribution, activity distribution) are provided in Supporting
Figure S1 of the accompanying manuscript.

Requests for training-library access under a material-transfer agreement,
for the purpose of academic collaboration, should be directed to the
corresponding author (Eli Chapman, University of Florida,
`echapman@cop.ufl.edu`).

The published Bayesian-optimized model checkpoints included in this
repository can be applied to any external compound library **without** access
to the underlying training data.

---

## Reproducing published results

### With access to the training library (internal collaborators only)

```bash
python code/run_pipeline.py                    # steps 01–06
python code/generate_publication_outputs.py    # all figures and tables
```

### Applying the published models to a new external library

Prepare an input CSV with a `SMILES` column (and optionally a Glide docking
output CSV):

```bash
python code/pipeline/step06_screen_otava.py \
    --library path/to/your_library.csv \
    --docking path/to/your_docking_output.csv
```

Compounds without docking poses are handled by the SMILES-only branch
(`code/pipeline/step11b_screen_fda_smiles_only.py`).

---

## Citing this code

The `CITATION.cff` file in this repository is rendered by GitHub's
**"Cite this repository"** button in the right sidebar. If the paper is
accepted, please cite the published article as well.

---

## Contact

- **Corresponding author:** Eli Chapman — `echapman@cop.ufl.edu`
- **First author (code questions):** Haofeng Ye — `hye1@ufl.edu`

Open a GitHub issue on this repository for bug reports or method questions.
rts or method questions.
