# GRP78 substrate-binding domain: scaffold-grouped nested cross-validation of Extra Trees, CatBoost and a graph neural network

Code for the corrected evaluation, feature attributions, retrospective re-ranking, figures and supplementary tables of the manuscript on
machine-learning prioritization and fluorescence-polarization testing of GRP78 substrate-binding-domain ligands.

The chemical structures of the training library are proprietary and are not part of this repository.

## Layout

```
GRP78/
├── README.md, LICENSE, requirements.txt
├── config.py                        all paths (override with GRP78_* environment variables)
├── run_all.py                       runs every merge, analysis, figure and table step after the nested-CV runs exist
│
├── main_code/                       MAIN CODE
│   ├── nested_cv.py                 featurization, fold-local preprocessing, Extra Trees / CatBoost, Bayesian optimization,
│   │                                scaffold-grouped nested cross-validation, metrics, nearest-active-neighbor baseline
│   ├── gnn_model.py                 MAIN MODEL (graph neural network): GATv2 graph branch + tabular branch
│   ├── merge_results.py             merges tree and GNN runs, adds the consensus and paired differences
│   └── common.py                    loaders for the training library and its features
│
├── analysis/                        ANALYSES AND TABLES
│   ├── shap_analysis.py             TreeSHAP, feature importance, class-level Shapley values, GNN expected gradients, fold stability
│   ├── retrospective_screening.py   re-ranking of the external libraries with the corrected models
│   ├── tables.py                    CSV tables behind the supplementary tables (performance, cutoff sensitivity, model agreement, partitions,
│   │                                search spaces, masking diagnostics, nearest neighbours, FP results, corrected OTAVA ranking, top-50 audit)
│   └── si_tables.py                 assembles Supplementary Tables S1 to S3 (Excel) from those CSV tables
│
├── figures/                         FIGURE CODE
│   ├── figure1.py                   workflow
│   ├── figure2.py                   chemical space, nearest-neighbor similarity, feature-set ablation
│   ├── figure3.py                   feature-class summary, residue contacts, SHAP summary plots
│   ├── figure4.py                   FP dose response, predicted versus measured inhibition, MD
│   ├── figure5.py                   FP-active compounds and their nearest training compounds
│   ├── figure_supplementary.py      Figures S2 to S5
│   └── style.py                     shared style and save helper
│
├── Input/                           see Input/README.md
└── Output/                          created by the scripts (tables/, figures/, one folder per nested-CV run)
```

## Run order (from the repository root)

1. **Nested cross-validation** (`main_code/nested_cv.py`; test with `--smoke` on synthetic data). Fixed settings: five repeats (seeds 0-4),
   five scaffold-grouped outer folds, three inner folds, hit cutoff 38% inhibition at 25 µM. The Bayesian-optimization budgets
   (6 random + 14 Bayesian trials for the tree models, 3 + 5 for the GNN) are set in `BO_SPACES`.
   ```
   python main_code/nested_cv.py --activity Input/training_library/training_activity_merged_result.csv \
          --ifp Input/training_library/interaction_fingerprints.csv --search bayes \
          --models ExtraTrees CatBoost --out Output/nested_cv_bo_trees_a          # second run with the remaining seeds/sets: nested_cv_bo_trees_b
   python main_code/nested_cv.py ... --models GNN --search bayes --out Output/nested_cv_bo_gnn
   python main_code/nested_cv.py ... --models GNN --search bayes --gnn-all-sets --feature-sets docking_only ifp_only ligand+dock --out Output/nested_cv_bo_gnn_extra
   python main_code/nested_cv.py ... --models ExtraTrees CatBoost --search bayes --feature-sets morgan_only --out Output/nested_cv_bo_morgan_trees
   python main_code/nested_cv.py ... --models GNN --search bayes --feature-sets morgan_only --out Output/nested_cv_bo_morgan_gnn
   ```
   By default the GNN is run on the Morgan-only, ligand-only and full feature sets; `--gnn-all-sets` runs it on every set requested with `--feature-sets`.
   Training of the GNN on CPU is not deterministic, so a rerun with the same seed does not reproduce its predictions exactly.
   The diagnostics of the historical activity-dependent masking use `--mask-ifp` and `--mask-train-only` and are written to
   `Output/nested_cv_masked_diagnostic*`, `Output/nested_cv_masked_train_only`; the random-search comparator is `Output/nested_cv_final_all`.
2. `python run_all.py` merges the runs, runs the analyses, draws the figures and writes the tables in the order they depend on each other
   (`python run_all.py --list` prints the steps; `python run_all.py figures` runs only the figure steps).

Every script in `analysis/` and `figures/` is made of named stages; `python <script> <stage> [<stage> ...]` runs only those stages, and no argument runs all of them.

## Requirements

`pip install -r requirements.txt` (Python 3.14). The graph neural network needs PyTorch and PyTorch Geometric; the tree models run without them.

## Data availability

See the Data and Software Availability section of the manuscript. Deposited derived tables contain no chemical structures.
