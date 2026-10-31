# Input

Chemical structures of the proprietary 1,488-compound focused library are **not distributed**. The code reads the files below, which have to be supplied by the user (the library can be obtained from the corresponding author under a material-transfer agreement).

| File | Content | Distributed |
|---|---|---|
| `training_library/training_activity_merged_result.csv` | columns `Name`, `SMILES`, `A5` (percent inhibition at 25 µM); one row per compound | no |
| `training_library/interaction_fingerprints.csv` | Maestro interaction-fingerprint table (`Title`, `A<residue>_<type>` columns) and the numeric Glide/Epik columns, one row per docked pose | no |
| `5E85.pdb` | PDB entry 5E85 (chain A), used only to read residue names | download from the PDB |
| `external_libraries/` | OTAVA library table and the FDA GSRS-derived library table (structures, IDs, original three-model predictions) used by `analysis/retrospective_screening.py` | no (OTAVA Chemicals; PubChem GSRS source) |
| `md_simulations/MD_Analysis_Summary_1.csv`, `_2.csv`, `_3.csv` | GROMACS analysis summaries of the three simulations per complex (time, ligand RMSD, Coulomb, Lennard-Jones, total interaction energy) | on request |
| `original_selection/Table_S3_otava_top50_triple_consensus.csv`, `Table_S4_fda_top50_triple_consensus.csv` | top 50 compounds of the original selection of the two external libraries | yes |
| `original_selection/Table_S2A_algorithm_survey.csv` | scores of the 17 regression algorithms of the original selection (single scaffold split, masked pipeline); a historical record, not produced by this code | yes |

Paths can be changed with environment variables (see `config.py`), for example `GRP78_TRAINING`, `GRP78_IFP`, `GRP78_OUTPUT`.
