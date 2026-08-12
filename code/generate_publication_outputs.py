"""
Consolidated publication-output runner for the GRP78 ML paper.

This script keeps the final figure-generation workflow in one place. It assumes
this package sits beside the working `GRP78_ML_Pipeline` folder:

    Xiaoyi/
      GRP78_ML_Pipeline/
      GRP78_Publication_Package/

Run from this folder or anywhere:

    python code/generate_publication_outputs.py

The original pipeline scripts write figures to `GRP78_ML_Pipeline/Output/`.
After generation, this script copies the paper-facing outputs back into this
package's `output/` folders.
"""

from pathlib import Path
import shutil
import subprocess
import sys


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PACKAGE_DIR.parent / "GRP78_ML_Pipeline"
PYTHON = sys.executable

MAIN_DIR = PACKAGE_DIR / "output" / "main_figures"
SI_DIR = PACKAGE_DIR / "output" / "supplementary_figures"
SOURCE_DIR = PACKAGE_DIR / "output" / "source_tables"


FINAL_SCRIPTS = [
    "step08a_fig_model_comparison.py",
    "step08j_fig4_interpretation_composites.py",
    "step09_fig5_otava_screening_validation.py",
    "step10_assemble_supplementary_figures.py",
    "step10b_si_shap_global_maps.py",
    "step10c_fig5_shap_global_maps_main.py",
]


def run_script(script_name):
    script_path = PIPELINE_DIR / script_name
    print(f"Running {script_path.name}")
    subprocess.run([PYTHON, str(script_path)], cwd=str(PIPELINE_DIR), check=True)


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print(f"Missing, skipped: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_glob(src_dir, pattern, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.glob(pattern):
        shutil.copy2(src, dst_dir / src.name)


def refresh_package_outputs():
    main = MAIN_DIR
    root = PIPELINE_DIR

    copy_if_exists(root / "Output/07_chemical_space/Figure_3_training_chemical_space.png", main / "Figure_2A_chemical_space.png")
    copy_if_exists(root / "Output/07_chemical_space/Figure_3_training_chemical_space.pdf", main / "Figure_2A_chemical_space.pdf")
    copy_if_exists(root / "Output/08_figures_bayesian/01_model_comparison/bayesian_model_comparison_overall_r2.png", main / "Figure_2B_model_comparison_r2.png")
    copy_if_exists(root / "Output/08_figures_bayesian/01_model_comparison/bayesian_model_comparison_overall_r2.pdf", main / "Figure_2B_model_comparison_r2.pdf")

    for model in ["extra_trees", "catboost", "gnn"]:
        copy_if_exists(
            root / f"Output/08_figures_bayesian/02_validation_observed_predicted/validation_scaffold_split_{model}.png",
            main / f"Figure_3_validation_scaffold_{model}.png",
        )
        copy_if_exists(
            root / f"Output/08_figures_bayesian/02_validation_observed_predicted/validation_scaffold_split_{model}.pdf",
            main / f"Figure_3_validation_scaffold_{model}.pdf",
        )

    copy_if_exists(root / "Output/08_figures_bayesian/11_figure4_interpretation/figure4_feature_class_contributions_scaffold.png", main / "Figure_4A_feature_class_contributions.png")
    copy_if_exists(root / "Output/08_figures_bayesian/11_figure4_interpretation/figure4_feature_class_contributions_scaffold.pdf", main / "Figure_4A_feature_class_contributions.pdf")
    copy_if_exists(root / "Output/08_figures_bayesian/11_figure4_interpretation/figure4_normalized_cross_model_feature_importance_scaffold.png", main / "Figure_4B_normalized_cross_model_feature_heatmap.png")
    copy_if_exists(root / "Output/08_figures_bayesian/11_figure4_interpretation/figure4_normalized_cross_model_feature_importance_scaffold.pdf", main / "Figure_4B_normalized_cross_model_feature_heatmap.pdf")

    copy_if_exists(root / "Output/08_figures_bayesian/12_figure5_shap_global_maps/figure5_tree_model_shap_global_contribution_maps_scaffold.png", main / "Figure_5_SHAP_global_contribution_maps.png")
    copy_if_exists(root / "Output/08_figures_bayesian/12_figure5_shap_global_maps/figure5_tree_model_shap_global_contribution_maps_scaffold.pdf", main / "Figure_5_SHAP_global_contribution_maps.pdf")

    for suffix in ["png", "pdf"]:
        copy_if_exists(root / f"Output/09_figure5_otava_screening/figure5a_otava_top3_model_overlap_predA5_gt38.{suffix}", main / f"Figure_6A_OTAVA_model_overlap.{suffix}")
        copy_if_exists(root / f"Output/09_figure5_otava_screening/figure5b_otava_top10_consensus_ranking.{suffix}", main / f"Figure_6B_OTAVA_top10_consensus.{suffix}")
        copy_if_exists(root / f"Output/09_figure5_otava_screening/figure5c_confirmed_active_otava_structures.{suffix}", main / f"Figure_6C_confirmed_active_structures.{suffix}")

    copy_glob(root / "Output/10_supplementary_figures", "*", SI_DIR)
    copy_glob(root / "Output/08_figures_bayesian/10_source_tables", "*", SOURCE_DIR)
    copy_glob(root / "Output/09_figure5_otava_screening", "*.csv", SOURCE_DIR)
    copy_glob(root / "Output/08_figures_bayesian/11_figure4_interpretation", "*.csv", SOURCE_DIR)
    copy_glob(root / "Output/08_figures_bayesian/12_figure5_shap_global_maps", "*.csv", SOURCE_DIR)


def main():
    if not PIPELINE_DIR.exists():
        raise FileNotFoundError(f"Cannot find pipeline folder: {PIPELINE_DIR}")
    for folder in [MAIN_DIR, SI_DIR, SOURCE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    for script in FINAL_SCRIPTS:
        run_script(script)
    refresh_package_outputs()
    print(f"Publication package refreshed: {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
