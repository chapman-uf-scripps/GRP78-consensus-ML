"""
Runs every merge, analysis, figure and table step after the nested cross-validation runs exist in Output/ (README, step 1).

    python run_all.py              all steps in order
    python run_all.py figures      only the steps whose command contains "figures"
    python run_all.py --list       print the steps

Each script also runs alone, and its stages can be chosen: python analysis/tables.py nested_cv_results model_agreement
"""
import subprocess
import sys

STEPS = [
    "main_code/merge_results.py nested_cv_final_bo nested_cv_bo_gnn,nested_cv_bo_gnn_extra nested_cv_bo_trees_a,nested_cv_bo_trees_b",
    "main_code/merge_results.py nested_cv_bo_morgan_merged nested_cv_bo_morgan_gnn nested_cv_bo_morgan_trees",
    "analysis/retrospective_screening.py",
    "analysis/shap_analysis.py",
    "figures/figure1.py", "figures/figure2.py", "figures/figure3.py", "figures/figure4.py",
    "analysis/tables.py",
    "figures/figure5.py", "figures/figure_supplementary.py",
    "analysis/si_tables.py",
]

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--list"]:
        print("\n".join(STEPS))
        sys.exit()
    for step in [s for s in STEPS if not args or args[0] in s]:
        print("\n>>>", step, flush=True)
        if subprocess.call([sys.executable, *step.split()]):
            sys.exit(f"step failed: {step}")
