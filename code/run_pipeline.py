import argparse
import subprocess
import sys
from pathlib import Path

from config import PIPELINE_DIR


PYTHON = Path(r"C:\Users\hye1\AppData\Local\Programs\Python\Python314\python.exe")

STEPS = [
    ("01", "Prepare docking + IFP feature table", "01_prepare_inputs_features.py"),
    ("02a", "Bayesian optimize strong classical models", "02a_bayesian_optimize_strong_models.py"),
    ("02b", "Bayesian optimize remaining classical models", "02b_bayesian_optimize_remaining_models.py"),
    ("02c", "Bayesian optimize deep-learning models", "02c_bayesian_optimize_deep_models.py"),
    ("03", "Select final Bayesian top 3", "03_select_best3_models.py"),
    ("04", "Screen OTAVA with final best 3", "04_screen_otava_with_best3.py"),
]


def main():
    parser = argparse.ArgumentParser(description="Run the organized Bayesian docking+IFP OTAVA screening pipeline.")
    parser.add_argument("--from-step", default="01", help="First step id to run, e.g. 01, 02a, 03.")
    parser.add_argument("--to-step", default="04", help="Last step id to run, e.g. 02c, 04.")
    parser.add_argument("--list", action="store_true", help="List pipeline steps and exit.")
    args = parser.parse_args()

    if args.list:
        for step_id, label, script in STEPS:
            print(f"{step_id}: {label} ({script})")
        return

    ids = [step_id for step_id, _, _ in STEPS]
    if args.from_step not in ids or args.to_step not in ids:
        raise ValueError(f"Step ids must be one of: {ids}")
    start = ids.index(args.from_step)
    end = ids.index(args.to_step)
    if start > end:
        raise ValueError("--from-step must come before --to-step")

    for step_id, label, script in STEPS[start : end + 1]:
        print(f"\n=== {step_id}: {label} ===", flush=True)
        subprocess.run([str(PYTHON), str(PIPELINE_DIR / script)], check=True)


if __name__ == "__main__":
    main()
