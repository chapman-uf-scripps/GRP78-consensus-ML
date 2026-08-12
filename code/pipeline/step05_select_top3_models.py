import json
import sys

from config import OUTPUT_DIR, VENV_SITE_PACKAGES


if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import pandas as pd


RANKING_PATH = OUTPUT_DIR / "02c_bayesian_optimized_deep_models" / "bayesian_optimized_all_models_including_deep_ranking.csv"
SELECT_OUTPUT_DIR = OUTPUT_DIR / "03_selected_best3"
TOP_N = 3


def main():
    if not RANKING_PATH.exists():
        raise FileNotFoundError(
            f"Final ranking not found: {RANKING_PATH}. Run 02a, 02b, and 02c Bayesian optimization first."
        )
    ranking = pd.read_csv(RANKING_PATH)
    required = {"Overall_Rank", "Model", "Test_R2", "Test_RMSE", "Test_MAE"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"Ranking file is missing required columns: {sorted(missing)}")

    selected = ranking.sort_values(["Overall_Rank"]).head(TOP_N).copy()
    SELECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_out = SELECT_OUTPUT_DIR / "final_bayesian_model_ranking.csv"
    selected_csv = SELECT_OUTPUT_DIR / "selected_best3_models.csv"
    selected_json = SELECT_OUTPUT_DIR / "selected_best3_models.json"
    ranking.to_csv(ranking_out, index=False)
    selected.to_csv(selected_csv, index=False)
    payload = {
        "selection_rule": "Top 3 by scaffold-split Test_R2 after Bayesian optimization; Test_RMSE/Test_MAE used for ranking ties.",
        "ranking_source": str(RANKING_PATH),
        "selected_models": selected["Model"].tolist(),
        "selected_rows": selected.to_dict(orient="records"),
        "outputs": {
            "ranking_csv": str(ranking_out),
            "selected_csv": str(selected_csv),
            "selected_json": str(selected_json),
        },
    }
    selected_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Selected final Bayesian top 3:")
    print(selected[["Overall_Rank", "Model", "Test_R2", "Test_RMSE", "Test_MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
