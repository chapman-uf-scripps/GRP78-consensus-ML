"""
08_run_all_figures_bayesian.py
==============================
Master runner — trains the top-3 Bayesian-optimized models (Extra Trees,
CatBoost, GNN) ONCE, saves source tables to
Output/08_figures_bayesian/10_source_tables/, then generates every figure.

Run from the pipeline_bayesian_docking_ifp_final/ directory:
    python 08_run_all_figures_bayesian.py

Outputs (PNG + PDF) are written to Output/08_figures_bayesian/<sub-folder>/.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from torch_geometric.loader import DataLoader as GraphDataLoader

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from utils_figures import (
    MODEL_COMPARISON_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    SOURCE_TABLE_DIR,
    compute_catboost_shap,
    figure_path,
    load_step11,
    np,
    rmse,
    shap,
    set_plot_style,
    train_top3_predictions,
)


def _load(filename, module_name):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mc  = _load("step08a_fig_model_comparison.py",    "fig_model_comparison")
_vp  = _load("step08b_fig_observed_predicted.py",  "fig_observed_predicted")
_tm  = _load("step08c_fig_train_test_metrics.py",  "fig_train_test_metrics")
_mae = _load("step08d_fig_mae_dotplots.py",        "fig_mae_dotplots")
_pc  = _load("step08e_fig_prediction_curves.py",   "fig_prediction_curves")
_sg  = _load("step08f_fig_shap_global.py",         "fig_shap_global")
_si  = _load("step08i_fig_shap_interactions.py",   "fig_shap_interactions")
_sd  = _load("step08h_fig_shap_dependency.py",     "fig_shap_dependency")
_sh  = _load("step08g_fig_shap_heatmaps.py",       "fig_shap_heatmaps")

plot_model_comparison_overall           = _mc.plot_model_comparison_overall
plot_individual_validation              = _vp.plot_individual_validation
plot_train_test_r2_bars                 = _tm.plot_train_test_r2_bars
plot_train_test_mae_lines               = _tm.plot_train_test_mae_lines
plot_individual_mae_dotplots            = _mae.plot_individual_mae_dotplots
plot_prediction_curve_bars              = _pc.plot_prediction_curve_bars
plot_shap_global_bar                    = _sg.plot_shap_global_bar
plot_feature_group_percent              = _sg.plot_feature_group_percent
plot_feature_group_percent_from_values  = _sg.plot_feature_group_percent_from_values
plot_importance_bar                     = _sg.plot_importance_bar
plot_interaction_matrix                 = _si.plot_interaction_matrix
plot_main_vs_interaction_effect         = _si.plot_main_vs_interaction_effect
plot_shap_network                       = _si.plot_shap_network
plot_single_feature_dependencies        = _sd.plot_single_feature_dependencies
plot_shap_heatmap                       = _sh.plot_shap_heatmap


def save_source_tables(comparison_metrics, top3_metrics, predictions):
    SOURCE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    comparison_metrics.to_csv(figure_path("10_source_tables", "source_all_model_comparison_metrics.csv"), index=False)
    top3_metrics.to_csv(figure_path("10_source_tables", "top3_retrained_metrics.csv"), index=False)
    predictions.to_csv(figure_path("10_source_tables", "top3_retrained_predictions.csv"), index=False)
    figure_path("10_source_tables", "top3_retrained_metrics.json").write_text(
        json.dumps(top3_metrics.to_dict(orient="records"), indent=2)
    )
    print(f"Source tables saved -> {SOURCE_TABLE_DIR}")


def run_catboost_shap_figures(trained):
    feature_names = list(trained[("random_split", "CatBoost")]["x_test"].columns)
    for split in ["random_split", "scaffold_split"]:
        print(f"  SHAP CatBoost [{split}]")
        _, x_shap, shap_values = compute_catboost_shap(trained, split)
        pd.DataFrame(shap_values, columns=feature_names).to_csv(
            SOURCE_TABLE_DIR / f"catboost_{split}_shap_values.csv", index=False
        )
        x_shap.to_csv(SOURCE_TABLE_DIR / f"catboost_{split}_shap_feature_matrix.csv", index=False)
        plot_shap_global_bar(shap_values, feature_names, split, model_label="CatBoost", model_slug="catboost")
        plot_feature_group_percent(shap_values, feature_names, split, model_label="CatBoost", model_slug="catboost")
        plot_main_vs_interaction_effect(shap_values, feature_names, split, "CatBoost", "catboost")
        plot_single_feature_dependencies(shap_values, x_shap, feature_names, split, "CatBoost", "catboost")
        plot_interaction_matrix(shap_values, feature_names, split, "CatBoost", "catboost")
        plot_shap_network(shap_values, feature_names, split, "CatBoost", "catboost")
        plot_shap_heatmap(shap_values, feature_names, split, "CatBoost", "catboost")


def compute_tree_shap(trained, split, model_name):
    item = trained[(split, model_name)]
    x_shap = item["x_test"].copy()
    explainer = shap.TreeExplainer(item["model"])
    shap_values = explainer.shap_values(x_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return x_shap, np.asarray(shap_values)


def run_extra_trees_shap_figures(trained):
    feature_names = list(trained[("random_split", "Extra Trees")]["x_test"].columns)
    for split in ["random_split", "scaffold_split"]:
        print(f"  SHAP Extra Trees [{split}]")
        x_shap, shap_values = compute_tree_shap(trained, split, "Extra Trees")
        pd.DataFrame(shap_values, columns=feature_names).to_csv(
            SOURCE_TABLE_DIR / f"extra_trees_{split}_shap_values.csv", index=False
        )
        x_shap.to_csv(SOURCE_TABLE_DIR / f"extra_trees_{split}_shap_feature_matrix.csv", index=False)
        plot_shap_global_bar(shap_values, feature_names, split, model_label="Extra Trees", model_slug="extra_trees")
        plot_feature_group_percent(shap_values, feature_names, split, model_label="Extra Trees", model_slug="extra_trees")
        plot_main_vs_interaction_effect(shap_values, feature_names, split, "Extra Trees", "extra_trees")
        plot_single_feature_dependencies(shap_values, x_shap, feature_names, split, "Extra Trees", "extra_trees")
        plot_interaction_matrix(shap_values, feature_names, split, "Extra Trees", "extra_trees")
        plot_shap_network(shap_values, feature_names, split, "Extra Trees", "extra_trees")
        plot_shap_heatmap(shap_values, feature_names, split, "Extra Trees", "extra_trees")


def predict_gnn_with_features(step11, model, meta_test, x_values, y_test):
    dataset = step11.GraphTabularDataset(meta_test, x_values, y_test)
    loader = GraphDataLoader(dataset, batch_size=256, shuffle=False)
    _, predictions = step11.predict_gnn(model, loader, next(model.parameters()).device)
    return predictions


def run_gnn_permutation_figures(trained):
    step11 = load_step11()
    for split in ["random_split", "scaffold_split"]:
        print(f"  Permutation importance GNN [{split}]")
        item = trained[(split, "GNN + active-filtered IFP")]
        model = item["model"]
        x_test = item["x_test"].copy()
        y_test = item["y_test"]
        meta_test = item["meta_test"]
        feature_names = list(x_test.columns)
        scaler = model.scaler
        x_scaled = scaler.transform(x_test.values).astype(np.float32)
        baseline_pred = predict_gnn_with_features(step11, model, meta_test, x_scaled, y_test)
        baseline_rmse = rmse(y_test, baseline_pred)
        rng = np.random.default_rng(RANDOM_STATE)
        importances = []
        for col_idx in range(x_scaled.shape[1]):
            x_perm = x_scaled.copy()
            x_perm[:, col_idx] = rng.permutation(x_perm[:, col_idx])
            perm_pred = predict_gnn_with_features(step11, model, meta_test, x_perm, y_test)
            importances.append(max(0.0, rmse(y_test, perm_pred) - baseline_rmse))
        importances = np.asarray(importances)
        pd.DataFrame({"feature": feature_names, "permutation_rmse_increase": importances,
                      "baseline_rmse": baseline_rmse}).to_csv(
            SOURCE_TABLE_DIR / f"gnn_{split}_permutation_importance.csv", index=False
        )
        plot_importance_bar(importances, feature_names, split,
                            model_label="GNN", model_slug="gnn",
                            value_label="Permutation RMSE increase")
        plot_feature_group_percent_from_values(importances, feature_names, split,
                                               model_label="GNN", model_slug="gnn",
                                               value_name="permutation_rmse_increase")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    print("=" * 60)
    print("Step 1/6 — Training Top-3 models (Extra Trees, CatBoost, GNN)")
    print("          with Bayesian-optimized hyperparameters")
    print("=" * 60)
    comparison_metrics = pd.read_csv(MODEL_COMPARISON_PATH)
    top3_metrics, predictions, trained = train_top3_predictions()
    save_source_tables(comparison_metrics, top3_metrics, predictions)

    print("\nStep 2/6 — Model performance figures")
    plot_model_comparison_overall(comparison_metrics)
    plot_individual_validation(predictions)
    plot_train_test_r2_bars(top3_metrics)
    plot_train_test_mae_lines(top3_metrics)
    plot_individual_mae_dotplots(predictions)
    plot_prediction_curve_bars(predictions)

    print("\nStep 3/6 — CatBoost SHAP figures")
    run_catboost_shap_figures(trained)

    print("\nStep 4/6 — Extra Trees SHAP figures")
    run_extra_trees_shap_figures(trained)

    print("\nStep 5/6 — GNN permutation importance figures")
    run_gnn_permutation_figures(trained)

    print("\nStep 6/6 — Done!")
    print(f"All figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
