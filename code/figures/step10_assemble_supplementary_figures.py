from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).parent
FIG_DIR = ROOT / "Output" / "08_figures_bayesian"
SCREEN_DIR = ROOT / "Output" / "09_figure5_otava_screening"
OUT_DIR = ROOT / "Output" / "10_supplementary_figures"


def copy_pair(source_stem, target_stem, records, caption):
    source_stem = Path(source_stem)
    target_stem = OUT_DIR / target_stem
    target_stem.parent.mkdir(parents=True, exist_ok=True)
    copied = []
    for suffix in [".png", ".pdf", ".csv"]:
        src = source_stem.with_suffix(suffix)
        if src.exists():
            dst = target_stem.with_suffix(suffix)
            shutil.copy2(src, dst)
            copied.append(dst.name)
    records.append(
        {
            "si_figure": target_stem.stem,
            "caption": caption,
            "source": str(source_stem),
            "files": "; ".join(copied),
        }
    )


def add_panel_label(ax, label):
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3),
    )


def make_image_grid(image_paths, target_stem, title, ncols=3):
    image_paths = [Path(path) for path in image_paths]
    nrows = (len(image_paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.1 * ncols, 4.8 * nrows))
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for idx, ax in enumerate(axes):
        ax.set_axis_off()
        if idx >= len(image_paths):
            continue
        image = plt.imread(image_paths[idx])
        ax.imshow(image)
        add_panel_label(ax, chr(ord("A") + idx))
    fig.suptitle(title, fontsize=24, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    target = OUT_DIR / target_stem
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    copy_pair(
        FIG_DIR / "01_model_comparison" / "bayesian_model_comparison_overall_r2",
        "Figure_S1_all_model_scaffold_r2_ranking",
        records,
        "Complete Bayesian-optimized model ranking for scaffold-split test R2.",
    )

    random_validation = [
        FIG_DIR / "02_validation_observed_predicted" / f"validation_random_split_{model}.png"
        for model in ["extra_trees", "catboost", "gnn"]
    ]
    make_image_grid(
        random_validation,
        "Figure_S2_random_split_observed_predicted_validation",
        "Random-split observed vs predicted validation",
        ncols=3,
    )
    records.append(
        {
            "si_figure": "Figure_S2_random_split_observed_predicted_validation",
            "caption": "Observed versus predicted GRP78 inhibition for the top three models under random splitting.",
            "source": "; ".join(map(str, random_validation)),
            "files": "Figure_S2_random_split_observed_predicted_validation.png; Figure_S2_random_split_observed_predicted_validation.pdf",
        }
    )

    metric_panels = [
        FIG_DIR / "03_train_test_metrics" / "top3_train_test_r2_bar_random_split.png",
        FIG_DIR / "03_train_test_metrics" / "top3_train_test_r2_bar_scaffold_split.png",
        FIG_DIR / "03_train_test_metrics" / "top3_train_test_mae_line_random_split.png",
        FIG_DIR / "03_train_test_metrics" / "top3_train_test_mae_line_scaffold_split.png",
    ]
    make_image_grid(
        metric_panels,
        "Figure_S3_train_test_metric_summary",
        "Train-test performance summary",
        ncols=2,
    )
    records.append(
        {
            "si_figure": "Figure_S3_train_test_metric_summary",
            "caption": "Train and test R2/MAE summaries for the top three models under random and scaffold splits.",
            "source": "; ".join(map(str, metric_panels)),
            "files": "Figure_S3_train_test_metric_summary.png; Figure_S3_train_test_metric_summary.pdf",
        }
    )

    prediction_curves = [
        FIG_DIR / "05_prediction_curves" / f"test_data_curve_{split}_{model}.png"
        for split in ["random_split", "scaffold_split"]
        for model in ["extra_trees", "catboost", "gnn"]
    ]
    make_image_grid(
        prediction_curves,
        "Figure_S4_test_prediction_curves",
        "Rank-ordered test-set prediction curves",
        ncols=3,
    )
    records.append(
        {
            "si_figure": "Figure_S4_test_prediction_curves",
            "caption": "Observed and predicted GRP78 inhibition for test compounds sorted by observed inhibition.",
            "source": "; ".join(map(str, prediction_curves)),
            "files": "Figure_S4_test_prediction_curves.png; Figure_S4_test_prediction_curves.pdf",
        }
    )

    mae_dotplots = [
        FIG_DIR / "04_mae_dotplots" / f"mae_dotplot_{split}_{model}.png"
        for split in ["random_split", "scaffold_split"]
        for model in ["extra_trees", "catboost", "gnn"]
    ]
    make_image_grid(
        mae_dotplots,
        "Figure_S5_absolute_error_dotplots",
        "Absolute error distributions across test compounds",
        ncols=3,
    )
    records.append(
        {
            "si_figure": "Figure_S5_absolute_error_dotplots",
            "caption": "Per-compound absolute prediction error for top models under random and scaffold splits.",
            "source": "; ".join(map(str, mae_dotplots)),
            "files": "Figure_S5_absolute_error_dotplots.png; Figure_S5_absolute_error_dotplots.pdf",
        }
    )

    feature_importance = [
        FIG_DIR / "06_shap_global" / f"feature_importance_{model}_{split}.png"
        for split in ["random_split", "scaffold_split"]
        for model in ["extra_trees", "catboost", "gnn"]
    ]
    make_image_grid(
        feature_importance,
        "Figure_S6_full_feature_importance_top15",
        "Full top-feature importance profiles",
        ncols=3,
    )
    records.append(
        {
            "si_figure": "Figure_S6_full_feature_importance_top15",
            "caption": "Top 15 global features for Extra Trees, CatBoost, and GNN under random and scaffold splits.",
            "source": "; ".join(map(str, feature_importance)),
            "files": "Figure_S6_full_feature_importance_top15.png; Figure_S6_full_feature_importance_top15.pdf",
        }
    )

    feature_groups = [
        FIG_DIR / "06_shap_global" / f"feature_group_percent_{model}_{split}.png"
        for split in ["random_split", "scaffold_split"]
        for model in ["extra_trees", "catboost", "gnn"]
    ]
    make_image_grid(
        feature_groups,
        "Figure_S7_feature_class_contributions_all_splits",
        "Feature-class contribution summaries",
        ncols=3,
    )
    records.append(
        {
            "si_figure": "Figure_S7_feature_class_contributions_all_splits",
            "caption": "Feature-class contribution summaries for top models under random and scaffold splits.",
            "source": "; ".join(map(str, feature_groups)),
            "files": "Figure_S7_feature_class_contributions_all_splits.png; Figure_S7_feature_class_contributions_all_splits.pdf",
        }
    )

    shap_heatmaps = [
        FIG_DIR / "09_shap_heatmaps" / f"shap_heatmap_features_by_instances_{model}_{split}.png"
        for split in ["random_split", "scaffold_split"]
        for model in ["extra_trees", "catboost"]
    ]
    make_image_grid(
        shap_heatmaps,
        "Figure_S8_shap_instance_heatmaps",
        "SHAP feature-by-instance heatmaps",
        ncols=2,
    )
    records.append(
        {
            "si_figure": "Figure_S8_shap_instance_heatmaps",
            "caption": "Feature-by-instance SHAP heatmaps for Extra Trees and CatBoost.",
            "source": "; ".join(map(str, shap_heatmaps)),
            "files": "Figure_S8_shap_instance_heatmaps.png; Figure_S8_shap_instance_heatmaps.pdf",
        }
    )

    shap_interactions = [
        FIG_DIR / "07_shap_interactions" / f"shap_interaction_composite_matrix_{model}_scaffold_split.png"
        for model in ["extra_trees", "catboost"]
    ] + [
        FIG_DIR / "07_shap_interactions" / f"shap_main_vs_interaction_bar_{model}_scaffold_split.png"
        for model in ["extra_trees", "catboost"]
    ]
    make_image_grid(
        shap_interactions,
        "Figure_S9_scaffold_shap_interaction_summaries",
        "Scaffold-split SHAP interaction summaries",
        ncols=2,
    )
    records.append(
        {
            "si_figure": "Figure_S9_scaffold_shap_interaction_summaries",
            "caption": "SHAP interaction matrices and main-versus-interaction effect summaries for tree models.",
            "source": "; ".join(map(str, shap_interactions)),
            "files": "Figure_S9_scaffold_shap_interaction_summaries.png; Figure_S9_scaffold_shap_interaction_summaries.pdf",
        }
    )

    for model in ["extra_trees", "catboost", "gnn"]:
        for split in ["random_split", "scaffold_split"]:
            copy_pair(
                FIG_DIR / "06_shap_global" / f"feature_importance_{model}_{split}",
                f"individual_feature_importance_{model}_{split}",
                records,
                f"Individual full feature-importance plot for {model} under {split}.",
            )
            copy_pair(
                FIG_DIR / "06_shap_global" / f"feature_group_percent_{model}_{split}",
                f"individual_feature_group_percent_{model}_{split}",
                records,
                f"Individual feature-class contribution plot for {model} under {split}.",
            )

    top50 = ROOT / "Output" / "04_screen_otava_best3" / "otava_bayesian_top3_extra_catboost_gnn_predictions.csv"
    if top50.exists():
        shutil.copy2(top50, OUT_DIR / "Table_S1_full_otava_top3_model_predictions.csv")
        records.append(
            {
                "si_figure": "Table_S1_full_otava_top3_model_predictions",
                "caption": "Full OTAVA screening prediction table from the top three models.",
                "source": str(top50),
                "files": "Table_S1_full_otava_top3_model_predictions.csv",
            }
        )

    pd.DataFrame(records).to_csv(OUT_DIR / "supplementary_figure_manifest.csv", index=False)
    print(f"Supplementary figures assembled in: {OUT_DIR}")
    print(pd.DataFrame(records)[["si_figure", "caption"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
