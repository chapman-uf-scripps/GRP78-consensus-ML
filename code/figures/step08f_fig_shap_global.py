import pandas as pd

from utils_figures import (
    SHAP_TOP_N,
    SOURCE_TABLE_DIR,
    SPLIT_LABELS,
    feature_group,
    figure_path,
    np,
    save_figure,
    set_plot_style,
    short_feature_name,
    style_axes,
)
from utils_figures import plt


def compact_split_label(split):
    return SPLIT_LABELS[split].replace(" Split", "")


def plot_shap_global_bar(shap_values, feature_names, split, model_label="Extra Trees", model_slug="extra_trees", value_label="Mean absolute SHAP value"):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    plot_importance_bar(mean_abs, feature_names, split, model_label, model_slug, value_label)


def plot_importance_bar(values, feature_names, split, model_label, model_slug, value_label):
    values = np.asarray(values)
    top_idx = np.argsort(values)[::-1][:SHAP_TOP_N]
    names = [short_feature_name(feature_names[i]) for i in top_idx][::-1]
    values = values[top_idx][::-1]
    fig, ax = plt.subplots(figsize=(8.8, max(6.0, 0.28 * len(names))))
    ax.barh(names, values, color="#D99A25")
    ax.set_xlabel(value_label)
    ax.set_title(f"Global Feature Contributions\n{model_label} | {SPLIT_LABELS[split]}")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_path("06_shap_global", f"feature_importance_{model_slug}_{split}.png"))


def plot_feature_group_percent(shap_values, feature_names, split, model_label="Extra Trees", model_slug="extra_trees", value_name="mean_abs_shap"):
    values = np.mean(np.abs(shap_values), axis=0)
    plot_feature_group_percent_from_values(values, feature_names, split, model_label, model_slug, value_name)


def plot_feature_group_percent_from_values(values, feature_names, split, model_label, model_slug, value_name):
    groups = pd.DataFrame({"feature": feature_names, value_name: values})
    groups["group"] = groups["feature"].map(feature_group)
    grouped = groups.groupby("group", as_index=False)[value_name].sum()
    grouped["percent"] = grouped[value_name] / grouped[value_name].sum() * 100.0
    grouped = grouped.sort_values("percent")

    colors = {
        "Morgan fingerprint": "#159A9C",
        "RDKit descriptor": "#7B5FB5",
        "Docking score/pose": "#D95E55",
        "Active-filtered IFP": "#D99A25",
        "Other": "#7A8793",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(grouped["group"], grouped["percent"], color=[colors.get(g, "#7A8793") for g in grouped["group"]])
    for y_pos, value in enumerate(grouped["percent"]):
        ax.text(value + 0.8, y_pos, f"{value:.1f}%", va="center", fontweight="bold")
    ax.set_xlim(0, max(100, float(grouped["percent"].max()) + 10))
    ax.set_xlabel("Percent of total feature contribution")
    ax.set_title(f"{model_label} {compact_split_label(split)} Feature Class")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_path("06_shap_global", f"feature_group_percent_{model_slug}_{split}.png"))
    grouped.to_csv(figure_path("06_shap_global", f"feature_group_percent_{model_slug}_{split}.csv"), index=False)


def main():
    set_plot_style()
    for model_label, model_slug in [("Extra Trees", "extra_trees"), ("CatBoost", "catboost")]:
        for split in ["random_split", "scaffold_split"]:
            shap_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{split}_shap_values.csv")
            shap_values = shap_df.to_numpy()
            feature_names = list(shap_df.columns)
            plot_shap_global_bar(shap_values, feature_names, split, model_label, model_slug)
            plot_feature_group_percent(shap_values, feature_names, split, model_label, model_slug)


if __name__ == "__main__":
    main()
