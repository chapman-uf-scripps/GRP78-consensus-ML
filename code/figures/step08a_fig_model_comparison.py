import pandas as pd

from utils_figures import (
    MODEL_COLORS,
    MODEL_COMPARISON_PATH,
    MODEL_LABELS,
    TOP3,
    figure_path,
    save_figure,
    set_plot_style,
    style_axes,
)
from utils_figures import plt

PLOT_MODELS = [
    "Extra Trees",
    "CatBoost",
    "GNN + active-filtered IFP",
    "VAE + active-filtered IFP",
    "Hist Gradient Boosting",
    "Gradient Boosting",
    "LightGBM",
    "XGBoost",
    "Random Forest",
    "KNN",
    "SVR",
    "MLP + active-filtered IFP",
    "Elastic Net",
    "Ridge",
    "SMILES-CNN",
    "SMILES-LSTM",
    "SMILES-Transformer",
]

MIN_TEST_R2_FOR_BAR = 0.4


def plot_model_comparison_overall(comparison_metrics):
    plot_df = comparison_metrics.copy()
    # Drop the GNN baseline row — keep only bayesian_optimized entries
    plot_df = plot_df[plot_df["Label"] != "baseline_same_docking_ifp_table"].copy()
    plot_df["Model_label"] = plot_df["Model"].map(MODEL_LABELS).fillna(plot_df["Model"])
    subset = plot_df[plot_df["Model"].isin(PLOT_MODELS)].copy()
    subset = subset[subset["Test_R2"] >= MIN_TEST_R2_FOR_BAR].copy()
    subset = subset.sort_values("Test_R2", ascending=True)

    fig, ax = plt.subplots(figsize=(9.8, max(8.0, 0.52 * len(subset))))
    colors = [
        MODEL_COLORS.get(model, "#AAB4BF") if model in TOP3 else "#CDD5DF"
        for model in subset["Model"]
    ]
    ax.barh(subset["Model_label"], subset["Test_R2"], color=colors, edgecolor="white", linewidth=0.7)
    ax.set_title("Bayesian-Optimized Model Ranking\n(Scaffold Split)", pad=22)
    ax.set_xlabel(r"Test R$^2$")
    ax.set_xlim(0.40, 1.00)
    ax.set_ylim(-0.7, len(subset) - 0.1)
    style_axes(ax)
    for y_pos, value in enumerate(subset["Test_R2"]):
        offset = 0.015 if value >= 0 else -0.055
        ax.text(value + offset, y_pos, f"{value:.4f}", va="center", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, figure_path("01_model_comparison", "bayesian_model_comparison_overall_r2.png"))


def main():
    set_plot_style()
    plot_model_comparison_overall(pd.read_csv(MODEL_COMPARISON_PATH))


if __name__ == "__main__":
    main()
