import networkx as nx
import pandas as pd

from utils_figures import (
    SHAP_TOP_N,
    SOURCE_TABLE_DIR,
    SPLIT_LABELS,
    figure_path,
    np,
    save_figure,
    set_plot_style,
    short_feature_name,
    style_axes,
    top_shap_indices,
)
from utils_figures import plt


def plot_main_vs_interaction_effect(shap_values, feature_names, split, model_label="Extra Trees", model_slug="extra_trees"):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:SHAP_TOP_N]
    selected = shap_values[:, top_idx]
    corr = np.corrcoef(selected, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    interaction_proxy = np.mean(np.abs(corr), axis=1) * mean_abs[top_idx]
    main_proxy = np.maximum(mean_abs[top_idx] - interaction_proxy, 0)

    order = np.argsort(mean_abs[top_idx])[::-1][:14][::-1]
    names = [short_feature_name(feature_names[top_idx[i]]) for i in order]
    main_vals = main_proxy[order]
    interaction_vals = interaction_proxy[order]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.barh(names, main_vals, color="#159A9C", label="Main effect")
    ax.barh(names, interaction_vals, left=main_vals, color="#D95E55", label="Interaction effect proxy")
    ax.set_xlabel("Mean SHAP magnitude")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_path("07_shap_interactions", f"shap_main_vs_interaction_bar_{model_slug}_{split}.png"))


def plot_interaction_matrix(shap_values, feature_names, split, model_label="Extra Trees", model_slug="extra_trees"):
    top_idx = top_shap_indices(shap_values, 16)
    selected_shap = shap_values[:, top_idx]
    names = [short_feature_name(feature_names[i]) for i in top_idx]
    corr = np.corrcoef(selected_shap, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_yticklabels(names)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation of SHAP effects", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figure_path("07_shap_interactions", f"shap_interaction_composite_matrix_{model_slug}_{split}.png"))


def plot_shap_network(shap_values, feature_names, split, model_label="Extra Trees", model_slug="extra_trees"):
    top_idx = top_shap_indices(shap_values, 18)
    selected_shap = shap_values[:, top_idx]
    names = [short_feature_name(feature_names[i]) for i in top_idx]
    mean_abs = np.mean(np.abs(selected_shap), axis=0)
    corr = np.corrcoef(selected_shap, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    graph = nx.Graph()
    for i, name in enumerate(names):
        graph.add_node(name, weight=float(mean_abs[i]))
    edge_candidates = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            edge_candidates.append((names[i], names[j], abs(float(corr[i, j])), float(corr[i, j])))
    edge_candidates = sorted(edge_candidates, key=lambda item: item[2], reverse=True)[:26]
    for source, target, weight, signed in edge_candidates:
        graph.add_edge(source, target, weight=weight, signed=signed)

    pos = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    max_node = max(mean_abs.max(), 1e-9)
    node_sizes = [800 + 4200 * graph.nodes[node]["weight"] / max_node for node in graph.nodes]
    edge_weights = [graph.edges[edge]["weight"] for edge in graph.edges]
    max_edge = max(edge_weights or [1.0])
    edge_colors = ["#D95E55" if graph.edges[edge]["signed"] >= 0 else "#3366CC" for edge in graph.edges]
    edge_widths = [0.8 + 4.2 * graph.edges[edge]["weight"] / max_edge for edge in graph.edges]

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.45)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#F2B447", edgecolors="#111827", linewidths=1.1, node_size=node_sizes)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10, font_weight="bold")
    ax.text(
        0.5,
        -0.06,
        "Node size = mean |SHAP|; edge width = correlated SHAP co-effect; red = positive, blue = negative",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_axis_off()
    fig.tight_layout()
    save_figure(fig, figure_path("07_shap_interactions", f"shap_feature_importance_interaction_network_{model_slug}_{split}.png"))


def main():
    set_plot_style()
    for model_label, model_slug in [("Extra Trees", "extra_trees"), ("CatBoost", "catboost")]:
        for split in ["random_split", "scaffold_split"]:
            shap_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{split}_shap_values.csv")
            shap_values = shap_df.to_numpy()
            feature_names = list(shap_df.columns)
            plot_main_vs_interaction_effect(shap_values, feature_names, split, model_label, model_slug)
            plot_interaction_matrix(shap_values, feature_names, split, model_label, model_slug)
            plot_shap_network(shap_values, feature_names, split, model_label, model_slug)


if __name__ == "__main__":
    main()
