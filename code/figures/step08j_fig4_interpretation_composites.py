from pathlib import Path
import re

import pandas as pd

from utils_figures import (
    FEATURE_TABLE_PATH,
    OUTPUT_DIR,
    SOURCE_TABLE_DIR,
    feature_group,
    np,
    set_plot_style,
    short_feature_name,
)
from utils_figures import plt


SPLIT = "scaffold_split"
OUT_DIR = OUTPUT_DIR / "11_figure4_interpretation"
TOP_N = 15
NORMALIZED_HEATMAP_TOP_N = 8

MODELS = [
    {
        "label": "Extra Trees",
        "slug": "extra_trees",
        "kind": "shap",
        "metric_label": "Mean absolute SHAP value",
    },
    {
        "label": "CatBoost",
        "slug": "catboost",
        "kind": "shap",
        "metric_label": "Mean absolute SHAP value",
    },
    {
        "label": "GNN",
        "slug": "gnn",
        "kind": "permutation",
        "metric_label": "Permutation RMSE\nincrease",
    },
]

GROUP_ORDER = [
    "Active-filtered IFP",
    "Morgan fingerprint",
    "Docking score/pose",
    "RDKit descriptor",
]

GROUP_COLORS = {
    "Active-filtered IFP": "#D99A25",
    "Morgan fingerprint": "#159A9C",
    "Docking score/pose": "#D95E55",
    "RDKit descriptor": "#7B5FB5",
    "Other": "#7A8793",
}

DEPENDENCE_FEATURES = [
    "ifp_total_donor_residue_any",
    "ifp_A428_contact_mean",
    "morgan_0411",
    "desc_logp",
]

MORGAN_EXAMPLE_BIT = 411


def load_model_importance(model):
    if model["kind"] == "shap":
        shap_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model['slug']}_{SPLIT}_shap_values.csv")
        values = np.mean(np.abs(shap_df.to_numpy()), axis=0)
        return pd.DataFrame({"feature": list(shap_df.columns), "importance": values})

    perm_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model['slug']}_{SPLIT}_permutation_importance.csv")
    return perm_df.rename(columns={"permutation_rmse_increase": "importance"})[
        ["feature", "importance"]
    ].copy()


def add_feature_metadata(df):
    out = df.copy()
    out["feature_label"] = out["feature"].map(short_feature_name)
    out["feature_group"] = out["feature"].map(feature_group)
    return out


def group_percent_table(importances_by_model):
    rows = []
    for model in MODELS:
        df = add_feature_metadata(importances_by_model[model["slug"]])
        grouped = df.groupby("feature_group", as_index=False)["importance"].sum()
        total = grouped["importance"].sum()
        grouped["percent"] = grouped["importance"] / total * 100.0
        grouped["model"] = model["label"]
        rows.append(grouped)
    combined = pd.concat(rows, ignore_index=True)
    return combined


def style_bar_axis(ax):
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")


def add_panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_feature_class_composite(group_df):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    axes = axes.ravel()

    for panel_idx, model in enumerate(MODELS):
        ax = axes[panel_idx]
        sub = group_df[group_df["model"] == model["label"]].copy()
        sub = sub.set_index("feature_group").reindex(GROUP_ORDER).reset_index()
        sub["percent"] = sub["percent"].fillna(0.0)
        sub = sub.sort_values("percent")
        colors = [GROUP_COLORS.get(group, "#7A8793") for group in sub["feature_group"]]
        ax.barh(sub["feature_group"], sub["percent"], color=colors)
        for y_pos, value in enumerate(sub["percent"]):
            ax.text(value + 1.2, y_pos, f"{value:.1f}%", va="center", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent of total feature contribution", fontweight="bold")
        ax.set_title(f"{model['label']} | Scaffold split", fontsize=20, fontweight="bold")
        style_bar_axis(ax)
        add_panel_label(ax, chr(ord("A") + panel_idx))

    ax = axes[3]
    matrix = group_df.pivot_table(
        index="feature_group", columns="model", values="percent", fill_value=0.0
    ).reindex(index=GROUP_ORDER, columns=[model["label"] for model in MODELS])
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=70, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.1f}%", ha="center", va="center", fontweight="bold")
    ax.set_title("Feature-class summary", fontsize=20, fontweight="bold")
    add_panel_label(ax, "D", x=-0.18, y=1.10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Percent of Total\nFeature Contribution", fontsize=13, fontweight="bold")

    fig.suptitle("Figure 4. Feature-class contributions across top models", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_composite(fig, OUT_DIR / "figure4_feature_class_contributions_scaffold")
    matrix.to_csv(OUT_DIR / "figure4_feature_class_contributions_scaffold_matrix.csv")


def plot_individual_feature_class_panels(group_df):
    for panel_idx, model in enumerate(MODELS):
        sub = group_df[group_df["model"] == model["label"]].copy()
        sub = sub.set_index("feature_group").reindex(GROUP_ORDER).reset_index()
        sub["percent"] = sub["percent"].fillna(0.0)
        sub = sub.sort_values("percent")

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        colors = [GROUP_COLORS.get(group, "#7A8793") for group in sub["feature_group"]]
        ax.barh(sub["feature_group"], sub["percent"], color=colors)
        for y_pos, value in enumerate(sub["percent"]):
            ax.text(value + 1.2, y_pos, f"{value:.1f}%", va="center", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent of total feature contribution", fontweight="bold")
        ax.set_title(f"{model['label']} | Scaffold split", fontsize=20, fontweight="bold")
        style_bar_axis(ax)
        fig.tight_layout()
        save_composite(
            fig,
            OUT_DIR / f"figure4_panel_{chr(ord('a') + panel_idx)}_feature_class_{model['slug']}_scaffold",
        )

    matrix = group_df.pivot_table(
        index="feature_group", columns="model", values="percent", fill_value=0.0
    ).reindex(index=GROUP_ORDER, columns=[model["label"] for model in MODELS])

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=70, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.1f}%", ha="center", va="center", fontweight="bold")
    ax.set_title("Feature-class summary", fontsize=20, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Percent of Total\nFeature Contribution", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    save_composite(fig, OUT_DIR / "figure4_panel_d_feature_class_summary_scaffold")


def plot_global_importance_composite(importances_by_model):
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 8.2))

    for panel_idx, model in enumerate(MODELS):
        ax = axes[panel_idx]
        df = add_feature_metadata(importances_by_model[model["slug"]])
        top = df.sort_values("importance", ascending=False).head(TOP_N).iloc[::-1]
        colors = [GROUP_COLORS.get(group, "#7A8793") for group in top["feature_group"]]
        ax.barh(top["feature_label"], top["importance"], color=colors)
        ax.set_xlabel(model["metric_label"], fontweight="bold")
        ax.set_title(
            f"Global feature contributions\n{model['label']} | Scaffold split",
            fontsize=18,
            fontweight="bold",
            pad=12,
        )
        style_bar_axis(ax)
        add_panel_label(ax, chr(ord("A") + panel_idx), x=-0.13, y=0.99)
        if model["kind"] == "permutation":
            x_max = max(float(top["importance"].max()) * 1.08, 0.01)
            ax.set_xlim(0, x_max)

        top.assign(model=model["label"]).to_csv(
            OUT_DIR / f"figure4_global_top{TOP_N}_{model['slug']}_scaffold.csv",
            index=False,
        )

    fig.tight_layout()
    save_composite(fig, OUT_DIR / "figure4_global_feature_importance_scaffold")


def plot_normalized_importance_heatmap(importances_by_model):
    normalized_frames = []
    raw_frames = []
    selected_features = []

    for model in MODELS:
        df = add_feature_metadata(importances_by_model[model["slug"]])
        raw = df[["feature", "feature_label", "feature_group", "importance"]].copy()
        raw["model"] = model["label"]
        raw_frames.append(raw)
        max_importance = max(float(df["importance"].max()), 1e-12)
        df["normalized_importance"] = df["importance"] / max_importance * 100.0
        df["model"] = model["label"]
        normalized_frames.append(df)
        selected_features.extend(
            df.sort_values("importance", ascending=False).head(NORMALIZED_HEATMAP_TOP_N)["feature"].tolist()
        )

    selected_features = list(dict.fromkeys(selected_features))
    combined = pd.concat(normalized_frames, ignore_index=True)
    raw_combined = pd.concat(raw_frames, ignore_index=True)
    selected = combined[combined["feature"].isin(selected_features)].copy()
    selected_raw = raw_combined[raw_combined["feature"].isin(selected_features)].copy()

    label_lookup = selected.drop_duplicates("feature").set_index("feature")["feature_label"]
    group_lookup = selected.drop_duplicates("feature").set_index("feature")["feature_group"]
    matrix = selected.pivot_table(
        index="feature", columns="model", values="normalized_importance", fill_value=0.0
    ).reindex(columns=[model["label"] for model in MODELS])
    raw_matrix = selected_raw.pivot_table(
        index="feature", columns="model", values="importance", fill_value=0.0
    ).reindex(columns=[model["label"] for model in MODELS])
    matrix["row_max"] = matrix.max(axis=1)
    matrix = matrix.sort_values("row_max", ascending=True).drop(columns="row_max")
    raw_matrix = raw_matrix.reindex(index=matrix.index)

    row_labels = [label_lookup.loc[feature] for feature in matrix.index]
    row_groups = [group_lookup.loc[feature] for feature in matrix.index]

    height = max(7.8, 0.34 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(8.6, height))
    image = ax.imshow(matrix.to_numpy(), cmap="magma_r", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="x", labelsize=14, labelrotation=0)
    ax.tick_params(axis="y", labelsize=11)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(0)
        tick_label.set_ha("center")
    for tick_label, group in zip(ax.get_yticklabels(), row_groups):
        tick_label.set_color(GROUP_COLORS.get(group, "#111827"))
        tick_label.set_fontweight("bold")
    ax.set_title(
        "Normalized cross-model feature importance\nTop feature in each model scaled to 100",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            if value == 0:
                label = "0"
            elif value < 10:
                label = f"{value:.1f}"
            else:
                label = f"{value:.0f}"
            text_color = "white" if value > 55 else "black"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Relative importance within model (%)", fontsize=13, fontweight="bold")
    colorbar.ax.tick_params(labelsize=11)

    present_groups = [group for group in GROUP_ORDER if group in set(row_groups)]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[group], label=group)
        for group in present_groups
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        fontsize=9,
    )

    fig.tight_layout(rect=(0.02, 0.12, 1, 1))
    save_composite(fig, OUT_DIR / "figure4_normalized_cross_model_feature_importance_scaffold")

    export = matrix.copy()
    export.insert(0, "feature_label", row_labels)
    export.insert(1, "feature_group", row_groups)
    export.to_csv(OUT_DIR / "figure4_normalized_cross_model_feature_importance_scaffold.csv")

    export_raw = raw_matrix.copy()
    export_raw.insert(0, "feature_label", row_labels)
    export_raw.insert(1, "feature_group", row_groups)
    export_raw.to_csv(OUT_DIR / "figure4_raw_cross_model_feature_importance_scaffold.csv")

    export_combined = pd.concat(
        {
            "normalized_percent": matrix,
            "raw_importance": raw_matrix,
        },
        axis=1,
    )
    export_combined.insert(0, ("feature", "feature_label"), row_labels)
    export_combined.insert(1, ("feature", "feature_group"), row_groups)
    export_combined.to_csv(OUT_DIR / "figure4_cross_model_feature_importance_raw_and_normalized_scaffold.csv")


def plot_catboost_shap_dependence_panels():
    model_label = "CatBoost"
    model_slug = "catboost"
    shap_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{SPLIT}_shap_values.csv")
    feature_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model_slug}_{SPLIT}_shap_feature_matrix.csv")

    available = [feature for feature in DEPENDENCE_FEATURES if feature in shap_df.columns]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.2))
    axes = axes.ravel()

    for panel_idx, feature in enumerate(available[:4]):
        ax = axes[panel_idx]
        x_values = feature_df[feature].to_numpy(dtype=float)
        shap_values = shap_df[feature].to_numpy(dtype=float)
        x_plot = x_values.copy()
        unique_values = np.unique(x_values[~np.isnan(x_values)])
        if len(unique_values) <= 4:
            rng = np.random.default_rng(42 + panel_idx)
            x_plot = x_plot + rng.normal(0, 0.025, size=len(x_plot))

        scatter = ax.scatter(
            x_plot,
            shap_values,
            c=shap_values,
            cmap="coolwarm",
            s=46,
            alpha=0.82,
            edgecolors="white",
            linewidth=0.35,
        )
        ax.axhline(0, color="#111827", linestyle="--", linewidth=1.1)
        ax.set_xlabel(short_feature_name(feature), fontweight="bold")
        ax.set_ylabel("SHAP value", fontweight="bold")
        ax.set_title(short_feature_name(feature), fontsize=16, fontweight="bold")
        style_bar_axis(ax)

        rows.append(
            {
                "feature": feature,
                "feature_label": short_feature_name(feature),
                "feature_group": feature_group(feature),
                "mean_abs_shap": float(np.mean(np.abs(shap_values))),
                "mean_shap": float(np.mean(shap_values)),
                "min_feature_value": float(np.nanmin(x_values)),
                "max_feature_value": float(np.nanmax(x_values)),
            }
        )

        single_fig, single_ax = plt.subplots(figsize=(6.2, 5.0))
        single_ax.scatter(
            x_plot,
            shap_values,
            c=shap_values,
            cmap="coolwarm",
            s=48,
            alpha=0.84,
            edgecolors="white",
            linewidth=0.35,
        )
        single_ax.axhline(0, color="#111827", linestyle="--", linewidth=1.1)
        single_ax.set_xlabel(short_feature_name(feature), fontweight="bold")
        single_ax.set_ylabel("SHAP value", fontweight="bold")
        single_ax.set_title(f"{model_label} SHAP dependence\n{short_feature_name(feature)}", fontsize=17, fontweight="bold")
        style_bar_axis(single_ax)
        single_fig.tight_layout()
        save_composite(
            single_fig,
            OUT_DIR / f"figure4_shap_dependence_{model_slug}_scaffold_{feature}",
        )

    for ax in axes[len(available[:4]):]:
        ax.set_axis_off()

    fig.suptitle("Representative CatBoost SHAP dependence plots | Scaffold split", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_composite(fig, OUT_DIR / "figure4_catboost_representative_shap_dependence_scaffold")
    pd.DataFrame(rows).to_csv(OUT_DIR / "figure4_catboost_representative_shap_dependence_features.csv", index=False)


def morgan_bit_environment(mol, bit_id=MORGAN_EXAMPLE_BIT, radius=2, n_bits=1024):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
    environments = bit_info.get(bit_id, [])
    if not environments:
        return [], [], None

    atom_id, env_radius = max(environments, key=lambda item: item[1])
    if env_radius == 0:
        return [atom_id], [], {"center_atom": atom_id, "radius": env_radius}

    bond_ids = list(Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, atom_id))
    atom_ids = {atom_id}
    for bond_id in bond_ids:
        bond = mol.GetBondWithIdx(int(bond_id))
        atom_ids.add(bond.GetBeginAtomIdx())
        atom_ids.add(bond.GetEndAtomIdx())
    return sorted(atom_ids), bond_ids, {"center_atom": atom_id, "radius": env_radius}


def draw_morgan_example(mol, atom_ids, bond_ids):
    from rdkit.Chem import Draw

    return Draw.MolToImage(
        mol,
        size=(900, 620),
        highlightAtoms=atom_ids,
        highlightBonds=bond_ids,
        highlightColor=(0.08, 0.60, 0.61),
    )


def representative_morgan_bit_row():
    from rdkit import Chem

    usecols = [
        "Name",
        "SMILES",
        "A5",
        "morgan_0411",
        "desc_logp",
        "ifp_total_donor_residue_any",
        "ifp_A428_contact_mean",
    ]
    df = pd.read_csv(FEATURE_TABLE_PATH, usecols=usecols, low_memory=False)
    df["morgan_0411"] = pd.to_numeric(df["morgan_0411"], errors="coerce").fillna(0)
    df["A5"] = pd.to_numeric(df["A5"], errors="coerce")
    candidates = df[df["morgan_0411"] > 0].sort_values("A5", ascending=False)

    for _, row in candidates.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]))
        if mol is None:
            continue
        atom_ids, bond_ids, env = morgan_bit_environment(mol)
        if env is not None:
            return row, mol, atom_ids, bond_ids, env

    raise ValueError("No valid molecule with Morgan bit 0411 was found in the feature table.")


def plot_feature_meaning_key():
    row, mol, atom_ids, bond_ids, env = representative_morgan_bit_row()
    mol_image = draw_morgan_example(mol, atom_ids, bond_ids)

    fig = plt.figure(figsize=(13.2, 6.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.08)
    mol_ax = fig.add_subplot(grid[0, 0])
    text_ax = fig.add_subplot(grid[0, 1])

    mol_ax.imshow(mol_image)
    mol_ax.set_axis_off()
    mol_ax.set_title("Morgan 0411: representative substructure", fontsize=19, fontweight="bold", pad=12)
    mol_ax.text(
        0.5,
        -0.04,
        f"Training example: {row['Name']} | radius {env['radius']} hashed Morgan bit",
        transform=mol_ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    text_ax.set_axis_off()
    text_ax.set_title("Feature meaning key", fontsize=21, fontweight="bold", loc="left", pad=12)
    entries = [
        (
            "Morgan 0411",
            "A radius-2 circular fingerprint bit. The highlighted atoms show one real local chemical environment that activates this hashed bit; the same bit can represent more than one environment.",
            GROUP_COLORS["Morgan fingerprint"],
        ),
        (
            "IFP",
            "Interaction fingerprint from docking poses. Residue-level features record whether the ligand contacts GRP78 residues and what type of contact is observed.",
            GROUP_COLORS["Active-filtered IFP"],
        ),
        (
            "ifp_total_donor_residue_any",
            "Number of GRP78 residues with at least one donor-type interaction in any pose.",
            GROUP_COLORS["Active-filtered IFP"],
        ),
        (
            "ifp_A428_contact_mean",
            "Mean contact signal with residue A428 across docking/IFP poses.",
            GROUP_COLORS["Active-filtered IFP"],
        ),
        (
            "logP",
            "RDKit Crippen MolLogP, an estimated octanol/water partition coefficient; higher values indicate greater lipophilicity.",
            GROUP_COLORS["RDKit descriptor"],
        ),
    ]

    y = 0.90
    for title, body, color in entries:
        text_ax.text(0.00, y, title, fontsize=15.5, fontweight="bold", color=color, va="top")
        text_ax.text(0.00, y - 0.045, body, fontsize=12.2, va="top", wrap=True, linespacing=1.25)
        y -= 0.175

    fig.suptitle("Interpreting key GRP78 model features", fontsize=23, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_composite(fig, OUT_DIR / "figure5_feature_meaning_key_morgan_ifp_logp")

    pd.DataFrame(
        [
            {
                "feature": "morgan_0411",
                "representative_compound": row["Name"],
                "representative_smiles": row["SMILES"],
                "A5": row["A5"],
                "morgan_radius": 2,
                "morgan_n_bits": 1024,
                "morgan_bit": MORGAN_EXAMPLE_BIT,
                "environment_center_atom": env["center_atom"],
                "environment_radius": env["radius"],
                "highlight_atoms": ";".join(map(str, atom_ids)),
                "highlight_bonds": ";".join(map(str, bond_ids)),
            }
        ]
    ).to_csv(OUT_DIR / "figure5_feature_meaning_key_morgan0411_source.csv", index=False)


def safe_feature_filename(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_").lower()


def plot_shap_dependence_on_axis(ax, model, feature, panel_label=None):
    shap_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model['slug']}_{SPLIT}_shap_values.csv")
    feature_df = pd.read_csv(SOURCE_TABLE_DIR / f"{model['slug']}_{SPLIT}_shap_feature_matrix.csv")

    x_values = feature_df[feature].to_numpy(dtype=float)
    shap_values = shap_df[feature].to_numpy(dtype=float)
    x_plot = x_values.copy()
    unique_values = np.unique(x_values[~np.isnan(x_values)])
    if len(unique_values) <= 4:
        rng = np.random.default_rng(520 + len(feature))
        x_plot = x_plot + rng.normal(0, 0.025, size=len(x_plot))

    scatter = ax.scatter(
        x_plot,
        shap_values,
        c=shap_values,
        cmap="coolwarm",
        s=48,
        alpha=0.84,
        edgecolors="white",
        linewidth=0.35,
    )
    ax.axhline(0, color="#111827", linestyle="--", linewidth=1.1)
    ax.set_xlabel(short_feature_name(feature), fontweight="bold")
    ax.set_ylabel("SHAP value", fontweight="bold")
    ax.set_title(f"{model['label']} top feature\n{short_feature_name(feature)}", fontsize=16, fontweight="bold")
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.10, y=1.08)
    style_bar_axis(ax)
    return scatter


def plot_gnn_top_feature_on_axis(ax, feature, importance, panel_label=None):
    feature_df = pd.read_csv(FEATURE_TABLE_PATH, usecols=["Name", "A5", feature], low_memory=False)
    feature_df["A5"] = pd.to_numeric(feature_df["A5"], errors="coerce")
    feature_df[feature] = pd.to_numeric(feature_df[feature], errors="coerce")
    feature_df = feature_df.dropna(subset=["A5", feature])

    x_values = feature_df[feature].to_numpy(dtype=float)
    y_values = feature_df["A5"].to_numpy(dtype=float)
    x_plot = x_values.copy()
    unique_values = np.unique(x_values[~np.isnan(x_values)])
    if len(unique_values) <= 4:
        rng = np.random.default_rng(892)
        x_plot = x_plot + rng.normal(0, 0.035, size=len(x_plot))

    scatter = ax.scatter(
        x_plot,
        y_values,
        c=y_values,
        cmap="viridis",
        s=50,
        alpha=0.82,
        edgecolors="white",
        linewidth=0.35,
    )
    ax.axhline(38, color="#C65A5A", linestyle="--", linewidth=1.2)
    ax.text(
        0.98,
        38,
        "A5 = 38",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#C65A5A",
    )
    ax.set_xlabel(short_feature_name(feature), fontweight="bold")
    ax.set_ylabel("Measured GRP78 inhibition (A5)", fontweight="bold")
    ax.set_title(f"GNN top feature\n{short_feature_name(feature)}", fontsize=16, fontweight="bold")
    ax.text(
        0.04,
        0.94,
        f"Permutation RMSE increase = {importance:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#374151",
    )
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.10, y=1.08)
    style_bar_axis(ax)
    return scatter


def plot_top1_feature_per_model(importances_by_model):
    top_rows = []
    for model in MODELS:
        df = add_feature_metadata(importances_by_model[model["slug"]])
        row = df.sort_values("importance", ascending=False).iloc[0]
        top_rows.append(
            {
                "model": model["label"],
                "model_slug": model["slug"],
                "feature": row["feature"],
                "feature_label": row["feature_label"],
                "feature_group": row["feature_group"],
                "importance": row["importance"],
                "importance_metric": model["metric_label"].replace("\n", " "),
            }
        )
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(OUT_DIR / "figure6_top1_feature_per_model_scaffold.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5))
    for panel_idx, model in enumerate(MODELS):
        top = top_df[top_df["model_slug"] == model["slug"]].iloc[0]
        if model["kind"] == "shap":
            plot_shap_dependence_on_axis(axes[panel_idx], model, top["feature"], chr(ord("A") + panel_idx))
        else:
            plot_gnn_top_feature_on_axis(
                axes[panel_idx],
                top["feature"],
                float(top["importance"]),
                chr(ord("A") + panel_idx),
            )

    fig.suptitle("Figure 6. Top model-specific feature interpretation", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_composite(fig, OUT_DIR / "figure6_top1_model_feature_interpretation_scaffold")

    for model in MODELS:
        top = top_df[top_df["model_slug"] == model["slug"]].iloc[0]
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        if model["kind"] == "shap":
            plot_shap_dependence_on_axis(ax, model, top["feature"])
        else:
            plot_gnn_top_feature_on_axis(ax, top["feature"], float(top["importance"]))
        fig.tight_layout()
        save_composite(
            fig,
            OUT_DIR
            / f"figure6_top1_{model['slug']}_scaffold_{safe_feature_filename(top['feature'])}",
        )


def save_composite(fig, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    set_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    importances_by_model = {model["slug"]: load_model_importance(model) for model in MODELS}
    group_df = group_percent_table(importances_by_model)
    group_df.to_csv(OUT_DIR / "figure4_feature_class_contributions_scaffold_long.csv", index=False)
    plot_feature_class_composite(group_df)
    plot_individual_feature_class_panels(group_df)
    plot_global_importance_composite(importances_by_model)
    plot_normalized_importance_heatmap(importances_by_model)
    plot_catboost_shap_dependence_panels()
    plot_feature_meaning_key()
    plot_top1_feature_per_model(importances_by_model)


if __name__ == "__main__":
    main()
