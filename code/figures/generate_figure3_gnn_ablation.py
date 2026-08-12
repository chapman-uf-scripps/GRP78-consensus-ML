"""
Add GNN bars to Figure 3 by re-training the IFP-augmented GNN at L1/L2/L3.

Uses utils_models from pipeline_core to reproduce the exact graph featurization,
GNN architecture, and training loop, swapping out the tabular feature schema
per level. Same scaffold split and BO hyperparameters as the tree models.

Outputs:
  - Extends output/source_tables/figure3_feature_ablation_metrics.csv
  - Re-renders output/main_figures/Figure_3_feature_ablation.{png,pdf}
"""

import warnings, sys
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add pipeline_core to path so we can import its exact GNN definition
PIPELINE_CORE = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package\code\pipeline_core")
sys.path.insert(0, str(PIPELINE_CORE))

import utils_models as um   # noqa: E402

ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
FEATURE_CSV = ROOT / "output" / "full_pipeline_output" / "01_features" / "training_features_docking_ifp_threshold38.csv"
SRC_DIR  = ROOT / "output" / "source_tables"
OUT_DIR  = ROOT / "output" / "main_figures"
SEED = 42
TEST_SIZE = 0.20  # exact pipeline setting (step04a-c); yields N_train=1361, N_test=127

# BO-optimized GNN hyperparameters from source_all_model_comparison_metrics.csv
# {"batch_size": 16, "dropout": 0.341, "hidden_dim": 64, "lr": 0.001208, "weight_decay": 7.07e-06}
GNN_BO = dict(batch_size=16, dropout=0.34097295564859825, hidden_dim=64,
              lr=0.001208301829462281, weight_decay=7.068974950624601e-06)


def feature_cols_for_level(df, level):
    morgan = [c for c in df.columns if c.startswith("morgan_")]
    ifp    = [c for c in df.columns if c.startswith("ifp_") and c != "ifp_allowed_by_A5_threshold"]
    dock   = [c for c in df.columns if c.startswith("dock_")]
    blocked = {"Name", "SMILES", "A5", "ifp_allowed_by_A5_threshold"}
    rdkit_descs = [c for c in df.columns
                   if c not in blocked and not c.startswith(("morgan_","ifp_","dock_"))]
    if level == "L1": return morgan
    if level == "L2": return morgan + rdkit_descs
    if level == "L3": return morgan + rdkit_descs + ifp
    if level == "L4": return morgan + rdkit_descs + ifp + dock


def train_gnn_at_level(meta_train, meta_test, x_train_arr, x_test_arr, y_train, y_test):
    """Reproduce utils_models.train_gnn but with BO hyperparameters from config."""
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    um.set_seed(SEED)
    device = torch.device("cpu")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_arr).astype(np.float32)
    x_test_scaled  = scaler.transform(x_test_arr).astype(np.float32)

    fit_idx, val_idx = train_test_split(
        np.arange(len(meta_train)), test_size=0.15, random_state=SEED, shuffle=True
    )
    train_ds = um.GraphTabularDataset(meta_train.iloc[fit_idx].reset_index(drop=True),
                                      x_train_scaled[fit_idx], y_train[fit_idx])
    val_ds   = um.GraphTabularDataset(meta_train.iloc[val_idx].reset_index(drop=True),
                                      x_train_scaled[val_idx], y_train[val_idx])
    full_tr_ds = um.GraphTabularDataset(meta_train.reset_index(drop=True),
                                        x_train_scaled, y_train)
    test_ds  = um.GraphTabularDataset(meta_test.reset_index(drop=True),
                                      x_test_scaled, y_test)

    from torch_geometric.loader import DataLoader as GraphDataLoader
    train_loader = GraphDataLoader(train_ds, batch_size=GNN_BO["batch_size"], shuffle=True)
    val_loader   = GraphDataLoader(val_ds,   batch_size=GNN_BO["batch_size"], shuffle=False)
    full_loader  = GraphDataLoader(full_tr_ds, batch_size=GNN_BO["batch_size"], shuffle=False)
    test_loader  = GraphDataLoader(test_ds,  batch_size=GNN_BO["batch_size"], shuffle=False)

    sample = train_ds[0]
    model = um.GNNTabularRegressor(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        tabular_dim=sample.tabular.shape[-1],
        hidden_dim=GNN_BO["hidden_dim"],
        dropout=GNN_BO["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=GNN_BO["lr"], weight_decay=GNN_BO["weight_decay"])

    best_state, best_val_rmse, stale = None, float("inf"), 0
    import torch.nn.functional as F
    for epoch in range(um.GNN_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.tabular)
            loss = F.mse_loss(pred.view(-1), batch.y.view(-1))
            loss.backward()
            optimizer.step()
        val_true, val_pred = um.predict_gnn(model, val_loader, device)
        v_rmse = um.rmse(val_true, val_pred)
        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 15:
            break

    model.load_state_dict(best_state)
    train_true, train_pred = um.predict_gnn(model, full_loader, device)
    test_true,  test_pred  = um.predict_gnn(model, test_loader,  device)
    from sklearn.metrics import r2_score, mean_absolute_error
    return dict(
        Train_R2 = r2_score(train_true, train_pred),
        Test_R2  = r2_score(test_true,  test_pred),
        Train_MAE= mean_absolute_error(train_true, train_pred),
        Test_MAE = mean_absolute_error(test_true,  test_pred),
    )


def main():
    print("Loading", FEATURE_CSV.name)
    df = pd.read_csv(FEATURE_CSV)
    y = df["A5"].values.astype(float)
    meta = df[["Name","SMILES","A5"]].copy()
    meta["A5"] = y

    # Same scaffold split as Figure 3 ablation
    smiles = df["SMILES"].values
    print("Computing scaffold split...")
    from sklearn.model_selection import GroupShuffleSplit
    scaffolds = [um.scaffold_from_smiles(s) for s in smiles]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(np.arange(len(smiles)), groups=scaffolds))
    print(f"  N_train = {len(tr_idx)}, N_test = {len(te_idx)}")

    metrics_csv = SRC_DIR / "figure3_feature_ablation_metrics.csv"
    out = pd.read_csv(metrics_csv)
    # Drop any prior GNN rows (the L4 placeholder included earlier)
    out = out[out["Model"] != "GNN"].copy()

    for level in ["L1","L2","L3","L4"]:
        cols = feature_cols_for_level(df, level)
        X = df[cols].values.astype(float)
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        print(f"\n=== GNN {level} ({len(cols)} features) ===")
        m = train_gnn_at_level(meta.iloc[tr_idx], meta.iloc[te_idx], Xtr, Xte, ytr, yte)
        print(f"  Train R2 = {m['Train_R2']:.4f}  Test R2 = {m['Test_R2']:.4f}  Test MAE = {m['Test_MAE']:.4f}")
        out = pd.concat([out, pd.DataFrame([dict(Level=level, NumFeatures=len(cols), Model="GNN", **m)])],
                        ignore_index=True)

    out.to_csv(metrics_csv, index=False)
    print("\nUpdated", metrics_csv)

    # ---- re-render Figure 3 with full bars ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False)
    levels = ["L1","L2","L3","L4"]
    level_labels = ["L1\nMorgan", "L2\n+ RDKit", "L3\n+ IFP", "L4\n+ docking\n(full)"]
    models = ["ExtraTrees","CatBoost","GNN"]
    colors = {"ExtraTrees":"#4c72b0","CatBoost":"#dd8452","GNN":"#55a868"}

    def get_val(lv, m, col):
        sub = out[(out["Level"]==lv)&(out["Model"]==m)]
        return float(sub[col].iloc[0]) if not sub.empty else np.nan

    width = 0.27
    x = np.arange(len(levels))

    ax = axes[0]
    for i, m in enumerate(models):
        vals = [get_val(lv, m, "Test_R2") for lv in levels]
        bars = ax.bar(x + (i-1)*width, vals, width, label=m, color=colors[m],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                yt = v + 0.005 if v >= 0 else v - 0.05
                va = "bottom" if v >= 0 else "top"
                ax.text(b.get_x() + b.get_width()/2, yt, f"{v:.3f}",
                        ha="center", va=va, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(level_labels)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Scaffold-split Test R²"); ax.set_title("Test R²")
    ax.set_ylim(-0.25, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)

    ax = axes[1]
    for i, m in enumerate(models):
        vals = [get_val(lv, m, "Test_MAE") for lv in levels]
        bars = ax.bar(x + (i-1)*width, vals, width, label=m, color=colors[m],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(level_labels)
    ax.set_ylabel("Scaffold-split Test MAE (percent inhibition)")
    ax.set_title("Test MAE")
    ax.grid(axis="y", alpha=0.3); ax.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("Figure 3. Feature-class ablation ladder (scaffold split)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Figure_3_feature_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_3_feature_ablation.pdf", bbox_inches="tight")
    print("Re-rendered Figure 3 with all three models at all four levels.")


if __name__ == "__main__":
    main()
