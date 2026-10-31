"""
Merge tree-model and GNN nested-CV outputs (same seeds => identical outer folds),
add consensus rows over a FIXED model set, and write the summary tables.

Consensus definitions (both reported, on the same outer folds):
  Consensus(ET+Cat)      mean of ExtraTrees and CatBoost, every feature set
  Consensus(ET+Cat+GNN)  mean of all three, only feature sets the GNN was run on
Also writes paired per-repeat differences vs. the ligand-only baseline and vs. each
constituent, so claims about added value rest on identical folds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

HIT = 38.0
KEY = ["repeat", "fold", "Name", "y"]


def metrics(g):
    y, p = g.y.values, g.pred.values
    hit = (y >= HIT).astype(int)
    r = {"auroc": roc_auc_score(hit, p) if 0 < hit.sum() < len(hit) else np.nan}
    order = np.argsort(-p)
    for k in (25, 50, 100):
        r[f"precision@{k}"] = hit[order[:k]].mean()
        r[f"enrichment@{k}"] = hit[order[:k]].mean() / hit.mean()
    if not g.is_score_only.iloc[0]:
        r.update(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                 RMSE=float(np.sqrt(mean_squared_error(y, p))), Spearman=spearmanr(y, p)[0])
    return pd.Series(r)


def main(out, gnn_dirs, tree_dirs):
    """Merge nested-CV runs of identical outer folds. gnn_dirs and tree_dirs are comma-separated run directories under Output/
    (several GNN runs may be given, for example the main run and the run on the remaining feature sets)."""
    base = cfg.OUTPUT
    gnn = [d for d in gnn_dirs.split(",") if d]
    parts = [pd.read_csv(base / d / "outer_predictions.csv") for d in tree_dirs.split(",") + gnn]
    P = pd.concat(parts, ignore_index=True)
    P = P[~P.model.isin(["Consensus(mean)"])]                       # drop per-run consensus
    P = P.drop_duplicates(["repeat", "fold", "Name", "model", "feature_set"])   # NN baseline appears in both runs
    # sanity: identical outer folds in both runs
    f = P.groupby(["repeat", "Name"]).fold.nunique()
    assert (f == 1).all(), "a compound sits in different folds across models"

    gm = ["ExtraTrees", "CatBoost"] + (["GNN"] if gnn else [])
    wide = P[P.model.isin(gm)].pivot_table(
        index=KEY + ["feature_set"], columns="model", values="pred").reset_index()
    c2 = wide.dropna(subset=["ExtraTrees", "CatBoost"]).assign(
        pred=lambda d: d[["ExtraTrees", "CatBoost"]].mean(axis=1), model="Consensus(ET+Cat)")
    cons_parts = [c2]
    if gnn:
        cons_parts.append(wide.dropna(subset=["ExtraTrees", "CatBoost", "GNN"]).assign(
            pred=lambda d: d[["ExtraTrees", "CatBoost", "GNN"]].mean(axis=1), model="Consensus(ET+Cat+GNN)"))
    cons = pd.concat(cons_parts)[KEY + ["feature_set", "model", "pred"]].assign(is_score_only=False)
    P = pd.concat([P, cons], ignore_index=True)

    out = cfg.OUTPUT / out; out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame([{"repeat": k[0], "model": k[1], "feature_set": k[2], **metrics(g).to_dict()}
                        for k, g in P.groupby(["repeat", "model", "feature_set"])])
    per.to_csv(out / "metrics_per_repeat.csv", index=False)
    cols = [c for c in per.select_dtypes("number").columns if c != "repeat"]
    s = per.groupby(["model", "feature_set"])[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    s.to_csv(out / "metrics_summary_mean_sd.csv", index=False)

    # paired differences per repeat (same folds): each config minus ligand-only of the same model
    d = per.pivot_table(index=["repeat", "model"], columns="feature_set", values=["R2", "auroc", "precision@50"])
    rows = []
    for m in ["ExtraTrees", "CatBoost", "GNN", "Consensus(ET+Cat)", "Consensus(ET+Cat+GNN)"]:
        for fs in ["ligand+dock", "ligand+dock+ifp", "ifp_only", "docking_only"]:
            for met in ["R2", "auroc", "precision@50"]:
                try:
                    x = (d[(met, fs)] - d[(met, "ligand_only")]).xs(m, level="model").dropna()
                except KeyError:
                    continue
                if len(x):
                    rows.append(dict(model=m, feature_set=fs, metric=met, vs="ligand_only",
                                     mean_diff=x.mean(), sd=x.std(), n_repeats=len(x),
                                     repeats_positive=int((x > 0).sum())))
    pd.DataFrame(rows).to_csv(out / "paired_diff_vs_ligand_only.csv", index=False)
    P.to_csv(out / "outer_predictions_all.csv", index=False)

    # per-outer-fold R2 for the model/feature combos (shows scaffold-fold instability)
    fold = pd.DataFrame([
        {"repeat": k[0], "fold": k[1], "model": k[2], "feature_set": k[3], "n": len(g),
         "R2": r2_score(g.y, g.pred),
         "auroc": roc_auc_score(g.y >= HIT, g.pred) if 0 < (g.y >= HIT).sum() < len(g) else np.nan}
        for k, g in P[~P.is_score_only].groupby(["repeat", "fold", "model", "feature_set"])])
    fold.to_csv(out / "metrics_per_outer_fold.csv", index=False)
    pd.set_option("display.width", 220)
    show = s[["model", "feature_set"] + [c for c in s.columns if c.split("_")[0] in ("R2", "MAE", "auroc", "precision@50", "enrichment@50") and c.endswith(("mean", "std"))]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])   # output folder, GNN runs, tree runs
