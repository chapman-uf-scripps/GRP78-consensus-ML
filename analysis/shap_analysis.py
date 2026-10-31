"""
SHAP analyses of the unmasked models (held-out outer-fold compounds, repeat seed 0)

Stages: tree_shap_values, feature_importance, class_shapley, gnn_expected_gradients, fold_stability.

Run from the repository root:  python analysis/shap_analysis.py [stage ...]   (no stage = all stages in order)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402
from main_code.common import load_features, load_training                    # noqa: E402

import numpy as np
import pandas as pd
import shap
from main_code import nested_cv as n                                          # noqa: E402
from analysis.retrospective_screening import robust_params            # noqa: E402
from rdkit import Chem
from rdkit.Chem import AllChem
import itertools
import math
from main_code import gnn_model as nested_cv_gnn                                           # noqa: E402
import torch
from torch_geometric.loader import DataLoader
import itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from main_code import nested_cv as n


def run_tree_shap_values():
    """Per-compound TreeSHAP values and feature values on HELD-OUT outer-fold compounds (repeat seed 0, five scaffold-grouped folds)"""
    OUT = cfg.OUTPUT / "shap_unmasked"
    OUT.mkdir(parents=True, exist_ok=True)
    act = load_training()
    y = np.clip(act.A5.values.astype(float), 0, 100)
    mg, ds, mols, ifp, dock = load_features(act)
    X = pd.concat([mg, ds, ifp, dock], axis=1)
    scaf = np.array([n.scaffold(m) for m in mols])
    outer = n.group_folds(scaf, 5, 0)
    trees_hp = sorted(cfg.OUTPUT.glob("nested_cv_bo_trees_*/selected_hyperparameters.csv"))
    hp = {"catboost_full": ("CatBoost", ["morgan", "desc", "dock", "ifp"], robust_params("CatBoost", trees_hp, "ligand+dock+ifp")[0], None),
          "extratrees_full": ("ExtraTrees", ["morgan", "desc", "dock", "ifp"], robust_params("ExtraTrees", trees_hp, "ligand+dock+ifp")[0], 120),
          "catboost_ligand": ("CatBoost", ["morgan", "desc"], robust_params("CatBoost", trees_hp, "ligand_only")[0], None)}
    rng = np.random.RandomState(0)
    for name, (model, blocks, params, cap) in hp.items():
        S, V = [], []
        for f in range(5):
            te, tr = np.where(outer == f)[0], np.where(outer != f)[0]
            pre = n.FoldPreprocessor(blocks).fit(X.iloc[tr])
            Xtr, Xte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
            m = n.make_model(model, params, 0).fit(Xtr, y[tr])
            if cap and len(Xte) > cap:
                Xte = Xte.iloc[rng.choice(len(Xte), cap, replace=False)]
            sv = shap.TreeExplainer(m).shap_values(Xte, check_additivity=False)
            S.append(pd.DataFrame(sv, index=Xte.index, columns=Xte.columns))
            V.append(Xte)
            print(name, "fold", f, "done", flush=True)
        pd.concat(S).fillna(0.0).to_csv(OUT / f"{name}_shap.csv")
        pd.concat(V).to_csv(OUT / f"{name}_values.csv")
    print("done", flush=True)


def run_feature_importance():
    """TreeSHAP of the corrected (unmasked) models for the new Figure 3, computed on HELD-OUT outer-fold compounds"""
    OUT = cfg.OUTPUT / "shap_unmasked"
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ET_MAX_EXPLAIN = 120          # held-out compounds explained per fold for Extra Trees (exact TreeSHAP is slow on deep forests)

    act = load_training()
    y = np.clip(act.A5.values.astype(float), 0, 100)
    mg, ds, mols, ifp, dock = load_features(act)
    X = pd.concat([mg, ds, ifp, dock], axis=1)
    Xm = X.copy()
    ic = [c for c in X.columns if c.startswith("ifp_") and c != "ifp_has_pose"]
    Xm.loc[y < n.HIT_CUTOFF, ic] = 0.0                              # historical masking (diagnostic)
    scaf = np.array([n.scaffold(m) for m in mols])
    outer = n.group_folds(scaf, 5, 0)

    trees_hp = sorted(cfg.OUTPUT.glob("nested_cv_bo_trees_*/selected_hyperparameters.csv"))
    hp = {("ExtraTrees", "full"): robust_params("ExtraTrees", trees_hp, "ligand+dock+ifp")[0],
          ("CatBoost", "full"): robust_params("CatBoost", trees_hp, "ligand+dock+ifp")[0],
          ("CatBoost", "ligand"): robust_params("CatBoost", trees_hp, "ligand_only")[0]}
    for k, v in hp.items():
        print(k, v, flush=True)


    def explain(model_name, params, blocks, Xsrc, tr, te, rng):
        pre = n.FoldPreprocessor(blocks).fit(Xsrc.iloc[tr])
        Xtr, Xte = pre.transform(Xsrc.iloc[tr]), pre.transform(Xsrc.iloc[te])
        m = n.make_model(model_name, params, 0).fit(Xtr, y[tr])
        if model_name == "ExtraTrees" and len(Xte) > ET_MAX_EXPLAIN:
            Xte = Xte.iloc[rng.choice(len(Xte), ET_MAX_EXPLAIN, replace=False)]
        sv = shap.TreeExplainer(m).shap_values(Xte, check_additivity=False)
        return pd.Series(np.abs(sv).mean(axis=0), index=Xte.columns)


    rows = []
    rng = np.random.RandomState(0)
    for f in range(5):
        te, tr = np.where(outer == f)[0], np.where(outer != f)[0]
        jobs = [("CatBoost", "ligand-only", "unmasked", ["morgan", "desc"], X, hp[("CatBoost", "ligand")]),
                ("CatBoost", "full", "unmasked", ["morgan", "desc", "dock", "ifp"], X, hp[("CatBoost", "full")]),
                ("CatBoost", "full", "active_filtered", ["morgan", "desc", "dock", "ifp"], Xm, hp[("CatBoost", "full")]),
                ("ExtraTrees", "full", "unmasked", ["morgan", "desc", "dock", "ifp"], X, hp[("ExtraTrees", "full")]),
                ("ExtraTrees", "full", "active_filtered", ["morgan", "desc", "dock", "ifp"], Xm, hp[("ExtraTrees", "full")])]
        for model, fs, setting, blocks, Xsrc, params in jobs:
            s = explain(model, params, blocks, Xsrc, tr, te, rng)
            rows.append(pd.DataFrame({"model": model, "features": fs, "setting": setting, "fold": f,
                                      "feature": s.index, "mean_abs_shap": s.values}))
            print(f"fold {f} | {model} | {fs} | {setting} done", flush=True)
        pd.concat(rows).to_csv(OUT / "importance_long.csv", index=False)

    # example substructure for each Morgan bit (first occurrence in the training set)
    ex = {}
    for mol in mols:
        bi = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bi)
        for bit, envs in bi.items():
            name = f"morgan_{bit:04d}"
            if name in ex:
                continue
            a, r = envs[0]
            if r == 0:
                ex[name] = mol.GetAtomWithIdx(a).GetSymbol()
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, a)
                sub = Chem.PathToSubmol(mol, env) if env else None
                ex[name] = Chem.MolToSmiles(sub) if sub is not None else "?"
    pd.Series(ex, name="example_substructure").rename_axis("feature").reset_index().to_csv(OUT / "bit_examples.csv", index=False)
    print("done", flush=True)


def run_class_shapley():
    """Feature-class Shapley values for the three models on HELD-OUT outer-fold compounds (repeat seed 0)."""
    OUT = cfg.OUTPUT / "shap_unmasked"
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    N_TE, N_BG = 100, 25
    CLASSES = ["morgan", "desc", "dock", "ifp"]

    act = load_training()
    y = np.clip(act.A5.values.astype(float), 0, 100)
    mg, ds, mols, ifp, dock = load_features(act)
    X = pd.concat([mg, ds, ifp, dock], axis=1)
    Xm = X.copy()
    ic = [c for c in X.columns if c.startswith("ifp_") and c != "ifp_has_pose"]
    Xm.loc[y < n.HIT_CUTOFF, ic] = 0.0
    scaf = np.array([n.scaffold(m) for m in mols])
    outer = n.group_folds(scaf, 5, 0)
    nested_cv_gnn.set_context(act.SMILES, scaf)

    trees_hp = sorted(cfg.OUTPUT.glob("nested_cv_bo_trees_*/selected_hyperparameters.csv"))
    gnn_hp = [cfg.OUTPUT / "nested_cv_bo_gnn/selected_hyperparameters.csv"]
    HP = {"ExtraTrees": robust_params("ExtraTrees", trees_hp, "ligand+dock+ifp")[0],
          "CatBoost": robust_params("CatBoost", trees_hp, "ligand+dock+ifp")[0],
          "GNN": robust_params("GNN", gnn_hp, "ligand+dock+ifp")[0]}
    print(HP, flush=True)


    def class_of(col):
        return "morgan" if col.startswith("morgan_") else "desc" if col.startswith("desc_") else "dock" if col.startswith("dock_") else "ifp"


    def shapley(predict, Xte, Xbg, cls_cols, graph_idx_te=None, graph_idx_bg=None):
        """Exact class-level Shapley values. predict(DataFrame) -> array. Returns phi (n_te x n_players) and player names."""
        players = list(CLASSES) + (["graph"] if graph_idx_te is not None else [])
        p = len(players)
        nte, nbg, d = len(Xte), len(Xbg), Xte.shape[1]
        colmask = {c: np.array([class_of(col) == c for col in cls_cols]) for c in CLASSES}
        v = {}
        for r in range(p + 1):
            for S in itertools.combinations(range(p), r):
                cm = np.zeros(d, bool)
                for i in S:
                    if players[i] in colmask:
                        cm |= colmask[players[i]]
                rows = np.where(cm[None, None, :], Xte.values[:, None, :], Xbg.values[None, :, :]).reshape(nte * nbg, d)
                if graph_idx_te is not None:
                    g_te = np.repeat(graph_idx_te[:, None], nbg, axis=1)
                    g_bg = np.repeat(graph_idx_bg[None, :], nte, axis=0)
                    gidx = (g_te if players.index("graph") in S else g_bg).reshape(-1)
                else:
                    gidx = np.tile(np.arange(nte * nbg), 1)
                frame = pd.DataFrame(rows, columns=cls_cols, index=gidx)
                v[S] = predict(frame).reshape(nte, nbg).mean(axis=1)
        phi = np.zeros((nte, p))
        for i in range(p):
            others = [j for j in range(p) if j != i]
            for r in range(p):
                w = math.factorial(r) * math.factorial(p - r - 1) / math.factorial(p)
                for S in itertools.combinations(others, r):
                    phi[:, i] += w * (v[tuple(sorted(S + (i,)))] - v[tuple(sorted(S))])
        return phi, players


    rng = np.random.RandomState(0)
    rows = []
    for f in range(5):
        te_all, tr = np.where(outer == f)[0], np.where(outer != f)[0]
        te = rng.choice(te_all, min(N_TE, len(te_all)), replace=False)
        bg = rng.choice(tr, N_BG, replace=False)
        for setting, Xsrc in [("unmasked", X), ("active_filtered", Xm)]:
            pre = n.FoldPreprocessor(CLASSES).fit(Xsrc.iloc[tr])
            Xtr = pre.transform(Xsrc.iloc[tr])
            Xte, Xbg = pre.transform(Xsrc.iloc[te]), pre.transform(Xsrc.iloc[bg])
            for model in ["ExtraTrees", "CatBoost", "GNN"]:
                m = n.make_model(model, HP[model], 0).fit(Xtr, y[tr])
                if model == "GNN":
                    phi, players = shapley(lambda fr: m.predict(fr), Xte, Xbg, list(Xte.columns), te, bg)
                else:
                    phi, players = shapley(lambda fr: m.predict(fr.reset_index(drop=True)), Xte, Xbg, list(Xte.columns))
                imp = np.abs(phi).mean(axis=0)
                share = 100 * imp / imp.sum()
                for pl, a, s in zip(players, imp, share):
                    rows.append({"model": model, "setting": setting, "fold": f, "player": pl, "mean_abs_phi": a, "share": s})
                print(f"fold {f} | {setting} | {model}: " + ", ".join(f"{pl} {s:.1f}%" for pl, s in zip(players, share)), flush=True)
        pd.DataFrame(rows).to_csv(OUT / "class_shapley_long.csv", index=False)
    print("done", flush=True)


def run_gnn_expected_gradients():
    """Feature-level SHAP-type attributions for the GNN (tabular branch) by Expected Gradients, on HELD-OUT outer-fold compounds"""
    OUT = cfg.OUTPUT / "shap_unmasked"
    OUT.mkdir(parents=True, exist_ok=True)
    N_TE, N_BG, K = 100, 50, 100
    act = load_training()
    y = np.clip(act.A5.values.astype(float), 0, 100)
    mg, ds, mols, ifp, dock = load_features(act)
    X = pd.concat([mg, ds, ifp, dock], axis=1)
    scaf = np.array([n.scaffold(m) for m in mols])
    outer = n.group_folds(scaf, 5, 0)
    nested_cv_gnn.set_context(act.SMILES, scaf)
    hp = robust_params("GNN", [cfg.OUTPUT / "nested_cv_bo_gnn/selected_hyperparameters.csv"], "ligand+dock+ifp")[0]
    print(hp, flush=True)
    blocks = ["morgan", "desc", "dock", "ifp"]
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    S, V = [], []
    for f in range(5):
        te_all, tr = np.where(outer == f)[0], np.where(outer != f)[0]
        te = rng.choice(te_all, min(N_TE, len(te_all)), replace=False)
        bg = rng.choice(tr, N_BG, replace=False)
        pre = n.FoldPreprocessor(blocks).fit(X.iloc[tr])
        Xtr, Xte, Xbg = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te]), pre.transform(X.iloc[bg])
        m = n.make_model("GNN", hp, 0).fit(Xtr, y[tr])
        m.net.eval()
        sc = lambda fr: np.clip(m.scaler.transform(fr.values), -10, 10).astype(np.float32)
        xs, bs = sc(Xte), sc(Xbg)
        phi = np.zeros_like(xs)
        for i, idx in enumerate(Xte.index):
            b = bs[rng.randint(0, len(bs), K)]
            a = rng.uniform(0, 1, (K, 1)).astype(np.float32)
            pts = b + a * (xs[i][None, :] - b)
            data = m._data(pd.DataFrame(index=[idx] * K, columns=Xte.columns), pts)
            batch = next(iter(DataLoader(data, batch_size=K)))
            tab = batch.tabular.detach().clone().requires_grad_(True)
            z = m.net(batch.x, batch.edge_index, batch.edge_attr, batch.batch, tab).view(-1)
            pred = torch.sign(z) * torch.expm1(torch.abs(z))
            pred.sum().backward()
            phi[i] = ((torch.from_numpy(xs[i][None, :] - b) * tab.grad).mean(dim=0)).numpy()
        S.append(pd.DataFrame(phi, index=Xte.index, columns=Xte.columns))
        V.append(Xte)
        print("fold", f, "done | mean |phi| sum per compound:", float(np.abs(phi).sum(axis=1).mean()), flush=True)
    pd.concat(S).fillna(0.0).to_csv(OUT / "gnn_full_shap.csv")
    pd.concat(V).to_csv(OUT / "gnn_full_values.csv")
    print("done", flush=True)


def run_fold_stability():
    """Stability of SHAP feature rankings across the five outer folds (repeat seed 0) for the unmasked ligand+docking+IFP models."""
    SH = cfg.OUTPUT / "shap_unmasked"
    act = load_training()
    mg, ds, mols = n.morgan_desc(act.SMILES)
    scaf = np.array([n.scaffold(m) for m in mols])
    fold = pd.Series(n.group_folds(scaf, 5, 0), index=range(len(act)))
    rows = []
    for name in ["catboost_full", "extratrees_full", "gnn_full"]:
        sv = pd.read_csv(SH / f"{name}_shap.csv", index_col=0)
        f = fold.reindex(sv.index)
        imp = pd.DataFrame({k: sv[f == k].abs().mean() for k in range(5)})
        top = {k: set(imp[k].nlargest(15).index) for k in range(5)}
        overall = set(sv.abs().mean().nlargest(15).index)
        jac = [len(top[a] & top[b]) / len(top[a] | top[b]) for a, b in itertools.combinations(range(5), 2)]
        rho = [spearmanr(imp[a], imp[b])[0] for a, b in itertools.combinations(range(5), 2)]
        inn = [len(top[k] & overall) for k in range(5)]
        cnt = pd.Series([c for k in range(5) for c in top[k]]).value_counts()
        rows.append(dict(model=name, jaccard_top15_mean=np.mean(jac), jaccard_min=min(jac), spearman_all_mean=np.mean(rho),
                         overall_top15_in_fold_top15_mean=np.mean(inn), overall_top15_in_fold_top15_min=min(inn),
                         features_in_top15_of_all5=int((cnt == 5).sum()), n_ifp_top15_any_fold=int(sum(c.startswith("ifp_") for c in cnt.index)),
                         n_per_fold=f.value_counts().sort_index().tolist()))
    r = pd.DataFrame(rows); print(r.T.to_string())
    r.to_csv(cfg.TABLES / "Supplementary_Table_shap_fold_stability.csv", index=False)


STAGES = {
    "tree_shap_values": run_tree_shap_values,
    "feature_importance": run_feature_importance,
    "class_shapley": run_class_shapley,
    "gnn_expected_gradients": run_gnn_expected_gradients,
    "fold_stability": run_fold_stability,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    for w in wanted:
        print("=== stage", w, flush=True)
        STAGES[w]()
