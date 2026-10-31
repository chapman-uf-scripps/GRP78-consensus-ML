"""
Nested, scaffold-grouped, repeated cross-validation for the GRP78 models.

Replaces the single-split / test-selected evaluation (step02, step04a-c, step05).

Design
------
* Outer loop : repeated scaffold-GROUP K-fold (prespecified seeds). Every
  compound is predicted exactly once per repeat by a model that never saw its
  scaffold. Outer folds are used ONLY for evaluation.
* Inner loop : scaffold-group K-fold inside each outer-training set. All
  hyper-parameter selection happens here.
* Preprocessing: every learned step (Morgan/IFP frequency filter, missing-value
  filter, constant-column removal, median imputation) is fit on the training
  portion of the current (inner or outer) fold and then frozen.
* IFP block: computed identically for every compound from the pose only. NO
  activity-dependent zeroing. Missing poses stay NaN (+ a has_pose indicator),
  they are not zero-filled.
* The 38 % cutoff is NOT a feature-building parameter. It is only used to define
  "hit" for screening metrics (precision@k, enrichment, AUROC).
* Baselines at the same budget: nearest-active-neighbour (Tanimoto), ligand-only
  models, docking-only, IFP-only, plus the consensus (mean) of tree models.

Inputs  (CSV)
-------------
--activity  : columns Name, SMILES, A5  (A5 = % inhibition)
--ifp       : Maestro IFP table with Title + A<res>_<type> columns
--docking   : (optional) docking table with Title (or Compound_ID) + numeric cols

Usage
-----
python nested_cv.py --activity act.csv --ifp ifp.csv [--docking dock.csv] \
       --out Output/nested_cv --seeds 0 1 2 3 4 --outer 5 --inner 3
python nested_cv.py --smoke      # synthetic-data self test
"""
import argparse
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

HIT_CUTOFF = 38.0          # defines "hit" for screening metrics only
ID_LIKE = re.compile(          # record identifiers / pose indices / atom serials: never features
    r"(^|[\s_])(title|name|id|index|idx|molecule|entry|pose|posenum|confnum|lignum|rank|rack|"
    r"plate|well|file|variant|smiles|source|flags|version|job|atom)([\s_\d]|$)", re.I)
IFP_RE = re.compile(r"^A\d+_(contact|backbone|sidechain|polar|hydrophobic|acceptor|donor|aromatic|charged)$")

FEATURE_SETS = {                      # name -> feature blocks
    "morgan_only": ["morgan"],
    "ligand_only": ["morgan", "desc"],
    "docking_only": ["dock"],
    "ifp_only": ["ifp"],
    "ligand+dock": ["morgan", "desc", "dock"],
    "ligand+dock+ifp": ["morgan", "desc", "dock", "ifp"],
}


# --------------------------------------------------------------------------
# Feature construction (outcome-independent; no labels touched here)
# --------------------------------------------------------------------------
def count_cell(v):
    if pd.isna(v):
        return 0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(np.isfinite(v) and v != 0)
    t = str(v).strip()
    if t.startswith("{") and t.endswith("}"):
        inner = t[1:-1].strip()
        return len([p for p in inner.split(",") if p.strip()]) if inner else 0
    try:
        return int(float(t) != 0)
    except ValueError:
        return int(bool(t) and t.lower() not in {"nan", "none"})


def morgan_desc(smiles):
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    fps = np.zeros((len(mols), 1024), dtype=np.int8)
    rd = np.full((len(mols), 10), np.nan)
    for i, m in enumerate(mols):
        if m is None:
            continue
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), fps[i])
        rd[i] = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                 Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                 Descriptors.NumRotatableBonds(m), Descriptors.HeavyAtomCount(m),
                 Descriptors.RingCount(m), Descriptors.NumAromaticRings(m),
                 Descriptors.FractionCSP3(m)]
    morgan = pd.DataFrame(fps, columns=[f"morgan_{i:04d}" for i in range(1024)])
    desc = pd.DataFrame(rd, columns=["desc_" + c for c in
                        ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "heavy", "rings", "arom", "fsp3"]])
    return morgan, desc, mols


def scaffold(mol):
    if mol is None:
        return "invalid"
    s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return s or "acyclic"


def ifp_features(path, names):
    raw = pd.read_csv(path, low_memory=False)
    cols = [c for c in raw.columns if IFP_RE.match(c)]
    cnt = raw[cols].map(count_cell).astype(np.float32)
    cnt.insert(0, "Name", raw["Title"].astype(str).str.replace(r"_\d+$", "", regex=True).values)
    binary = cnt.copy()
    binary[cols] = (binary[cols] > 0).astype(np.float32)
    any_ = binary.groupby("Name")[cols].max().add_prefix("ifp_").add_suffix("_any")
    parts = [any_]
    for fam in sorted({c.rsplit("_", 1)[1] for c in cols}):
        fc = [c for c in cols if c.endswith("_" + fam)]
        p = pd.DataFrame(index=any_.index)
        p[f"ifp_total_{fam}_residue_any"] = binary.groupby("Name")[fc].max().sum(axis=1)
        parts.append(p)
    out = pd.concat(parts, axis=1).reindex(names)          # missing pose -> NaN, not 0
    out["ifp_has_pose"] = out.notna().any(axis=1).astype(float)
    return out.reset_index(drop=True)


def best_pose_features(path, names):
    """Top-docking-score pose per compound (lowest Glide docking score), as used for the
    external libraries. IFP bits and Glide terms both come from that SAME pose row."""
    raw = pd.read_csv(path, low_memory=False)
    raw["Name"] = raw["Title"].astype(str).str.replace(r"_\d+$", "", regex=True)
    raw = raw.sort_values("r_i_docking_score").drop_duplicates("Name", keep="first").set_index("Name")
    raw = raw.reindex(names)                                   # compounds without any pose -> NaN rows
    has = raw["r_i_docking_score"].notna().values
    cols = [c for c in raw.columns if IFP_RE.match(c)]
    bits = (raw[cols].map(count_cell) > 0).astype(np.float32)
    bits.loc[~has, :] = np.nan                                 # missing pose stays missing, not "no contact"
    ifp = bits.add_prefix("ifp_").add_suffix("_any")
    for fam in sorted({c.rsplit("_", 1)[1] for c in cols}):
        fc = [c for c in cols if c.endswith("_" + fam)]
        tot = bits[fc].sum(axis=1)
        tot[~has] = np.nan
        ifp[f"ifp_total_{fam}_residue_any"] = tot
    ifp["ifp_has_pose"] = has.astype(float)
    dcols = [c for c in raw.columns if c.startswith("r_") and not ID_LIKE.search(c)]
    dock = raw[dcols].apply(pd.to_numeric, errors="coerce").add_prefix("dock_")
    return ifp.reset_index(drop=True), dock.reset_index(drop=True)


def docking_features(path, names):
    raw = pd.read_csv(path, low_memory=False)
    idc = "Title" if "Title" in raw.columns else "Compound_ID"
    num = {}
    for c in raw.columns:
        if c == idc or ID_LIKE.search(c) or c.startswith("SIFT "):
            continue                                         # drop record identifiers / pose indices
        v = pd.to_numeric(raw[c], errors="coerce")
        if v.notna().any():
            num[c] = v
    df = pd.DataFrame(num)
    df.insert(0, "Name", raw[idc].astype(str).str.replace(r"_\d+$", "", regex=True).values)
    out = df.groupby("Name").mean().reindex(names).add_prefix("dock_")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Fold-local preprocessing
# --------------------------------------------------------------------------
class FoldPreprocessor:
    """Learned on the training fold only, then frozen."""

    def __init__(self, blocks, min_freq=0.01, max_missing=0.5):
        self.prefixes = tuple(b + "_" for b in blocks)
        self.min_freq, self.max_missing = min_freq, max_missing

    def fit(self, X):
        cols = [c for c in X.columns if c.startswith(self.prefixes) or c == "ifp_has_pose"
                and "ifp_" in self.prefixes]
        X = X[cols]
        keep = X.columns[X.isna().mean() <= self.max_missing]
        X = X[keep]
        med = X.median()
        Xf = X.fillna(med)
        keep = Xf.columns[Xf.nunique() > 1]
        Xf = Xf[keep]
        drop = []
        for c in Xf.columns:
            if c.startswith(("morgan_", "ifp_")) and set(Xf[c].unique()) <= {0, 1, 0.0, 1.0}:
                f = Xf[c].mean()
                if f < self.min_freq or f > 1 - self.min_freq:
                    drop.append(c)
        self.cols_ = [c for c in Xf.columns if c not in drop]
        self.med_ = med[self.cols_]
        return self

    def transform(self, X):
        return X.reindex(columns=self.cols_).fillna(self.med_).astype(np.float32)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def group_folds(groups, k, seed):
    """Balanced scaffold-group K-fold: whole scaffold groups per fold."""
    rng = np.random.RandomState(seed)
    g = pd.Series(groups)
    sizes = g.value_counts()
    order = sorted(sizes.index, key=lambda s: (-sizes[s], rng.rand()))
    load = np.zeros(k)
    assign = {}
    for s in order:
        f = int(np.argmin(load + rng.rand(k) * 1e-6))
        assign[s] = f
        load[f] += sizes[s]
    return g.map(assign).values


# --------------------------------------------------------------------------
# Models + inner search
# --------------------------------------------------------------------------
GNN_SETS = ("morgan_only", "ligand_only", "ligand+dock+ifp")     # feature sets the (slow) GNN is run on


def sample_params(model, rng):
    if model == "GNN":
        from main_code.gnn_model import sample_params as gnn_params
        return gnn_params(rng)
    pick = lambda opts: opts[rng.randint(len(opts))]      # keeps native python types
    if model == "ExtraTrees":
        return dict(n_estimators=pick([300, 500, 800]),
                    max_depth=pick([None, 8, 12, 20]),
                    min_samples_leaf=pick([1, 2, 3, 5]),
                    max_features=pick([0.2, 0.33, 0.5, "sqrt"]))
    return dict(iterations=pick([300, 600, 1000]),
                depth=pick([4, 6, 8]),
                learning_rate=pick([0.03, 0.06, 0.1]),
                l2_leaf_reg=pick([1, 3, 10]))


def make_model(model, params, seed):
    if model == "ExtraTrees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1, **params)
    if model == "GNN":
        from main_code.gnn_model import GNNRegressor
        return GNNRegressor(random_state=seed, **params)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_seed=seed, verbose=0, thread_count=-1, allow_writing_files=False, **params)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def _score_params(model, blocks, X, y, folds, k_inner, p, seed):
    """Mean R2 over the inner scaffold-grouped folds (fold-local preprocessing)."""
    sc = []
    for f in range(k_inner):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        pre = FoldPreprocessor(blocks).fit(X.iloc[tr])
        m = make_model(model, p, seed).fit(pre.transform(X.iloc[tr]), y[tr])
        sc.append(r2_score(y[va], m.predict(pre.transform(X.iloc[va]))))
    return float(np.mean(sc))


def inner_search(model, blocks, X, y, groups, n_iter, k_inner, seed):
    """Random search scored by inner folds."""
    rng = np.random.RandomState(seed)
    folds = group_folds(groups, k_inner, seed)
    best, best_score = None, -np.inf
    for _ in range(n_iter):
        p = sample_params(model, rng)
        s = _score_params(model, blocks, X, y, folds, k_inner, p, seed)
        if s > best_score:
            best, best_score = p, s
    return best, best_score


# ---- Bayesian optimization (same surrogate / acquisition / search spaces as the original pipeline) ----
# Original pipeline (step04a/04c): GP surrogate (Constant x Matern-5/2 + White noise), Expected Improvement over
# 384 random unit-cube candidates; 6 random + 14 Bayesian trials for classical models, 3 + 5 for deep models.
# The ONLY change here is the objective: mean R2 over INNER scaffold-grouped folds instead of the test-set R2.
def _decode_extra(u):
    return dict(n_estimators=int(round(300 + u[0] * 900)), max_features=float(0.20 + u[1] * 0.80),
                min_samples_leaf=int(round(1 + u[2] * 10)), min_samples_split=int(round(2 + u[3] * 18)),
                max_depth=None if u[4] > 0.82 else int(round(8 + u[4] * 45)), bootstrap=bool(u[5] > 0.5))


def _decode_catboost(u):
    return dict(iterations=int(round(250 + u[0] * 750)), depth=int(round(3 + u[1] * 5)),
                learning_rate=float(10 ** (-2.3 + u[2] * 1.4)), l2_leaf_reg=float(10 ** (-1 + u[3] * 2.2)),
                random_strength=float(u[4] * 3), bagging_temperature=float(u[5] * 2))


def _decode_gnn(u):
    return dict(hidden_dim=int([64, 96, 128, 192][min(3, int(u[0] * 4))]), dropout=float(0.05 + u[1] * 0.30),
                lr=float(10 ** (-4.0 + u[2] * 1.3)), weight_decay=float(10 ** (-6 + u[3] * 4)),
                batch_size=int([16, 32, 64][min(2, int(u[4] * 3))]))


BO_SPACES = {"ExtraTrees": (6, _decode_extra, 6, 14), "CatBoost": (6, _decode_catboost, 6, 14),
             "GNN": (5, _decode_gnn, 3, 5)}          # (dim, decoder, n_initial, n_bayes)
BO_CANDIDATES = 384


def _expected_improvement(gp, cand, best):
    from scipy.stats import norm
    mu, sd = gp.predict(cand, return_std=True)
    sd = np.maximum(sd, 1e-9)
    imp = mu - best
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def inner_search_bayes(model, blocks, X, y, groups, n_iter, k_inner, seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    dim, decode, n_init, n_bo = BO_SPACES[model]
    rng = np.random.RandomState(seed)
    folds = group_folds(groups, k_inner, seed)
    pts, scores, best, best_score = [], [], None, -np.inf
    for t in range(n_init + n_bo):
        if t < n_init:
            u = rng.random_sample(dim)
        else:
            gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(1e-5),
                                          alpha=1e-6, normalize_y=True, random_state=seed, n_restarts_optimizer=1)
            gp.fit(np.asarray(pts), np.asarray(scores))
            cand = rng.random_sample((BO_CANDIDATES, dim))
            u = cand[int(np.argmax(_expected_improvement(gp, cand, max(scores))))]
        p = decode(u)
        s = _score_params(model, blocks, X, y, folds, k_inner, p, seed)
        pts.append(u); scores.append(s)
        if s > best_score:
            best, best_score = p, s
    return best, best_score


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def screening_metrics(y, score, ks=(10, 25, 50, 100)):
    hit = (y >= HIT_CUTOFF).astype(int)
    out = {"n": len(y), "hit_rate": float(hit.mean())}
    if 0 < hit.sum() < len(hit):
        out["auroc"] = float(roc_auc_score(hit, score))
    order = np.argsort(-score)
    for k in ks:
        if k <= len(y):
            p = hit[order[:k]].mean()
            out[f"precision@{k}"] = float(p)
            out[f"enrichment@{k}"] = float(p / hit.mean()) if hit.mean() > 0 else np.nan
    return out


def regression_metrics(y, p):
    rho = spearmanr(y, p)[0]
    return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p),
            "RMSE": rmse(y, p), "Spearman": float(rho)}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(act, ifp, dock, out_dir, seeds, k_outer, k_inner, n_iter, models, pose="best", only_sets=None, mask_ifp=False, mask_train_only=False, search="random"):
    out_dir.mkdir(parents=True, exist_ok=True)
    morgan, desc, mols = morgan_desc(act["SMILES"])
    X = pd.concat([morgan, desc], axis=1)
    if pose == "best":
        ifp_df, dock_df = best_pose_features(ifp, act["Name"])
        dock = dock or ifp                                    # Glide terms come from the IFP table's best pose
        print(f"best-pose coverage: {int(ifp_df.ifp_has_pose.sum())}/{len(act)} compounds")
        X = pd.concat([X, ifp_df, dock_df], axis=1)
    else:                                                     # legacy: aggregate over all poses
        X = pd.concat([X, ifp_features(ifp, act["Name"]) if ifp is not None else pd.DataFrame(index=X.index)], axis=1)
        if dock is not None:
            X = pd.concat([X, docking_features(dock, act["Name"])], axis=1)
    y = np.clip(act["A5"].astype(float).values, 0, 100)
    if mask_ifp:      # DIAGNOSTIC ONLY: reproduces the historical activity-dependent zeroing (uses each row's own label)
        ic = [c for c in X.columns if c.startswith("ifp_") and c != "ifp_has_pose"]
        X.loc[y < HIT_CUTOFF, ic] = 0.0
        print(f"MASKED-IFP DIAGNOSTIC: zeroed IFP block for {(y < HIT_CUTOFF).sum()} compounds below {HIT_CUTOFF}%")
    scaf = np.array([scaffold(m) for m in mols])
    sg = pd.Series(scaf).value_counts()
    sg.rename("n_compounds").to_csv(out_dir / "scaffold_group_sizes.csv")
    print(f"{len(y)} compounds, {sg.size} scaffold groups; largest group {sg.iloc[0]}; "
          f"singletons {(sg == 1).sum()}")

    if "GNN" in models:
        from main_code import gnn_model as nested_cv_gnn
        nested_cv_gnn.set_context(act["SMILES"], scaf)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None for m in mols]
    sets = {n: b for n, b in FEATURE_SETS.items()
            if (("dock" not in b or dock is not None) and ("ifp" not in b or ifp is not None))
            and (only_sets is None or n in only_sets)}
    pred_rows, cfg_rows = [], []

    for seed in seeds:
        outer = group_folds(scaf, k_outer, seed)
        for f in range(k_outer):
            te, tr = np.where(outer == f)[0], np.where(outer != f)[0]
            assert not set(scaf[tr]) & set(scaf[te]), "scaffold leakage across outer split"
            Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]
            if mask_train_only:       # DEPLOYMENT-REALISTIC: model trained on masked rows, applied to unmasked new compounds
                ic = [c for c in X.columns if c.startswith("ifp_") and c != "ifp_has_pose"]
                Xtr = Xtr.copy(); Xtr.loc[ytr < HIT_CUTOFF, ic] = 0.0
            base = dict(repeat=seed, fold=f)
            # baseline: nearest-active-neighbour (training actives only)
            act_fps = [fps[i] for i in tr if y[i] >= HIT_CUTOFF and fps[i] is not None]
            nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], act_fps)) if fps[i] and act_fps else 0.0
                           for i in te])
            pred_rows.append(pd.DataFrame({**base, "Name": act["Name"].values[te], "y": y[te],
                                           "model": "NearestActiveNeighbour", "feature_set": "ecfp4",
                                           "pred": nn * 100, "is_score_only": True}))
            for fs, blocks in sets.items():
                for mdl in models:
                    if mdl == "GNN" and fs not in GNN_SETS:
                        continue
                    searcher = inner_search_bayes if search == "bayes" else inner_search
                    p, inner_r2 = searcher(mdl, blocks, Xtr, ytr, scaf[tr], n_iter, k_inner, seed + 1000 * f)
                    pre = FoldPreprocessor(blocks).fit(Xtr)
                    m = make_model(mdl, p, seed).fit(pre.transform(Xtr), ytr)
                    pred = np.clip(m.predict(pre.transform(Xte)), 0, 100)
                    pred_rows.append(pd.DataFrame({**base, "Name": act["Name"].values[te], "y": y[te],
                                                   "model": mdl, "feature_set": fs, "pred": pred,
                                                   "is_score_only": False}))
                    cfg_rows.append({**base, "model": mdl, "feature_set": fs, "n_features": len(pre.cols_),
                                     "inner_R2": inner_r2, "params": json.dumps(p, default=str)})
            print(f"repeat {seed} fold {f}: done", flush=True)

    P = pd.concat(pred_rows, ignore_index=True)
    # consensus = mean of the tree models within each feature set (same outer folds)
    cons = (P[~P.is_score_only].groupby(["repeat", "fold", "Name", "y", "feature_set"], as_index=False)["pred"].mean()
            .assign(model="Consensus(mean)", is_score_only=False))
    P = pd.concat([P, cons], ignore_index=True)
    P.to_csv(out_dir / "outer_predictions.csv", index=False)
    pd.DataFrame(cfg_rows).to_csv(out_dir / "selected_hyperparameters.csv", index=False)

    rows = []
    for (rep, mdl, fs), g in P.groupby(["repeat", "model", "feature_set"]):
        r = {"repeat": rep, "model": mdl, "feature_set": fs, **screening_metrics(g.y.values, g.pred.values)}
        if not g.is_score_only.iloc[0]:
            r.update(regression_metrics(g.y.values, g.pred.values))
        rows.append(r)
    per_rep = pd.DataFrame(rows)
    per_rep.to_csv(out_dir / "metrics_per_repeat.csv", index=False)
    summ = per_rep.groupby(["model", "feature_set"]).agg(["mean", "std"]).drop(columns="repeat")
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(out_dir / "metrics_summary_mean_sd.csv", index=False)
    print(summ[[c for c in summ.columns if c.startswith(("R2", "MAE", "auroc", "precision@50"))]]
          .round(3).to_string())


def smoke_data(n=240, seed=0):
    rng = np.random.RandomState(seed)
    frags = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2[nH]ccc2c1", "C1CCOCC1", "c1ccsc1"]
    subs = ["", "C", "CC", "O", "N", "(Cl)", "C(=O)O", "OC", "(F)"]
    smi = []
    while len(smi) < n:                                   # keep only valid molecules
        c = rng.choice(frags) + rng.choice(subs) + rng.choice(subs)
        if Chem.MolFromSmiles(c) is not None:
            smi.append(c)
    names = [f"S{i}" for i in range(n)]
    y = rng.uniform(-10, 90, n)
    act = pd.DataFrame({"Name": names, "SMILES": smi, "A5": y})
    cols = [f"A{r}_{t}" for r in range(420, 440) for t in ["contact", "hydrophobic", "donor"]]
    ifp = pd.DataFrame(rng.randint(0, 2, (n, len(cols))), columns=cols)
    ifp.insert(0, "Title", names)
    dock = pd.DataFrame({"Title": names, "docking score": rng.randn(n), "Molecule": np.arange(n)})
    return act, ifp, dock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity"); ap.add_argument("--ifp"); ap.add_argument("--docking")
    ap.add_argument("--out", default=str(cfg.OUTPUT / "nested_cv"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--models", nargs="+", default=["ExtraTrees", "CatBoost", "GNN"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pose", choices=["best", "mean"], default="best")
    ap.add_argument("--feature-sets", nargs="+", default=None)
    ap.add_argument("--search", choices=["random", "bayes"], default="random", help="inner hyperparameter search")
    ap.add_argument("--mask-train-only", action="store_true", help="diagnostic: mask only training rows, test rows keep real IFP")
    ap.add_argument("--mask-ifp", action="store_true", help="diagnostic: historical activity-dependent IFP zeroing")
    ap.add_argument("--gnn-all-sets", action="store_true", help="run the GNN on every requested feature set (default: only GNN_SETS)")
    a = ap.parse_args()
    if a.gnn_all_sets:
        global GNN_SETS
        GNN_SETS = tuple(FEATURE_SETS)
    if a.smoke:
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        act, ifp, dock = smoke_data()
        ifp.to_csv(tmp / "ifp.csv", index=False); dock.to_csv(tmp / "dock.csv", index=False)
        run(act, tmp / "ifp.csv", tmp / "dock.csv", Path(a.out), [0], 3, 2, 2, ["ExtraTrees", "GNN"], pose="mean")
        return
    act = pd.read_csv(a.activity).dropna(subset=["A5"]).drop_duplicates("Name").reset_index(drop=True)
    run(act, Path(a.ifp) if a.ifp else None, Path(a.docking) if a.docking else None,
        Path(a.out), a.seeds, a.outer, a.inner, a.n_iter, a.models,
        pose="mean" if a.pose == "mean" else "best", only_sets=a.feature_sets, mask_ifp=a.mask_ifp, mask_train_only=a.mask_train_only, search=a.search)


if __name__ == "__main__":
    main()
