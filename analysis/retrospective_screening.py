"""
(2) Retrospective re-ranking of the known tested hits with the corrected (unmasked,
    ligand-only) models, on the SAME candidate pool for every scorer, and
(3) TreeSHAP for the corrected ligand-only CatBoost model with fold-to-fold stability.

Everything here is RETROSPECTIVE: the hit identities were known before this analysis.
Hyper-parameters = the most frequently selected configuration across the nested-CV
outer folds for the 'ligand_only' feature set (no test-set selection).
"""
import json
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import spearmanr

from main_code import nested_cv as ncv                         # noqa: E402
from main_code import gnn_model as gnn                     # noqa: E402

OUT = cfg.OUTPUT / "corrected_screen_shap_bo"
LIB = cfg.EXTERNAL_LIBRARIES
HITS_OTAVA = ["P2195430", "P1157904", "P2195267", "P2195222"]
HIT_FDA_CID = 73951493                          # desisobutyl-n-butyl bortezomib
BLOCKS = ["morgan", "desc"]


def robust_params(model, hp_csvs, tag="ligand_only"):
    """Aggregate the hyper-parameters selected by Bayesian optimization in the 25 outer folds
    (5 repeats x 5 folds, ligand_only feature set): median for numeric values, mode for discrete/categorical.
    Uses only inner-fold selections; no outer-fold performance enters."""
    import numpy as np
    h = pd.concat([pd.read_csv(f) for f in hp_csvs])
    h = h[(h.model == model) & (h.feature_set == tag)]
    ps = [json.loads(x) for x in h.params]
    out = {}
    for k in ps[0]:
        vals = [p[k] for p in ps]
        if k in ("bootstrap", "hidden_dim", "batch_size"):
            out[k] = Counter(vals).most_common(1)[0][0]
        elif k == "max_depth":                       # None = unlimited
            n_none = sum(v is None for v in vals)
            out[k] = None if n_none > len(vals) / 2 else int(np.median([v for v in vals if v is not None]))
        else:
            med = float(np.median(vals))
            out[k] = int(round(med)) if all(isinstance(v, int) for v in vals) else med
    return out, len(ps)


def lib_features(smiles):
    mg, ds, mols = ncv.morgan_desc(smiles)
    return pd.concat([mg, ds], axis=1), mols


def ranks(score, ids, target_ids):
    s = pd.Series(np.asarray(score, float), index=ids)
    r = s.rank(ascending=False, method="min")
    return {t: int(r[t]) for t in target_ids}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    act = pd.read_csv(cfg.TRAINING).dropna(subset=["A5"])
    act = act.drop_duplicates("Name").reset_index(drop=True)
    y = np.clip(act.A5.values, 0, 100)
    Xtr_raw, mols = lib_features(act.SMILES)
    scaf = np.array([ncv.scaffold(m) for m in mols])

    trees_hp = sorted(cfg.OUTPUT.glob("nested_cv_bo_trees_*/selected_hyperparameters.csv"))
    gnn_hp = [cfg.OUTPUT / "nested_cv_bo_gnn/selected_hyperparameters.csv"]
    et_p, n1 = robust_params("ExtraTrees", trees_hp)
    cb_p, n2 = robust_params("CatBoost", trees_hp)
    gn_p, n3 = robust_params("GNN", gnn_hp)
    print("Bayesian-optimization hyper-parameters (median/mode over outer folds):", f"ET n={n1}", f"Cat n={n2}", f"GNN n={n3}")
    print(" ET", et_p); print(" Cat", cb_p); print(" GNN", gn_p)

    # --- libraries -------------------------------------------------------
    ot = pd.read_csv(LIB / "otava_best3_model_predictions.csv", usecols=["source_id", "SMILES"]).reset_index(drop=True)
    fda = pd.read_csv(LIB / "d_best3_model_predictions.csv", usecols=["Compound_CID", "Name", "IUPAC_Name", "SMILES"])
    fda = fda.drop_duplicates("Compound_CID").reset_index(drop=True)
    X_ot, m_ot = lib_features(ot.SMILES)
    X_fda, m_fda = lib_features(fda.SMILES)

    # --- final models on all 1488 (fold-free preprocessing is fit on all training rows) ---
    pre = ncv.FoldPreprocessor(BLOCKS).fit(Xtr_raw)
    Xtr = pre.transform(Xtr_raw)
    et = ncv.make_model("ExtraTrees", et_p, 0).fit(Xtr, y)
    cb = ncv.make_model("CatBoost", cb_p, 0).fit(Xtr, y)
    n_tr, n_ot = len(act), len(ot)
    gnn.set_context(list(act.SMILES) + list(ot.SMILES),
                    np.concatenate([scaf, np.array(["lib"] * n_ot)]))
    gn = ncv.make_model("GNN", gn_p, 0).fit(Xtr, y)

    def to_idx(X, off):
        X = pre.transform(X)
        X.index = np.arange(off, off + len(X))
        return X

    Xo = to_idx(X_ot, n_tr)
    P = pd.DataFrame({"id": ot.source_id,
                      "ExtraTrees": et.predict(Xo), "CatBoost": cb.predict(Xo), "GNN": gn.predict(Xo)})
    P[["ExtraTrees", "CatBoost", "GNN"]] = P[["ExtraTrees", "CatBoost", "GNN"]].clip(0, 100)
    P["Consensus(ET+Cat)"] = P[["ExtraTrees", "CatBoost"]].mean(axis=1)
    P["Consensus(ET+Cat+GNN)"] = P[["ExtraTrees", "CatBoost", "GNN"]].mean(axis=1)
    act_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m, v in zip(mols, y) if v >= ncv.HIT_CUTOFF]
    fp_ot = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in m_ot]
    P["NearestActiveNeighbour"] = [max(DataStructs.BulkTanimotoSimilarity(f, act_fps)) for f in fp_ot]
    P.to_csv(OUT / "otava_corrected_predictions.csv", index=False)

    old = pd.read_csv(LIB / "otava_best3_model_predictions.csv")
    old_r = {"Original masked consensus (historical)": ranks(old.consensus_A5_mean, old.source_id, HITS_OTAVA)}
    rows = {**old_r}
    for c in P.columns[1:]:
        rows[c] = ranks(P[c].values, P.id.values, HITS_OTAVA)
    rows["Random expectation (median rank)"] = {h: (n_ot + 1) // 2 for h in HITS_OTAVA}
    R = pd.DataFrame(rows).T
    R["best_rank"] = R[HITS_OTAVA].min(axis=1); R["median_rank"] = R[HITS_OTAVA].median(axis=1)
    R["hits_in_top50"] = (R[HITS_OTAVA] <= 50).sum(axis=1); R["hits_in_top100"] = (R[HITS_OTAVA] <= 100).sum(axis=1)
    R.to_csv(OUT / "otava_hit_ranks_same_pool.csv")
    print(f"\nOTAVA pool n={n_ot}; ranks of the 4 tested hits (1 = best)")
    print(R.to_string())

    # --- FDA-derived library (trees + NN only; GNN omitted for 124k graphs) ---
    Xf = pre.transform(X_fda)
    F = pd.DataFrame({"id": fda.Compound_CID.values,
                      "ExtraTrees": np.clip(et.predict(Xf), 0, 100), "CatBoost": np.clip(cb.predict(Xf), 0, 100)})
    F["Consensus(ET+Cat)"] = F[["ExtraTrees", "CatBoost"]].mean(axis=1)
    fp_f = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in m_fda]
    F["NearestActiveNeighbour"] = [max(DataStructs.BulkTanimotoSimilarity(f, act_fps)) for f in fp_f]
    fr = {c: ranks(F[c].values, F.id.values, [HIT_FDA_CID]) for c in F.columns[1:]}
    FR = pd.DataFrame(fr).T.rename(columns={HIT_FDA_CID: "rank_of_CID_73951493"})
    FR["pool"] = len(F)
    FR.to_csv(OUT / "fda_hit_rank_same_pool.csv")
    # merged pool (OTAVA + FDA): all scorers ranked together; GNN omitted (not run on 124k graphs)
    M = pd.concat([P.assign(lib="OTAVA")[["id", "lib", "ExtraTrees", "CatBoost", "Consensus(ET+Cat)", "NearestActiveNeighbour"]],
                   F.assign(lib="FDA")[["id", "lib", "ExtraTrees", "CatBoost", "Consensus(ET+Cat)", "NearestActiveNeighbour"]]],
                  ignore_index=True)
    M["id"] = M["id"].astype(str)
    tgt = HITS_OTAVA + [str(HIT_FDA_CID)]
    MR = pd.DataFrame({c: ranks(M[c].values, M["id"].values, tgt) for c in M.columns[2:]}).T
    MR["pool"] = len(M)
    MR.to_csv(OUT / "merged_pool_hit_ranks.csv")
    print(f"\nMERGED pool n={len(M)}; ranks of the 5 confirmed hits (CID 73951493 = bortezomib analogue)")
    print(MR.to_string())
    M.to_csv(OUT / "merged_pool_corrected_scores.csv", index=False)
    print(f"\nFDA-derived pool n={len(F)}; rank of the tested bortezomib analogue")
    print(FR.to_string())

    # --- (3) SHAP: final model + fold stability (repeat seed 0, same outer folds) ---
    import shap
    feats = list(Xtr.columns)
    expl = shap.TreeExplainer(cb)
    sv = expl.shap_values(Xtr)
    g = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    # describe Morgan bits by the most common environment seen in the training set
    desc = {}
    for i, m in enumerate(mols):
        bi = {}
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024, bitInfo=bi)
        for bit, envs in bi.items():
            name = f"morgan_{bit:04d}"
            if name in desc and desc[name][1] >= 1:
                continue
            a, r = envs[0]
            if r == 0:
                smi = m.GetAtomWithIdx(a).GetSymbol()
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(m, r, a)
                amap = {}
                sub = Chem.PathToSubmol(m, env, atomMap=amap) if env else None
                smi = Chem.MolToSmiles(sub) if sub is not None else "?"
            desc[name] = (smi, 1)
    g["bit_substructure(example)"] = g.feature.map(lambda f: desc.get(f, ("", 0))[0])
    g.to_csv(OUT / "shap_global_catboost_ligand_only.csv", index=False)
    print("\nTop 15 features, final CatBoost (ligand-only):")
    print(g.head(15).to_string(index=False))

    outer = ncv.group_folds(scaf, 5, 0)
    fold_imp, fold_pred = {}, None
    for f in range(5):
        tr, te = np.where(outer != f)[0], np.where(outer == f)[0]
        p_ = ncv.FoldPreprocessor(BLOCKS).fit(Xtr_raw.iloc[tr])
        m_ = ncv.make_model("CatBoost", cb_p, 0).fit(p_.transform(Xtr_raw.iloc[tr]), y[tr])
        s_ = shap.TreeExplainer(m_).shap_values(p_.transform(Xtr_raw.iloc[te]))      # held-out compounds only
        fold_imp[f] = pd.Series(np.abs(s_).mean(0), index=p_.cols_)
    FI = pd.DataFrame(fold_imp).reindex(feats)
    FI.to_csv(OUT / "shap_importance_per_outer_fold.csv")
    stab = []
    for a in range(5):
        for b in range(a + 1, 5):
            ta, tb = set(FI[a].nlargest(20).index), set(FI[b].nlargest(20).index)
            d = FI[[a, b]].dropna()
            stab.append({"fold_a": a, "fold_b": b, "top20_overlap": len(ta & tb),
                         "spearman_all": spearmanr(d[a], d[b])[0]})
    stab = pd.DataFrame(stab)
    stab.to_csv(OUT / "shap_fold_stability.csv", index=False)
    in_all = [f for f in FI.index if all(f in FI[c].nlargest(20).index for c in FI)]
    print(f"\nSHAP fold stability: top-20 overlap mean {stab.top20_overlap.mean():.1f}/20, "
          f"Spearman mean {stab.spearman_all.mean():.2f}; features in top-20 of ALL 5 folds: {in_all}")
    cat = lambda f: f.split("_")[0]
    cls = g.assign(block=g.feature.map(cat)).groupby("block").mean_abs_shap.sum()
    print("\nShare of total |SHAP| by block:", (cls / cls.sum()).round(3).to_dict())


if __name__ == "__main__":
    main()
