"""
Tables behind the supplementary tables (CSV files in Output/tables). The Excel files are assembled from them by analysis/si_tables.py.

Stages
  nested_cv_results       mean ± SD of every model and feature set                           (Table S1A)
  threshold_sensitivity   AUROC at hit cutoffs of 30, 38 and 50%                             (Table S1B)
  model_agreement         error correlation, rank overlap and same-budget precision          (Table S1C)
  partition_agreement     adjusted Rand index between the scaffold-grouped partitions of the five repeats (note of Table S1A)
  search_spaces           hyperparameter search spaces and the configuration selected in every outer fold (Table S1D)
  masked_vs_unmasked      historical IFP masking versus the corrected input                  (Table S2B)
  deployment_test         masked training evaluated on compounds with their real IFP         (Table S2B)
  nearest_neighbours      nearest training compounds of the FP-active compounds              (Table S3C)
  experimental_results    FP results of all tested compounds (needs figures/figure4.py)      (Table S3C)
  otava_corrected_ranking corrected scores and ranks of the 1,909 OTAVA compounds (needs analysis/retrospective_screening.py) (Table S3C)
  fda_top50_audit         parent structures, boron content and name checks of the FDA GSRS-derived top 50 (Table S3E)

Inputs are the nested-CV runs in Output/ (see README). Run from the repository root:  python analysis/tables.py [stage ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

OUT, TAB = cfg.OUTPUT, cfg.TABLES
SEL = cfg.INPUT / "original_selection"
S3_FILE, S4_FILE = SEL / "Table_S3_otava_top50_triple_consensus.csv", SEL / "Table_S4_fda_top50_triple_consensus.csv"
MASKED = ["nested_cv_masked_diagnostic", "nested_cv_masked_diagnostic_cat", "nested_cv_masked_diagnostic_gnn"]
# the masking diagnostics were run with random-search hyperparameters, so the like-for-like unmasked comparator is the random-search run
UNMASKED_RANDOM = "nested_cv_final_all"
SEARCH_SPACES = [
    ("Extra Trees", "Number of trees", "300-1,200", "integer, uniform"),
    ("Extra Trees", "Maximum feature fraction", "0.20-1.00", "uniform"),
    ("Extra Trees", "Minimum leaf size", "1-11", "integer, uniform"),
    ("Extra Trees", "Minimum split size", "2-20", "integer, uniform"),
    ("Extra Trees", "Maximum depth", "8-45, or unlimited", "integer, uniform; unlimited for the upper 18% of the unit range"),
    ("Extra Trees", "Bootstrap sampling", "on or off", "binary, equal probability"),
    ("CatBoost", "Iterations", "250-1,000", "integer, uniform"),
    ("CatBoost", "Depth", "3-8", "integer, uniform"),
    ("CatBoost", "Learning rate", "0.0050-0.126", "log-uniform (10^-2.3 to 10^-0.9)"),
    ("CatBoost", "L2 leaf regularization", "0.10-15.8", "log-uniform (10^-1 to 10^1.2)"),
    ("CatBoost", "Random strength", "0-3", "uniform"),
    ("CatBoost", "Bagging temperature", "0-2", "uniform"),
    ("GNN", "Hidden size", "64, 96, 128 or 192", "categorical, equal probability"),
    ("GNN", "Dropout", "0.05-0.35", "uniform"),
    ("GNN", "Learning rate", "1e-4 to 2.0e-3", "log-uniform (10^-4 to 10^-2.7)"),
    ("GNN", "Weight decay", "1e-6 to 1e-2", "log-uniform (10^-6 to 10^-2)"),
    ("GNN", "Batch size", "16, 32 or 64", "categorical, equal probability"),
]
# raw values of the six inactive OTAVA compounds in the three independent experiments (highest concentration, then 25 uM)
RAW_NEGATIVES = {
    "P2195233": [[39.65116, 38.48837, 44.88372], [12.32558, 9.651163, 13.83721]],
    "P3459742": [[10.32967033, 16.7032967, 12.85714286], [4.395604396, 5.164835165, 2.307692308]],
    "P2195764": [[7.472527473, 4.065934066, 14.50549451], [0, 1.538461538, 5.934065934]],
    "P2195728": [[25.27472527, 25.93406593, 30.21978022], [9.120879121, 10.65934066, 11.53846154]],
    "P2195223": [[48.46153846, 44.17582418, 48.13186813], [21.53846154, 19.56043956, 21.0989011]],
    "P2195277": [[3.548387, 3.655914, 5.16129], [3.763441, 1.72043, -3.763441]],
}
TESTED = {"P2195430": "FP-active", "P2195267": "FP-active", "P2195222": "FP-active", "P1157904": "FP-active", "P2195277": "inactive",
          "P2195233": "inactive", "P2195223": "inactive", "P2195728": "inactive", "P2195764": "inactive", "P3459742": "inactive"}


def _summary(d):
    return pd.read_csv(OUT / d / "metrics_summary_mean_sd.csv")


def _pm(m, sd, d):
    return "" if pd.isna(m) else f"{m:.{d}f} ± {sd:.{d}f}"


# ----------------------------------------------------------------------------------------------------------- Table S1
def run_nested_cv_results():
    s = pd.concat([_summary("nested_cv_final_bo"), _summary("nested_cv_bo_morgan_merged")], ignore_index=True)
    models = [("ExtraTrees", "Extra Trees"), ("CatBoost", "CatBoost"), ("GNN", "GNN"), ("Consensus(ET+Cat)", "Mean of Extra Trees and CatBoost"),
              ("Consensus(ET+Cat+GNN)", "Mean of three models"), ("NearestActiveNeighbour", "Nearest-active-neighbor baseline")]
    fsets = [("morgan_only", "Morgan-only"), ("ligand_only", "Ligand-only (Morgan + RDKit)"), ("docking_only", "Docking-only"), ("ifp_only", "IFP-only"),
             ("ligand+dock", "Ligand + docking"), ("ligand+dock+ifp", "Ligand + docking + IFP"), ("ecfp4", "ECFP4 similarity")]
    metrics = [("R2", "R²", 3), ("MAE", "MAE", 1), ("RMSE", "RMSE", 1), ("auroc", "AUROC", 3), ("precision@50", "Precision at 50", 3),
               ("precision@100", "Precision at 100", 3), ("enrichment@50", "Enrichment at 50", 2)]
    rows = []
    for fk, fl in fsets:
        for mk, ml in models:
            r = s[(s.model == mk) & (s.feature_set == fk)]
            if not r.empty:
                r = r.iloc[0]
                rows.append({"Feature set": fl, "Model": ml, **{lab: _pm(r[f"{k}_mean"], r[f"{k}_std"], d).replace("-", "−") for k, lab, d in metrics}})
    pd.DataFrame(rows).to_csv(TAB / "nested_cv_results.csv", index=False)
    print("nested_cv_results.csv", len(rows), "rows")


def run_threshold_sensitivity():
    P = pd.read_csv(OUT / "nested_cv_final_bo" / "outer_predictions_all.csv")
    rows = []
    for model in ["CatBoost", "ExtraTrees", "GNN", "Consensus(ET+Cat+GNN)", "Consensus(ET+Cat)", "NearestActiveNeighbour"]:
        g = P[(P.model == model) & (P.feature_set == ("ecfp4" if model == "NearestActiveNeighbour" else "ligand_only"))]
        per = {c: [roc_auc_score(gr.y >= c, gr.pred) for _, gr in g.groupby("repeat")] for c in (30, 38, 50)}
        means = {c: np.mean(v) for c, v in per.items()}
        rows.append({"model": model, **{f"AUROC@{c}": f"{means[c]:.3f}±{np.std(per[c], ddof=1):.3f}" for c in per},
                     "max_range": round(max(means.values()) - min(means.values()), 3)})
    pd.DataFrame(rows).to_csv(TAB / "hit_cutoff_sensitivity.csv", index=False)
    print("hit_cutoff_sensitivity.csv written")


def run_model_agreement():
    P = pd.read_csv(OUT / "nested_cv_final_bo/outer_predictions_all.csv")
    L = P[P.feature_set == "ligand_only"]
    names = {"ExtraTrees": "Extra Trees", "CatBoost": "CatBoost", "GNN": "GNN"}
    f = lambda v, d: f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"
    rows = []
    for a, b in [("ExtraTrees", "CatBoost"), ("ExtraTrees", "GNN"), ("CatBoost", "GNN")]:
        er, rs, ov = [], [], []
        for r in range(5):
            A = L[(L.model == a) & (L["repeat"] == r)].set_index("Name")
            B = L[(L.model == b) & (L["repeat"] == r)].set_index("Name").loc[A.index]
            er.append(np.corrcoef(A.pred - A.y, B.pred - B.y)[0, 1])
            rs.append(spearmanr(A.pred, B.pred)[0])
            ov.append(len(set(A.pred.sort_values(ascending=False).index[:50]) & set(B.pred.sort_values(ascending=False).index[:50])))
        rows.append({"Model pair (ligand-only)": f"{names[a]} vs {names[b]}", "Pearson r of prediction errors": f(er, 2),
                     "Spearman rho of predictions": f(rs, 2), "Shared compounds among each model's top 50": f(ov, 1)})
    pd.DataFrame(rows).to_csv(TAB / "model_agreement.csv", index=False)

    items = [("Extra Trees", "ExtraTrees", "ligand_only"), ("CatBoost", "CatBoost", "ligand_only"), ("GNN", "GNN", "ligand_only"),
             ("Mean of three models", "Consensus(ET+Cat+GNN)", "ligand_only"), ("Mean of Extra Trees and CatBoost", "Consensus(ET+Cat)", "ligand_only"),
             ("Extra Trees, docking-only", "ExtraTrees", "docking_only"), ("CatBoost, docking-only", "CatBoost", "docking_only"),
             ("Extra Trees, IFP-only", "ExtraTrees", "ifp_only"), ("CatBoost, IFP-only", "CatBoost", "ifp_only"),
             ("Nearest-active-neighbor baseline", "NearestActiveNeighbour", "ecfp4")]
    pk = lambda g, k: (g.y.values >= 38)[np.argsort(-g.pred.values)[:k]].mean()
    rows = []
    for lab, m, fs in items:
        g = P[(P.model == m) & (P.feature_set == fs)]
        d = {"Scorer": lab}
        for k in (10, 25, 50):
            d[f"Precision at {k}"] = f([pk(x, k) for _, x in g.groupby("repeat")], 2)
        d["AUROC"] = f([roc_auc_score(x.y >= 38, x.pred) for _, x in g.groupby("repeat")], 3)
        rows.append(d)
    pd.DataFrame(rows).to_csv(TAB / "same_budget_precision.csv", index=False)
    print("model_agreement.csv and same_budget_precision.csv written")


def run_partition_agreement():
    """Adjusted Rand index between the outer-fold assignments of the five repeats (all compounds, and without the dominant scaffold group)."""
    from itertools import combinations
    from sklearn.metrics import adjusted_rand_score as ari
    from main_code import nested_cv as n
    from main_code.common import load_training
    scaf = np.array([n.scaffold(m) for m in n.morgan_desc(load_training().SMILES)[2]])
    rest = scaf != pd.Series(scaf).value_counts().index[0]
    folds = {s: n.group_folds(scaf, 5, s) for s in range(5)}
    d = pd.DataFrame([dict(seed_a=a, seed_b=b, ARI_all=ari(folds[a], folds[b]), ARI_excluding_dominant=ari(folds[a][rest], folds[b][rest]))
                      for a, b in combinations(range(5), 2)])
    d.loc[len(d)] = dict(seed_a="mean", seed_b="", ARI_all=d.ARI_all.mean(), ARI_excluding_dominant=d.ARI_excluding_dominant.mean())
    d.round(3).to_csv(TAB / "partition_agreement.csv", index=False)
    print(d.tail(1).round(2).to_string(index=False))


def run_search_spaces():
    pd.DataFrame(SEARCH_SPACES, columns=["Model", "Hyperparameter", "Search range", "Scale / type"]).to_csv(TAB / "search_spaces.csv", index=False)
    runs = sorted(OUT.glob("nested_cv_bo_trees_*/selected_hyperparameters.csv")) + sorted(OUT.glob("nested_cv_bo_gnn*/selected_hyperparameters.csv"))
    pd.concat([pd.read_csv(f) for f in runs if "gnn_all" not in str(f)], ignore_index=True).to_csv(TAB / "selected_hyperparameters.csv", index=False)
    print("search_spaces.csv and selected_hyperparameters.csv written")


# ----------------------------------------------------------------------------------------------------------- Table S2
def run_masked_vs_unmasked():
    keep = ["model", "feature_set", "R2_mean", "R2_std", "MAE_mean", "MAE_std", "auroc_mean", "auroc_std"]
    u = _summary(UNMASKED_RANDOM)
    u = u[u.model.isin(["CatBoost", "ExtraTrees", "GNN"]) & u.feature_set.isin(["ifp_only", "ligand+dock+ifp", "ligand_only"])][keep]
    u = u.assign(input="unmasked").sort_values(["model", "feature_set"])
    m = pd.concat([_summary(d) for d in MASKED])
    m = m[m.model.isin(["CatBoost", "ExtraTrees", "GNN"]) & m.feature_set.isin(["ifp_only", "ligand+dock+ifp"])][keep].assign(input="masked (historical diagnostic)")
    pd.concat([u, m]).to_csv(OUT / "masked_vs_unmasked_ablation_table.csv", index=False)
    print("masked_vs_unmasked_ablation_table.csv written")


def run_deployment_test():
    keep = ["model", "feature_set", "R2_mean", "R2_std", "MAE_mean", "auroc_mean", "auroc_std"]
    sets = {"all masked (historical)": pd.concat([_summary(d) for d in MASKED]),
            "train masked / test REAL (new-compound)": _summary("nested_cv_masked_train_only"),
            "unmasked (corrected)": _summary(UNMASKED_RANDOM)}
    T = pd.concat([s[s.model.isin(["CatBoost", "ExtraTrees"]) & s.feature_set.isin(["ifp_only", "ligand+dock+ifp"])][keep].assign(setting=k)
                   for k, s in sets.items()])
    T = T.assign(_o=T.setting.map({k: i for i, k in enumerate(sets)})).sort_values(["model", "feature_set", "_o"]).drop(columns="_o")
    T.round(3).to_csv(OUT / "masked_deployment_test_table.csv", index=False)
    print("masked_deployment_test_table.csv written")


# ----------------------------------------------------------------------------------------------------------- Table S3
def run_nearest_neighbours():
    """Nearest training neighbours of the five FP-active compounds (ECFP4, 2048 bits): identity, similarity, activity, scaffolds."""
    from main_code.common import load_training
    tr = load_training()
    s3, s4 = pd.read_csv(S3_FILE).set_index("Compound ID"), pd.read_csv(S4_FILE).set_index("Name")
    q = {k: s3.loc[k, "SMILES"] for k in ["P2195430", "P2195267", "P2195222", "P1157904"]}
    q["Desisobutyl-n-butyl Bortezomib"] = s4.loc["Desisobutyl-n-butyl Bortezomib", "SMILES"]
    fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    scaf = lambda m: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    tm = [Chem.MolFromSmiles(s) for s in tr.SMILES]
    tf = [fp(m) for m in tm]
    active, rows = tr.A5.values >= 38, []
    for k, s in q.items():
        m = Chem.MolFromSmiles(s)
        sim = np.array(DataStructs.BulkTanimotoSimilarity(fp(m), tf))
        for lab, mask in [("any training compound", np.ones(len(tr), bool)), ("training active (>=38%)", active)]:
            j = np.where(mask)[0][np.argmax(sim[mask])]
            rows.append(dict(hit=k, neighbour_type=lab, neighbour=tr.Name[j], tanimoto=round(sim[j], 2), neighbour_inhibition=round(tr.A5[j], 1),
                             hit_scaffold=scaf(m), neighbour_scaffold=scaf(tm[j]), same_scaffold=scaf(m) == scaf(tm[j]),
                             neighbour_smiles=tr.SMILES[j], hit_smiles=s))
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "Nearest_training_neighbours_of_hits.csv", index=False)
    print(d[["hit", "neighbour_type", "neighbour", "tanimoto", "neighbour_inhibition", "same_scaffold"]].to_string(index=False))


def run_experimental_results():
    """All ten OTAVA compounds and the FDA GSRS-derived analog, with the raw FP results (fits from figures/figure4.py)."""
    from figures import figure4 as g
    fit = pd.read_csv(TAB / "Figure4_AB_source_table.csv").set_index("compound")
    col = {"hi": "Inhibition at highest concentration (%, mean ± SD, n = 3)", "at25": "Inhibition at 25 µM (%, mean ± SD, n = 3)"}
    rows = []
    for _, r in pd.read_csv(S3_FILE).head(10).iterrows():
        cid = r["Compound ID"]
        row = {"OTAVA rank": r["rank"], "Compound": cid, "Plate": r["Plate"], "Rack position": r["Rack Position"],
               "Consensus predicted inhibition (%)": r.iloc[4], "Highest concentration tested (µM)": "", col["hi"]: "", col["at25"]: "", "IC50 (µM)": ">50", "Class": "Inactive / weak"}
        if cid in g.DATA:                                                  # FP-active: dose-response fit
            d = g.DATA[cid]; reps = np.array(d["reps"]); m, sd = reps.mean(1), reps.std(1, ddof=1); top = int(np.argmax(d["logC"])); f = fit.loc[cid]
            row.update({"Highest concentration tested (µM)": f"{10 ** d['logC'][top] * 1e6:.0f}", col["hi"]: _pm(m[top], sd[top], 1),
                        col["at25"]: _pm(f.measured_25uM, f.sd_25uM, 1) + ("" if f.observed_directly else " (interpolated from fit; 25 µM not tested)"),
                        "IC50 (µM)": f"{f.ic50_uM:.1f}", "Class": "FP-active"})
        elif cid in RAW_NEGATIVES:
            hi, at25 = np.array(RAW_NEGATIVES[cid])
            row.update({"Highest concentration tested (µM)": "100", col["hi"]: _pm(hi.mean(), hi.std(ddof=1), 1), col["at25"]: _pm(at25.mean(), at25.std(ddof=1), 1)})
        rows.append(row)
    name = "Desisobutyl-n-butyl Bortezomib"
    f, reps = fit.loc[name], np.array(g.DATA[name]["reps"])
    rows.append({"OTAVA rank": "FDA rank 2", "Compound": name, "Plate": "", "Rack position": "", "Consensus predicted inhibition (%)": 66.65,
                 "Highest concentration tested (µM)": "100", col["hi"]: _pm(reps.mean(1)[0], reps.std(1, ddof=1)[0], 1), col["at25"]: _pm(f.measured_25uM, f.sd_25uM, 1),
                 "IC50 (µM)": f"{f.ic50_uM:.1f}", "Class": "FP-active"})
    out = pd.DataFrame(rows)
    vendor = cfg.EXTERNAL_LIBRARIES / "otava_library.xlsx"                  # OTAVA shipment list (optional): lot and 1H NMR purity
    if vendor.exists():
        v = pd.read_excel(vendor).set_index("COMPOUND ID")
        out["Lot number (OTAVA)"] = out.Compound.map(v["LOT NUMBER"])
        out["Purity by 1H NMR (%, vendor)"] = out.Compound.map(v["ANALYTICAL RESULT"])
    out["Supplier / identifier"] = ["MilliporeSigma; CAS 1104011-35-9; PubChem CID 73951493" if c.startswith("Desisobutyl") else "OTAVA (PO#484641)" for c in out.Compound]
    out.to_csv(TAB / "tested_compounds_results.csv", index=False)
    print(out.to_string(index=False))


def run_otava_corrected_ranking():
    """Corrected (unmasked) scores and ranks of all 1,909 OTAVA compounds with the original selection rank and the tested outcome."""
    m = pd.read_csv(OUT / "corrected_screen_shap_bo" / "otava_corrected_predictions.csv")
    for k in ["ExtraTrees", "CatBoost", "GNN", "Consensus(ET+Cat)", "Consensus(ET+Cat+GNN)", "NearestActiveNeighbour"]:
        m["rank_" + k] = m[k].rank(ascending=False, method="min").astype(int)
    m = m.sort_values("rank_Consensus(ET+Cat+GNN)")
    m["tested_outcome"] = m.id.map(TESTED).fillna("")
    s3 = pd.read_csv(S3_FILE).rename(columns={"Compound ID": "id", "rank": "original_selection_rank_S3"})
    s3 = s3[["id", "original_selection_rank_S3", *s3.columns[4:8]]]
    s3.columns = ["id", "original_selection_rank_S3", "orig_consensus_mean", "orig_ExtraTrees", "orig_CatBoost", "orig_GNN"]
    m.merge(s3, on="id", how="left").to_csv(TAB / "otava_corrected_ranking.csv", index=False)
    print("otava_corrected_ranking.csv written")


def run_fda_top50_audit():
    """Parent structures (salt stripping), boron content and name-structure checks of the FDA GSRS-derived top 50."""
    d = pd.read_csv(S4_FILE)
    parent = lambda smi: Chem.MolToSmiles(max(Chem.GetMolFrags(Chem.MolFromSmiles(smi), asMols=True), key=lambda x: x.GetNumHeavyAtoms()))
    d["parent_smiles"] = d.SMILES.map(parent)
    d["parent_group"] = d.groupby("parent_smiles")["rank"].transform(lambda x: "/".join(map(str, x)) if len(x) > 1 else "")
    d["contains_boron"] = d.SMILES.map(lambda s: any(a.GetSymbol() == "B" for a in Chem.MolFromSmiles(s).GetAtoms()))
    d["name_structure_check"] = ""
    d.loc[d["rank"] == 10, "name_structure_check"] = "MISMATCH: name = piperazine-isoxazole carbamate; SMILES = peptide-like hydroperoxide"
    d.loc[d["rank"].isin([6, 8]), "name_structure_check"] = "CHECK: bortezomib-like scaffold with C(OH) in place of B(OH)2"
    d.to_csv(TAB / "fda_top50_audit.csv", index=False)
    print("fda_top50_audit.csv written")


STAGES = {f.__name__[4:]: f for f in (run_nested_cv_results, run_threshold_sensitivity, run_model_agreement, run_partition_agreement, run_search_spaces, run_masked_vs_unmasked,
                                      run_deployment_test, run_nearest_neighbours, run_experimental_results, run_otava_corrected_ranking, run_fda_top50_audit)}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    for w in wanted:
        print("=== stage", w, flush=True)
        STAGES[w]()
