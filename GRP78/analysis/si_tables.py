"""
Supplementary Tables S1 to S3 (Excel) assembled from the CSV tables of analysis/tables.py, analysis/shap_analysis.py and figures/figure3.py, figures/figure4.py.

Stages
  table_s1   Supplementary_Table_S1.xlsx  Nested CV results, Hit cutoff sensitivity, Model agreement, Search spaces
  table_s2   Supplementary_Table_S2.xlsx  Algorithm survey, IFP masking diagnostic
  table_s3   Supplementary_Table_S3.xlsx  Class-level Shapley, Ranking stability, Tested compounds, OTAVA top 50, GSRS-derived top 50

Each sheet has the table title in the first row, the table, a general note and lettered footnotes below it (ACS style).
Run from the repository root:  python analysis/si_tables.py [stage ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

TAB = cfg.TABLES
SEL = cfg.INPUT / "original_selection"
THIN = Side(style="thin", color="000000")
# IC50 as mean ± SD of the IC50 values of the three independent experiments (GraphPad Prism fits; Table 1 of the manuscript)
IC50_PRISM = {"P2195430": "4.4 ± 0.5", "P1157904": "7.5 ± 0.3", "P2195267": "12.8 ± 0.4", "P2195222": "16.1 ± 1.1",
              "Desisobutyl-n-butyl bortezomib": "4.8 ± 0.8"}
# residues with a ligand contact in at least 50% of the frames in each of the three simulations
CONTACTS = {"P2195430": "T428, V429, S452", "P1157904": "T428, S452", "P2195267": "V429, S452", "P2195222": "I426, T428, S452",
            "Desisobutyl-n-butyl bortezomib": "T428, V429, S452"}


def _put(ws, r, df, textcols=1, wrap=True):
    for j, c in enumerate(df.columns, 1):
        x = ws.cell(r, j, c)
        x.font = Font(bold=True, name="Arial", size=10)
        x.alignment = Alignment(wrap_text=True, vertical="center", horizontal="left" if j <= textcols else "center")
        x.border = Border(top=THIN, bottom=THIN)
    for i, row in enumerate(df.itertuples(index=False), r + 1):
        for j, v in enumerate(row, 1):
            x = ws.cell(i, j, v.item() if hasattr(v, "item") else v)
            x.font = Font(name="Arial", size=10)
            x.alignment = Alignment(wrap_text=wrap, vertical="top", horizontal="left" if j <= textcols else "center")
    for j in range(1, len(df.columns) + 1):
        ws.cell(r + len(df), j).border = Border(bottom=THIN)
    return r + len(df) + 1


def _note(ws, r, text, ncol):
    for k, line in enumerate(text.split("\n")):
        c = ws.cell(r + k, 1, line)
        c.font = Font(name="Arial", size=9)
        ws.merge_cells(start_row=r + k, start_column=1, end_row=r + k, end_column=ncol)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r + k].height = 13 * max(1, len(line) // (18 * min(ncol, 8)) + 1)


def _sheet(wb, name, title, blocks, note, textcols=1, wrap=True):
    """blocks: list of (subtitle or None, DataFrame); the note goes below the last block."""
    ws = wb.create_sheet(name)
    ws.cell(1, 1, title).font = Font(bold=True, name="Arial", size=11)
    r, ncol = 3, max(len(b.columns) for _, b in blocks)
    for sub, df in blocks:
        if sub:
            ws.cell(r, 1, sub).font = Font(italic=True, name="Arial", size=10)
            r += 1
        r = _put(ws, r, df, textcols, wrap) + 1
    _note(ws, r, note, min(ncol, 9))
    ws.sheet_view.showGridLines = False
    ws.row_dimensions[4 if blocks[0][0] else 3].height = 48
    for j in range(1, ncol + 1):
        cols = [b for _, b in blocks if j <= len(b.columns)]
        width = max([len(str(b.columns[j - 1])) // 2 for b in cols] + [len(str(v)) for b in cols for v in b.iloc[:, j - 1]])
        ws.column_dimensions[get_column_letter(j)].width = min(max(14, width + 2), 60 if wrap else 42)


def _book():
    wb = Workbook()
    wb.remove(wb.active)
    return wb


def _fp_result():
    """FP outcome of every tested compound, used to label the top-50 tables."""
    ex = pd.read_csv(TAB / "tested_compounds_results.csv")
    return ex, ex.set_index("Compound")["Class"]


# ------------------------------------------------------------------------------------------------------------------ Table S1
def build_table_s1():
    wb = _book()
    ari = pd.read_csv(TAB / "partition_agreement.csv").iloc[-1]
    nest = pd.read_csv(TAB / "nested_cv_results.csv").fillna("n/a ᶜ")
    nest = nest.rename(columns={"MAE": "MAE (% inhibition)", "RMSE": "RMSE (% inhibition)", "Precision at 50": "Precision at 50 ᵃ",
                                "Precision at 100": "Precision at 100 ᵃ", "Enrichment at 50": "Enrichment at 50 ᵇ"})
    _sheet(wb, "Nested CV results", "Table S1A. Performance in scaffold-grouped nested cross-validation.", [(None, nest)],
           "Mean ± SD over five repeats (seeds 0-4) of scaffold-grouped nested cross-validation (five outer folds, three inner folds) with hyperparameters "
           "selected by Bayesian optimization on the inner folds. A hit is a measured inhibition of at least 38% at 25 µM. The five repeats use different "
           f"partitions (mean adjusted Rand index between repeats {ari.ARI_all:.2f} for all 1,488 compounds and {ari.ARI_excluding_dominant:.2f} after excluding "
           "the dominant scaffold group of 544 compounds). The GNN always uses the molecular graph in addition to the listed feature block. Rerunning one repeat "
           "of the GNN with the same seed changed its R² from 0.35 to 0.51, so small differences between feature sets for the GNN are not interpreted.\n"
           "ᵃ Fraction of hits among the k compounds with the highest predictions.\nᵇ Precision at 50 divided by the hit rate of the library (39.6%).\n"
           "ᶜ The nearest-active-neighbor baseline yields a ranking only, so R², MAE and RMSE are not given.", textcols=2)
    cut = pd.read_csv(TAB / "hit_cutoff_sensitivity.csv")
    cut["model"] = cut["model"].replace({"Consensus(ET+Cat+GNN)": "Mean of three models", "Consensus(ET+Cat)": "Mean of Extra Trees and CatBoost",
                                         "NearestActiveNeighbour": "Nearest-active-neighbor baseline", "ExtraTrees": "Extra Trees"})
    for c in ("AUROC@30", "AUROC@38", "AUROC@50"):
        cut[c] = cut[c].str.replace("±", " ± ")
    cut = cut.rename(columns={"model": "Model", "AUROC@30": "AUROC at 30% cutoff", "AUROC@38": "AUROC at 38% cutoff", "AUROC@50": "AUROC at 50% cutoff",
                              "max_range": "Largest difference in mean AUROC"})
    _sheet(wb, "Hit cutoff sensitivity", "Table S1B. Sensitivity of AUROC to the hit cutoff.", [(None, cut)],
           "Mean ± SD over five repeats of nested scaffold-grouped cross-validation with ligand-only features (ECFP4 for the nearest-active-neighbor baseline). "
           "Hits were redefined at cutoffs of 30%, 38% (used throughout) and 50% without retraining the models.")
    agree = pd.read_csv(TAB / "model_agreement.csv").rename(columns={
        "Model pair (ligand-only)": "Model pair", "Spearman rho of predictions": "Spearman ρ of predictions",
        "Shared compounds among each model's top 50": "Shared compounds among the top 50 of each model"})
    budget = pd.read_csv(TAB / "same_budget_precision.csv").rename(columns={f"Precision at {k}": f"Precision at {k} ᵃ" for k in (10, 25, 50)})
    _sheet(wb, "Model agreement", "Table S1C. Agreement among the three models and comparison at the same budget.",
           [(None, agree), (None, budget)],
           "Mean ± SD over five repeats of scaffold-grouped nested cross-validation (outer-fold predictions; Bayesian-optimized models). Hits are compounds with a "
           "measured inhibition of at least 38%. Docking-only and IFP-only scorers use the corresponding feature block alone.\n"
           "ᵃ 39.6% of the library are hits, so precision is close to saturation for the learned models.")
    _sheet(wb, "Search spaces", "Table S1D. Hyperparameter search spaces.", [(None, pd.read_csv(TAB / "search_spaces.csv"))],
           "Ranges explored by Bayesian optimization on the inner folds (Gaussian-process surrogate, Matérn-5/2 kernel with white noise, expected improvement over "
           "384 random candidates). Budget per outer fold: 6 random and 14 Bayesian trials for Extra Trees and CatBoost, 3 random and 5 Bayesian trials for the GNN.",
           textcols=2)
    wb.save(TAB / "Supplementary_Table_S1.xlsx")
    print("Supplementary_Table_S1.xlsx written")


# ------------------------------------------------------------------------------------------------------------------ Table S2
def build_table_s2():
    wb = _book()
    alg = pd.read_csv(SEL / "Table_S2A_algorithm_survey.csv")
    _sheet(wb, "Algorithm survey", "Table S2A. Historical survey of 17 regression algorithms used to select the three models.", [(None, alg)],
           "Single scaffold split of the training library with the original feature set (activity-dependent IFP masking) and uniform Bayesian hyperparameter "
           "optimization. These scores come from the original masked pipeline and were used only to choose the three models carried forward; performance is "
           "reported in Table S1.", textcols=2)
    fs = {"ifp_only": "IFP-only", "ligand+dock+ifp": "Ligand + docking + IFP"}
    mdl = {"ExtraTrees": "Extra Trees"}
    pm = lambda r, k, d: f"{r[k + '_mean']:.{d}f} ± {r[k + '_std']:.{d}f}"
    a = pd.read_csv(cfg.OUTPUT / "masked_vs_unmasked_ablation_table.csv")
    a = a[a.feature_set.isin(fs)]
    a = pd.DataFrame({"Model": a.model.replace(mdl), "Feature set": a.feature_set.map(fs),
                      "IFP input": a.input.map({"unmasked": "Unmasked", "masked (historical diagnostic)": "Masked"}),
                      "R²": a.apply(pm, axis=1, k="R2", d=3), "MAE (% inhibition)": a.apply(pm, axis=1, k="MAE", d=1), "AUROC": a.apply(pm, axis=1, k="auroc", d=3)})
    a = a.assign(_o=a["IFP input"].eq("Masked")).sort_values(["_o", "Model", "Feature set"]).drop(columns="_o")
    b = pd.read_csv(cfg.OUTPUT / "masked_deployment_test_table.csv")
    b = b[b.setting.str.contains("REAL")]
    b = pd.DataFrame({"Model": b.model.replace(mdl), "Feature set": b.feature_set.map(fs),
                      "Setting": "Trained with masked IFP, tested on held-out compounds with their actual IFP",
                      "R² ᵃ": b.apply(pm, axis=1, k="R2", d=3), "AUROC": b.apply(pm, axis=1, k="auroc", d=3)}).sort_values(["Model", "Feature set"])
    _sheet(wb, "IFP masking diagnostic", "Table S2B. Effect of the activity-dependent IFP masking on cross-validated performance.",
           [("Masked and unmasked evaluation", a), ("Masked training, evaluation on compounds with their actual IFP", b)],
           "Mean ± SD over five repeats of scaffold-grouped nested cross-validation with random-search hyperparameters, so values differ slightly from Table S1A. "
           "Unmasked rows use the IFP block of every compound as calculated; in the masked (historical) rule the IFP block of every compound below the activity "
           "cutoff was set to zero, which depends on the measured outcome and inflates masked evaluation (AUROC close to 1.0).\n"
           "ᵃ A negative R² means that predictions are worse than predicting the mean inhibition of the training compounds.", textcols=2)
    wb.save(TAB / "Supplementary_Table_S2.xlsx")
    print("Supplementary_Table_S2.xlsx written")


# ------------------------------------------------------------------------------------------------------------------ Table S3
def build_table_s3():
    wb = _book()
    cs = pd.read_csv(TAB / "Supplementary_Table_class_shapley.csv").fillna("n/a")
    cs["Input"] = cs["Input"].map(lambda v: "Unmasked" if v == "Unmasked" else "Masked")
    cs = cs.rename(columns={"Input": "IFP", "Extra Trees": "Extra Trees (%)", "CatBoost": "CatBoost (%)", "GNN": "GNN (%)"})
    _sheet(wb, "Class-level Shapley", "Table S3A. Contribution of each feature class to the predictions of the three models.", [(None, cs)],
           "Class-level exact Shapley values (percent of the summed absolute contribution, mean ± SD) computed on held-out outer-fold compounds of one repeat (seed 0) "
           "for the ligand + docking + IFP models. Unmasked rows use the corrected evaluation; masked rows use the historical rule that set the IFP block of inactive "
           "compounds to zero, shown only as a diagnostic.", textcols=2)
    st = pd.read_csv(TAB / "Supplementary_Table_shap_fold_stability.csv")
    st = st.assign(Model=st.model.map({"extratrees_full": "Extra Trees", "catboost_full": "CatBoost", "gnn_full": "GNN"})).set_index("Model").loc[
        ["Extra Trees", "CatBoost", "GNN"]].reset_index()
    stab = pd.DataFrame({"Model": st.Model, "Mean Jaccard index of the top 15 features ᵃ": st.jaccard_top15_mean.round(2),
                         "Minimum Jaccard index": st.jaccard_min.round(2), "Mean Spearman ρ of all feature importances": st.spearman_all_mean.round(2),
                         "Overall top 15 features found in the top 15 of a fold (mean) ᵇ": st.overall_top15_in_fold_top15_mean.round(1),
                         "Features in the top 15 of all five folds": st.features_in_top15_of_all5, "IFP features in the top 15 of any fold": st.n_ifp_top15_any_fold})
    _sheet(wb, "Ranking stability", "Table S3B. Stability of the feature rankings across the five outer folds.", [(None, stab)],
           "Mean absolute SHAP value (TreeSHAP for Extra Trees and CatBoost, expected gradients for the GNN) of each feature in the held-out compounds of each of the "
           "five outer folds of one repeat (seed 0), for the unmasked ligand + docking + IFP models.\n"
           "ᵃ Jaccard index between the sets of 15 most important features of two folds, averaged over all pairs of folds.\n"
           "ᵇ Number of the 15 most important features of the pooled analysis that are also among the 15 most important features of an individual fold.")
    # tested compounds
    ex, outcome = _fp_result()
    md = pd.read_csv(TAB / "MD_summary_three_replicates.csv").assign(key=lambda d: d.compound.str.lower()).set_index("key")
    nn = pd.read_csv(TAB / "Nearest_training_neighbours_of_hits.csv")
    nn = nn[nn.neighbour_type == "any training compound"].assign(key=lambda d: d.hit.str.lower()).set_index("key")
    rk = pd.read_csv(TAB / "otava_corrected_ranking.csv").set_index("id")
    rows = []
    for _, r in ex.iterrows():
        c, k = r["Compound"], r["Compound"].lower()
        fda = k.startswith("desisobutyl")
        row = {"Rank in library": "2 (FDA GSRS-derived)" if fda else str(r["OTAVA rank"]), "Compound": c, "Source": "FDA GSRS-derived" if fda else "OTAVA",
               "Class": r["Class"], "Consensus prediction at selection (%) ᵃ": r["Consensus predicted inhibition (%)"],
               "Highest concentration tested (µM)": r["Highest concentration tested (µM)"],
               "Inhibition at highest concentration (%) ᵇ": r["Inhibition at highest concentration (%, mean ± SD, n = 3)"],
               "Inhibition at 25 µM (%) ᵇ": r["Inhibition at 25 µM (%, mean ± SD, n = 3)"],
               "IC50 (µM) ᶜ": IC50_PRISM.get(c.replace("Bortezomib", "bortezomib"), r["IC50 (µM)"])}
        has_md = k in md.index
        row["MD ligand RMSD (Å) ᵈ"] = f"{md.loc[k].rmsd_mean:.2f} ± {md.loc[k].rmsd_sd:.2f}" if has_md else ""
        row["MD interaction energy (kJ/mol) ᵈ"] = f"{md.loc[k].energy_mean:.0f} ± {md.loc[k].energy_sd:.0f}" if has_md else ""
        row["Persistent contacts ᵉ"] = CONTACTS.get(c.replace("Bortezomib", "bortezomib"), "")
        row["Nearest training compound"] = nn.loc[k].neighbour if k in nn.index else ""
        row["Tanimoto similarity ᶠ"] = f"{nn.loc[k].tanimoto:.2f}" if k in nn.index else ""
        row["Rank, mean of corrected models (of 1,909) ᵍ"] = int(rk.loc[c, "rank_Consensus(ET+Cat+GNN)"]) if c in rk.index else ""
        row["Rank, nearest-active-neighbor baseline (of 1,909) ᵍ"] = int(rk.loc[c, "rank_NearestActiveNeighbour"]) if c in rk.index else ""
        row["Inhibition of the nearest training compound (%)"] = nn.loc[k].neighbour_inhibition if k in nn.index else ""
        row["Same Murcko scaffold"] = ("yes" if nn.loc[k].same_scaffold else "no") if k in nn.index else ""
        row["Lot number"] = r.get("Lot number (OTAVA)", "")
        row["Purity by 1H NMR (%, vendor)"] = r.get("Purity by 1H NMR (%, vendor)", "")
        row["Supplier"] = r["Supplier / identifier"]
        rows.append(row)
    tested = pd.DataFrame(rows).fillna("")
    _sheet(wb, "Tested compounds", "Table S3C. Experimental and simulation results for the compounds tested by fluorescence polarization.", [(None, tested)],
           "The ten OTAVA compounds with the highest consensus predictions and one FDA GSRS-derived compound (rank 2 of the FDA GSRS-derived library) were tested. "
           "IC50 is given as > 50 µM when the compound did not reach 50% inhibition.\n"
           "ᵃ Mean of the three models used to select the compounds, before the corrected analysis.\n"
           "ᵇ Mean ± SD of three independent experiments (24 wells each); the value for P1157904 at 25 µM was interpolated from the fitted dose-response curve because "
           "this concentration was not tested.\nᶜ Mean ± SD of the IC50 values of three independent experiments.\n"
           "ᵈ Mean ± SD of three independent 100 ns simulations; interaction energy is the sum of the short-range Lennard-Jones and Coulomb energies between ligand and "
           "protein and is not a binding free energy. MD was run only for the FP-active compounds.\n"
           "ᵉ Residues with a ligand contact in at least 50% of the frames in each of the three simulations.\nᶠ ECFP4, 2,048 bits.\n"
           "ᵍ Rank among the 1,909 OTAVA compounds with the mean of the three corrected (unmasked, ligand-only) models and with the nearest-active-neighbor baseline "
           "(ECFP4); the analysis is retrospective because the outcomes were known.", textcols=3, wrap=False)
    # top-50 tables
    ot = pd.read_csv(SEL / "Table_S3_otava_top50_triple_consensus.csv")
    ot.columns = ["Rank", "Compound ID", "Plate", "Rack position", "Consensus predicted inhibition (%) ᵃ", "Extra Trees (%)", "CatBoost (%)", "GNN (%)", "SMILES"]
    ot.insert(2, "FP result ᵇ", ot["Compound ID"].map(outcome).fillna("Not tested"))
    ot = ot[["Rank", "Compound ID", "FP result ᵇ", "Consensus predicted inhibition (%) ᵃ", "Extra Trees (%)", "CatBoost (%)", "GNN (%)", "SMILES"]]
    note = ("Selection used the original pipeline of the three models; it is given as the historical record of how the tested compounds were chosen.\n"
            "ᵃ Mean of the predictions of the three models.\nᵇ Result of the fluorescence-polarization experiments (Table S3C).")
    _sheet(wb, "OTAVA top 50", "Table S3D. Fifty OTAVA compounds with the highest consensus predictions at selection.", [(None, ot)], note, textcols=2, wrap=False)
    fd = pd.read_csv(TAB / "fda_top50_audit.csv")
    fd = pd.DataFrame({"Rank": fd["rank"], "PubChem CID": fd["PubChem CID"], "Name": fd["Name"],
                       "Consensus predicted inhibition (%) ᵃ": fd["Consensus predicted GRP78 inhibition (mean)"],
                       "Extra Trees (%)": fd["Predicted GRP78 inhibition (Extra Trees)"], "CatBoost (%)": fd["Predicted GRP78 inhibition (CatBoost)"],
                       "GNN (%)": fd["Predicted GRP78 inhibition (GNN)"],
                       "FP result ᵇ": fd["PubChem CID"].map({73951493: "FP-active"}).fillna("Not tested"), "SMILES": fd["SMILES"]})
    _sheet(wb, "GSRS-derived top 50", "Table S3E. Fifty FDA GSRS-derived compounds with the highest consensus predictions at selection.", [(None, fd)], note,
           textcols=3, wrap=False)
    wb.save(TAB / "Supplementary_Table_S3.xlsx")
    print("Supplementary_Table_S3.xlsx written")


STAGES = {"table_s1": build_table_s1, "table_s2": build_table_s2, "table_s3": build_table_s3}

if __name__ == "__main__":
    for w in (sys.argv[1:] or list(STAGES)):
        print("=== stage", w, flush=True)
        STAGES[w]()
