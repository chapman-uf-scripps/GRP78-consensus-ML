"""
Figure 4: panels A and B (raw 3-experiment FP data, dose-response fits) and panels C and D (MD, 3 simulations per complex).
Module-level DATA and fit helpers are shared with analysis/supplementary_tables.py.

Run from the repository root:  python figures/figure4.py [panels_ab panels_cd combined]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402

import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

matplotlib.use("Agg")

PAN = cfg.FIGURES / "panels" / "figure4"
PAN.mkdir(parents=True, exist_ok=True)
FIG_SIZE = (7.2, 5.0)                                   # the four panels have the same canvas size
MARGINS = dict(left=0.15, right=0.97, top=0.97, bottom=0.16)
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold"})
DATA = {
    "P2195430": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [95.34884, 91.16279, 95.0],
            [86.74419, 85.46512, 86.62791],
            [60.0, 56.39535, 51.86047],
            [33.48837, 25.11628, 30.11628],
            [7.906977, 9.186047, 6.744186],
            [2.674419, 2.906977, 6.511628],
            [-3.255814, 4.069767, 2.55814],
            [-1.511628, -0.9302326, 3.837209],
        ],
        predicted=64.6,
    ),
    "P2195267": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [88.70968, 89.35484, 89.56989],
            [61.07527, 61.72043, 63.22581],
            [34.08602, 34.4086, 34.51613],
            [12.7957, 16.98925, 13.76344],
            [9.462365, 7.204301, 3.225806],
            [1.075269, 3.010753, -0.4301075],
            [-4.516129, 3.978495, 3.870968],
            [0.7526882, -0.9677419, 4.301075],
        ],
        predicted=63.0,
    ),
    "Desisobutyl-n-butyl Bortezomib": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [98.02325, 101.3953, 101.5116],
            [76.74419, 80.11628, 78.60465],
            [48.02325, 55.34884, 51.27907],
            [26.86047, 29.76744, 28.37209],
            [14.06977, 15.81395, 13.13953],
            [6.162791, 14.4186, 11.86047],
            [0.4651163, 7.790698, 8.488372],
            [4.534883, 5.813953, 5.813953],
        ],
        predicted=66.6,
    ),
    "P2195222": dict(
        logC=[-4, -4.60206, -5.20412, -5.80618, -6.40824, -7.0103, -7.61236, -8.21442],
        reps=[
            [86.7033, 87.14286, 81.97802],
            [55.82418, 60.76923, 56.37363],
            [31.75824, 24.61539, 36.48352],
            [10.54945, 7.472528, 16.7033],
            [3.076923, 3.846154, 6.153846],
            [0, 0.5494506, 0.8791209],
            [-2.747253, -0.2197802, 0.6593407],
            [-2.307692, 0.5494506, 2.197802],
        ],
        predicted=62.7,
    ),
    "P1157904": dict(
        logC=[-4.30103, -4.90309, -5.50515, -6.10721, -6.70927, -7.31133, -7.91339, -8.51545],
        reps=[
            [99.55882, 97.5, 96.61765],
            [61.91177, 56.32353, 66.91177],
            [23.08824, 26.32353, 24.26471],
            [13.97059, 15.44118, 5.0],
            [9.852942, 4.411765, 11.47059],
            [-2.5, 5.147059, 7.647059],
            [3.529412, 1.323529, 3.235294],
            [3.382353, -12.05882, 4.411765],
        ],
        predicted=61.8,
    ),
}


COL = {"P2195430": "#4C5BFF", "P2195267": "#4B1769", "P2195222": "#B933E8", "P1157904": "#A09A62", "Desisobutyl-n-butyl Bortezomib": "#FF8C8C"}
MK = {"P2195430": "o", "P2195267": "s", "P2195222": "^", "P1157904": "D", "Desisobutyl-n-butyl Bortezomib": "*"}
# 3-parameter logistic (top fixed at 100, bottom fixed at 0)
# Parameters: logIC50 (log10 of IC50 in M), hill
def hill_3pl(logC, logIC50, hill):
    return 100.0 / (1.0 + 10.0 ** ((logIC50 - logC) * hill))


def hill_4pl(logC, bottom, top, logIC50, hill):
    """4PL used only for plotting smooth curves; main IC50 comes from 3PL."""
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((logIC50 - logC) * hill))


def fit_dose_response(logC_list, mean_pct):
    """3-parameter logistic with top=100 and bottom=0 fixed (FP convention)."""
    p0 = [-5.5, 1.0]
    bounds = ([-9.0, 0.4], [-3.0, 2.5])
    try:
        popt, pcov = curve_fit(hill_3pl, logC_list, mean_pct,
                               p0=p0, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        # Return in 4PL format (bottom=0, top=100, logIC50, hill) for plotting
        return np.array([0.0, 100.0, popt[0], popt[1]]), np.array([0.0, 0.0, perr[0], perr[1]])
    except Exception as exc:
        print(f"  fit failed: {exc}")
        return np.array([0.0, 100.0, -5.5, 1.0]), np.array([np.nan]*4)


def measured_at_25uM(name, logC, mean_pct, fit_popt):
    """Return measured percent inhibition at 25 uM:
       - if 25 uM is one of the sampled concentrations (-4.60206), use the
         observed replicate mean directly
       - otherwise predict from the 4PL fit
    """
    target_logC = np.log10(25e-6)   # = -4.60206
    for lc, mp in zip(logC, mean_pct):
        if abs(lc - target_logC) < 1e-3:
            return float(mp), True
    return float(hill_4pl(target_logC, *fit_popt)), False


def sd_at_25uM(name, logC, reps):
    target_logC = np.log10(25e-6)
    # If 25 uM is one of the sampled concentrations, use the observed SD.
    for lc, rep in zip(logC, reps):
        if abs(lc - target_logC) < 1e-3:
            return float(np.std(rep, ddof=1))
    # Otherwise: fit 3PL to each replicate individually, read the predicted
    # value at 25 uM, return the SD across the 3 per-replicate fits.
    reps_arr = np.asarray(reps)               # shape (n_conc, 3)
    pred_at_25 = []
    for r in range(reps_arr.shape[1]):
        y_r = reps_arr[:, r]
        try:
            popt, _ = curve_fit(hill_3pl, logC, y_r,
                                p0=[-5.5, 1.0],
                                bounds=([-9.0, 0.4], [-3.0, 2.5]),
                                maxfev=20000)
            pred_at_25.append(float(hill_3pl(target_logC, *popt)))
        except Exception:
            pass
    if len(pred_at_25) >= 2:
        return float(np.std(pred_at_25, ddof=1))
    return np.nan


def run_panels_ab():
    """Panels A and B (FP dose response and predicted versus measured inhibition)."""
    rows, fits = [], {}
    for name, d in DATA.items():
        logC, reps = np.array(d["logC"]), np.array(d["reps"])
        mean = reps.mean(axis=1)
        popt, perr = fit_dose_response(logC, mean)
        fits[name] = popt
        m25, obs = measured_at_25uM(name, logC, mean, popt)
        rows.append(dict(compound=name, predicted=d["predicted"], ic50_uM=10 ** popt[2] / 1e-6, hill=popt[3], measured_25uM=m25,
                         sd_25uM=sd_at_25uM(name, logC, reps), observed_directly=obs))
    fit = pd.DataFrame(rows)
    fit.to_csv(cfg.TABLES / "Figure4_AB_source_table.csv", index=False)
    print(fit.to_string(index=False))


    def save(fig, nm):
        for ext in ("png", "pdf"):
            fig.savefig(PAN / f"{nm}.{ext}", dpi=300)
        plt.close(fig)


    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.linspace(-9, -3, 400)
    for name, d in DATA.items():
        logC, reps = np.array(d["logC"]), np.array(d["reps"])
        ax.plot(x, hill_4pl(x, *fits[name]), color=COL[name], lw=2.2, zorder=3)
        ax.errorbar(logC, reps.mean(axis=1), yerr=reps.std(axis=1, ddof=1), fmt=MK[name], color=COL[name], ms=8 if name[0] != "D" else 11,
                    mec="black", mew=0.8, capsize=3.5, elinewidth=1.2, zorder=5, label=name)
    ax.axvline(np.log10(25e-6), color="#5a6573", ls=":", lw=1.3)
    ax.text(np.log10(25e-6) + 0.08, 4, "25 µM", fontsize=13, color="#5a6573")
    ax.set_xlabel("log$_{10}$ [Compound] (M)", fontsize=15); ax.set_ylabel("GRP78 inhibition (%)", fontsize=15)
    ax.set_ylim(-12, 112); ax.set_xlim(-8.8, -3.6)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    fig.subplots_adjust(**MARGINS)
    save(fig, "Fig4_A_FP_dose_response")

    order = ["P2195430", "Desisobutyl-n-butyl Bortezomib", "P1157904", "P2195267", "P2195222"]
    f = fit.set_index("compound").loc[order]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    pos = np.arange(5); bw = 0.38
    ax.bar(pos - bw / 2, f.predicted, bw, color="#9AA3AD", label="Predicted", zorder=2)
    ax.bar(pos + bw / 2, f.measured_25uM, bw, color=[COL[c] for c in order], yerr=f.sd_25uM, capsize=4, label="Measured", zorder=2,
           error_kw={"elinewidth": 1.3})
    for i, (p, m, s) in enumerate(zip(f.predicted, f.measured_25uM, f.sd_25uM)):
        ax.text(i - bw / 2, p + 1.5, f"{p:.1f}", ha="center", fontsize=9.5)
        ax.text(i + bw / 2, m + s + 1.5, f"{m:.1f}" + chr(10) + f"±{s:.1f}", ha="center", fontsize=9.5, linespacing=0.95)
    ax.set_xticks(pos); ax.set_xticklabels([c.replace("Desisobutyl-n-butyl Bortezomib", "Desisobutyl-n-butyl" + chr(10) + "bortezomib") for c in order], fontsize=10)
    ax.set_ylabel("GRP78 inhibition at 25 µM (%)", fontsize=15); ax.set_ylim(0, 112)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.subplots_adjust(**MARGINS)
    save(fig, "Fig4_B_predicted_vs_measured")


def run_panels_cd():
    """Figure 4 panels C and D: ligand-heavy-atom RMSD and total protein-ligand interaction energy (Coulomb + Lennard-Jones) over 100 ns,"""
    matplotlib.use("Agg")

    SRC = cfg.MD_SIMULATIONS
    PAN = cfg.FIGURES / "panels" / "figure4"
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold"})
    NAMES = ["P1157904", "P2195222", "P2195267", "P2195430", "Desisobutyl-n-butyl bortezomib"]
    COL = {"P2195430": "#4C5BFF", "P2195267": "#4B1769", "P2195222": "#B933E8", "P1157904": "#A09A62", "Desisobutyl-n-butyl bortezomib": "#FF8C8C"}
    ORDER = ["P1157904", "P2195430", "P2195267", "P2195222", "Desisobutyl-n-butyl bortezomib"]


    def load(i):
        d = pd.read_csv(SRC / f"MD_Analysis_Summary_{i}.csv", header=None, skiprows=0 if i == 1 else 3).apply(pd.to_numeric, errors="coerce")
        return d


    runs = [load(i) for i in (1, 2, 3)]
    t = runs[0].iloc[:, 0].values
    RM = {n: np.stack([r.iloc[:, 2 + k].values[:len(t)] for r in runs]) for k, n in enumerate(NAMES)}
    EN = {n: np.stack([r.iloc[:, 20 + k].values[:len(t)] for r in runs]) for k, n in enumerate(NAMES)}

    rows = []
    for n in ORDER:
        rm_run, en_run = RM[n].mean(axis=1), EN[n].mean(axis=1)
        rows.append(dict(compound=n, rmsd_mean=rm_run.mean(), rmsd_sd=rm_run.std(ddof=1), energy_mean=en_run.mean(), energy_sd=en_run.std(ddof=1),
                         rmsd_runs=np.round(rm_run, 2).tolist(), energy_runs=np.round(en_run, 1).tolist()))
    S = pd.DataFrame(rows); S.to_csv(cfg.TABLES / "MD_summary_three_replicates.csv", index=False); print(S.round(2).to_string(index=False))


    def smooth(a, w=100):
        return pd.Series(a).rolling(w, center=True, min_periods=1).mean().values


    def panel(D, ylabel, name, ylim=None, legend_loc="upper center"):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for n in ORDER:
            sm = np.stack([smooth(x) for x in D[n]])
            m, s = sm.mean(0), sm.std(0, ddof=1)
            ax.plot(t, m, color=COL[n], lw=2.0, label=n, zorder=3)
            ax.fill_between(t, m - s, m + s, color=COL[n], alpha=0.18, lw=0, zorder=2)
        ax.set_xlabel("Time (ns)", fontsize=15); ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlim(0, 100)
        if ylim: ax.set_ylim(*ylim)
        ax.tick_params(labelsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5, ncol=2, loc=legend_loc, columnspacing=1.0, handlelength=1.4)
        fig.subplots_adjust(**MARGINS)
        for ext in ("png", "pdf"):
            fig.savefig(PAN / f"{name}.{ext}", dpi=300)
        plt.close(fig)


    panel(EN, "Interaction energy (kJ/mol)", "Fig4_C_interaction_energy", ylim=(-570, -170), legend_loc="lower center")
    panel(RM, "Ligand RMSD (Å)", "Fig4_D_RMSD", ylim=(0, 12), legend_loc="upper center")

    # C and D in one figure: stacked panels with a shared time axis and a single legend
    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True, gridspec_kw={"hspace": 0.10})
    for ax, D, ylabel, ylim in [(axes[0], EN, "Interaction energy" + chr(10) + "(kJ/mol)", (-490, -170)), (axes[1], RM, "Ligand RMSD" + chr(10) + "(Å)", (0, 9.5))]:
        for n in ORDER:
            sm = np.stack([smooth(x) for x in D[n]])
            m, sd = sm.mean(0), sm.std(0, ddof=1)
            ax.plot(t, m, color=COL[n], lw=1.8, label=n, zorder=3)
            ax.fill_between(t, m - sd, m + sd, color=COL[n], alpha=0.18, lw=0, zorder=2)
        ax.set_ylabel(ylabel, fontsize=13); ax.set_ylim(*ylim); ax.set_xlim(0, 100)
        ax.yaxis.set_label_coords(-0.095, 0.5)                    # both y labels on the same vertical line
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Time (ns)", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=10, columnspacing=1.2, handlelength=1.4, bbox_to_anchor=(0.56, 1.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(PAN / f"Fig4_CD_combined.{ext}", dpi=300)
    plt.close(fig)
    print("saved")


def run_combined():
    """Figure 4 as one figure at the final size (7.2 in wide): A dose response, B predicted versus measured, C interaction energy and D ligand RMSD, one font size and one legend."""
    SRC = cfg.MD_SIMULATIONS
    FS_LAB, FS_TICK, FS_TXT, FS_LEG = 8, 7, 6.2, 7.5
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold",
                         "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.linewidth": 0.7})
    MD_NAMES = ["P1157904", "P2195222", "P2195267", "P2195430", "Desisobutyl-n-butyl bortezomib"]
    ORD = ["P1157904", "P2195430", "P2195267", "P2195222", "Desisobutyl-n-butyl bortezomib"]
    COLORS = {"P2195430": "#4C5BFF", "P2195267": "#4B1769", "P2195222": "#B933E8", "P1157904": "#A09A62", "Desisobutyl-n-butyl bortezomib": "#FF8C8C"}
    MARK = {"P2195430": "o", "P2195267": "s", "P2195222": "^", "P1157904": "D", "Desisobutyl-n-butyl bortezomib": "*"}
    dkey = lambda n: "Desisobutyl-n-butyl Bortezomib" if n.startswith("Desis") else n

    fits, rows = {}, []
    for name, d in DATA.items():
        logC, reps = np.array(d["logC"]), np.array(d["reps"])
        popt, _ = fit_dose_response(logC, reps.mean(axis=1))
        fits[name] = popt
        m25, obs = measured_at_25uM(name, logC, reps.mean(axis=1), popt)
        rows.append(dict(compound=name, predicted=d["predicted"], measured_25uM=m25, sd_25uM=sd_at_25uM(name, logC, reps)))
    fit = pd.DataFrame(rows).set_index("compound")

    runs = [pd.read_csv(SRC / f"MD_Analysis_Summary_{i}.csv", header=None, skiprows=0 if i == 1 else 3).apply(pd.to_numeric, errors="coerce")
            for i in (1, 2, 3)]
    t = runs[0].iloc[:, 0].values
    RM = {n: np.stack([r.iloc[:, 2 + k].values[:len(t)] for r in runs]) for k, n in enumerate(MD_NAMES)}
    EN = {n: np.stack([r.iloc[:, 20 + k].values[:len(t)] for r in runs]) for k, n in enumerate(MD_NAMES)}
    smooth = lambda a, w=100: pd.Series(a).rolling(w, center=True, min_periods=1).mean().values

    fig = plt.figure(figsize=(7.2, 7.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.20, left=0.12, right=0.98, top=0.92, bottom=0.07)
    top = outer[0].subgridspec(1, 2, wspace=0.32, width_ratios=[1.0, 1.18])
    bot = outer[1].subgridspec(2, 1, hspace=0.10)
    axA, axB = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    axC = fig.add_subplot(bot[0]); axD = fig.add_subplot(bot[1], sharex=axC)

    # A: dose response
    x = np.linspace(-9, -3, 400)
    for n in ORD:
        d = DATA[dkey(n)]; logC, reps = np.array(d["logC"]), np.array(d["reps"])
        axA.plot(x, hill_4pl(x, *fits[dkey(n)]), color=COLORS[n], lw=1.3, zorder=3)
        axA.errorbar(logC, reps.mean(axis=1), yerr=reps.std(axis=1, ddof=1), fmt=MARK[n], color=COLORS[n], ms=3.6 if n[0] != "D" else 5.5,
                     mec="black", mew=0.4, capsize=1.6, elinewidth=0.7, zorder=5)
    axA.axvline(np.log10(25e-6), color="#5a6573", ls=":", lw=0.9)
    axA.text(np.log10(25e-6) + 0.08, 5, "25 µM", fontsize=FS_TXT + 0.5, color="#5a6573")
    axA.set_xlabel("log$_{10}$ [Compound] (M)", fontsize=FS_LAB, labelpad=2); axA.set_ylabel("GRP78 inhibition (%)", fontsize=FS_LAB, labelpad=2)
    axA.set_ylim(-12, 112); axA.set_xlim(-8.8, -3.6)

    # B: predicted versus measured inhibition at 25 uM
    order = ["P2195430", "Desisobutyl-n-butyl bortezomib", "P1157904", "P2195267", "P2195222"]
    pos, bw = np.arange(5), 0.36
    f = fit.loc[[dkey(n) for n in order]]
    axB.bar(pos - bw / 2, f.predicted, bw, color="#9AA3AD", zorder=2)
    axB.bar(pos + bw / 2, f.measured_25uM, bw, color=[COLORS[n] for n in order], yerr=f.sd_25uM, capsize=2, zorder=2, error_kw={"elinewidth": 0.8})
    for i, (pr, m, sd) in enumerate(zip(f.predicted, f.measured_25uM, f.sd_25uM)):
        axB.text(i - bw / 2, pr + 1.2, f"{pr:.1f}", ha="center", fontsize=FS_TXT)
        axB.text(i + bw / 2, m + sd + 1.2, f"{m:.1f}" + chr(10) + f"±{sd:.1f}", ha="center", fontsize=FS_TXT, linespacing=0.95)
    axB.set_xticks(pos)
    axB.set_xticklabels([("Desisobutyl-" + chr(10) + "n-butyl" + chr(10) + "bortezomib") if n.startswith("Desis") else n for n in order], fontsize=FS_TICK - 0.9)
    axB.set_ylabel("GRP78 inhibition at 25 µM (%)", fontsize=FS_LAB, labelpad=2); axB.set_ylim(0, 112)
    axB.grid(axis="y", alpha=0.25, zorder=0)

    # C and D: MD (mean of three simulations, band = SD between simulations)
    for ax, D, ylabel, ylim in [(axC, EN, "Interaction energy" + chr(10) + "(kJ/mol)", (-490, -170)), (axD, RM, "Ligand RMSD" + chr(10) + "(Å)", (0, 9.5))]:
        for n in ORD:
            sm = np.stack([smooth(v) for v in D[n]])
            m, sd = sm.mean(0), sm.std(0, ddof=1)
            ax.plot(t, m, color=COLORS[n], lw=1.1, zorder=3)
            ax.fill_between(t, m - sd, m + sd, color=COLORS[n], alpha=0.18, lw=0, zorder=2)
        ax.set_ylabel(ylabel, fontsize=FS_LAB, labelpad=2); ax.set_ylim(*ylim); ax.set_xlim(0, 100)
    axD.set_xlabel("Time (ns)", fontsize=FS_LAB, labelpad=2)
    plt.setp(axC.get_xticklabels(), visible=False)

    for ax in (axA, axB, axC, axD):
        ax.tick_params(width=0.7, length=2.5, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[n], lw=1.3, marker=MARK[n], ms=3.6 if n[0] != "D" else 5.5, mec="black", mew=0.4) for n in ORD]
    fig.legend(handles, ORD, loc="upper center", ncol=3, frameon=False, fontsize=FS_LEG, columnspacing=1.2, handlelength=2.0, bbox_to_anchor=(0.54, 1.0))
    fig.canvas.draw()
    for ax, letter, dx in [(axA, "A", 0.09), (axB, "B", 0.10), (axC, "C", 0.09)]:
        pb = ax.get_position()
        fig.text(pb.x0 - dx, pb.y1 + 0.012, letter, fontsize=13, fontweight="bold")
    for ext, dpi in (("png", 600), ("pdf", 300)):
        fig.savefig(PAN / f"Fig4_combined.{ext}", dpi=dpi)

    # the same panels as separate files (cropped from the combined figure, so the font sizes are identical)
    from matplotlib.transforms import Bbox
    fig.legends[0].set_visible(True)
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()

    def crop(artists, name, pad=0.06):
        bb = Bbox.union([a.get_tightbbox(renderer) if hasattr(a, "get_tightbbox") else a.get_window_extent(renderer) for a in artists]).transformed(inv)
        bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
        for ext, dpi in (("png", 600), ("pdf", 300)):
            fig.savefig(PAN / f"{name}.{ext}", dpi=dpi, bbox_inches=bb)

    for tx in list(fig.texts):
        tx.set_visible(False)                                   # panel letters are added by hand when the figure is assembled
    crop([axA], "Fig4_A_same_font")
    sel = [4, 0, 1, 2, 3]                                          # the long name goes first so that it sits above the curves
    leg_in = axA.legend([handles[i] for i in sel], [ORD[i] for i in sel], loc="upper left", fontsize=FS_TXT, frameon=False, handlelength=1.6, labelspacing=0.3,
                        borderaxespad=0.1, handletextpad=0.4)
    crop([axA], "Fig4_A_with_legend")                              # panel A with its own legend
    leg_in.remove()
    crop([axB], "Fig4_B_same_font")
    crop([axC, axD], "Fig4_C_same_font")
    leg_c = axD.legend(handles, ORD, loc="upper right", ncol=2, fontsize=FS_TXT, frameon=False, handlelength=1.8, labelspacing=0.3,
                       columnspacing=1.2, borderaxespad=0.1, handletextpad=0.4)
    crop([axC, axD], "Fig4_C_with_legend")                         # panel C with its own legend (inside the RMSD plot)
    leg_c.remove()
    crop([fig.legends[0]], "Fig4_legend_same_font")
    plt.close(fig)
    print("saved Fig4_combined and the separate panels")


STAGES = {"panels_ab": run_panels_ab, "panels_cd": run_panels_cd, "combined": run_combined}

if __name__ == "__main__":
    for w in (sys.argv[1:] or list(STAGES)):
        print("=== stage", w, flush=True)
        STAGES[w]()
