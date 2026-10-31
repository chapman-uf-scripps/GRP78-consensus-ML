"""
Figure 5 panels (7.2 in wide, Arial 7-8 pt)

Stage: panels.

Run from the repository root:  python figures/figure5.py [stage ...]   (no stage = all stages in order)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg                                  # noqa: E402
from main_code.common import load_training                    # noqa: E402

import io
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import umap
from PIL import Image


def run_panels():
    """Figure 5 (old Figure 6): FP-active compounds and their nearest training compounds, redrawn."""
    matplotlib.use("Agg")

    PAN = cfg.FIGURES / "panels" / "figure5"
    PAN.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold"})
    OT, FD = "#159A9C", "#D99A25"

    nn = pd.read_csv(cfg.TABLES / "Nearest_training_neighbours_of_hits.csv")
    nn = nn[nn.neighbour_type == "any training compound"].set_index("hit")
    IC50 = {"P2195430": "4.3 (3.9-4.8)", "P1157904": "7.5 (6.1-9.1)", "P2195267": "12.8 (11.5-14.2)", "P2195222": "16.1 (14.4-18.0)",
            "Desisobutyl-n-butyl Bortezomib": "4.7 (3.8-5.9)"}
    ORDER = ["P1157904", "P2195430", "P2195267", "P2195222", "Desisobutyl-n-butyl Bortezomib"]

    tr = load_training()
    tra = tr[tr.A5 >= 38].reset_index(drop=True)


    def fp1024(s):
        a = np.zeros(1024, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024), a)
        return a


    X = np.vstack([np.stack([fp1024(s) for s in tra.SMILES]), np.stack([fp1024(nn.loc[h, "hit_smiles"]) for h in ORDER])]).astype(bool)
    xy = umap.UMAP(n_neighbors=35, min_dist=0.12, metric="jaccard", random_state=42, low_memory=True).fit_transform(X)
    txy, hxy = xy[:len(tra)], xy[len(tra):]
    print("training actives", len(tra))


    def mol_img(smi, w=640, h=400):
        m = Chem.MolFromSmiles(smi)
        d = rdMolDraw2D.MolDraw2DCairo(w, h)
        o = d.drawOptions(); o.bondLineWidth = 2; o.baseFontSize = 0.62
        d.DrawMolecule(m); d.FinishDrawing()
        return Image.open(io.BytesIO(d.GetDrawingText()))

    _src = pd.read_csv(cfg.TABLES / "Figure4_AB_source_table.csv").set_index("compound")
    INH = {h: f"{_src.loc[h, 'measured_25uM']:.1f}%" for h in ORDER}
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8})
    # ---------------- A
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.scatter(txy[:, 0], txy[:, 1], s=5, c="#C8CED4", alpha=0.7, edgecolor="none", zorder=2)
    for i, h in enumerate(ORDER):
        ax.scatter(hxy[i, 0], hxy[i, 1], s=45, c=FD if h.startswith("Desis") else OT, edgecolor="black", lw=0.8, zorder=6)
    LAB = {"P1157904": (0.6, 0.45, "P1157904"), "P2195430": (1.6, 1.0, "P2195430"), "P2195267": (2.2, -0.25, "P2195267"),
           "P2195222": (2.5, -1.0, "P2195222"), "Desisobutyl-n-butyl Bortezomib": (0.8, 0.9, "Desisobutyl-n-butyl bortezomib")}
    for i, h in enumerate(ORDER):
        dx, dy, t = LAB[h]
        ax.annotate(t, xy=hxy[i], xytext=(hxy[i, 0] + dx, hxy[i, 1] + dy), fontsize=8, fontweight="bold", color=FD if h.startswith("Desis") else OT,
                    arrowprops=dict(arrowstyle="-", color="#555", lw=0.5), zorder=7)
    ax.set_xlabel("UMAP-1", fontsize=8, fontweight="bold", labelpad=1); ax.set_ylabel("UMAP-2", fontsize=8, fontweight="bold", labelpad=1)
    ax.tick_params(labelsize=7, width=0.6, length=2.5, pad=1.5)
    for l_ in ax.get_xticklabels() + ax.get_yticklabels(): l_.set_fontweight("bold")
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_linewidth(0.6)
    hd = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color="#C8CED4"), plt.Line2D([], [], marker="o", ls="", ms=5, mfc=OT, mec="black", mew=0.6),
          plt.Line2D([], [], marker="o", ls="", ms=5, mfc=FD, mec="black", mew=0.6)]
    ax.legend(hd, ["Training actives (n = 589)", "OTAVA, FP-active", "Approved-drug-derived, FP-active"], loc="lower left", frameon=False,
              prop={"size": 7, "weight": "bold"}, handletextpad=0.3, labelspacing=0.25)
    for ext, dpi in (("png", 600), ("pdf", 300)):
        fig.savefig(PAN / f"Fig5_A_final_size.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ---------------- B
    fig = plt.figure(figsize=(7.2, 3.3))
    gB = fig.add_gridspec(2, 5, hspace=0.12, wspace=0.03)
    for k, h in enumerate(ORDER):
        for r, (smi, t1, t2) in enumerate([
                (nn.loc[h, "hit_smiles"], h if not h.startswith("Desis") else "Desisobutyl-n-butyl" + chr(10) + "bortezomib", f"Inhibition {INH[h]}"),
                (nn.loc[h, "neighbour_smiles"], nn.loc[h, "neighbour"], f"Inhibition {nn.loc[h, 'neighbour_inhibition']:.1f}%" + chr(10) + f"Tanimoto {nn.loc[h, 'tanimoto']:.2f}")]):
            a = fig.add_subplot(gB[r, k]); a.imshow(mol_img(smi)); a.axis("off")
            a.set_title(t1, fontsize=8, fontweight="bold", color=(FD if h.startswith("Desis") else OT) if r == 0 else "#333", pad=1, linespacing=0.9)
            a.text(0.5, 0.02, t2, transform=a.transAxes, ha="center", va="top", fontsize=7, fontweight="normal", color="#1f6b3a" if r == 0 else "#333", linespacing=1.0)
    for ext, dpi in (("png", 600), ("pdf", 300)):
        fig.savefig(PAN / f"Fig5_B_final_size.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved separate final-size panels")


STAGES = {
    "panels": run_panels,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    for w in wanted:
        print("=== stage", w, flush=True)
        STAGES[w]()
