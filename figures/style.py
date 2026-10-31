"""Figure style shared by the supplementary figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("ExtraTrees", "Extra Trees", "#1E9AA0"), ("CatBoost", "CatBoost", "#D99A25"), ("GNN", "GNN", "#7B5FB5")]


def setup(**extra):
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "normal", "axes.spines.top": False, "axes.spines.right": False, **extra})


def save(fig, outdir, name, dpi=300):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
