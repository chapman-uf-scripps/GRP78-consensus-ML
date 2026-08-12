"""
Figure 8 top panel: 2D structures of the five confirmed actives.

Sources of SMILES:
    OTAVA hits (4): figure5b_otava_top10_consensus_ranking.csv
    FDA hit  (1):   fda_corrected_docking_top3_extra_catboost_gnn_top200_hits.csv
                    or the predictions CSV.
"""

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\19294\PyCharmMiscProject\GRP78_Publication_Package")
SRC  = ROOT / "output" / "source_tables"
MAIN = ROOT / "output" / "main_figures"

# --- Pull SMILES for the 5 confirmed actives ---
otava = pd.read_csv(SRC / "figure5b_otava_top10_consensus_ranking.csv")
otava_actives = otava[otava["status"] == "Confirmed active"].copy()

fda_full = pd.read_csv(
    ROOT / "output" / "full_pipeline_output"
         / "11c_screen_fda_kevin_docking_corrected_20260605"
         / "fda_bayesian_top3_extra_catboost_gnn_predictions.csv"
)
fda_hit = fda_full[fda_full["Compound_CID"] == 73951493].iloc[0]

# Compose ordered hit list (consistent with Table 1 / Table 2 ordering)
hits = [
    ("P2195430",                            otava_actives.loc[otava_actives["COMPOUND ID"] == "P2195430", "SMILES"].iloc[0],
     "4.3 ± 0.02 µM"),
    ("P1157904",                            otava_actives.loc[otava_actives["COMPOUND ID"] == "P1157904", "SMILES"].iloc[0],
     "7.5 ± 0.04 µM"),
    ("P2195267",                            otava_actives.loc[otava_actives["COMPOUND ID"] == "P2195267", "SMILES"].iloc[0],
     "12.8 ± 0.2 µM"),
    ("P2195222",                            otava_actives.loc[otava_actives["COMPOUND ID"] == "P2195222", "SMILES"].iloc[0],
     "16.1 ± 0.2 µM"),
    ("Desisobutyl-n-butyl\nBortezomib",     fda_hit["canonical_smiles"],
     "4.7 ± 0.04 µM"),
]

# --- Render each compound to a panel image with a label strip below ---
TILE_W, TILE_H = 360, 360
LABEL_STRIP_H  = 90
total_w = TILE_W * len(hits)
total_h = TILE_H + LABEL_STRIP_H
canvas  = Image.new("RGB", (total_w, total_h), "white")

# Try to load a clean font; fall back to default if Arial isn't found
def load_font(size, bold=False):
    candidates = ["arialbd.ttf", "arial.ttf", "Arial Bold.ttf", "Arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()

font_name = load_font(20, bold=True)
font_ic50 = load_font(18, bold=True)

for i, (name, smi, ic50_label) in enumerate(hits):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for {name}: {smi}")
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(TILE_W, TILE_H), kekulize=True)
    canvas.paste(img, (i * TILE_W, 0))

    # Label strip beneath this tile
    strip = Image.new("RGB", (TILE_W, LABEL_STRIP_H), "white")
    draw  = ImageDraw.Draw(strip)
    # Compound name (centred, multi-line OK)
    text_lines = name.split("\n")
    y_cursor = 8
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font_name)
        w = bbox[2] - bbox[0]
        draw.text(((TILE_W - w) / 2, y_cursor), line, fill="black", font=font_name)
        y_cursor += (bbox[3] - bbox[1]) + 4
    # IC50 line in a slightly larger green-ish colour
    # (plain "IC50" avoids the missing-glyph rendering issue in Arial)
    ic50_text = f"IC50 = {ic50_label}"
    bbox = draw.textbbox((0, 0), ic50_text, font=font_ic50)
    w = bbox[2] - bbox[0]
    draw.text(((TILE_W - w) / 2, LABEL_STRIP_H - (bbox[3] - bbox[1]) - 8),
              ic50_text, fill="#1a5d2c", font=font_ic50)

    canvas.paste(strip, (i * TILE_W, TILE_H))

# Save outputs
out_png = MAIN / "Figure_8_confirmed_actives_structures.png"
out_pdf = MAIN / "Figure_8_confirmed_actives_structures.pdf"
canvas.save(out_png, dpi=(300, 300))
# Save a PDF by converting the PIL image
canvas.save(out_pdf)
print("Wrote", out_png)
print("Wrote", out_pdf)
