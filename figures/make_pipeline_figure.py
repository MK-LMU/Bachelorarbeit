# -*- coding: utf-8 -*-
"""Schematic at the start of Chapter 4: SpEx tree -> converter -> Tree SHAP ->
importance heatmap Z, IDC gates -> Z, both -> the same metric code, with the
validation chain under the converter. Needs no data.

Writes fig_pipeline.png next to this script.

Usage (repo root):  python figures/make_pipeline_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_pipeline.png")

BLUE, ORANGE, INK, INK2, MUTED, LINE, SURF, GREEN = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb", "#3a9d5d"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.facecolor": SURF, "savefig.facecolor": SURF})

fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, title, sub="", color=INK2, fill="#ffffff", lw=1.2, fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=color, lw=lw))
    ax.text(x + w / 2, y + h / 2 + (2.1 if sub else 0), title, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center", fontsize=7.6, color=INK2, linespacing=1.25)


def arrow(x1, y1, x2, y2, color=INK2, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, linestyle=ls))


# ---- lanes
ax.text(1, 46.5, "SpEx (rule paths)", color=BLUE, fontsize=9.5, fontweight="bold")
ax.text(1, 18.5, "IDC (feature gates)", color=ORANGE, fontsize=9.5, fontweight="bold")

# SpEx lane (top)
box(1, 33, 17, 10, "Spectral Reference", "clustering of X\n(seed enters here)", color=BLUE)
box(21, 33, 17, 10, "SpEx Tree", "K leaves, axis-parallel cuts\nrule path per leaf", color=BLUE)
box(41, 33, 18, 10, "Converter", "children, split feature,\nthreshold, value vector,\nnode sample weights", color=BLUE, fill="#eef5ff", lw=1.6)
box(62, 33, 16, 10, "Tree SHAP", "path-dependent,\n|φ| of assigned cluster", color=BLUE)
arrow(18.4, 38, 20.6, 38, BLUE); arrow(38.4, 38, 40.6, 38, BLUE); arrow(59.4, 38, 61.6, 38, BLUE)

# IDC lane (bottom)
box(1, 5, 17, 10, "Autoencoder", "embedding of X", color=ORANGE)
box(21, 5, 17, 10, "Gating Network", "gate vector g ∈ [0,1]ᴰ\nper sample", color=ORANGE)
box(41, 5, 18, 10, "Clustering Head", "K clusters on the\ngated embedding", color=ORANGE)
box(62, 5, 16, 10, "Gate Heatmap", "rows = samples,\ncolumns = features", color=ORANGE)
arrow(18.4, 10, 20.6, 10, ORANGE); arrow(38.4, 10, 40.6, 10, ORANGE); arrow(59.4, 10, 61.6, 10, ORANGE)

# shared object and metrics
box(80.5, 19, 18.5, 12, "Importance Heatmap Z", "N × D, non-negative\none row per sample", color=INK, fill="#f3f2ee", lw=1.8, fs=9.0)
arrow(78.4, 38, 80.5, 29, BLUE); arrow(78.4, 10, 80.5, 21, ORANGE)
box(80.5, 3, 18.5, 9, "Same Metric Code", "faithfulness, diversity,\nuniqueness, generalizability", color=INK, fill="#ffffff")
arrow(89.75, 18.6, 89.75, 12.4, INK)

# validation chain under the converter
ax.add_patch(FancyBboxPatch((41, 20.5), 35, 9, boxstyle="round,pad=0.4,rounding_size=1.0", fc="#f0faf0", ec=GREEN, lw=1.2))
ax.text(58.5, 27.6, "Validating the Pipeline", ha="center", va="center", fontsize=8.4, color=GREEN, fontweight="bold")
ax.text(58.5, 23.4, "equivalence · node weights · additivity" + chr(10) + "brute-force Shapley · production wiring",
        ha="center", va="center", fontsize=7.4, color=INK2, linespacing=1.3)
arrow(50, 32.6, 50, 29.4, GREEN, lw=1.0, ls="--")

# labels at the merge
ax.text(73.5, 46.0, "rule path →\nattribution row", fontsize=7.4, color=BLUE, ha="center", va="center")
ax.text(73.5, 17.5, "gate vector →\nselection row", fontsize=7.4, color=ORANGE, ha="center", va="center")

fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=220)
print("written", OUT)
