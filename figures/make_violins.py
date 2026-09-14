# -*- coding: utf-8 -*-
"""Violin plots of Section 5.3 (Faithfulness under zero versus column-mean fill).

Data: results/faithfulness_curves.json (seeds 1-10, both fills, both methods).
Per representation and reading, two violins for IDC's ten seeds and the
seed-constant SpEx value as a bar. Writes, next to this script:

    fig_violin_main.png       HAR and Two Moons (Figure 5.2)
    fig_violin_appendix.png   the remaining seven representations (Figure A.3)

Usage (repo root):  python figures/make_violins.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results", "faithfulness_curves.json")

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 8, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": False, "legend.frameon": False, "legend.fontsize": 7.5, "pdf.fonttype": 42,
})

ORDER = [("har_best", "HAR"), ("two_moons_best", "Two Moons"), ("blobs_best", "Gaussian Blobs"),
         ("iris_best", "Iris"), ("breast_cancer_best", "Breast Cancer"), ("digits_best", "Digits"),
         ("cifar10_best", "CIFAR-10 (feat.)"), ("mnist_feats_best", "MNIST (feat.)"), ("mnist", "MNIST (pixels)")]
MAIN = ["har_best", "two_moons_best"]

raw = json.load(open(RES, encoding="utf-8"))


def values(key, side, fill, met):
    e = raw[key][side]
    out = []
    for s in e:
        if not s.isdigit():
            continue
        v = e[s][fill].get(met)
        if v is not None:
            out.append(float(v))
    return out


def panel(ax, key, name, met, show_ylabel):
    # x positions: 0 = zero fill, 1 = column-mean fill
    for xi, fill in ((0, "zero"), (1, "mean")):
        idc = values(key, "idc", fill, met)
        spx = values(key, "spex", fill, met)
        if idc:
            parts = ax.violinplot([idc], positions=[xi], widths=0.62, showmeans=False,
                                  showmedians=False, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(ORANGE)
                b.set_edgecolor(ORANGE)
                b.set_alpha(0.28 if fill == "zero" else 0.14)
                b.set_linewidth(1.0)
                if fill == "mean":
                    b.set_hatch("////")
            ax.scatter(np.full(len(idc), xi) + np.random.default_rng(0).uniform(-0.09, 0.09, len(idc)), idc,
                       s=9, color=ORANGE, edgecolor=SURFACE, linewidth=0.4, zorder=4)
            ax.plot([xi - 0.22, xi + 0.22], [np.median(idc)] * 2, color=ORANGE, lw=1.6, zorder=5)
        if spx:
            sv = float(np.mean(spx))
            ax.plot([xi - 0.31, xi + 0.31], [sv, sv], color=BLUE, lw=2.2, zorder=6,
                    solid_capstyle="round", linestyle="-" if fill == "zero" else (0, (3, 2)))
    ax.axhline(0, color=AXIS, lw=0.7, zorder=1)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["zero fill", "column-mean fill"])
    ax.set_ylim(-0.12, 0.72)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis='y', labelleft=show_ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"{name}: {'top-1 drop' if met == 'top1drop' else 'AOPC'}", fontsize=8.2, color=INK2, loc="left", pad=5)


def legend(fig, y=1.0):
    handles = [Line2D([], [], color=BLUE, lw=2.2, label="SpEx, zero fill (seed-constant)"),
               Line2D([], [], color=BLUE, lw=2.2, linestyle=(0, (3, 2)), label="SpEx, column-mean fill"),
               Patch(facecolor=ORANGE, alpha=0.28, edgecolor=ORANGE, label="IDC, zero fill (ten seeds, median as bar)"),
               Patch(facecolor=ORANGE, alpha=0.14, edgecolor=ORANGE, hatch="////", label="IDC, column-mean fill")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, y), handletextpad=0.6, columnspacing=1.6)


def fig_main():
    keys = [(k, n) for k, n in ORDER if k in MAIN]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7), sharey=True)
    i = 0
    for k, n in keys:
        for met in ("top1drop", "aopc"):
            panel(axes[i], k, n, met, show_ylabel=(i == 0))
            i += 1
    axes[0].set_ylabel("accuracy drop", color=INK2, fontsize=7.5)
    legend(fig, 1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(HERE, "fig_violin_main.png"), dpi=220)
    plt.close(fig)


def fig_appendix():
    """Two representations per row, i.e. four columns, so that the figure is
    almost square and fills the text width instead of staying narrow and tall."""
    keys = [(k, n) for k, n in ORDER if k not in MAIN]
    rows = (len(keys) + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(7.4, 2.2 * rows), sharey=True)
    used = set()
    for idx, (k, n) in enumerate(keys):
        r, base = idx // 2, (idx % 2) * 2
        for off, met in enumerate(("top1drop", "aopc")):
            c = base + off
            panel(axes[r][c], k, n, met, show_ylabel=(c == 0))
            used.add((r, c))
    # the last occupied row of each column keeps its x labels, the rows above do not
    last_row = {c: max(r for (r, cc) in used if cc == c) for c in range(4)}
    for (r, c) in used:
        if r < last_row[c]:
            axes[r][c].set_xticklabels([])
    for r in range(rows):
        if (r, 0) in used:
            axes[r][0].set_ylabel("accuracy drop", color=INK2, fontsize=7.5)
    for r in range(rows):
        for c in range(4):
            if (r, c) not in used:
                axes[r][c].set_axis_off()
    legend(fig, 1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(HERE, "fig_violin_appendix.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_main()
    fig_appendix()
    print("written to", HERE)
