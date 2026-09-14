# -*- coding: utf-8 -*-
"""Figure 5.1 of the thesis: the deletion curves under zero masking, one panel
per representation.

Reads only results/faithfulness_curves.json (written by
figures/compute_faithfulness_curves.py; the per-seed scalars in it equal those
of results_multiseed_*.json). Writes fig_faithfulness_curves.png next to this
script. The fill dependence of the two readings is shown by figures/make_violins.py.

Usage (repo root):  python figures/make_faithfulness_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from wpaths import RESULTS  # noqa: E402

OUT = HERE
BLUE, ORANGE, BG = "#2a78d6", "#eb6834", "#fcfcfb"
INK, MUTED, GRID = "#222222", "#666666", "#e3e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})

ORDER = ["two_moons_best", "blobs_best", "iris_best", "breast_cancer_best",
         "digits_best", "har_best", "cifar10_best", "mnist_feats_best", "mnist"]
SHORT = {"two_moons_best": "Two Moons", "blobs_best": "Blobs", "iris_best": "Iris",
         "breast_cancer_best": "Breast Cancer", "digits_best": "Digits",
         "har_best": "HAR", "cifar10_best": "CIFAR-10 feat.",
         "mnist_feats_best": "MNIST feat.", "mnist": "MNIST pixels"}

with open(os.path.join(RESULTS, "faithfulness_curves.json"), encoding="utf-8") as f:
    CURVES = json.load(f)


def curves(entry, side, fill):
    """List of (seed, baseline, acc array) for non-empty curves."""
    out = []
    for s, c in CURVES[entry][side].items():
        cc = c[fill]
        if cc["n_used"] == 0 or not cc["acc"]:
            continue
        out.append((int(s), float(cc["baseline"]), np.asarray(cc["acc"], float)))
    return out


# --------------------------------------------------------------------------
# deletion curves, small multiples, log x (features masked)
# --------------------------------------------------------------------------
def fig_curves(fill="zero", name="fig_faithfulness_curves.png"):
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.2), sharey=True)
    for ax, entry in zip(axes.ravel(), ORDER):
        e = CURVES[entry]
        K, D = e["K"], e["D"]
        ax.set_xscale("symlog", linthresh=1, linscale=0.5)
        ax.axhline(1.0 / K, color=MUTED, lw=0.6, ls=":", zorder=1)
        # IDC: per-seed thin lines + median
        idc = curves(entry, "idc", fill)
        for _, b, acc in idc:
            x = np.arange(0, len(acc) + 1)
            ax.step(x, np.r_[b, acc], where="post", color=ORANGE, lw=0.7, alpha=0.35, zorder=2)
        if idc:
            n = max(len(a) for _, _, a in idc)
            med = []
            for k in range(0, n + 1):
                vals = [(b if k == 0 else acc[k - 1]) for _, b, acc in idc if len(acc) >= k]
                med.append(np.median(vals))
            ax.step(np.arange(0, n + 1), med, where="post", color=ORANGE, lw=1.6, zorder=4)
        # SpEx: identical over seeds -> one line
        sp = curves(entry, "spex", fill)
        if sp:
            _, b, acc = sp[0]
            x = np.arange(0, len(acc) + 1)
            ax.step(x, np.r_[b, acc], where="post", color=BLUE, lw=1.6, zorder=5)
            ax.plot(x, np.r_[b, acc], "o", color=BLUE, ms=2.5, zorder=6)
        ax.set_title(f"{SHORT[entry]}  (D={D}, K={K})", loc="left", fontweight="bold", pad=3)
        n_empty = len(e["idc"]) - len(idc)
        if n_empty:
            ax.text(0.98, 0.03, f"IDC: {n_empty}/{len(e['idc'])} seeds no gates",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=ORANGE)
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 1, 10, 100, 1000])
        ax.set_xticklabels(["0", "1", "10", "100", "1000"])
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y")
        ax.tick_params(length=2)
    axes[-1, 1].set_xlabel("features masked, cumulative in importance order (count, log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("clustering accuracy")
    handles = [Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=2.5, label="SpEx (tree; identical over seeds)"),
               Line2D([], [], color=ORANGE, lw=1.6, label="IDC median over seeds"),
               Line2D([], [], color=ORANGE, lw=0.7, alpha=0.5, label="IDC single seed"),
               Line2D([], [], color=MUTED, lw=0.6, ls=":", label="chance level 1/K")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_curves("zero")
    print("written to", OUT)
