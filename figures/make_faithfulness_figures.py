# -*- coding: utf-8 -*-
"""Figures 5.1 and A.2-style companion of the thesis: the deletion curves and
the fill dependence of the drop-based Faithfulness readings.

Reads only results/faithfulness_curves.json (written by
figures/compute_faithfulness_curves.py; the per-seed scalars in it equal those
of results_multiseed_*.json). Writes, next to this script:

    fig_faithfulness_curves.png   deletion curves under zero masking, one panel
                                  per representation (Figure 5.1)
    fig_faithfulness_fill.png     top-1 drop and AOPC under zero versus
                                  column-mean fill, paired per representation

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


def scalar(entry, side, fill, key):
    return np.array([c[fill][key] for c in CURVES[entry][side].values()
                     if c[fill]["n_used"] > 0 and c[fill][key] is not None], float)


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


# --------------------------------------------------------------------------
# fill dependence, paired dumbbells (zero -> mean) for top-1 drop and AOPC
# --------------------------------------------------------------------------
def fig_fill_dumbbell(name="fig_faithfulness_fill.png"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), sharey=True)
    ys = np.arange(len(ORDER))[::-1]
    for ax, key, ttl in zip(axes, ["top1drop", "aopc"], ["top-1 drop", "AOPC"]):
        ax.axvline(0, color=MUTED, lw=0.6)
        for y, entry in zip(ys, ORDER):
            for side, col, dy in [("spex", BLUE, +0.18), ("idc", ORANGE, -0.18)]:
                z = scalar(entry, side, "zero", key)
                m = scalar(entry, side, "mean", key)
                if len(z) == 0:
                    continue
                if side == "idc":
                    ax.plot(z, np.full_like(z, y + dy + 0.06), "|", color=col, ms=4, mew=0.6, alpha=0.5)
                    ax.plot(m, np.full_like(m, y + dy - 0.06), "|", color=col, ms=4, mew=0.6, alpha=0.5)
                zc, mc = np.median(z), np.median(m)
                ax.plot([zc, mc], [y + dy, y + dy], "-", color=col, lw=1.2, zorder=2)
                ax.plot([zc], [y + dy], "o", color=col, ms=5, zorder=3)
                ax.plot([mc], [y + dy], "o", color=BG, mec=col, mew=1.2, ms=5, zorder=3)
        ax.set_title(ttl, loc="left", fontweight="bold")
        ax.set_xlim(-0.08, 0.72)
        ax.set_xlabel("accuracy drop (baseline minus masked)")
        ax.grid(True, axis="x")
        ax.tick_params(length=2)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([SHORT[k] for k in ORDER])
    handles = [Line2D([], [], color=BLUE, lw=1.2, label="SpEx"),
               Line2D([], [], color=ORANGE, lw=1.2, label="IDC (median of seeds; ticks = seeds)"),
               Line2D([], [], color=INK, marker="o", ls="none", ms=5, label="zero fill (published)"),
               Line2D([], [], color=BG, mec=INK, mew=1.2, marker="o", ls="none", ms=5, label="column-mean fill")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_curves("zero")
    fig_fill_dumbbell()
    print("written to", OUT)
