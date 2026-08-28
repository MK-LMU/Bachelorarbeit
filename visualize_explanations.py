# -*- coding: utf-8 -*-
"""Proposal working step 3: visualize and interpret the explanations of SpEx
and IDC on the low-dimensional and classical tabular datasets.

Outputs (PNG, 300 dpi, light chart chrome) into notes/figures/:
  <ds>_partition.png   2-D datasets: spectral reference | SpEx tree partition
                       with its axis-parallel cuts drawn | IDC clusters (tuned)
  <ds>_heatmap.png     tabular: per-sample explanation matrices, SpEx |SHAP|
                       vs IDC gates, one shared sample order -> the
                       piecewise-constant-vs-per-sample contrast is visible
  <ds>_tree.png        SpEx tree as a rule diagram with real feature names
                       (thresholds in MinMax-scaled units)

Colors follow the validated reference palette of the dataviz method:
categorical slots in fixed order for cluster identity (slot 4 yellow is
skipped in scatter/all-pairs use — documented yellow-beside-orange failure —
identity is additionally direct-labeled), one blue light->dark ramp for
magnitude, text in ink tokens, hairline axes.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SpEx"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

from sklearn.datasets import load_iris, load_breast_cancer
from spex_pipeline import spex_side

import wpaths
FIGDIR = wpaths.FIGURES

# ---- reference palette (dataviz skill, light mode) ----
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#008300",
       "#4a3aa7", "#e34948", "#eda100"]          # slots 1,2,3,5,6,7,8,4
SEQ = LinearSegmentedColormap.from_list("seq", [
    "#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b"])
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.grid": False, "font.size": 9,
})


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_cuts(ax, cut, xmin, xmax, ymin, ymax):
    """Recursively draw the tree's axis-parallel cuts, clipped to their region."""
    if cut.left is None:
        return
    if cut.coordinate == 0:
        ax.plot([cut.threshold, cut.threshold], [ymin, ymax],
                color=INK, lw=1.4, zorder=3)
        draw_cuts(ax, cut.left, xmin, cut.threshold, ymin, ymax)
        draw_cuts(ax, cut.right, cut.threshold, xmax, ymin, ymax)
    else:
        ax.plot([xmin, xmax], [cut.threshold, cut.threshold],
                color=INK, lw=1.4, zorder=3)
        draw_cuts(ax, cut.left, xmin, xmax, ymin, cut.threshold)
        draw_cuts(ax, cut.right, xmin, xmax, cut.threshold, ymax)


def scatter_clusters(ax, X, labels, title):
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(X[m, 0], X[m, 1], s=9, lw=0, alpha=0.85,
                   color=CAT[int(c) % len(CAT)], zorder=2)
        cx, cy = X[m, 0].mean(), X[m, 1].mean()
        ax.text(cx, cy, str(int(c)), fontsize=11, fontweight="bold",
                color=INK, ha="center", va="center", zorder=4,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)])
    ax.set_title(title, fontsize=10, color=INK)
    style_ax(ax)


def align_to_ref(labels, ref, K):
    """One-to-one relabel via Hungarian matching on the overlap matrix, so the
    same blob keeps the same color across panels (color follows the entity)."""
    from scipy.optimize import linear_sum_assignment
    Kp = max(K, int(labels.max()) + 1)
    ov = np.zeros((Kp, Kp))
    for c in range(Kp):
        for r in range(Kp):
            ov[c, r] = np.sum((labels == c) & (ref == r))
    rows, cols = linear_sum_assignment(-ov)
    mapping = {int(c): int(r) for c, r in zip(rows, cols)}
    return np.array([mapping[int(c)] for c in labels])


def fig_partition(ds, idc_npz):
    d = np.load(os.path.join(wpaths.IDC_OUT, idc_npz))
    X = np.ascontiguousarray(d["X"], float)
    y, K = d["y_true"].astype(int), int(d["K"])
    tree, spex_labels, _, ref = spex_side(X, K, y, seed=0, return_ref=True)
    spex_labels = align_to_ref(spex_labels, ref, K)
    idc_labels = align_to_ref(d["labels_pred"].astype(int), ref, K)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    scatter_clusters(axes[0], X, ref, "Spectral reference clustering")
    scatter_clusters(axes[1], X, spex_labels,
                     "SpEx tree partition (cuts drawn)")
    pad = 0.03
    draw_cuts(axes[1], tree.tree.root, X[:, 0].min() - pad, X[:, 0].max() + pad,
              X[:, 1].min() - pad, X[:, 1].max() + pad)
    scatter_clusters(axes[2], X, idc_labels, "IDC clusters (tuned config)")
    for ax in axes:
        ax.set_xlabel("feature 0 (MinMax)", fontsize=8)
    axes[0].set_ylabel("feature 1 (MinMax)", fontsize=8)
    out = os.path.join(FIGDIR, f"{ds}_partition.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def fig_heatmap(ds, idc_npz, feature_names, max_feats=None):
    d = np.load(os.path.join(wpaths.IDC_OUT, idc_npz))
    X = np.ascontiguousarray(d["X"], float)
    y, K = d["y_true"].astype(int), int(d["K"])
    _, spex_labels, spex_gates = spex_side(X, K, y, seed=0)
    idc_gates = np.ascontiguousarray(d["gates"], float)

    # one shared sample order: by true class, then by SpEx leaf within class
    order = np.lexsort((spex_labels, y))
    ys = y[order]
    feats = np.arange(X.shape[1])
    if max_feats is not None and X.shape[1] > max_feats:
        score = spex_gates.mean(0) + idc_gates.mean(0)
        feats = np.sort(np.argsort(score)[::-1][:max_feats])

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 1.4 + 0.42 * len(feats)),
                             constrained_layout=True, sharex=True)
    panels = [("SpEx |SHAP|  —  cell = how much this feature (row) contributed "
               "to this sample's (column) cluster assignment", spex_gates,
               "|SHAP| value"),
              ("IDC gates  —  cell = learned feature selector, 0 = feature off, "
               "1 = fully used for this sample", idc_gates, "gate value")]
    class_bounds = np.where(np.diff(ys) != 0)[0]
    for ax, (title, G, cbar_label) in zip(axes, panels):
        M = G[order][:, feats].T
        vmax = M.max() if M.max() > 0 else 1.0
        im = ax.imshow(M, aspect="auto", cmap=SEQ, vmin=0, vmax=vmax,
                       interpolation="nearest")
        for b in class_bounds:
            ax.axvline(b + 0.5, color=INK, lw=0.8)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([feature_names[f] for f in feats], fontsize=7)
        ax.set_title(title, fontsize=8.5, loc="left", color=INK, pad=18)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        cb.set_label(cbar_label, fontsize=7, color=INK2)
        cb.ax.tick_params(labelsize=6, color=MUTED)
        cb.outline.set_edgecolor(AXIS)
        style_ax(ax)
    # class labels centered above their block, on a tick-free secondary axis
    edges = np.concatenate([[0], class_bounds + 1, [len(ys)]])
    sec = axes[0].secondary_xaxis("top")
    sec.set_xticks((edges[:-1] + edges[1:]) / 2)
    sec.set_xticklabels([f"class {ys[e]}" for e in edges[:-1]],
                        fontsize=7, color=INK2)
    sec.tick_params(length=0)
    sec.spines["top"].set_visible(False)
    axes[1].set_xlabel("samples, sorted by true class (vertical bars = class "
                       "boundaries); color scales are per panel, not comparable "
                       "across panels", fontsize=8)
    out = os.path.join(FIGDIR, f"{ds}_heatmap.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def fig_tree(ds, idc_npz, feature_names, class_names):
    d = np.load(os.path.join(wpaths.IDC_OUT, idc_npz))
    X = np.ascontiguousarray(d["X"], float)
    y, K = d["y_true"].astype(int), int(d["K"])
    tree, _, _ = spex_side(X, K, y, seed=0)
    root = tree.tree.root

    def n_leaves(c):
        return 1 if c.left is None else n_leaves(c.left) + n_leaves(c.right)

    def depth(c):
        return 1 if c.left is None else 1 + max(depth(c.left), depth(c.right))

    total_w, total_d = n_leaves(root), depth(root)
    fig, ax = plt.subplots(figsize=(1.0 + 2.2 * total_w, 0.8 + 1.15 * total_d),
                           constrained_layout=True)
    ax.set_xlim(0, total_w); ax.set_ylim(-total_d - 0.3, 0.5); ax.axis("off")

    def draw(c, idx, x0, dep):
        w = n_leaves(c)
        xc, yc = x0 + w / 2.0, -dep
        if c.left is None:
            maj = np.bincount(y[idx], minlength=len(class_names)).argmax() \
                if idx.size else 0
            fc = CAT[int(maj) % len(CAT)]
            ax.text(xc, yc, f"{class_names[maj]}\nn={idx.size}",
                    ha="center", va="center", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=fc, lw=1.6))
            return xc, yc
        rule = f"{feature_names[c.coordinate]}\n≤ {c.threshold:.3f} ?"
        ax.text(xc, yc, rule, ha="center", va="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=AXIS, lw=1.0))
        m = X[idx, c.coordinate] <= c.threshold
        for child, cidx, cx0, lab in ((c.left, idx[m], x0, "yes"),
                                      (c.right, idx[~m], x0 + n_leaves(c.left), "no")):
            xch, ych = draw(child, cidx, cx0, dep + 1)
            ax.annotate("", xy=(xch, ych + 0.28), xytext=(xc, yc - 0.28),
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=1.0))
            ax.text((xc + xch) / 2, (yc + ych) / 2, lab, fontsize=7,
                    color=MUTED, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none"))
        return xc, yc

    draw(root, np.arange(len(X)), 0.0, 0)
    ax.set_title(f"SpEx decision tree — {ds} (thresholds in MinMax units; "
                 f"leaves show majority true class)", fontsize=10, color=INK)
    out = os.path.join(FIGDIR, f"{ds}_tree.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def fig_importance_space(ds, idc_npz):
    """Explanations shown IN the 2-D data space: each point colored by how
    important feature 0 / feature 1 is for it — the most direct way to read
    'what does the explanation say, and where'."""
    d = np.load(os.path.join(wpaths.IDC_OUT, idc_npz))
    X = np.ascontiguousarray(d["X"], float)
    y, K = d["y_true"].astype(int), int(d["K"])
    _, _, spex_gates = spex_side(X, K, y, seed=0)
    idc_gates = np.ascontiguousarray(d["gates"], float)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2), constrained_layout=True,
                             sharex=True, sharey=True)
    rows = [("SpEx |SHAP|", spex_gates, "|SHAP| value"),
            ("IDC gate", idc_gates, "gate value")]
    for r, (mname, G, cbar_label) in enumerate(rows):
        vmax = G.max() if G.max() > 0 else 1.0
        for f in range(2):
            ax = axes[r, f]
            sc = ax.scatter(X[:, 0], X[:, 1], c=G[:, f], cmap=SEQ, vmin=0,
                            vmax=vmax, s=8, lw=0)
            ax.set_title(f"{mname} for feature {f}", fontsize=10, color=INK)
            style_ax(ax)
        cb = fig.colorbar(sc, ax=axes[r, :], shrink=0.85, pad=0.01)
        cb.set_label(cbar_label, fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, color=MUTED)
        cb.outline.set_edgecolor(AXIS)
    for ax in axes[1]:
        ax.set_xlabel("feature 0 (MinMax)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("feature 1 (MinMax)", fontsize=8)
    fig.suptitle(f"Where is which feature important? — {ds}: dark = this "
                 f"feature matters for this sample's cluster assignment",
                 fontsize=10, color=INK)
    out = os.path.join(FIGDIR, f"{ds}_importance_space.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def spex_leaf_paths(tree, X, labels):
    """Root-to-leaf rule paths of the SpEx tree: {cluster_id: (path, member_idx)}
    with path = [(feature, threshold, op)], op '<=' (go left / must be LOW) or
    '>' (go right / must be HIGH)."""
    paths = {}

    def walk(cut, idx, path):
        if cut.left is None:
            paths[int(np.bincount(labels[idx]).argmax())] = (path, idx)
            return
        m = X[idx, cut.coordinate] <= cut.threshold
        walk(cut.left, idx[m], path + [(int(cut.coordinate), float(cut.threshold), "<=")])
        walk(cut.right, idx[~m], path + [(int(cut.coordinate), float(cut.threshold), ">")])

    walk(tree.tree.root, np.arange(len(X)), [])
    return paths


def fig_digits_pixels(tag=""):
    """Digits (8x8): the explanations shown as what they ARE. SpEx row = the
    tree's RULE PATH per cluster (markers on the queried pixels: blue = must be
    dark, orange = must be light) over the cluster's mean digit; IDC row = gate
    strength overlaid on the cluster's mean digit, used clusters only.
    tag='_best' renders the tuned-IDC variant for the before/after pair."""
    from matplotlib.patches import Rectangle, Patch
    d = np.load(os.path.join(wpaths.IDC_OUT, f"idc_out_digits{tag}_seed0.npz"))
    X = np.ascontiguousarray(d["X"], float)
    y, K = d["y_true"].astype(int), int(d["K"])
    tree, spex_labels, _ = spex_side(X, K, y, seed=0)
    idc_gates = np.ascontiguousarray(d["gates"], float)
    idc_labels = d["labels_pred"].astype(int)

    BLUE, ORANGE = CAT[0], CAT[1]
    paths = spex_leaf_paths(tree, X, spex_labels)

    def majority(idx):
        cnt = np.bincount(y[idx], minlength=10)
        dig = int(cnt.argmax())
        return dig, cnt.max() / max(1, idx.size)

    def bg(ax, idx):
        img = X[idx].mean(0).reshape(8, 8)
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=img.max() * 1.8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GRID)

    spex_order = sorted(paths, key=lambda c: majority(paths[c][1])[0])
    idc_used = sorted([c for c in np.unique(idc_labels)],
                      key=lambda c: majority(np.where(idc_labels == c)[0])[0])

    fig, axes = plt.subplots(2, K, figsize=(1.2 * K, 3.6),
                             constrained_layout=True)
    # ---- row A: SpEx rule paths ----
    for col, c in enumerate(spex_order):
        ax = axes[0, col]
        path, idx = paths[c]
        dig, pur = majority(idx)
        bg(ax, idx)
        for f, thr, op in path:
            r, cc = divmod(f, 8)
            ax.add_patch(Rectangle((cc - 0.5, r - 0.5), 1, 1, fill=False, lw=2.2,
                                   edgecolor=BLUE if op == ">" else ORANGE))
        ax.set_title(f"'{dig}' · {pur:.0%} · n={idx.size}", fontsize=7.5, color=INK)
    axes[0, 0].set_ylabel("SpEx: rule path\nof the tree", fontsize=8, color=INK2)

    # ---- row B: IDC gates, used clusters only ----
    for col in range(K):
        ax = axes[1, col]
        if col < len(idc_used):
            m = idc_labels == idc_used[col]
            idx = np.where(m)[0]
            dig, pur = majority(idx)
            bg(ax, idx)
            g8 = idc_gates[m].mean(0).reshape(8, 8)
            ax.imshow(g8, cmap=SEQ, vmin=0, vmax=max(g8.max(), 1e-9),
                      alpha=(g8 / max(g8.max(), 1e-9)) * 0.8)
            ax.set_title(f"'{dig}' · {pur:.0%} · n={idx.size}", fontsize=7.5,
                         color=INK)
        elif col == len(idc_used):
            why = "(default-config\ncollapse)" if not tag else \
                  "(even tuned, IDC does\nnot use all of them)"
            ax.text(0.5, 0.5, f"{K - len(idc_used)} of {K} clusters\nunused\n"
                    f"{why}", ha="center", va="center",
                    fontsize=8, color=INK2, transform=ax.transAxes)
            ax.axis("off")
        else:
            ax.axis("off")
    axes[1, 0].set_ylabel("IDC: opened\ngates", fontsize=8, color=INK2)

    fig.legend(handles=[
        Patch(facecolor="none", edgecolor=BLUE, lw=2.2,
              label="tree question: patch must be DARK"),
        Patch(facecolor="none", edgecolor=ORANGE, lw=2.2,
              label="tree question: patch must be LIGHT"),
        Patch(facecolor="#3987e5", alpha=0.8, label="IDC: blue intensity = gate open")],
        loc="lower center", ncol=3, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Digits — what the explanations actually say: background = "
                 "mean digit of the cluster; title = "
                 "majority digit · purity · size", fontsize=9.5, color=INK)
    out = os.path.join(FIGDIR, f"digits_pixel_importance{tag}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_mnist_pixels():
    """MNIST (28x28), paper-style: ONE sample per digit, per-sample explanations
    on the sample's own image. Four rows:
      1  IDC's top-15 gates (the paper's |S|=15 view) — IDC runs its VALIDATED
         config here, so the gates are sparse AND per-sample (the open-gate count
         is printed per panel).
      2  the converter's output: |SHAP| for the sample's assigned cluster, size
         and colour encoding the magnitude. This is the row the whole bridge
         exists for.
      3  the tree's rule path for the same sample (blue = must be dark,
         orange = must be light).
      4  the full SpEx tree the marks in row 3 come from.
    The fair high-dimensional counterpart to the Digits figure."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    d = np.load(os.path.join(wpaths.IDC_OUT, "idc_out_mnist_seed0.npz"))
    X = np.ascontiguousarray(d["X"], float)
    y, K = d["y_true"].astype(int), int(d["K"])
    idc_gates = np.ascontiguousarray(d["gates"], float)
    tree, spex_labels, spex_gates = spex_side(X, K, y, seed=0)
    paths = spex_leaf_paths(tree, X, spex_labels)

    BLUE, ORANGE = CAT[0], CAT[1]
    samples = [int(np.where(y == dig)[0][0]) for dig in range(10)]

    fig = plt.figure(figsize=(12.5, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 10, height_ratios=[1, 1, 1, 1.75])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(10)]
                     for r in range(3)])
    ax_tree = fig.add_subplot(gs[3, :])

    def base(ax, img):
        # lighter strokes so the importance overlays carry the contrast
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=img.max() * 1.6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GRID)

    for col, i in enumerate(samples):
        img = X[i].reshape(28, 28)
        n_open = int((idc_gates[i] > 0).sum())
        top = np.argsort(idc_gates[i])[::-1][:15]
        top = top[idc_gates[i][top] > 0]

        ax = axes[0, col]
        base(ax, img)
        ax.scatter(top % 28, top // 28, s=22, marker="s", lw=0.4,
                   color=BLUE, edgecolors=SURFACE)
        ax.set_title(f"'{y[i]}' · {n_open} gates", fontsize=7.5, color=INK)

        # middle row: the ACTUAL TreeSHAP converter output, |SHAP| magnitudes —
        # size AND ramp-color encode the contribution (double-encoded for pop)
        ax = axes[1, col]
        base(ax, img)
        nz = np.where(spex_gates[i] > 0)[0]
        v = spex_gates[i][nz] / max(spex_gates[i].max(), 1e-12)
        ax.scatter(nz % 28, nz // 28, s=30 + 90 * v, marker="s",
                   color=SEQ(0.45 + 0.55 * v), edgecolors=SURFACE, lw=0.6)
        ax.set_title(f"{len(nz)} pixels · max {spex_gates[i].max():.2f}",
                     fontsize=7.5, color=INK)

        ax = axes[2, col]
        base(ax, img)
        path, _ = paths[int(spex_labels[i])]
        for f, thr, op in path:
            ax.scatter([f % 28], [f // 28], s=52, marker="s", facecolors="none",
                       edgecolors=BLUE if op == ">" else ORANGE, lw=1.8)
        ax.set_title(f"{len(path)} tree questions", fontsize=7.5, color=INK)

    axes[0, 0].set_ylabel("IDC: top-15\ngates (|S|=15)", fontsize=8, color=INK2)
    axes[1, 0].set_ylabel("SpEx: |SHAP|\n(converter output)", fontsize=8, color=INK2)
    axes[2, 0].set_ylabel("SpEx: rule path\nof the tree", fontsize=8, color=INK2)

    # ---- row 4: the FULL SpEx decision tree (the 9 questions the rows above
    # mark on the samples), leaves = majority digit + purity + size ----
    root = tree.tree.root

    def n_leaves(c):
        return 1 if c.left is None else n_leaves(c.left) + n_leaves(c.right)

    total_w = n_leaves(root)
    ax_tree.set_xlim(0, total_w)
    ax_tree.set_ylim(-5.4, 0.5)
    ax_tree.axis("off")

    def draw(c, idx, x0, dep):
        w = n_leaves(c)
        xc, yc = x0 + w / 2.0, -dep * 1.15
        if c.left is None:
            maj = int(np.bincount(y[idx], minlength=10).argmax()) if idx.size else 0
            pur = np.bincount(y[idx], minlength=10).max() / max(1, idx.size)
            ax_tree.text(xc, yc, f"'{maj}'\n{pur:.0%} · n={idx.size}",
                         ha="center", va="center", fontsize=6.8, color=INK,
                         bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE,
                                   ec=CAT[maj % len(CAT)], lw=1.6))
            return xc, yc
        r_, c_ = divmod(int(c.coordinate), 28)
        ax_tree.text(xc, yc, f"Px({r_},{c_})\n≤ {c.threshold:.2f}?",
                     ha="center", va="center", fontsize=6.8, color=INK,
                     bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=AXIS, lw=1.0))
        m = X[idx, int(c.coordinate)] <= c.threshold
        for child, cidx, cx0, lab in ((c.left, idx[m], x0, "light"),
                                      (c.right, idx[~m], x0 + n_leaves(c.left), "dark")):
            xch, ych = draw(child, cidx, cx0, dep + 1)
            ax_tree.annotate("", xy=(xch, ych + 0.34), xytext=(xc, yc - 0.34),
                             arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.9))
            ax_tree.text((xc + xch) / 2, (yc + ych) / 2, lab, fontsize=6,
                         color=ORANGE if lab == "light" else BLUE,
                         ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none"))
        return xc, yc

    draw(root, np.arange(len(X)), 0.0, 0)
    ax_tree.set_title("The full SpEx tree: 9 pixel questions for 10,000 "
                      "samples (leaves = majority digit · purity · size; "
                      "edge color = answer light/dark)", fontsize=8.5,
                      color=INK, loc="left")

    fig.legend(handles=[
        Line2D([], [], marker="s", ls="", color=BLUE, ms=7,
               label="IDC: gate open (top-15, validated config)"),
        Line2D([], [], marker="s", ls="", color="#1c5cab", ms=9,
               label="SpEx: size+color = |SHAP| contribution"),
        Patch(facecolor="none", edgecolor=BLUE, lw=1.8,
              label="tree question: must be DARK"),
        Patch(facecolor="none", edgecolor=ORANGE, lw=1.8,
              label="tree question: must be LIGHT")],
        loc="lower center", ncol=4, fontsize=7.5, frameon=False,
        bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("MNIST — per-sample explanations, ONE example per digit: "
                 "IDC's gates vs. SpEx's Tree-SHAP attributions vs. the "
                 "underlying tree rules", fontsize=9.5, color=INK)
    out = os.path.join(FIGDIR, "mnist_pixel_importance.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    fig_partition("two_moons", "idc_out_two_moons_tuned_seed0.npz")
    fig_partition("blobs", "idc_out_blobs_tuned_seed0.npz")
    fig_importance_space("two_moons", "idc_out_two_moons_tuned_seed0.npz")
    fig_digits_pixels()
    fig_digits_pixels(tag="_best")
    fig_mnist_pixels()

    iris = load_iris()
    fig_heatmap("iris", "idc_out_iris_seed0.npz", iris.feature_names)
    fig_tree("iris", "idc_out_iris_seed0.npz", iris.feature_names,
             list(iris.target_names))

    bc = load_breast_cancer()
    fig_heatmap("breast_cancer", "idc_out_breast_cancer_seed0.npz",
                bc.feature_names, max_feats=10)
    fig_tree("breast_cancer", "idc_out_breast_cancer_seed0.npz",
             bc.feature_names, list(bc.target_names))


if __name__ == "__main__":
    main()
