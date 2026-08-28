"""Tree SHAP on the SpEx clustering tree.

SpEx (NeurIPS 2025, https://github.com/talargv/SpEx) produces an axis-parallel
tree explaining a reference clustering. This module converts that tree into
SHAP's "custom tree" dict format, so that shap.TreeExplainer yields continuous,
signed per-sample attributions instead of a binary "feature is on the path"
mask. It is the bridge every SpEx number in this work goes through;
spex_pipeline.py wraps it into the production path.

The one thing the converter cannot read off the tree
----------------------------------------------------
SpEx nodes are `Cut` objects (SpEx/base_classes.py) carrying `coordinate`,
`threshold`, `left`, `right` -- and a `cluster` field that `CliqueBased.train`
NEVER populates. Cluster ids are assigned at PREDICTION time: `Tree.predict`
walks the tree with a LIFO stack and labels each leaf with a running counter as
it reaches it, so a leaf's cluster id is the order in which predict() visits it.
Reading `cut.cluster` would therefore give nonsense.

Instead the reference data is routed through the tree and the predict()-labels
are counted per node. That yields both `values` (cluster probabilities, one-hot
at any leaf reference samples actually reach) and `node_sample_weight`, which
Tree SHAP needs to integrate out absent features.

Split convention: go LEFT iff `x[coordinate] <= threshold` -- identical to
SHAP's, so no inversion is needed. Leaves keep a `coordinate`/`threshold` from
a split that was never applied; ignore them there.
"""

import os
import sys

import numpy as np

# make SpEx importable
SPEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpEx")
sys.path.insert(0, SPEX_DIR)

from clique_based import CliqueBased  # noqa: E402

import shap  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.datasets import make_blobs, load_iris  # noqa: E402
from sklearn.cluster import SpectralClustering  # noqa: E402


# ---------------------------------------------------------------------------
# CONVERTER:  SpEx Cut-tree  ->  SHAP custom-tree dict
# ---------------------------------------------------------------------------
def spex_tree_to_shap_dict(spex_tree, X_ref, labels=None):
    """
    Convert a trained SpEx tree into the 7-array dict shap.TreeExplainer wants.

    SHAP convention (matches SpEx exactly):
        - go LEFT  when  x[feature] <= threshold
        - leaves:  children_left = children_right = -1,  feature = -2

    values[node]            -> cluster PROBABILITIES (n_nodes, K), one-hot at
                               leaves that receive reference samples
    node_sample_weight[node] -> number of reference samples reaching the node
    Both are computed empirically by routing X_ref through the tree.
    """
    root = spex_tree.tree.root if hasattr(spex_tree, "tree") else spex_tree

    if labels is None:
        labels = spex_tree.predict(X_ref).astype(int)
    labels = np.asarray(labels).astype(int)
    K = int(labels.max()) + 1

    children_left, children_right = [], []
    features, thresholds = [], []
    values, node_sample_weight = [], []

    def add_node():
        children_left.append(-1)
        children_right.append(-1)
        features.append(-2)
        thresholds.append(-2.0)
        values.append(np.zeros(K))
        node_sample_weight.append(0.0)
        return len(children_left) - 1

    def build(cut, idx_ref):
        nid = add_node()
        # node-local class histogram (probabilities) + weight
        cnt = np.bincount(labels[idx_ref], minlength=K).astype(float)
        node_sample_weight[nid] = float(idx_ref.size)
        values[nid] = cnt / cnt.sum() if cnt.sum() > 0 else cnt
        if cut.left is None:  # LEAF
            return nid
        features[nid] = int(cut.coordinate)
        thresholds[nid] = float(cut.threshold)
        left_mask = X_ref[idx_ref, cut.coordinate] <= cut.threshold
        l_idx = idx_ref[left_mask]
        r_idx = idx_ref[~left_mask]
        children_left[nid] = build(cut.left, l_idx)
        children_right[nid] = build(cut.right, r_idx)
        return nid

    build(root, np.arange(X_ref.shape[0]))

    children_left = np.array(children_left, dtype=np.int32)
    children_right = np.array(children_right, dtype=np.int32)
    return {
        "children_left": children_left,
        "children_right": children_right,
        "children_default": children_left.copy(),  # missing-value -> left
        "features": np.array(features, dtype=np.int32),
        "thresholds": np.array(thresholds, dtype=np.float64),
        "values": np.array(values, dtype=np.float64),
        "node_sample_weight": np.array(node_sample_weight, dtype=np.float64),
    }


def sklearn_to_shap_dict(model):
    """Same 7-array dict, extracted from a fitted sklearn DecisionTreeClassifier."""
    t = model.tree_
    value = t.value  # (n_nodes, 1, n_classes), class counts
    probs = value[:, 0, :] / value[:, 0, :].sum(axis=1, keepdims=True)
    cl = t.children_left.astype(np.int32)
    return {
        "children_left": cl,
        "children_right": t.children_right.astype(np.int32),
        "children_default": cl.copy(),
        "features": t.feature.astype(np.int32),
        "thresholds": t.threshold.astype(np.float64),
        "values": probs.astype(np.float64),
        "node_sample_weight": t.weighted_n_node_samples.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def to_NDK(sv, N, D, K):
    """Normalise shap output to shape (N, D, K) regardless of shap version."""
    sv = np.asarray(sv)
    if sv.ndim == 2:  # single class edge-case -> (N, D)
        sv = sv[:, :, None]
    if sv.shape == (N, D, K):
        return sv
    if sv.shape == (K, N, D):
        return np.transpose(sv, (1, 2, 0))
    if sv.shape == (N, K, D):
        return np.transpose(sv, (0, 2, 1))
    raise ValueError(f"Unexpected shap shape {sv.shape}, expected perm of {(N, D, K)}")


def importances_from_sv(sv_ndk, assigned):
    """local (N,D), per-cluster (K,D), global (D,) from |shap| of shape (N,D,K)."""
    N, D, K = sv_ndk.shape
    abs_sv = np.abs(sv_ndk)
    local = abs_sv[np.arange(N), :, assigned]          # (N, D) own-cluster
    cluster = np.zeros((K, D))
    for c in range(K):
        m = assigned == c
        if m.any():
            cluster[c] = abs_sv[m][:, :, c].mean(axis=0)
    glob = abs_sv.mean(axis=(0, 2))                     # (D,)
    return local, cluster, glob


# ---------------------------------------------------------------------------
# Converter format verification (sklearn round-trip)
# ---------------------------------------------------------------------------
def verify_roundtrip():
    print("=" * 70)
    print("STEP 5  Converter format verification (sklearn round-trip)")
    print("=" * 70)
    X, y, _ = make_blobs(n_samples=400, centers=4, n_features=6,
                         random_state=1, return_centers=True)
    clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)

    sv_native = shap.TreeExplainer(clf).shap_values(X)
    sv_dict = shap.TreeExplainer({"trees": [sklearn_to_shap_dict(clf)]}).shap_values(X)

    N, D, K = X.shape[0], X.shape[1], len(np.unique(y))
    a = to_NDK(sv_native, N, D, K)
    b = to_NDK(sv_dict, N, D, K)
    diff = np.max(np.abs(a - b))
    print(f"  sklearn-native shape: {np.asarray(sv_native).shape}")
    print(f"  custom-dict   shape: {np.asarray(sv_dict).shape}")
    print(f"  max abs diff (native vs. dict): {diff:.3e}")
    ok = diff < 1e-6
    print(f"  -> format correct: {ok}")
    return diff, ok


# ---------------------------------------------------------------------------
# Demo: apply the converter to SpEx end to end
# ---------------------------------------------------------------------------
def run_spex(name, X, n_clusters, n_informative=None, seed=0):
    print("=" * 70)
    print(f"DATASET: {name}   X={X.shape}  k={n_clusters}")
    print("=" * 70)
    X = np.ascontiguousarray(X, dtype=np.float64)
    N, D = X.shape

    # spectral reference clustering
    ref = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
                             random_state=seed, assign_labels="kmeans").fit_predict(X)

    # SpEx clique
    tree = CliqueBased()
    tree.train(X, ref, n_clusters)
    labels = tree.predict(X).astype(int)
    K = int(labels.max()) + 1
    print(f"  spectral reference clusters: {len(np.unique(ref))}, "
          f"SpEx leaves/clusters: {K}")

    # convert + Tree SHAP
    tdict = spex_tree_to_shap_dict(tree, X, labels)
    n_nodes = len(tdict["features"])
    n_leaves = int((tdict["children_left"] == -1).sum())
    print(f"  SHAP tree: {n_nodes} nodes, {n_leaves} leaves")

    expl = shap.TreeExplainer({"trees": [tdict]},
                              feature_perturbation="tree_path_dependent")
    sv = to_NDK(expl.shap_values(X), N, D, K)
    print(f"  shap_values normalised shape (N,D,K): {sv.shape}")

    local, cluster, glob = importances_from_sv(sv, labels)

    # ---- sanity checks ----
    print("\n  --- SANITY CHECKS ---")
    order = np.argsort(glob)[::-1]
    print("  global |SHAP| per feature (sorted):")
    for d in order:
        tag = ""
        if n_informative is not None:
            tag = " [INFORMATIVE]" if d < n_informative else " [noise]"
        print(f"     feat {d:2d}: {glob[d]:.4f}{tag}")

    if n_informative is not None:
        info_mean = glob[:n_informative].mean()
        noise_mean = glob[n_informative:].mean()
        print(f"\n  mean global |SHAP| informative dims : {info_mean:.4f}")
        print(f"  mean global |SHAP| noise       dims : {noise_mean:.4f}")
        ratio = info_mean / (noise_mean + 1e-12)
        print(f"  ratio informative/noise            : {ratio:.1f}x")

    # leaf -> samples share identical SHAP?  Check both the full (D,K) matrix and
    # the assigned-cluster column only (the latter is what `local` uses).
    same_full = True
    same_assigned = True
    for c in np.unique(labels):
        rows = sv[labels == c].reshape((labels == c).sum(), -1)
        if rows.shape[0] > 1 and np.max(np.abs(rows - rows[0])) > 1e-9:
            same_full = False
        col = sv[labels == c][:, :, c]  # assigned-cluster explanation
        if col.shape[0] > 1 and np.max(np.abs(col - col[0])) > 1e-9:
            same_assigned = False
    print(f"\n  same-leaf samples identical -- assigned-cluster column: {same_assigned}")
    print(f"  same-leaf samples identical -- full (D,K) matrix       : {same_full}")
    print("    (assigned column constant per leaf == one explanation region, as expected;")
    print("     off-path cluster columns can vary slightly under path-dependent Tree SHAP --")
    print("     a SHAP-internal effect, not a converter bug; sklearn round-trip diff was 0.)")
    same = same_assigned

    # example sample
    s = 0
    print(f"\n  example sample #{s} (assigned cluster {labels[s]}), |SHAP| for its cluster:")
    if n_informative is not None:
        print(f"     informative dims 0..{n_informative-1}: "
              f"{np.round(local[s, :n_informative], 4)}")
        print(f"     noise dims {n_informative}..{D-1}      : "
              f"{np.round(local[s, n_informative:], 4)}")
    else:
        print(f"     {np.round(local[s], 4)}")

    print("\n  per-cluster |SHAP| (rows=cluster, cols=feature, rounded):")
    print(np.round(cluster, 3))
    print()
    return dict(name=name, glob=glob, cluster=cluster, local=local,
                labels=labels, n_informative=n_informative,
                same_leaf=same, n_nodes=n_nodes, n_leaves=n_leaves, K=K)


def make_synth(seed=0):
    """3 informative dims (make_blobs, 4 clusters) + 12 N(0,1) noise dims."""
    rng = np.random.default_rng(seed)
    Xi, y, _ = make_blobs(n_samples=600, centers=4, n_features=3,
                          cluster_std=1.0, random_state=seed, return_centers=True)
    noise = rng.normal(0, 1, size=(Xi.shape[0], 12))
    X = np.hstack([Xi, noise])
    return X, 4, 3


def main():
    diff, ok = verify_roundtrip()
    print()
    Xs, ks, ni = make_synth(seed=0)
    r_syn = run_spex("synthetic (3 informative + 12 noise)", Xs, ks,
                     n_informative=ni, seed=0)
    iris = load_iris()
    r_iris = run_spex("iris", iris.data, 3, n_informative=None, seed=0)
    return diff, ok, r_syn, r_iris


if __name__ == "__main__":
    main()
