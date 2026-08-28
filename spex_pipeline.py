"""The SpEx side of the comparison — the core pipeline function.

spex_side(): SpectralClustering reference -> CliqueBased tree -> Tree SHAP
-> |SHAP| gate matrix (N, D). Imported by evaluate.py, perturbation_stability.py,
reference_selection.py and the equivalence check in metrics.py. (Extracted from
the legacy compare_final.py, which lives in archive/ and re-imports from here.)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "SpEx"))

import shap
from sklearn.cluster import SpectralClustering

from clique_based import CliqueBased
from tree_shap_spex import spex_tree_to_shap_dict, to_NDK


def spex_side(X, K, y_true, seed=0, return_ref=False, ref=None):
    # NOTE: y_true is accepted but NEVER used -- the reference clustering and the
    # tree are label-free. It stays in the signature so every call site reads the
    # same as the IDC side; do not mistake it for label access.
    """Train SpEx on the SAME X; returns (tree, labels(N,), gates(N,D)).
    Callers wrap tree.predict themselves to get the inference_fn the metrics
    need -- see evaluate.py.
    With return_ref=True also returns the spectral reference labels, so callers can
    report ARI(ref, y) and ARI(tree, ref) separately -- without those two numbers
    the 'SpEx' clustering rows conflate reference quality with tree approximation.
    Pass `ref` to explain a PRECOMPUTED reference clustering instead of the
    default SpectralClustering (used by the symmetric reference-selection
    experiment, e.g. a k-means reference)."""
    X = np.ascontiguousarray(X, float)
    if ref is None:
        ref = SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                                 random_state=seed).fit_predict(X)
    tree = CliqueBased(); tree.train(X, ref, K)
    labels = tree.predict(X).astype(int)
    # leaf count, not K: the tree may end up with k' != K leaves (that is what
    # compare_kprime.py explores), and the SHAP value array is sized by it.
    Kp = int(labels.max()) + 1
    tdict = spex_tree_to_shap_dict(tree, X, labels)
    expl = shap.TreeExplainer({"trees": [tdict]}, feature_perturbation="tree_path_dependent")
    sv = to_NDK(expl.shap_values(X), X.shape[0], X.shape[1], Kp)
    gates = np.abs(sv[np.arange(len(labels)), :, labels])       # (N, D)
    if return_ref:
        return tree, labels, gates, ref
    return tree, labels, gates
