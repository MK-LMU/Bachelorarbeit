"""Correctness chain, part 3 -- the same checks on the trees that are actually
evaluated.

Part 1 (test_correctness.py) validates the converter on Breast Cancer, Wine and
Digits. Here every production tree is rebuilt exactly as evaluate.py builds it
(spex_side() on the X from the training npz, same K, spectral seed 0) and put
through node weights, functional equivalence, additivity, production wiring and
the brute-force Shapley definition.

Production wiring exists only here: part 1 never calls spex_side(), so a wrong
index in spex_pipeline.py would pass every check there and still corrupt every
number in the thesis. Its tolerance is exact equality, not 1e-6.

Two of part 1's checks are not repeated. The model-agnostic cross-check knows
nothing about trees and is far too slow at D = 784. The format round-trip needs
an sklearn tree because it validates the array FORMAT, which does not depend on
which production tree is converted.

The brute force is O(2^M) in the features the tree uses; trees above
MAX_BRUTEFORCE_FEATURES are reported and skipped instead of hanging.

Needs artifacts/idc_out/idc_out_<ds>_seed0.npz. Exit code 1 if any check fails.
Usage: test_correctness_evaluated.py [dataset ...]    (default: all 9)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SpEx"))

import shap

from wpaths import idc_out, CAMPAIGN_SEEDS
from spex_pipeline import spex_side
from tree_shap_spex import spex_tree_to_shap_dict, to_NDK
from test_correctness import (dict_tree_predict, exact_path_dependent_shap,
                              test_node_weights)

DATASETS = ["two_moons", "blobs", "iris", "breast_cancer", "digits", "har",
            "cifar10", "mnist", "mnist_feats"]
MAX_BRUTEFORCE_FEATURES = 14   # 2^14 subsets x 40 samples is about the runtime
                               # budget; beyond that the brute force dominates
TOL = 1e-6
N_BRUTE = 40


def check_dataset(ds):
    # Only X/y/K are taken from here (identical across seeds); the tree is
    # SpEx's own and deterministic. Read a REPORTED seed all the same.
    p = idc_out(f"idc_out_{ds}_seed{CAMPAIGN_SEEDS[0]}.npz")
    if not os.path.exists(p):
        print(f"[{ds}] no npz for seed {CAMPAIGN_SEEDS[0]}, skipped")
        return None
    d = np.load(p)
    X = np.ascontiguousarray(d["X"], float)
    y, K = d["y_true"].astype(int), int(d["K"])
    N, D = X.shape

    tree, labels, gates = spex_side(X, K, y, seed=0)           # production path
    Kp = int(labels.max()) + 1
    tdict = spex_tree_to_shap_dict(tree, X, labels)
    used = sorted({int(f) for f in tdict["features"] if f >= 0})
    print("=" * 72)
    print(f"{ds}: X={X.shape} K={K} -> tree {len(tdict['features'])} nodes, "
          f"{int((tdict['children_left'] == -1).sum())} leaves, K'={Kp}, "
          f"uses {len(used)} features {used}")

    ok = True
    # node weights -- checked first: every later number depends on them
    ok &= test_node_weights(ds, tdict, X)

    # functional equivalence
    dict_labels = dict_tree_predict(tdict, X).argmax(axis=1)
    n_mismatch = int((labels != dict_labels).sum())
    leaf_onehot = bool(np.all(np.isin(tdict["values"][tdict["children_left"] == -1], (0.0, 1.0))))
    a0 = n_mismatch == 0 and leaf_onehot
    ok &= a0
    print(f"[equivalence]  dict == SpEx tree: {n_mismatch} of {N} mismatch, one-hot: "
          f"{leaf_onehot} -> {'PASS' if a0 else 'FAIL'}")

    # additivity
    expl = shap.TreeExplainer({"trees": [tdict]}, feature_perturbation="tree_path_dependent")
    sv = to_NDK(expl.shap_values(X), N, D, Kp)
    base = np.atleast_1d(expl.expected_value)
    err_b = float(np.max(np.abs(base[None, :] + sv.sum(axis=1) - dict_tree_predict(tdict, X))))
    ok &= err_b < TOL
    print(f"[additivity]   {err_b:.3e} -> {'PASS' if err_b < TOL else 'FAIL'}")

    # production wiring: gates == |assigned-cluster SHAP| of the converted tree
    err_p = float(np.max(np.abs(gates - np.abs(sv[np.arange(N), :, labels]))))
    ok &= err_p == 0.0
    print(f"[production]   spex_side gates == |SHAP[assigned]|: {err_p:.3e} -> "
          f"{'PASS' if err_p == 0.0 else 'FAIL'}")

    # brute force
    if len(used) > MAX_BRUTEFORCE_FEATURES:
        print(f"[brute force]  skipped: tree uses {len(used)} > "
              f"{MAX_BRUTEFORCE_FEATURES} features (2^M subsets)")
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, min(N_BRUTE, N), replace=False)
        err_d = max(float(np.max(np.abs(sv[i] - exact_path_dependent_shap(tdict, X[i], D))))
                    for i in idx)
        ok &= err_d < TOL
        print(f"[brute force]  exact Shapley vs shipped ({len(idx)} samples): {err_d:.3e} -> "
              f"{'PASS' if err_d < TOL else 'FAIL'}")
    return ok


def main():
    res = {ds: check_dataset(ds) for ds in (sys.argv[1:] or DATASETS)}
    done = {k: v for k, v in res.items() if v is not None}
    failed = [k for k, v in done.items() if not v]
    print("=" * 72)
    print(f"RESULT: {len(done) - len(failed)}/{len(done)} evaluated trees pass "
          f"weights + equivalence + additivity + production + brute force"
          + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
