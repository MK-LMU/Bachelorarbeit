# -*- coding: utf-8 -*-
"""Multi-seed evaluation: ONE pass per dataset, writes the complete
results_multiseed_<ds>.json.

Both methods are scored by the same metric functions on the same X, with one
neighbour structure precomputed per dataset and reused across seeds and methods.
Three choices are worth knowing:

  - IDC's ACC/ARI/NMI are recomputed from the FINAL labels via Munkres, not read
    from the best-epoch values IDC logs during training -- those are selected
    under label access and would favour IDC over a tree that has no epochs.
  - Iris has duplicate rows, so its nearest-neighbour distance is 0 and the
    distance metrics divide by zero. A deduplicated variant is reported next to
    the raw one for both methods.
  - A plain k-means ARI is computed per dataset as a reality check; on several
    datasets it beats both explainable methods.

Seeds: the SpEx side is deterministic -- every tree-dependent metric has std
0.000 across the spectral seeds (on CIFAR/MNIST the reference itself moves by
~1e-4 without changing the tree). generalizability varies by design, its 70/30
split uses the seed identically for both methods.

Usage: evaluate.py [dataset ...]        (default: all)
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SpEx"))

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from wpaths import idc_out, results, CAMPAIGN_SEEDS
from metrics import (get_accuracy, diversity, generalizability, row_normalize,
                     faithfulness_k, faithfulness_drop, diversity_fixed,
                     precompute_nn, uniqueness_pre, stability_pre,
                     nn_identical_frac_pre)
from spex_pipeline import spex_side

DATASETS = ["two_moons", "two_moons_tuned", "two_moons_best",
            "blobs", "blobs_tuned", "blobs_best",
            "iris", "iris_best", "breast_cancer", "breast_cancer_best",
            "digits", "digits_best", "har", "har_best",
            "cifar10", "cifar10_best", "mnist",
            "mnist_feats", "mnist_feats_best"]
SEEDS = {}          # per-dataset override; empty means every dataset uses the
                    # campaign default below
DEFAULT_SEEDS = CAMPAIGN_SEEDS

_km_cache = {}


def base_of(ds):
    """'blobs_best' / 'blobs_tuned' -> 'blobs': same X, same geometry."""
    return ds.replace("_tuned", "").replace("_best", "")


def dist_block(gates, nn_d, nn_i):
    return {
        "uniqueness": uniqueness_pre(gates, nn_d, nn_i, k=2),
        "uniqueness_rownorm": uniqueness_pre(row_normalize(gates), nn_d, nn_i, k=2),
        "stability_k5": stability_pre(gates, nn_d, nn_i, k=5),
        "nn_identical_frac": nn_identical_frac_pre(gates, nn_i),
    }


def gener_split(X, gates, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    return float(generalizability(X[tr], gates[tr], y[tr], X[te], gates[te], y[te]))


def clustering_block(labels, y, K):
    return {"ACC": float(get_accuracy(labels, y, K)),
            "ARI": float(adjusted_rand_score(y, labels)),
            "NMI": float(normalized_mutual_info_score(y, labels))}


def agg(per_seed):
    """list of dicts -> {metric: {mean, std, n, values}} (nan-aware).

    mean/std are derived from the STORED `values`, not from the raw floats, so
    every consumer can re-derive them, and they keep 6 decimals so that no
    downstream formatter re-rounds an already-rounded number. Four decimals
    were enough for the JSON but not for the tables: a stored 0.4055 formats
    to 0.406 while the seeds average to 0.405499. Six decimals also drop float
    noise -- a std of 1e-16 over ten identical seeds becomes an honest 0.0."""
    out = {}
    for m in per_seed[0]:
        vals = [None if not np.isfinite(x) else round(float(x), 6)
                for x in np.array([d[m] for d in per_seed], float)]
        ok = np.array([x for x in vals if x is not None], float)
        out[m] = {"mean": round(float(ok.mean()), 6) if len(ok) else None,
                  "std": round(float(ok.std(ddof=1)), 6) if len(ok) > 1 else None,
                  "n": int(len(ok)), "values": vals}
    return out


def kmeans_baseline(base, X, y, K):
    if base not in _km_cache:                 # same X -> same baseline
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)
        _km_cache[base] = round(float(adjusted_rand_score(y, km)), 4)
    return _km_cache[base]


def evaluate_dataset(ds):
    base = base_of(ds)
    seeds = SEEDS.get(ds, DEFAULT_SEEDS)
    paths = {s: idc_out(f"idc_out_{ds}_seed{s}.npz") for s in seeds}
    have = [s for s in seeds if os.path.exists(paths[s])]
    if not have:
        print(f"[{ds}] no seed npz yet, skipped"); return
    print("=" * 72); print(f"DATASET {ds}  (seeds {have})"); print("=" * 72, flush=True)

    d0 = np.load(paths[have[0]])
    X = np.ascontiguousarray(d0["X"], float)
    y, K = d0["y_true"].astype(int), int(d0["K"])
    D = X.shape[1]
    nn_d, nn_i = precompute_nn(X, kmax=5)
    print("neighbour structure precomputed", flush=True)

    keep = None
    if base == "iris":                        # duplicate rows -> NN distance 0
        _, k_idx = np.unique(X.round(12), axis=0, return_index=True)
        keep = np.sort(k_idx)
        nn_d_dd, nn_i_dd = precompute_nn(X[keep], kmax=5)
        print(f"[{ds}] dedup: {len(X)} -> {len(keep)} samples")

    def extras(gates, row):
        """per-seed additions shared by both methods (formerly seed-0-only)."""
        row["diversity_fixed"] = diversity_fixed(y, gates, K)
        if keep is not None:
            row["uniqueness_dedup"] = uniqueness_pre(gates[keep], nn_d_dd, nn_i_dd, k=2)
            row["stability_k5_dedup"] = stability_pre(gates[keep], nn_d_dd, nn_i_dd, k=5)

    # Three ways a run can produce metric numbers that describe nothing. The
    # values are kept -- they are what the metric returns -- but flagged, so
    # the results table can say what the reader is looking at.
    #
    #   zero      the gate matrix is identically 0: no explanation at all, yet
    #             uniqueness 0, stability 0, nn_identical_frac 1.0, diversity
    #             100 -- "said nothing" reads as maximally stable and diverse.
    #   constant  the gate matrix has exactly ONE distinct row: every sample
    #             gets the same explanation. The distance metrics collapse to
    #             the same numbers as `zero`, but `np.any(gates > 0)` is True,
    #             so the zero test misses it entirely. This is the case on the
    #             tuned 2-D synthetics, where ARI/ACC look competitive.
    #   empty_med every cluster's MEDIAN gate vector is all-zero. IDC's
    #             original diversity then scores jaccard(empty, empty) = 0 for
    #             every pair and reports its maximum 100 -- for the opposite of
    #             the reason the name suggests. Independent of the other two:
    #             CIFAR-10 and MNIST-feats hit it with dense, varied gates.
    def degenerate(gm):
        """(all-zero, one distinct row, all median gate sets empty)"""
        zero = not np.any(gm > 0)
        const = len(np.unique(gm.round(12), axis=0)) == 1
        med_empty = all(not np.any(np.median(gm[y == c], axis=0) > 0)
                        for c in range(K) if np.any(y == c))
        return zero, const, med_empty

    # The ungated reference for generalizability. That metric trains a
    # LinearSVC on X * gates to predict the TRUE class labels, so it measures
    # how much of X's linear separability survives the gating -- a method that
    # gates nothing scores highest. Without this row a dense-gate method looks
    # like it "generalises better" when it has merely kept the input intact.
    # Same idea as the sigma=0 control: a condition whose expected result is
    # known (Adebayo et al. 2018; the metric-level version is Tomsett et al.
    # 2020, and Hooker et al. 2019 name the confound directly).
    plain_rows = []
    spex_rows, idc_rows = [], []
    spex_zero, idc_zero = [], []
    spex_const, idc_const = [], []
    spex_medempty, idc_medempty = [], []
    for s in have:
        # ---- SpEx side (spectral seed = s) ----
        tree, labels, gates, ref = spex_side(X, K, y, seed=s, return_ref=True)
        infer = lambda Xm, t=tree: t.predict(np.ascontiguousarray(Xm, float)).astype(int)
        row = clustering_block(labels, y, K)
        row["ref_ARI"] = float(adjusted_rand_score(y, ref))
        row["tree_vs_ref_ARI"] = float(adjusted_rand_score(ref, labels))
        row.update(dist_block(gates, nn_d, nn_i))
        row["diversity"] = float(diversity(y, gates, num_clusters=K, num_features=D))
        row["generalizability"] = gener_split(X, gates, y, s)
        # gates = 1 everywhere: X * gates == X, i.e. no explanation applied.
        plain_rows.append({"generalizability_plain_x":
                           gener_split(X, np.ones_like(gates), y, s)})
        try:
            # Guarded like faithfulness_corr below: masking can leave the tree
            # predicting fewer than K labels, and IDC's get_accuracy indexes a
            # K x K cost matrix with them. Unguarded that aborts the dataset
            # instead of costing one metric.
            fd = faithfulness_drop(gates, X, infer, y, D, K)
        except Exception as e:
            print(f"[{ds}] seed {s} faithfulness_drop error: {e}")
            fd = {"faithfulness_top1drop": float("nan"),
                  "faithfulness_aopc": float("nan")}
        row["faithfulness_top1drop"] = fd["faithfulness_top1drop"]
        row["faithfulness_aopc"] = fd["faithfulness_aopc"]
        try:
            row["faithfulness_corr"] = faithfulness_k(gates, X, infer, y,
                                                      num_features=D, n_clusters=K)
        except Exception as e:
            print(f"[{ds}] seed {s} faithfulness_corr error: {e}")
            row["faithfulness_corr"] = float("nan")
        extras(gates, row)
        z, c, me = degenerate(gates)
        if z: spex_zero.append(int(s))
        if c: spex_const.append(int(s))
        if me: spex_medempty.append(int(s))
        spex_rows.append(row)

        # ---- IDC side (training seed = s, from campaign npz) ----
        di = np.load(paths[s])
        g = np.ascontiguousarray(di["gates"], float)
        lp = di["labels_pred"].astype(int)
        row = clustering_block(lp, y, K)          # final model + Munkres, not IDC's
                                                  # best-epoch values
        row["ref_ARI"] = float("nan"); row["tree_vs_ref_ARI"] = float("nan")
        row.update(dist_block(g, nn_d, nn_i))
        row["diversity"] = float(diversity(y, g, num_clusters=K, num_features=D))
        row["generalizability"] = gener_split(X, g, y, s)
        row["faithfulness_top1drop"] = float(di["faithfulness_top1drop"]) if "faithfulness_top1drop" in di else float("nan")
        row["faithfulness_aopc"] = float(di["faithfulness_aopc"]) if "faithfulness_aopc" in di else float("nan")
        # The correlation faithfulness needs masked inference through the LIVE
        # torch model, which is gone by now, so run_idc.py computes it at
        # training time and it is only read across here. Forgetting this read
        # does not raise -- it silently prints nan for IDC in every row, while
        # SpEx shows a number.
        row["faithfulness_corr"] = float(di["faithfulness"]) if "faithfulness" in di else float("nan")
        extras(g, row)
        z, c, me = degenerate(g)
        if z: idc_zero.append(int(s))
        if c: idc_const.append(int(s))
        if me: idc_medempty.append(int(s))
        idc_rows.append(row)
        print(f"  seed {s} done", flush=True)

    res = {"dataset": ds, "N": int(X.shape[0]), "D": D, "K": K,
           "seeds": have,
           "idc_config_validated": bool(d0["config_validated"]) if "config_validated" in d0 else (ds == "mnist"),
           "spex": agg(spex_rows), "idc": agg(idc_rows),
           "spex_zero_gate_seeds": spex_zero, "idc_zero_gate_seeds": idc_zero,
           "spex_constant_gate_seeds": spex_const,
           "idc_constant_gate_seeds": idc_const,
           "spex_empty_median_seeds": spex_medempty,
           "idc_empty_median_seeds": idc_medempty,
           "kmeans_ari": kmeans_baseline(base, X, y, K),
           "generalizability_plain_x": agg(plain_rows)["generalizability_plain_x"]}

    jp = results(f"results_multiseed_{ds}.json")
    json.dump(res, open(jp, "w"), indent=2)

    print(f"\n{'metric':<22}{'SpEx mean±std':>22}{'IDC mean±std':>22}")
    print("-" * 66)
    for m in ["ACC", "ARI", "NMI", "ref_ARI", "tree_vs_ref_ARI", "uniqueness",
              "uniqueness_rownorm", "stability_k5", "nn_identical_frac",
              "generalizability", "faithfulness_top1drop", "faithfulness_aopc",
              "diversity_fixed", "faithfulness_corr"]:
        def f(a):
            if m not in a or a[m]["mean"] is None: return "nan"
            s_ = f"{a[m]['mean']:.3f}"
            return s_ + (f" ±{a[m]['std']:.3f}" if a[m]["std"] is not None else "")
        print(f"{m:<22}{f(res['spex']):>22}{f(res['idc']):>22}")
    for tag, sp, ic, what in (
            ("ALL-ZERO GATES", spex_zero, idc_zero,
             "the distance metrics describe an empty explanation"),
            ("CONSTANT GATES", spex_const, idc_const,
             "one explanation for every sample -- the distance metrics read "
             "like perfect stability"),
            ("EMPTY MEDIAN SETS", spex_medempty, idc_medempty,
             "IDC's original diversity reports its maximum for the opposite "
             "of the reason the name suggests")):
        if sp or ic:
            print(f"{tag} -- SpEx seeds {sp}, IDC seeds {ic}: {what}")
    print(f"k-means baseline ARI: {res['kmeans_ari']:.3f}")
    print(f"saved {os.path.basename(jp)}\n", flush=True)


def main():
    for ds in (sys.argv[1:] or DATASETS):
        evaluate_dataset(ds)


if __name__ == "__main__":
    main()
