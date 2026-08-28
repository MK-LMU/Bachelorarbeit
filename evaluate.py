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
0.000 across the 5 spectral seeds (on CIFAR/MNIST the reference itself moves by
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

from wpaths import idc_out, results
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
SEEDS = {}          # every dataset uses 5 seeds (MNIST started with 3 and was
                    # extended to 5 as well — see DEFAULT_SEEDS)
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

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
    """list of dicts -> {metric: {mean, std, n, values}} (nan-aware)."""
    out = {}
    for m in per_seed[0]:
        v = np.array([d[m] for d in per_seed], float)
        ok = v[np.isfinite(v)]
        out[m] = {"mean": round(float(ok.mean()), 4) if len(ok) else None,
                  "std": round(float(ok.std(ddof=1)), 4) if len(ok) > 1 else None,
                  "n": int(len(ok)), "values": [None if not np.isfinite(x) else
                                                round(float(x), 6) for x in v]}
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

    # Seeds whose gate matrix is identically zero. Such a run produced NO
    # explanation at all, yet every distance metric happily reports a number:
    # uniqueness 0, stability 0, nn_identical_frac 1.0 and diversity 100 --
    # 'said nothing' reads as maximally stable and maximally diverse. The
    # values are kept (they are what the metric returns) but flagged here so
    # the results table can mark them.
    spex_rows, idc_rows = [], []
    spex_zero, idc_zero = [], []
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
        fd = faithfulness_drop(gates, X, infer, y, D, K)
        row["faithfulness_top1drop"] = fd["faithfulness_top1drop"]
        row["faithfulness_aopc"] = fd["faithfulness_aopc"]
        try:
            row["faithfulness_corr"] = faithfulness_k(gates, X, infer, y,
                                                      num_features=D, n_clusters=K)
        except Exception as e:
            print(f"[{ds}] seed {s} faithfulness_corr error: {e}")
            row["faithfulness_corr"] = float("nan")
        extras(gates, row)
        if not np.any(gates > 0):
            spex_zero.append(int(s))
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
        if not np.any(g > 0):
            idc_zero.append(int(s))
        idc_rows.append(row)
        print(f"  seed {s} done", flush=True)

    res = {"dataset": ds, "N": int(X.shape[0]), "D": D, "K": K,
           "seeds": have,
           "idc_config_validated": bool(d0["config_validated"]) if "config_validated" in d0 else (ds == "mnist"),
           "spex": agg(spex_rows), "idc": agg(idc_rows),
           "spex_zero_gate_seeds": spex_zero, "idc_zero_gate_seeds": idc_zero,
           "kmeans_ari": kmeans_baseline(base, X, y, K)}

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
    if spex_zero or idc_zero:
        print(f"ALL-ZERO GATES -- SpEx seeds {spex_zero}, IDC seeds {idc_zero}: "
              f"the distance metrics for those seeds describe an empty explanation")
    print(f"k-means baseline ARI: {res['kmeans_ari']:.3f}")
    print(f"saved {os.path.basename(jp)}\n", flush=True)


def main():
    for ds in (sys.argv[1:] or DATASETS):
        evaluate_dataset(ds)


if __name__ == "__main__":
    main()
