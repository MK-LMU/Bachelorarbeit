# -*- coding: utf-8 -*-
"""Symmetric, label-free selection of SpEx's REFERENCE clusterer — the
pipeline's counterpart to IDC's config tuning (tune_idc.py -> reselect_best.py).

Phase 1: per dataset, two reference candidates (SpectralClustering as used
throughout, and k-means) are scored with the SAME unsupervised protocol as
IDC's grid: silhouette on (X, reference labels); ARI is logged for
transparency but never used for selection. Both candidates always produce
the requested K clusters, so the K-constraint of that protocol (a candidate
must use all K) is trivially met here.

Phase 2: wherever the selection flips away from the fixed Spectral choice, the
full SpEx pipeline (tree -> Tree SHAP -> all metrics, 5 seeds) is rerun on the
winning reference, so the effect on the comparison can be quantified.
Outcome: 6 of 8 datasets flip to k-means on silhouette (two_moons, blobs,
breast_cancer, digits, har, mnist); iris and CIFAR keep spectral. Which ones
flip cannot be predicted from ARI here -- ARI is exactly the signal this
protocol refuses to look at.

Output: results/tuning/reference_selection.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SpEx"))

from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score)

from wpaths import idc_out, RESULTS_TUNING
from metrics import (get_accuracy, diversity, generalizability, row_normalize,
                     faithfulness_drop, diversity_fixed,
                     precompute_nn, uniqueness_pre, stability_pre,
                     nn_identical_frac_pre)
from spex_pipeline import spex_side

DATASETS = ["two_moons", "blobs", "iris", "breast_cancer", "digits",
            "har", "cifar10", "mnist"]


def load(ds):
    d = np.load(idc_out(f"idc_out_{ds}_seed0.npz"))
    return (np.ascontiguousarray(d["X"], float), d["y_true"].astype(int),
            int(d["K"]))


def main():
    out_path = os.path.join(RESULTS_TUNING, "reference_selection.json")
    res = {}

    # ---- phase 1: label-free selection between the two reference candidates
    for ds in DATASETS:
        X, y, K = load(ds)
        cands = {}
        spec = SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                                  random_state=0).fit_predict(X)
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)
        for name, lab in (("spectral", spec), ("kmeans", km)):
            cands[name] = {
                "silhouette": round(float(silhouette_score(X, lab)), 4),
                "n_used_clusters": int(len(np.unique(lab))),
                "ari_logged_not_used": round(float(adjusted_rand_score(y, lab)), 4)}
        winner = max(cands, key=lambda n: cands[n]["silhouette"])
        res[ds] = {"criterion": "silhouette (unsupervised, same protocol as "
                                "IDC's grid in tune_idc.py); ARI logged only",
                   "candidates": cands, "winner": winner,
                   "flip_vs_fixed_spectral": winner != "spectral"}
        print(f"{ds:<14} spectral sil={cands['spectral']['silhouette']:+.3f} "
              f"(ARI {cands['spectral']['ari_logged_not_used']:.3f})   "
              f"kmeans sil={cands['kmeans']['silhouette']:+.3f} "
              f"(ARI {cands['kmeans']['ari_logged_not_used']:.3f})   "
              f"-> {winner}{'  <-- FLIP' if winner != 'spectral' else ''}",
              flush=True)
        json.dump(res, open(out_path, "w"), indent=2)

    # ---- phase 2: full pipeline on the winning reference where it flipped
    for ds in [d for d in DATASETS if res[d]["flip_vs_fixed_spectral"]]:
        print(f"\nPHASE 2: {ds} with the k-means reference (5 seeds)", flush=True)
        X, y, K = load(ds)
        nn_d, nn_i = precompute_nn(X, kmax=5)
        rows = []
        for s in range(5):
            ref = KMeans(n_clusters=K, n_init=10, random_state=s).fit_predict(X)
            tree, labels, gates = spex_side(X, K, y, seed=s, ref=ref)
            infer = lambda Xm, t=tree: t.predict(np.ascontiguousarray(Xm, float)).astype(int)
            row = {"ACC": float(get_accuracy(labels, y, K)),
                   "ARI": float(adjusted_rand_score(y, labels)),
                   "NMI": float(normalized_mutual_info_score(y, labels)),
                   "ref_ARI": float(adjusted_rand_score(y, ref)),
                   "tree_vs_ref_ARI": float(adjusted_rand_score(ref, labels)),
                   "uniqueness": uniqueness_pre(gates, nn_d, nn_i, k=2),
                   "uniqueness_rownorm": uniqueness_pre(row_normalize(gates), nn_d, nn_i, k=2),
                   "stability_k5": stability_pre(gates, nn_d, nn_i, k=5),
                   "nn_identical_frac": nn_identical_frac_pre(gates, nn_i),
                   "diversity_fixed": diversity_fixed(y, gates, K)}
            fd = faithfulness_drop(gates, X, infer, y, X.shape[1], K)
            row["faithfulness_top1drop"] = fd["faithfulness_top1drop"]
            row["faithfulness_aopc"] = fd["faithfulness_aopc"]
            rng = np.random.default_rng(s)
            idx = rng.permutation(len(X)); ntr = int(0.7 * len(X))
            tr, te = idx[:ntr], idx[ntr:]
            row["generalizability"] = float(generalizability(
                X[tr], gates[tr], y[tr], X[te], gates[te], y[te]))
            rows.append(row)
            print(f"  seed {s}: ARI {row['ARI']:.3f} (ref {row['ref_ARI']:.3f})",
                  flush=True)
        agg = {}
        for m in rows[0]:
            v = np.array([r[m] for r in rows], float)
            ok = v[np.isfinite(v)]
            agg[m] = {"mean": round(float(ok.mean()), 4) if len(ok) else None,
                      "std": round(float(ok.std(ddof=1)), 4) if len(ok) > 1 else None,
                      "n": int(len(ok))}
        res[ds]["kmeans_ref_pipeline"] = agg
        json.dump(res, open(out_path, "w"), indent=2)
        print(f"[{ds}] pipeline on the k-means reference: "
              f"ARI {agg['ARI']['mean']} ±{agg['ARI']['std']}", flush=True)

    print(f"\nDONE -> {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
