"""
SpEx with k' > k (over-segmentation) on MNIST, to show its effect on uniqueness.

Rationale: with k'=k, every SpEx cluster is exactly one leaf -> one constant
explanation per cluster -> uniqueness ~ 0. With k' > k, each of the K reference
clusters is split across several leaves, so nearby samples can fall in different
leaves and receive different |SHAP| gates -> uniqueness rises.

Design (fair vs IDC at K=10):
  - gates come from the k'-leaf tree (fine resolution -> many distinct gate rows)
  - ACC/ARI/NMI use the MAPPED assignment (each leaf -> its majority reference
    cluster, back to K=10). The metrics themselves get y_true as the label
    vector, as in the main evaluation; the mapping reaches them only through
    infer(), i.e. inside faithfulness
  - everything routed through the SHAP dict (stable leaf node ids) so masked
    inference in faithfulness is consistent.
"""
import os, sys
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJ, os.path.join(PROJ, "SpEx"), os.path.join(PROJ, "IDC")):
    sys.path.insert(0, _p)
import shap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from clique_based import CliqueBased
from tree_shap_spex import spex_tree_to_shap_dict, to_NDK
from metrics import all_metrics          # moved here unchanged from the archived
                                         # shared_metrics.py
from interpretability_metrics import get_accuracy


def route_leaf(td, X):
    """Route samples to stable leaf node ids of the dict tree."""
    cl, cr, feat, thr = (td["children_left"], td["children_right"],
                         td["features"], td["thresholds"])
    X = np.ascontiguousarray(X, float)
    out = np.empty(len(X), int)
    for i in range(len(X)):
        n = 0
        x = X[i]
        while cl[n] != -1:
            n = cl[n] if x[feat[n]] <= thr[n] else cr[n]
        out[i] = n
    return out


def spex_kprime(X, ref, kprime, chunk=400):
    """Train SpEx with kprime leaves; return dict, per-sample leaf-label, gates
    (|SHAP| for the sample's own leaf, chunked to bound memory), node->label."""
    tree = CliqueBased(); tree.train(X, ref, kprime)
    td = spex_tree_to_shap_dict(tree, X, tree.predict(X).astype(int))
    Kp = td["values"].shape[1]
    node_label = {n: int(np.argmax(td["values"][n]))
                  for n in range(len(td["features"])) if td["children_left"][n] == -1}
    leaf_node = route_leaf(td, X)
    labels = np.array([node_label[n] for n in leaf_node])         # 0..Kp-1

    expl = shap.TreeExplainer({"trees": [td]}, feature_perturbation="tree_path_dependent")
    N, D = X.shape
    gates = np.empty((N, D))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sv = to_NDK(expl.shap_values(X[s:e]), e - s, D, Kp)
        gates[s:e] = np.abs(sv[np.arange(e - s), :, labels[s:e]])
    return td, labels, gates, Kp, node_label


def main():
    from wpaths import idc_out
    d = np.load(idc_out("idc_out_mnist.npz"))
    X, y_true, K = np.ascontiguousarray(d["X"], float), d["y_true"].astype(int), int(d["K"])
    N, D = X.shape
    ref = SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                             random_state=0).fit_predict(X)

    idc = {k: float(d[k]) for k in ["acc", "ari", "nmi", "faithfulness",
                                    "diversity", "generalizability"]}
    # uniqueness is NOT read from the npz: run_idc.py runs all_metrics with
    # skip_distance_metrics=True, so every npz the current pipeline writes
    # stores nan there (only the legacy idc_out_mnist.npz still has a value).
    # Recompute it here from the stored gates, with the same fast path
    # evaluate.py uses -- otherwise this row silently prints nan.
    from metrics import precompute_nn, uniqueness_pre
    _nn_d, _nn_i = precompute_nn(X, kmax=2)
    idc["uniqueness"] = uniqueness_pre(np.ascontiguousarray(d["gates"], float),
                                       _nn_d, _nn_i, k=2)

    rows = []
    for kprime in [10, 50, 200]:
        td, labels, gates, Kp, node_label = spex_kprime(X, ref, kprime)
        # map each leaf-label to its majority reference cluster -> K=10 assignment
        lab2ref = np.array([np.bincount(ref[labels == l], minlength=K).argmax()
                            if (labels == l).any() else 0 for l in range(Kp)])
        mapped = lab2ref[labels]

        def infer(Xm, td=td, node_label=node_label, lab2ref=lab2ref):
            ln = route_leaf(td, Xm)
            return lab2ref[np.array([node_label[n] for n in ln])]

        m = all_metrics(f"SpEx k'={kprime}", X, gates, infer, y_true, K)
        acc = get_accuracy(mapped, y_true, K)
        ari = adjusted_rand_score(y_true, mapped)
        nmi = normalized_mutual_info_score(y_true, mapped)
        distinct = np.unique(gates.round(6), axis=0).shape[0]
        rows.append((kprime, Kp, distinct, acc, ari, nmi, m))
        print(f"done k'={kprime}: leaves={Kp}, distinct gate rows={distinct}, "
              f"uniqueness={m['uniqueness']:.3f}", flush=True)

    print("\n" + "=" * 90)
    print(f"MNIST: SpEx over-segmentation (k'>k) vs IDC   [N={N}, D={D}, K={K}]")
    print("=" * 90)
    hdr = f"{'method':<16}{'leaves':>7}{'distinctGates':>14}{'ACC':>7}{'ARI':>7}{'NMI':>7}{'uniq':>8}{'divers':>8}{'faith':>7}{'gen':>7}"
    print(hdr); print("-" * len(hdr))
    for kprime, Kp, distinct, acc, ari, nmi, m in rows:
        print(f"{'SpEx k=' + str(kprime):<16}{Kp:>7}{distinct:>14}{acc:>7.3f}{ari:>7.3f}"
              f"{nmi:>7.3f}{m['uniqueness']:>8.3f}{m['diversity']:>8.1f}"
              f"{m['faithfulness']:>7.3f}{m['generalizability']:>7.3f}")
    print(f"{'IDC':<16}{10:>7}{N:>14}{idc['acc']:>7.3f}{idc['ari']:>7.3f}{idc['nmi']:>7.3f}"
          f"{idc['uniqueness']:>8.3f}{idc['diversity']:>8.1f}{idc['faithfulness']:>7.3f}"
          f"{idc['generalizability']:>7.3f}")
    print("\nuniqueness rises with k' (more distinct explanations) but stays below IDC's")
    print("per-sample gates -- the rule-based-vs-gate-based granularity gap, quantified.")


if __name__ == "__main__":
    main()
