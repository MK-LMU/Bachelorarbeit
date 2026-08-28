"""One metric module applied IDENTICALLY to SpEx and IDC, so that no measured
difference can originate in the evaluation. Three layers:

  1. IDC's OWN interpretability_metrics.py, imported verbatim.
  2. Labelled additions of this thesis, each motivated by a defect of the
     original that is stated where the addition is defined.
  3. A bit-exact fast path for the O(N^2) distance metrics: the neighbour
     structure is precomputed once per dataset with IDC's OWN operations and
     reused across seeds and methods. Run this module directly to execute the
     equivalence proof against IDC's originals.

Only two inputs differ between the methods:
  - gates:        (N, D) non-negative importance matrix — IDC's learned local
                  gates, or SpEx's |SHAP| for the assigned cluster
  - inference_fn: X(masked) -> hard labels, each method's own predictor
"""
import os, sys
import numpy as np

# NumPy >= 2 removed np.infty / np.NaN / np.float and friends, but the vendored
# IDC code still uses them. That code must stay verbatim (layer 1 above is the
# whole point), so the aliases are restored here instead -- and this has to run
# BEFORE the IDC import below.
for _n, _v in [("infty", np.inf), ("NaN", np.nan), ("float", float),
               ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _n):
        setattr(np, _n, _v)

IDC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IDC")
if IDC_DIR not in sys.path:
    sys.path.insert(0, IDC_DIR)

# ---------------------------------------------------------------------------
# Layer 1: IDC's exact metric implementations (verbatim import)
# ---------------------------------------------------------------------------
from interpretability_metrics import (faithfulness, diversity, uniqueness,
                                       stability, generalizability, get_accuracy)

from scipy.spatial import distance_matrix


# ---------------------------------------------------------------------------
# Layer 2: labelled additions of this thesis
# ---------------------------------------------------------------------------
def faithfulness_k(gates_i, x, inference_fn, y, num_features, n_clusters):
    """IDC's faithfulness algorithm with the K fix: n_clusters is passed through
    to get_accuracy instead of IDC's hardcoded 10 (which only fits MNIST/Digits
    and errors for any other K).

    Two further, deliberate deviations from IDC's original:
      - `.astype(int)` on the inference output is REQUIRED, not cosmetic:
        SpEx's Tree.predict returns float64, and IDC's original would raise
        IndexError ("arrays used as indices must be of integer type").
      - <2 usable features returns nan instead of letting np.corrcoef produce
        a degenerate value; IDC returns nan there too, so this is explicit,
        not different."""
    importance_vec = np.sum(gates_i > 0, axis=0)
    importance_ind = np.where(importance_vec > 0)[0]
    importance_ind_sort = importance_ind[np.argsort(-importance_vec[importance_vec > 0])]
    mask = np.ones(num_features)
    acc = []
    for i in importance_ind_sort:
        mask[i] = 0
        yh = np.asarray(inference_fn(x * mask)).astype(int)
        acc.append(get_accuracy(yh, y, n_clusters))
    if len(acc) < 2:
        return float("nan")            # <2 selected features -> undefined (genuine degeneracy)
    return float(np.corrcoef(importance_vec[importance_ind_sort], np.array(acc))[0, 1])


def row_normalize(gates):
    """Scale every gate row to max 1 (IDC's gates already live in [0,1]; SpEx's
    |SHAP| rows are ~5x smaller). uniqueness/stability are homogeneous of degree 1
    in the gates, so comparing raw SpEx |SHAP| against IDC gates mixes scale into
    the granularity signal. Both variants are therefore reported side by side in
    RESULTS_MULTISEED.md, and the granularity claim itself rests on the
    scale-invariant nn_identical_frac rather than on either of them."""
    gmax = gates.max(axis=1, keepdims=True)
    return np.divide(gates, gmax, out=np.zeros_like(gates), where=gmax > 0)


def nn_identical_frac(X, gates):
    """Scale-invariant granularity: fraction of samples whose 1-NN (in X,
    excluding self) has an IDENTICAL gate row. This is the number that separates
    piecewise-constant (SpEx: NN usually shares a leaf) from per-sample (IDC)
    explanations without any dependence on gate magnitude."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, ind = nn.kneighbors(X)
    neigh = ind[:, 1]
    return float(np.mean(np.all(np.isclose(gates, gates[neigh], atol=1e-12), axis=1)))


def faithfulness_curve(gates_i, x, inference_fn, y, num_features, n_clusters):
    """Same masking loop as faithfulness_k, but RETURNS the accuracy curve
    (importance-sorted features, accuracy after cumulatively masking each) so a
    'cliff' (undefined correlation) can be shown instead of asserted. Also
    returns the drop-based scalar: baseline acc - acc after masking the top
    feature (well-defined even when the correlation is not)."""
    importance_vec = np.sum(gates_i > 0, axis=0)
    importance_ind = np.where(importance_vec > 0)[0]
    importance_ind_sort = importance_ind[np.argsort(-importance_vec[importance_vec > 0])]
    baseline = get_accuracy(np.asarray(inference_fn(x)).astype(int), y, n_clusters)
    mask = np.ones(num_features)
    acc = []
    for i in importance_ind_sort:
        mask[i] = 0
        yh = np.asarray(inference_fn(x * mask)).astype(int)
        acc.append(get_accuracy(yh, y, n_clusters))
    acc = np.array(acc)
    top1_drop = float(baseline - acc[0]) if len(acc) else float("nan")
    return importance_ind_sort, importance_vec[importance_ind_sort], acc, baseline, top1_drop


def faithfulness_drop(gates, X, inference_fn, y, num_features, n_clusters):
    """Drop-based faithfulness values that stay DEFINED where the correlation
    variant degenerates (<2 used features, e.g. Breast Cancer; or an exactly
    constant acc curve, e.g. SpEx on HAR):
      top1_drop = baseline acc - acc after masking the most important feature
      aopc      = mean(baseline - acc) over the cumulative masking curve
    Uses the SAME masking loop and zero-masking semantics as IDC's faithfulness,
    so the OOD caveat (zeroing is in-distribution for IDC's gated model, not for
    the tree) applies here too."""
    feats, imp, acc, base, top1 = faithfulness_curve(
        gates, X, inference_fn, y, num_features, n_clusters)
    aopc = float(np.mean(base - acc)) if len(acc) else float("nan")
    return {"faithfulness_top1drop": top1, "faithfulness_aopc": aopc,
            "faithfulness_n_used": int(len(feats)),
            "faithfulness_baseline_acc": float(base)}


def diversity_fixed(y, gates, num_clusters):
    """Repaired diversity. The original has two defects. (a) It accumulates the
    full K x K Jaccard matrix INCLUDING the diagonal — where a set is compared
    with itself and contributes exactly 1 — while dividing by the off-diagonal
    count K(K-1); the computed value is therefore the defined one minus
    100/(K-1), and the attainable maximum is 100*(1-1/(K-1)), not 100.
    (b) `sklearn.jaccard_score` returns 0 for two EMPTY sets, so a cluster whose
    median gate vector is all-zero RAISES the score — saying nothing scores best.

    This variant: weighted-Jaccard (Ruzicka) similarity between the clusters'
    MEAN gate vectors, averaged over OFF-DIAGONAL pairs only, as
    100 * (1 - mean similarity).
      - no `median > 0` thresholding -> the empty-set pathology cannot occur;
        if a cluster's mean gate vector is all-zero the metric is undefined
        (nan) instead of maximally diverse
      - diagonal excluded -> range [0, 100], 100 = fully disjoint feature use,
        directly comparable across different K
      - rows are normalised to max 1 FIRST. Without that the Ruzicka similarity
        also reacts to gate MAGNITUDE, so two clusters using the very same
        single feature with different |SHAP| magnitudes would score as diverse
        (measured on Breast Cancer: both SpEx clusters use only feature 27, yet
        the unnormalised value was 46.19 instead of 0). Normalising makes the
        metric answer the question it is named after -- which features, not how
        strongly -- and makes it scale-invariant per sample, not just jointly.
    """
    gates = row_normalize(np.asarray(gates, float))
    means = []
    for c in range(num_clusters):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            return float("nan")
        m = gates[idx].mean(axis=0)
        if m.sum() <= 0:
            return float("nan")
        means.append(m)
    sims = []
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                sims.append(np.minimum(means[i], means[j]).sum()
                            / np.maximum(means[i], means[j]).sum())
    return float(100.0 * (1.0 - np.mean(sims)))


def all_metrics(name, X, gates, inference_fn, y_true, K, seed=0,
                skip_distance_metrics=False):
    """Compute IDC's five interpretability metrics on one method's outputs.
    skip_distance_metrics=True omits the O(N^2) uniqueness/stability block
    (returns nan there) — used by the training runs, because those metrics
    depend only on (X, gates) and are recomputed offline from the saved npz
    via the fast path below (bit-exact, neighbour structure shared across
    seeds)."""
    N, D = X.shape
    out = {}

    # faithfulness: mask features by importance, watch accuracy drop.
    # faithfulness_k is IDC's algorithm with one fix: IDC's own faithfulness
    # passes the literal 10 to get_accuracy (interpretability_metrics.py:33),
    # which only fits K=10; here the dataset's real K is passed through.
    try:
        out["faithfulness"] = faithfulness_k(gates, X, inference_fn, y_true,
                                             num_features=D, n_clusters=K)
    except Exception as e:
        out["faithfulness"] = float("nan"); print(f"  [{name}] faithfulness: {e}")

    # diversity: cluster-level feature-set overlap. This is IDC's ORIGINAL, which
    # is defective (see diversity_fixed below); it is kept for comparability with
    # IDC's published numbers, and evaluate.py adds the repaired variant per seed.
    # Labels: ground truth for both methods, as in IDC's own evaluation.
    out["diversity"] = float(diversity(y_true, gates, num_clusters=K, num_features=D))

    # uniqueness/stability: how gates vary between nearest neighbours.
    # CAVEATS (methodological review):
    #  - at k=2 both reduce to the same single-neighbour ratio -> stability(k=2)
    #    IS uniqueness(k=2); stability is therefore reported at k=5 instead.
    #  - both are homogeneous of degree 1 in the gates -> NOT scale-invariant;
    #    raw |SHAP| rows are ~5x smaller than IDC's [0,1] gates, so the
    #    row-normalised value and the scale-invariant nn_identical_frac are
    #    reported alongside the raw value.
    if skip_distance_metrics:
        for k_ in ("uniqueness", "uniqueness_rownorm", "stability",
                   "stability_k5", "nn_identical_frac"):
            out[k_] = float("nan")
    else:
        out["uniqueness"] = float(uniqueness(X, gates, k=2, subset_size=N))
        out["uniqueness_rownorm"] = float(uniqueness(X, row_normalize(gates), k=2, subset_size=N))
        out["stability"] = out["uniqueness"]     # identical by construction at k=2
        out["stability_k5"] = float(stability(X, gates, k=5, subset_size=N))
        out["nn_identical_frac"] = nn_identical_frac(X, gates)

    # generalizability: LinearSVC on gated features, train/test split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N); ntr = int(0.7 * N)
    tr, te = idx[:ntr], idx[ntr:]
    out["generalizability"] = float(generalizability(
        X[tr], gates[tr], y_true[tr], X[te], gates[te], y_true[te]))
    return out


# ---------------------------------------------------------------------------
# Layer 3: bit-exact fast path for the distance metrics (multi-seed evaluation)
#
# WHY: IDC's uniqueness/stability rebuild and sort the full N x N distance
# matrix on X at EVERY call; in the multi-seed evaluation the same X is scored
# ~30 times. The neighbour structure depends only on X, so it is precomputed
# ONCE here — with IDC'S OWN OPERATIONS (`scipy.spatial.distance_matrix` +
# `np.sort`/`np.argsort`, same calls, same order) — and the per-sample ratio
# loops below are copied verbatim from IDC/interpretability_metrics.py.
# Result: numerically IDENTICAL values (bit-exact, including tie-breaking on
# integer-valued features, where an sklearn-based reimplementation deviated by
# ~2e-4). Run this module directly to execute the equivalence check.
# ---------------------------------------------------------------------------
def precompute_nn(X, kmax=5):
    """EXACTLY the neighbour structure IDC's uniqueness/stability build
    internally (self included in column 0), computed once for reuse."""
    dist_mat_x = distance_matrix(X, X, p=2)
    nn_dist_mat = np.sort(dist_mat_x, axis=1)[:, 0:kmax]
    nn_ind_mat = np.argsort(dist_mat_x, axis=1)[:, 0:kmax]
    return nn_dist_mat, nn_ind_mat


def _check_k(nn_ind_mat, k):
    """Guard: slicing [:k] out of a structure precomputed with kmax < k would
    silently return the value for the SMALLER k instead of failing."""
    assert nn_ind_mat.shape[1] >= k, (
        f"precompute_nn was called with kmax={nn_ind_mat.shape[1]} < k={k}; "
        "the result would silently be the k=kmax value")


def uniqueness_pre(gates, nn_dist_mat, nn_ind_mat, k=2):
    """IDC's uniqueness with the neighbour structure injected; the loop body is
    verbatim from interpretability_metrics.uniqueness."""
    _check_k(nn_ind_mat, k)
    vals = []
    for i in range(gates.shape[0]):
        vals.append(min(distance_matrix(gates[nn_ind_mat[i, :k], :],
                                        gates[nn_ind_mat[i, :k], :])[0][1:]
                        / nn_dist_mat[i, :k][1:]))
    return float(np.mean(np.array(vals)))


def stability_pre(gates, nn_dist_mat, nn_ind_mat, k=2):
    """IDC's stability with the neighbour structure injected (verbatim loop)."""
    _check_k(nn_ind_mat, k)
    lipchitz_constants = []
    for i in range(gates.shape[0]):
        lipchitz_constants.append(max(distance_matrix(gates[nn_ind_mat[i, :k], :],
                                                      gates[nn_ind_mat[i, :k], :])[0][1:]
                                      / nn_dist_mat[i, :k][1:]))
    return float(np.mean(np.array(lipchitz_constants)))


def nn_identical_frac_pre(gates, nn_ind_mat):
    """Scale-invariant granularity (share of 1-NN pairs with identical gate
    rows), using the same tie-breaking as the metrics above."""
    neigh = nn_ind_mat[:, 1]
    return float(np.mean(np.all(np.isclose(gates, gates[neigh], atol=1e-12), axis=1)))


def verify_equivalence(X, gates_list, ks=(2, 5)):
    """Assert bit-exact agreement with IDC's originals.

    Returns {"max_abs_diff", "n_numeric", "n_nan", "n_inf"}. The counts matter:
    on a dataset with duplicate rows the neighbour distance is 0, so most values
    are nan or inf and the comparison is vacuous even though nothing fails. Iris
    is the live example -- only 1 of its 16 comparisons is numeric, which a bare
    "max|diff| = 0.0" would hide. Both-nan and both-inf ARE agreements, just not
    numeric ones, so they are counted rather than silently dropped."""
    nn_d, nn_i = precompute_nn(X, kmax=max(ks))
    worst, n_num, n_nan, n_inf = 0.0, 0, 0, 0
    for g in gates_list:
        for k in ks:
            for slow, fast in ((uniqueness, uniqueness_pre), (stability, stability_pre)):
                a = slow(X, g, k=k, subset_size=X.shape[0])
                b = fast(g, nn_d, nn_i, k=k)
                if np.isnan(a) or np.isnan(b):
                    assert np.isnan(a) and np.isnan(b), (
                        f"{slow.__name__} k={k}: one side nan, the other not: "
                        f"{a!r} vs {b!r}")
                    n_nan += 1
                    continue
                assert a == b, f"{slow.__name__} k={k}: {a!r} != {b!r}"
                if np.isinf(a):
                    n_inf += 1
                else:
                    n_num += 1
                    worst = max(worst, abs(a - b))
    return {"max_abs_diff": worst, "n_numeric": n_num,
            "n_nan": n_nan, "n_inf": n_inf}


if __name__ == "__main__":
    # Equivalence proof over ALL datasets: X from the seed-0 training npz;
    # gates = real IDC gates, real SpEx |SHAP| gates (spex_pipeline, seed 0),
    # random dense and random sparse. Iris additionally on its duplicate-free
    # geometry (the one evaluate.py scores), plus TWO data-free adversarial
    # tie grids -- one with duplicate rows (everything undefined; checks that
    # both implementations agree on that) and one without (98 % of pairwise
    # distances still tie, and this is where the numeric tie-breaking proof
    # comes from). Any mismatch raises (verify_equivalence asserts a == b).
    # The JSON records per case how many comparisons were NUMERIC rather than
    # nan/inf -- iris alone manages 1 of 16, which a bare max|diff| would hide.
    # Output: results/equivalence_check.json. Budget SEVERAL HOURS -- for the
    # reason given at the top of layer 3 this costs O(N^2 * D) x 16 per dataset,
    # and HAR alone (N=10299, D=561) takes about an hour. The measured claim
    # does not move between runs; rerun only after touching the fast path.
    import json, warnings
    from wpaths import idc_out, results
    from spex_pipeline import spex_side          # pulls in shap -> only here
    warnings.filterwarnings("ignore")

    DATASETS = ["two_moons", "blobs", "iris", "breast_cancer", "digits", "har",
                "cifar10", "mnist", "mnist_feats"]
    only = sys.argv[1:] or DATASETS
    report, state = {}, {"overall": 0.0, "numeric": 0, "nonfinite": 0}

    def run_case(tag, X, gates_dict):
        res = {}
        for gname, g in gates_dict.items():
            r = verify_equivalence(X, [g])
            res[gname] = r
            state["overall"] = max(state["overall"], r["max_abs_diff"])
            state["numeric"] += r["n_numeric"]
            state["nonfinite"] += r["n_nan"] + r["n_inf"]
        num = sum(x["n_numeric"] for x in res.values())
        tot = sum(x["n_numeric"] + x["n_nan"] + x["n_inf"] for x in res.values())
        worst = max(x["max_abs_diff"] for x in res.values())
        print(f"{tag:<16} N={len(X):>6}  PASS  max|diff| = {worst:.1e}  "
              f"({num}/{tot} comparisons numeric, rest nan/inf)", flush=True)
        return res

    rng = np.random.default_rng(0)
    # TWO integer grids, because they probe different failure modes and the
    # comparison counters showed the first one alone proves nothing numerically.
    #   dense: values 0..2 on 4 columns = 81 possible rows for 400 samples, so
    #     ~80 % are duplicates -> neighbour distance 0 -> every ratio is inf/nan.
    #     Worth keeping: it checks that BOTH implementations agree the value is
    #     undefined. But it yields no numeric comparison at all.
    #   sparse: values 0..39 -> no duplicates, yet 98 % of all pairwise distances
    #     still tie, which is the case argsort tie-breaking actually has to get
    #     right. This one delivers the numeric proof.
    Xg = rng.integers(0, 3, size=(400, 4)).astype(float)
    report["tie_grid_duplicates_400x4"] = run_case("tie grid (dup)", Xg, {
        "sparse": (rng.random((400, 4)) < 0.3) * rng.random((400, 4)),
        "leaf_constant": np.repeat(rng.random((8, 4)), 50, axis=0)})
    Xt = rng.integers(0, 40, size=(400, 4)).astype(float)
    report["tie_grid_nodup_400x4"] = run_case("tie grid (no dup)", Xt, {
        "sparse": (rng.random((400, 4)) < 0.3) * rng.random((400, 4)),
        "leaf_constant": np.repeat(rng.random((8, 4)), 50, axis=0)})

    for ds in only:
        p = idc_out(f"idc_out_{ds}_seed0.npz")
        if not os.path.exists(p):
            print(f"{ds:<16} no seed-0 npz, skipped", flush=True)
            continue
        d = np.load(p)
        X = np.ascontiguousarray(d["X"], float)
        y, K = d["y_true"].astype(int), int(d["K"])
        g_idc = np.ascontiguousarray(d["gates"], float)
        _, _, g_spex = spex_side(X, K, y, seed=0)
        rd = rng.random(g_idc.shape)
        cases = {"idc": g_idc, "spex": g_spex, "rand_dense": rd,
                 "rand_sparse": rd * (rng.random(g_idc.shape) < 0.05)}
        report[ds] = run_case(ds, X, cases)
        if ds == "iris":                         # duplicate rows -> dedup geometry
            _, k_idx = np.unique(X.round(12), axis=0, return_index=True)
            keep = np.sort(k_idx)
            report["iris_dedup"] = run_case("iris (dedup)", X[keep],
                                            {k: v[keep] for k, v in cases.items()})

    # MERGE instead of overwrite. The full proof costs hours (see the note
    # above), so it has to be resumable: `python metrics.py har cifar10` updates
    # only those cases and leaves the rest of the file intact. Entries written
    # before the comparison counters existed carry a plain float instead of a
    # dict; they are kept, listed under "cases_without_counts", and excluded
    # from the totals rather than silently counted as zero.
    out = results("equivalence_check.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    cases = dict(prev.get("cases", {}))
    cases.update(report)

    def counted(v):
        return all(isinstance(x, dict) for x in v.values())

    legacy = sorted(c for c, v in cases.items() if not counted(v))
    num = sum(x["n_numeric"] for c, v in cases.items() if counted(v)
              for x in v.values())
    non = sum(x["n_nan"] + x["n_inf"] for c, v in cases.items() if counted(v)
              for x in v.values())
    worst = 0.0
    for v in cases.values():
        for x in v.values():
            worst = max(worst, x["max_abs_diff"] if isinstance(x, dict) else x)

    payload = {"max_abs_diff_overall": worst,
               "n_numeric_comparisons": num,
               "n_nonfinite_comparisons": non,
               "note": "n_nonfinite are both-nan or both-inf pairs: the two "
                       "implementations agree the value is undefined or infinite, "
                       "which is an agreement but not a numeric check. Datasets "
                       "with duplicate rows (iris) are almost entirely non-finite; "
                       "their dedup variant carries the numeric proof.",
               "cases": cases}
    if legacy:
        payload["cases_without_counts"] = legacy
        payload["note_on_legacy_cases"] = (
            "These cases were verified by an earlier run, before the comparison "
            "counters were added: their max_abs_diff is the measured 0.0, but the "
            "numeric/non-finite split was not recorded and they are NOT included "
            "in the totals above. Rerun `python metrics.py " + " ".join(legacy) +
            "` to fill them in (hours -- see the runtime note in this file).")
    json.dump(payload, open(out, "w"), indent=2)

    tail = (f"; {len(legacy)} case(s) still without counts: {', '.join(legacy)}"
            if legacy else "")
    print("")
    print(f"EQUIVALENCE PASS (bit-exact): max |diff| = {worst:.1e} vs IDC verbatim "
          f"over {len(cases)} cases -> {num} numeric comparisons plus {non} "
          f"both-undefined/both-inf agreements{tail} -> {os.path.basename(out)}")
