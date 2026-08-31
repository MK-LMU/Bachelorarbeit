# -*- coding: utf-8 -*-
"""Every reported number against the range its own definition allows.

The question that prompted this: `uniqueness (raw) = 2.431` looks wrong if one
expects a fraction. It is not — `uniqueness` divides a gate-space distance by
an input-space distance (IDC/interpretability_metrics.py:80) and is unbounded
above. But nothing in the table said so, and nothing checked the numbers that
ARE bounded. This module does both jobs from one source:

  - `gen_results_table.py` imports RANGES for the row labels, so the printed
    range and the checked range cannot drift apart
  - run this file directly to validate results/results_multiseed_*.json

Four metrics have K-DEPENDENT bounds, which is why every bound is a function
of K rather than a constant: ACC cannot fall below 1/K (Munkres picks the best
of K! assignments, and the average over all of them already reaches 1/K), the
two faithfulness drops are differences of two such accuracies and therefore
live in +-(1-1/K), and IDC's original `diversity` bottoms out at -100/(K-1).

Deliberately dependency-free (json/glob/math only). It must not import
metrics.py: that pulls in IDC's module and with it torch, and generating a
markdown table should not require a GPU library.

Usage: check_ranges.py [results_dir]      exit code 1 on any violation
"""
import os, sys, json, glob, math

HERE = os.path.dirname(os.path.abspath(__file__))

# The stored values are rounded (ACC is written as 0.333333 while the Munkres
# floor 1/3 is 0.3333333...), so an exact comparison reports violations that
# are pure display rounding. The tolerance has to cover that granularity.
EPS = 1e-6

# How far a gated generalizability may sit above the ungated reference before
# it stops being explainable by per-sample feature rescaling. Observed maximum
# in this campaign: +0.006.
PLAIN_X_MARGIN = 0.05

INF = math.inf

# metric -> (lo(K), hi(K), label for the table, why the bound is what it is)
RANGES = {
    "ACC": (lambda K: 1.0 / K, lambda K: 1.0, "[1/K,1]",
            "Munkres accuracy: the optimal 1:1 assignment is at least as good "
            "as the average over all K! permutations, which is 1/K"),
    "ARI": (lambda K: -0.5, lambda K: 1.0, "[-.5,1]", "adjusted Rand index"),
    "ref_ARI": (lambda K: -0.5, lambda K: 1.0, "[-.5,1]", "adjusted Rand index"),
    "tree_vs_ref_ARI": (lambda K: -0.5, lambda K: 1.0, "[-.5,1]",
                        "adjusted Rand index"),
    "NMI": (lambda K: 0.0, lambda K: 1.0, "[0,1]", "normalised mutual information"),
    "generalizability": (lambda K: 0.0, lambda K: 1.0, "[0,1]",
                         "LinearSVC score, i.e. an accuracy"),
    "nn_identical_frac": (lambda K: 0.0, lambda K: 1.0, "[0,1]",
                          "share of samples whose nearest neighbour has an "
                          "identical gate row"),
    "faithfulness_corr": (lambda K: -1.0, lambda K: 1.0, "[-1,1]",
                          "Pearson correlation"),
    "faithfulness_top1drop": (lambda K: -(1 - 1.0 / K), lambda K: 1 - 1.0 / K,
                              "[-(1-1/K),1-1/K]",
                              "difference of two Munkres accuracies, each >= 1/K; "
                              "negative means masking the top feature HELPED"),
    "faithfulness_aopc": (lambda K: -(1 - 1.0 / K), lambda K: 1 - 1.0 / K,
                          "[-(1-1/K),1-1/K]",
                          "mean over the same differences, same bound"),
    "diversity": (lambda K: -100.0 / (K - 1), lambda K: 100.0,
                  "[-100/(K-1),100]",
                  "IDC's original sums the Jaccard matrix INCLUDING the diagonal "
                  "over K(K-1) pairs; all-identical sets give the floor, all-EMPTY "
                  "sets give 100 (the empty-set defect), and the non-degenerate "
                  "ceiling is 100*(1-1/(K-1))"),
    "diversity_fixed": (lambda K: 0.0, lambda K: 100.0, "[0,100]",
                        "off-diagonal weighted Jaccard on non-negative gates"),
    "uniqueness": (lambda K: 0.0, lambda K: INF, "[0,inf)",
                   "gate-space distance over input-space distance -- a local "
                   "Lipschitz-type ratio, NOT a fraction"),
    "uniqueness_rownorm": (lambda K: 0.0, lambda K: INF, "[0,inf)",
                           "same ratio on row-normalised gates: the GATES are "
                           "normalised, the ratio is not"),
    "stability_k5": (lambda K: 0.0, lambda K: INF, "[0,inf)",
                     "same ratio with max instead of min over the neighbours"),
    "uniqueness_dedup": (lambda K: 0.0, lambda K: INF, "[0,inf)",
                         "uniqueness on the duplicate-free geometry"),
    "stability_k5_dedup": (lambda K: 0.0, lambda K: INF, "[0,inf)",
                           "stability on the duplicate-free geometry"),
}

SPEX_ONLY = ("ref_ARI", "tree_vs_ref_ARI")


def label(metric):
    """Range label for the results table; empty for anything not listed."""
    return RANGES[metric][2] if metric in RANGES else ""


def _undef(v):
    return v is None or (isinstance(v, float) and (v != v or math.isinf(v)))


def check_file(path):
    """(n_finite, n_undefined, [violation strings]) for one results file."""
    d = json.load(open(path))
    ds = os.path.basename(path)[len("results_multiseed_"):-len(".json")]
    K = d["K"]
    fin = und = 0
    bad = []

    for method in ("spex", "idc"):
        block = d.get(method, {})
        for metric, entry in block.items():
            if not isinstance(entry, dict):
                continue
            vals = entry.get("values") or []

            # --- range of every individual seed value
            if metric in RANGES:
                lo, hi = RANGES[metric][0](K), RANGES[metric][1](K)
                for i, v in enumerate(vals):
                    if _undef(v):
                        und += 1
                        continue
                    fin += 1
                    if v < lo - EPS or v > hi + EPS:
                        bad.append(f"{ds}/{method}/{metric} seed {i}: {v!r} "
                                   f"outside [{lo:.4f}, {hi}] (K={K})")

            # --- n must equal the number of defined values
            n = entry.get("n")
            n_def = sum(0 if _undef(v) else 1 for v in vals)
            if n is not None and n != n_def:
                bad.append(f"{ds}/{method}/{metric}: n={n} but {n_def} values "
                           f"are defined")

            # --- mean/std must be reproducible from values (4-decimal rounding)
            ok = [v for v in vals if not _undef(v)]
            m = entry.get("mean")
            if m is not None and ok:
                want = sum(ok) / len(ok)
                if abs(m - want) > 1e-4:
                    bad.append(f"{ds}/{method}/{metric}: mean={m} but values "
                               f"average to {want:.6f}")

            # --- SpEx-only metrics must not carry IDC numbers
            if method == "idc" and metric in SPEX_ONLY and ok:
                bad.append(f"{ds}/idc/{metric}: SpEx-only metric has "
                           f"{len(ok)} defined values")

    # --- the ungated reference is a CEILING for both methods
    px = d.get("generalizability_plain_x") or {}
    for i, v in enumerate(px.get("values") or []):
        if _undef(v):
            und += 1
            continue
        fin += 1
        if v < -EPS or v > 1 + EPS:
            bad.append(f"{ds}/generalizability_plain_x seed {i}: {v!r} outside [0, 1]")
        for method in ("spex", "idc"):
            g = ((d[method].get("generalizability") or {}).get("values") or [])
            # The ungated score is a REFERENCE, not a hard ceiling. X * gates
            # does not merely delete features, it rescales them per sample, and
            # that can help a regularised linear classifier a little -- measured
            # here at up to +0.006 on a few seeds. A LARGE excess would mean
            # something other than rescaling is going on, so the check keeps a
            # generous margin instead of asserting an invariant that does not
            # hold.
            if i < len(g) and not _undef(g[i]) and g[i] > v + PLAIN_X_MARGIN:
                bad.append(f"{ds}/{method}/generalizability seed {i}: {g[i]} "
                           f"exceeds the ungated reference {v} by more than "
                           f"{PLAIN_X_MARGIN}")

    # --- invariants across metrics
    for method in ("spex", "idc"):
        b = d.get(method, {})
        u = (b.get("uniqueness") or {}).get("values") or []
        s = (b.get("stability_k5") or {}).get("values") or []
        nn = (b.get("nn_identical_frac") or {}).get("values") or []
        for i, (a, c) in enumerate(zip(u, s)):
            if not _undef(a) and not _undef(c) and c < a - EPS:
                bad.append(f"{ds}/{method} seed {i}: stability_k5={c} < "
                           f"uniqueness={a} (max over a superset must not be "
                           f"smaller than min)")
        for i, (f, a) in enumerate(zip(nn, u)):
            if not _undef(f) and not _undef(a) and abs(f - 1.0) < EPS and a > EPS:
                bad.append(f"{ds}/{method} seed {i}: every neighbour has an "
                           f"identical gate row but uniqueness={a} > 0")
    return fin, und, bad


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    files = sorted(glob.glob(os.path.join(res_dir, "results_multiseed_*.json")))
    if not files:
        sys.exit(f"no results_multiseed_*.json in {res_dir}")

    fin = und = 0
    bad = []
    for p in files:
        a, b, v = check_file(p)
        fin += a; und += b; bad += v

    print("=" * 76)
    print(f"RANGE CHECK over {len(files)} result files")
    print("=" * 76)
    print(f"  {fin} finite values checked against their metric's range")
    print(f"  {und} undefined (null/nan/inf) -- not a violation, they are "
          f"reported as such")
    print()
    if bad:
        for v in bad:
            print(f"  VIOLATION  {v}")
        print(f"\n  {len(bad)} violations")
    else:
        print("  no violations")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
