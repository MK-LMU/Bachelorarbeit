# -*- coding: utf-8 -*-
"""Does the seed count change the conclusions? Compares the 5-seed campaign
against the 10-seed one.

Doubling the seeds is only worth reporting if one can say what it changed. The
comparison has two halves, and the first matters more:

  integrity   Seeds 0-4 compute the same data in both campaigns, so their
              per-seed values must be IDENTICAL. Any drift there is a bug, not
              statistics, and would invalidate the second half. Checked value
              by value across every metric.
  drift       For each metric: the two means, their difference, and whether
              that difference stays inside the 5-seed standard deviation. A
              conclusion that moves further than the old error bar was resting
              on the seed count.

For the head-to-head tests it additionally reports which verdicts changed and
how much the confidence intervals narrowed -- the actual point of n = 10.

The baseline defaults to notes/baseline_5seeds/ (kept out of the repository).
Any directory of results_multiseed_*.json works, so the 5-seed state can also
be recovered from git history:

    git checkout <commit-before-10-seeds> -- results/

Usage: seed_robustness.py [baseline_dir]
Output: results/seed_robustness.json + console report
"""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import results, RESULTS

DEFAULT_BASELINE = os.path.join(HERE, "notes", "baseline_5seeds")

# metrics worth tracking; the rest are bookkeeping fields
METRICS = ["ACC", "ARI", "NMI", "uniqueness", "uniqueness_rownorm",
           "stability_k5", "nn_identical_frac", "faithfulness_top1drop",
           "faithfulness_aopc", "faithfulness_corr", "diversity",
           "diversity_fixed", "generalizability", "ref_ARI",
           "tree_vs_ref_ARI",
           # Iris only -- it is the one dataset with duplicate rows, so the
           # plain uniqueness/stability are undefined there and these are the
           # only usable values it has. Leaving them out put 8 series outside
           # the integrity check without anyone noticing.
           "uniqueness_dedup", "stability_k5_dedup"]


def agg(block, metric):
    """(mean, std, n, values) for one metric, or None if absent."""
    v = block.get(metric)
    if not isinstance(v, dict):
        return None
    return (v.get("mean"), v.get("std"), v.get("n"), v.get("values") or [])


def same_on_shared_seeds(old_vals, old_seeds, new_vals, new_seeds):
    """Do both campaigns agree on the seeds they have in common?

    Matched by SEED NUMBER, not by position. The campaigns do not have to
    start at the same seed: the reported range moved from 0-9 to 1-10 when
    seed 0 became the configuration-search seed, so a positional comparison
    would report every value as changed while nothing had.

    None and nan both mean "undefined" and compare equal; a float comparison
    would call nan != nan and report a bug that is not one.
    """
    o_by = dict(zip(old_seeds, old_vals))
    n_by = dict(zip(new_seeds, new_vals))
    shared = sorted(set(o_by) & set(n_by))
    if not shared:
        # Not a pass. With no shared seed there is nothing to verify, and
        # silently returning ok would turn "we checked nothing" into "we found
        # nothing wrong" -- the failure mode this whole file exists to prevent.
        return False, 0, (f"no seed in common: baseline {old_seeds} vs "
                          f"current {new_seeds}")
    for k in shared:
        o, n = o_by[k], n_by[k]
        o_undef = o is None or (isinstance(o, float) and o != o)
        n_undef = n is None or (isinstance(n, float) and n != n)
        if o_undef and n_undef:
            continue
        if o_undef != n_undef or o != n:
            return False, len(shared), f"seed {k}: baseline {o!r} -> now {n!r}"
    return True, len(shared), ""


def compare_datasets(base_dir):
    rows, violations = [], []
    for p_new in sorted(glob.glob(results("results_multiseed_*.json"))):
        name = os.path.basename(p_new)
        p_old = os.path.join(base_dir, name)
        if not os.path.exists(p_old):
            continue
        old, new = json.load(open(p_old)), json.load(open(p_new))
        ds = name[len("results_multiseed_"):-len(".json")]
        o_seeds, n_seeds = old.get("seeds", []), new.get("seeds", [])
        for method in ("spex", "idc"):
            for m in METRICS:
                a, b = agg(old[method], m), agg(new[method], m)
                if a is None or b is None:
                    continue
                ok, n_shared, why = same_on_shared_seeds(a[3], o_seeds, b[3], n_seeds)
                if not ok:
                    violations.append({"dataset": ds, "method": method,
                                       "metric": m, "detail": why})
                if a[0] is None or b[0] is None:
                    continue
                delta = b[0] - a[0]
                within = None if not a[1] else abs(delta) <= a[1]
                rows.append({"dataset": ds, "method": method, "metric": m,
                             "mean_5": a[0], "std_5": a[1], "n_5": a[2],
                             "mean_10": b[0], "std_10": b[1], "n_10": b[2],
                             "delta": round(delta, 6),
                             "within_5seed_std": within})
    return rows, violations


def compare_significance(base_dir):
    p_old = os.path.join(base_dir, "significance.json")
    p_new = results("significance.json")
    if not (os.path.exists(p_old) and os.path.exists(p_new)):
        return None
    old = {x["dataset"]: x for x in json.load(open(p_old))["head_to_head"]}
    rows = []
    for x in json.load(open(p_new))["head_to_head"]:
        o = old.get(x["dataset"])
        if not o:
            continue
        # significance.py renamed ci95 -> idc_ari_ci95 once it became clear the
        # interval describes IDC's mean, not the difference. The 5-seed
        # baseline predates that, so accept either name.
        ci = lambda r: r.get("idc_ari_ci95") or r["ci95"]
        w_old = ci(o)[1] - ci(o)[0]
        w_new = ci(x)[1] - ci(x)[0]
        rows.append({"dataset": x["dataset"],
                     "test": x.get("test", "one-sample"),
                     "verdict_5": o["verdict"], "verdict_10": x["verdict"],
                     "verdict_changed": o["verdict"] != x["verdict"],
                     "p_holm_5": round(o["p_holm"], 6),
                     "p_holm_10": round(x["p_holm"], 6),
                     "idc_ci_width_5": round(w_old, 4),
                     "idc_ci_width_10": round(w_new, 4),
                     "idc_ci_narrowed_pct": round(100 * (1 - w_new / w_old), 1)
                     if w_old else None})
    return rows


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE
    if not os.path.isdir(base_dir):
        sys.exit(f"baseline directory not found: {base_dir}")
    rows, violations = compare_datasets(base_dir)
    sig = compare_significance(base_dir)

    print("=" * 78)
    print(f"INTEGRITY: do the seeds the campaigns SHARE still agree?")
    print("=" * 78)
    if violations:
        for v in violations:
            print(f"  MISMATCH {v['dataset']:<20} {v['method']:<5} "
                  f"{v['metric']:<22} {v['detail']}")
        print(f"\n  {len(violations)} violations -- the new campaign did not "
              f"merely ADD seeds. Investigate before reading anything below.")
    else:
        print(f"  clean: {len(rows)} metric series, every value the two "
              f"campaigns share by seed number reproduced exactly")

    moved = [r for r in rows if r["within_5seed_std"] is False]
    print()
    print("=" * 78)
    print("DRIFT: means that moved further than the 5-seed standard deviation")
    print("=" * 78)
    if moved:
        print(f"  {'dataset':<20}{'meth':<6}{'metric':<22}"
              f"{'5 seeds':>10}{'10 seeds':>10}{'delta':>9}")
        for r in sorted(moved, key=lambda r: -abs(r["delta"])):
            print(f"  {r['dataset']:<20}{r['method']:<6}{r['metric']:<22}"
                  f"{r['mean_5']:>10.3f}{r['mean_10']:>10.3f}{r['delta']:>+9.3f}")
    else:
        print("  none -- every metric stayed inside its own 5-seed error bar")
    print(f"\n  {len(moved)} of {len(rows)} metric means moved beyond one "
          f"5-seed sigma")

    if sig:
        print()
        print("=" * 78)
        print("HEAD-TO-HEAD: verdicts and interval width")
        print("=" * 78)
        print(f"  {'dataset':<22}{'verdict (5)':<15}{'verdict (10)':<15}"
              f"{'p_Holm 5':>10}{'p_Holm 10':>11}{'IDC CI':>8}")
        for r in sig:
            mark = "  <--" if r["verdict_changed"] else ""
            nar = (f"{r['idc_ci_narrowed_pct']:+.0f}%"
                   if r["idc_ci_narrowed_pct"] is not None else "-")
            print(f"  {r['dataset']:<22}{r['verdict_5']:<15}{r['verdict_10']:<15}"
                  f"{r['p_holm_5']:>10.3f}{r['p_holm_10']:>11.3f}{nar:>8}{mark}")
        ch = [r["dataset"] for r in sig if r["verdict_changed"]]
        print(f"\n  verdicts changed: {ch if ch else 'none'}")

    out = {"baseline_dir": os.path.relpath(base_dir, HERE),
           "integrity_violations": violations,
           "n_metric_series": len(rows),
           "n_moved_beyond_5seed_sigma": len(moved),
           "datasets": rows, "head_to_head": sig}
    jp = results("seed_robustness.json")
    json.dump(out, open(jp, "w"), indent=2)
    print(f"\nsaved {os.path.relpath(jp, HERE)}")
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
