# -*- coding: utf-8 -*-
"""Significance of the head-to-head ARI comparisons and of the architecture
spot check — recomputed from the result JSONs so that the p-values in the
thesis come from a listed, rerunnable command.

Head-to-head: the SpEx pipeline's ARI has been constant across seeds on every
dataset, which is what licenses a ONE-SAMPLE t-test of IDC's per-seed ARIs
against that constant. The code checks rather than assumes it and falls back
to Welch where it does not hold; `test` records which was used per comparison.
Either way n is small enough that normality cannot be checked, so 95 %
t-intervals are reported alongside. Across the 9 tests, raw p-values are kept
for comparability with the tables but `verdict` uses the Holm–Bonferroni
adjusted p (FWER 0.05).

Architecture check: the small/tiny networks against the default's `_best`
seeds — Welch t-tests, Holm-adjusted within their own family.

Output: results/significance.json + console table.
"""
import os, sys, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import results

ALPHA = 0.05
# (label, results file suffix) — IDC column = tuned `_best` run, MNIST = IDC's
# paper-validated config (no tuning), which is why MNIST has no `_best` entry.
COMPARISONS = [("Blobs", "blobs_best"), ("Digits", "digits_best"),
               ("CIFAR-10", "cifar10_best"), ("MNIST (ResNet feats)", "mnist_feats_best"),
               ("MNIST (validated)", "mnist"), ("Iris", "iris_best"),
               ("Breast Cancer", "breast_cancer_best"), ("Two Moons", "two_moons_best"),
               ("HAR", "har_best")]


def holm(pvals):
    """Holm–Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def head_to_head():
    rows = []
    for label, suffix in COMPARISONS:
        r = json.load(open(results(f"results_multiseed_{suffix}.json")))
        spex = np.array([v for v in r["spex"]["ARI"]["values"] if v is not None], float)
        idc = np.array([v for v in r["idc"]["ARI"]["values"] if v is not None], float)
        # SpEx has been seed-constant on every dataset so far, which is what
        # makes the one-sample test legitimate. Do not assume it: a wider seed
        # range could break it, and an assert here would kill the whole run for
        # one dataset. Fall back to Welch and say which test was used.
        if np.ptp(spex) == 0.0:
            test = "one-sample"
            t, p = stats.ttest_1samp(idc, spex[0])
        else:
            test = "welch"
            t, p = stats.ttest_ind(idc, spex, equal_var=False)
        ci = stats.t.interval(0.95, len(idc) - 1, loc=idc.mean(), scale=stats.sem(idc))
        rows.append({"dataset": label, "file": f"results_multiseed_{suffix}.json",
                     "test": test,
                     "spex_ari": round(float(spex.mean()), 4),
                     "spex_ari_std": round(float(spex.std(ddof=1)), 4) if len(spex) > 1 else 0.0,
                     "n_spex": int(len(spex)),
                     "idc_ari_mean": round(float(idc.mean()), 4),
                     "idc_ari_std": round(float(idc.std(ddof=1)), 4),
                     "n": int(len(idc)),
                     "delta": round(float(idc.mean() - spex.mean()), 4),
                     # Interval of IDC's MEAN, not of the difference: it is
                     # centred on idc.mean() and neither brackets `delta` nor
                     # need share its sign. Named accordingly so it cannot be
                     # read as an effect interval.
                     "idc_ari_ci95": [round(float(ci[0]), 6), round(float(ci[1]), 6)],
                     "t": round(float(t), 3), "p": float(p)})
    adj = holm([x["p"] for x in rows])
    for x, pa in zip(rows, adj):
        x["p_holm"] = float(pa)
        x["verdict"] = ("not resolved" if pa >= ALPHA else
                        "IDC better" if x["delta"] > 0 else "SpEx better")
    return rows


def default_arm(ds, stored):
    """The default-architecture ARIs for a dataset, from the CANONICAL results.

    arch_check.json also stores them, but as a copy made when the campaign ran
    5 seeds -- and that copy does not grow when the campaign does. Reading it
    left significance.json asserting n=5 for the same runs its head-to-head
    half reported at n=10, and one Holm verdict hung on the difference. The
    small/tiny arms have no canonical counterpart and still come from the file.
    """
    p = results(f"results_multiseed_{ds}_best.json")
    if not os.path.exists(p):
        return stored, "arch_check.json (no canonical _best file)"
    v = [x for x in json.load(open(p))["idc"]["ARI"]["values"] if x is not None]
    return v, f"results_multiseed_{ds}_best.json"


def architecture_check():
    # Absent whenever check_architecture.py was not run. Returning None makes
    # the JSON's architecture section null and leaves the head-to-head tests
    # untouched, so this file stays runnable without that script.
    p_arch = results("arch_check.json")
    if not os.path.exists(p_arch):
        return None
    a = json.load(open(p_arch))
    rows = []
    for ds in a:
        base, base_src = default_arm(ds, a[ds]["default"]["values"])
        for arch in ("small", "tiny"):
            if arch not in a[ds]:
                continue
            v = a[ds][arch]["values"]
            t, p = stats.ttest_ind(base, v, equal_var=False)       # Welch
            rows.append({"dataset": ds, "arch": arch, "default_from": base_src,
                         "n_default": len(base), "n_arch": len(v),
                         "default_mean": round(float(np.mean(base)), 4),
                         "arch_mean": round(float(np.mean(v)), 4),
                         "delta": round(float(np.mean(v) - np.mean(base)), 4),
                         "t_welch": round(float(t), 3), "p": float(p)})
    adj = holm([x["p"] for x in rows])
    for x, pa in zip(rows, adj):
        x["p_holm"] = float(pa)
        x["significant_holm"] = bool(pa < ALPHA)
    return rows


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    h2h = head_to_head()
    arch = architecture_check()
    out = {"alpha": ALPHA,
           "method_head_to_head": "two-sided t-test of IDC's per-seed ARI against "
                                  "SpEx's; one-sample against the constant where "
                                  "SpEx is seed-constant (field `test`), Welch "
                                  "otherwise; Holm-Bonferroni over the 9 comparisons. "
                                  "idc_ari_ci95 is the interval of IDC's MEAN "
                                  "ARI, not of the difference to SpEx",
           "method_architecture": "Welch two-sample t-test, default vs small/tiny "
                                  "(per-test n in n_default/n_arch); "
                                  "Holm-Bonferroni over the 4 tests",
           "head_to_head": h2h, "architecture_check": arch,
           "n_resolved_raw": int(sum(x["p"] < ALPHA for x in h2h)),
           "n_resolved_holm": int(sum(x["p_holm"] < ALPHA for x in h2h))}
    jp = results("significance.json")
    json.dump(out, open(jp, "w"), indent=2)

    print(f"{'dataset':<22}{'SpEx':>7}{'IDC mean±std':>16}{'95% CI (IDC mean)':>19}"
          f"{'p':>8}{'p_Holm':>8}  verdict")
    print("-" * 92)
    for x in h2h:
        print(f"{x['dataset']:<22}{x['spex_ari']:>7.3f}"
              f"{x['idc_ari_mean']:>9.3f} ±{x['idc_ari_std']:<5.3f}"
              f"  [{x['idc_ari_ci95'][0]:>6.3f},{x['idc_ari_ci95'][1]:>6.3f}]"
              f"{fmt_p(x['p']):>8}{fmt_p(x['p_holm']):>8}  {x['verdict']}")
    print(f"\nresolved at alpha={ALPHA}: {out['n_resolved_raw']}/9 raw, "
          f"{out['n_resolved_holm']}/9 Holm-adjusted")
    if arch:
        print(f"\n{'architecture check (Welch)':<30}{'default':>9}{'arch':>9}{'delta':>8}"
              f"{'p':>8}{'p_Holm':>8}")
        print("-" * 72)
        for x in arch:
            print(f"{x['dataset'] + ' ' + x['arch']:<30}{x['default_mean']:>9.3f}"
                  f"{x['arch_mean']:>9.3f}{x['delta']:>+8.3f}{fmt_p(x['p']):>8}"
                  f"{fmt_p(x['p_holm']):>8}{'  *' if x['significant_holm'] else ''}")
    print(f"\nsaved {os.path.relpath(jp, HERE)}")


if __name__ == "__main__":
    main()
