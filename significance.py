# -*- coding: utf-8 -*-
"""Significance of the head-to-head ARI comparisons and of the architecture
spot check — recomputed from the result JSONs so that the p-values in the
thesis come from a listed, rerunnable command.

Head-to-head: the SpEx pipeline's ARI is constant across seeds on every
dataset (asserted below from the `values` arrays), so each comparison is a
ONE-SAMPLE t-test of IDC's 5 per-seed ARIs against that constant, two-sided.
n = 5 is too small to check normality, so 95 % t-intervals are reported
alongside. Across the 9 tests, raw p-values are kept for comparability with
the tables but `verdict` uses the Holm–Bonferroni adjusted p (FWER 0.05).

Architecture check: 3 seeds of the small/tiny networks vs. the 5 `_best`
seeds of the default — Welch t-tests, Holm-adjusted within their own family.

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
        assert np.ptp(spex) == 0.0, f"{label}: SpEx ARI varies across seeds {spex}"
        t, p = stats.ttest_1samp(idc, spex[0])
        ci = stats.t.interval(0.95, len(idc) - 1, loc=idc.mean(), scale=stats.sem(idc))
        rows.append({"dataset": label, "file": f"results_multiseed_{suffix}.json",
                     "spex_ari": round(float(spex[0]), 4),
                     "idc_ari_mean": round(float(idc.mean()), 4),
                     "idc_ari_std": round(float(idc.std(ddof=1)), 4),
                     "n": int(len(idc)), "delta": round(float(idc.mean() - spex[0]), 4),
                     "ci95": [round(float(ci[0]), 6), round(float(ci[1]), 6)],
                     "t": round(float(t), 3), "p": float(p)})
    adj = holm([x["p"] for x in rows])
    for x, pa in zip(rows, adj):
        x["p_holm"] = float(pa)
        x["verdict"] = ("not resolved" if pa >= ALPHA else
                        "IDC better" if x["delta"] > 0 else "SpEx better")
    return rows


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
        base = a[ds]["default"]["values"]
        for arch in ("small", "tiny"):
            if arch not in a[ds]:
                continue
            v = a[ds][arch]["values"]
            t, p = stats.ttest_ind(base, v, equal_var=False)       # Welch
            rows.append({"dataset": ds, "arch": arch, "n_default": len(base), "n_arch": len(v),
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
           "method_head_to_head": "one-sample t-test (two-sided) of IDC per-seed ARI "
                                  "against the seed-constant SpEx ARI; Holm-Bonferroni "
                                  "over the 9 comparisons",
           "method_architecture": "Welch two-sample t-test default (5 seeds) vs small/tiny "
                                  "(3 seeds); Holm-Bonferroni over the 4 tests",
           "head_to_head": h2h, "architecture_check": arch,
           "n_resolved_raw": int(sum(x["p"] < ALPHA for x in h2h)),
           "n_resolved_holm": int(sum(x["p_holm"] < ALPHA for x in h2h))}
    jp = results("significance.json")
    json.dump(out, open(jp, "w"), indent=2)

    print(f"{'dataset':<22}{'SpEx':>7}{'IDC mean±std':>16}{'95% CI':>18}"
          f"{'p':>8}{'p_Holm':>8}  verdict")
    print("-" * 92)
    for x in h2h:
        print(f"{x['dataset']:<22}{x['spex_ari']:>7.3f}"
              f"{x['idc_ari_mean']:>9.3f} ±{x['idc_ari_std']:<5.3f}"
              f"  [{x['ci95'][0]:>6.3f},{x['ci95'][1]:>6.3f}]"
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
