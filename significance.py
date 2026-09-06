# -*- coding: utf-8 -*-
"""Significance of the head-to-head quality comparisons (ARI, NMI, ACC) and of
the architecture spot check — recomputed from the result JSONs so that the
p-values in the thesis come from a listed, rerunnable command.

Head-to-head: the SpEx pipeline's ARI has been constant across seeds on every
dataset, which is what licenses a ONE-SAMPLE t-test of IDC's per-seed ARIs
against that constant. The code checks rather than assumes it and falls back
to Welch where it does not hold; `test` records which was used per comparison.
Either way n is small enough that normality cannot be checked, so 95 %
t-intervals are reported alongside, and every comparison is repeated as a
one-sample Wilcoxon signed-rank test (`p_wilcoxon`, distribution-free). The
same test is run for each of the three quality measures the proposal names —
ARI (headline, block `head_to_head`), NMI and ACC (blocks `head_to_head_nmi`,
`head_to_head_acc`) — each as its own Holm family of 9. Raw p-values are kept
for comparability with the tables but `verdict` uses the Holm–Bonferroni
adjusted p (FWER 0.05).

Architecture check: the small/tiny networks against the default's `_best`
seeds — Welch t-tests, Holm-adjusted within their own family.

Family sensitivity: all reported tests (3 × 9 + 4) pooled into ONE Holm
family; the verdicts that change are the ones that hang on the family
definition and are reported as exploratory.

Output: results/significance.json + console table.
"""
import os, sys, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import results

ALPHA = 0.05
# The proposal names ACC, ARI and NMI as the clustering-quality measures. ARI
# is the headline (chance-corrected); NMI is reported next to it in the results
# chapter, ACC in the appendix. Each metric is its own Holm family.
METRICS = ["ARI", "NMI", "ACC"]
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


def verdict_of(p_adj, delta):
    return ("not resolved" if p_adj >= ALPHA else
            "IDC better" if delta > 0 else "SpEx better")


def head_to_head(metric="ARI"):
    """One row per dataset for one quality metric. Field names carry the metric
    (spex_ari, idc_ari_mean, idc_ari_ci95, ...), so the ARI block keeps the
    names every downstream reader has used since the 5-seed phase."""
    m = metric.lower()
    rows = []
    for label, suffix in COMPARISONS:
        r = json.load(open(results(f"results_multiseed_{suffix}.json")))
        spex = np.array([v for v in r["spex"][metric]["values"] if v is not None], float)
        idc = np.array([v for v in r["idc"][metric]["values"] if v is not None], float)
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
        # Distribution-free companion: one-sample Wilcoxon signed-rank of the
        # seed values against SpEx's mean (exact at n = 10 without zero
        # differences). A mean can hide a mixture -- six collapsed Iris seeds
        # against four working ones -- and this test does not care.
        diff = idc - spex.mean()
        if np.any(diff != 0):
            pw = float(stats.wilcoxon(diff, alternative="two-sided", method="auto").pvalue)
        else:
            pw = 1.0
        rows.append({"dataset": label, "file": f"results_multiseed_{suffix}.json",
                     "metric": metric, "test": test,
                     # 6 decimals, not 4: a mean of 0.4055 stored at 4 dp
                     # formats to 0.406 in a 3-dp table while the seeds average
                     # to 0.405499. Same reason as evaluate.agg.
                     f"spex_{m}": round(float(spex.mean()), 6),
                     f"spex_{m}_std": round(float(spex.std(ddof=1)), 6) if len(spex) > 1 else 0.0,
                     "n_spex": int(len(spex)),
                     f"idc_{m}_mean": round(float(idc.mean()), 6),
                     f"idc_{m}_std": round(float(idc.std(ddof=1)), 6),
                     "n": int(len(idc)),
                     "delta": round(float(idc.mean() - spex.mean()), 6),
                     # Interval of IDC's MEAN, not of the difference: it is
                     # centred on idc.mean() and neither brackets `delta` nor
                     # need share its sign. Named accordingly so it cannot be
                     # read as an effect interval.
                     f"idc_{m}_ci95": [round(float(ci[0]), 6), round(float(ci[1]), 6)],
                     "t": round(float(t), 3), "p": float(p),
                     "p_wilcoxon": pw})
    adj = holm([x["p"] for x in rows])
    adj_w = holm([x["p_wilcoxon"] for x in rows])
    for x, pa, pw in zip(rows, adj, adj_w):
        x["p_holm"] = float(pa)
        x["verdict"] = verdict_of(pa, x["delta"])
        x["p_wilcoxon_holm"] = float(pw)
        x["verdict_wilcoxon"] = verdict_of(pw, x["delta"])
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
                         "default_mean": round(float(np.mean(base)), 6),
                         "arch_mean": round(float(np.mean(v)), 6),
                         "delta": round(float(np.mean(v) - np.mean(base)), 6),
                         "t_welch": round(float(t), 3), "p": float(p)})
    adj = holm([x["p"] for x in rows])
    for x, pa in zip(rows, adj):
        x["p_holm"] = float(pa)
        x["significant_holm"] = bool(pa < ALPHA)
    return rows


def joint_family(h2h_by_metric, arch):
    """Sensitivity of the Holm verdicts to the family definition.

    Holm controls the FWER within the family it is applied to, and this file
    applies it four times -- 9 head-to-head tests per quality metric (ARI,
    NMI, ACC) and 4 architecture tests -- because the families answer
    different questions. That split is a choice, not a fact, so the
    alternative is computed as well: one family of all 31 tests. Whichever
    verdicts differ between the two are the ones that hang on the choice and
    have to be reported as exploratory rather than confirmed."""
    if not arch:
        return None
    tests = []
    for metric, rows in h2h_by_metric.items():
        fam = "head_to_head" if metric == "ARI" else f"head_to_head_{metric.lower()}"
        tests += [{"family": fam, "label": f"{x['dataset']} ({metric})", "p": x["p"],
                   "p_holm_split": x["p_holm"]} for x in rows]
    tests += [{"family": "architecture_check",
               "label": f"{x['dataset']} {x['arch']}", "p": x["p"],
               "p_holm_split": x["p_holm"]} for x in arch]
    changed = []
    for t, pa in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm_joint"] = float(pa)
        if (t["p_holm_split"] < ALPHA) != (pa < ALPHA):
            changed.append(t["label"])
    return {"n_tests": len(tests),
            "method": "Holm-Bonferroni over the head-to-head tests of all three "
                      "quality metrics and the architecture tests pooled into "
                      "ONE family, as a sensitivity check on the per-family "
                      "split used for the reported verdicts",
            "tests": tests, "verdicts_changed": changed}


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_h2h(metric, rows):
    m = metric.lower()
    print(f"\n{metric}: {'dataset':<22}{'SpEx':>7}{'IDC mean±std':>16}{'95% CI (IDC mean)':>19}"
          f"{'p':>8}{'p_Holm':>8}{'Wilc.':>8}  verdict")
    print("-" * 105)
    for x in rows:
        flag = "" if x["verdict"] == x["verdict_wilcoxon"] else "  (Wilcoxon differs)"
        print(f"{'':5}{x['dataset']:<22}{x[f'spex_{m}']:>7.3f}"
              f"{x[f'idc_{m}_mean']:>9.3f} ±{x[f'idc_{m}_std']:<5.3f}"
              f"  [{x[f'idc_{m}_ci95'][0]:>6.3f},{x[f'idc_{m}_ci95'][1]:>6.3f}]"
              f"{fmt_p(x['p']):>8}{fmt_p(x['p_holm']):>8}{fmt_p(x['p_wilcoxon_holm']):>8}"
              f"  {x['verdict']}{flag}")
    print(f"{'':5}resolved at alpha={ALPHA}: {sum(x['p'] < ALPHA for x in rows)}/9 raw, "
          f"{sum(x['p_holm'] < ALPHA for x in rows)}/9 Holm-adjusted")


def main():
    h2h = {metric: head_to_head(metric) for metric in METRICS}
    arch = architecture_check()
    joint = joint_family(h2h, arch)
    out = {"alpha": ALPHA,
           "metrics": METRICS,
           "method_head_to_head": "two-sided t-test of IDC's per-seed value against "
                                  "SpEx's; one-sample against the constant where "
                                  "SpEx is seed-constant (field `test`), Welch "
                                  "otherwise; Holm-Bonferroni over the 9 comparisons "
                                  "of each metric separately (blocks head_to_head = "
                                  "ARI, head_to_head_nmi, head_to_head_acc). "
                                  "p_wilcoxon: one-sample Wilcoxon signed-rank of the "
                                  "same seeds against the SpEx value, Holm within the "
                                  "same family. idc_<metric>_ci95 is the interval of "
                                  "IDC's MEAN, not of the difference to SpEx",
           "method_architecture": "Welch two-sample t-test, default vs small/tiny "
                                  "(per-test n in n_default/n_arch); "
                                  "Holm-Bonferroni over the 4 tests",
           "head_to_head": h2h["ARI"],
           "head_to_head_nmi": h2h["NMI"],
           "head_to_head_acc": h2h["ACC"],
           "architecture_check": arch,
           "family_sensitivity_joint": joint,
           "n_resolved_raw": int(sum(x["p"] < ALPHA for x in h2h["ARI"])),
           "n_resolved_holm": int(sum(x["p_holm"] < ALPHA for x in h2h["ARI"])),
           "n_resolved_holm_by_metric": {metric: int(sum(x["p_holm"] < ALPHA for x in rows))
                                         for metric, rows in h2h.items()},
           "verdicts_differ_from_ari": {
               metric: [x["dataset"] for x, a in zip(rows, h2h["ARI"])
                        if x["verdict"] != a["verdict"]]
               for metric, rows in h2h.items() if metric != "ARI"}}
    jp = results("significance.json")
    json.dump(out, open(jp, "w"), indent=2)

    for metric in METRICS:
        print_h2h(metric, h2h[metric])
    print("\nverdicts that differ from the ARI family: "
          + "; ".join(f"{k}: {', '.join(v) or 'none'}"
                      for k, v in out["verdicts_differ_from_ari"].items()))
    if arch:
        print(f"\n{'architecture check (Welch)':<30}{'default':>9}{'arch':>9}{'delta':>8}"
              f"{'p':>8}{'p_Holm':>8}")
        print("-" * 72)
        for x in arch:
            print(f"{x['dataset'] + ' ' + x['arch']:<30}{x['default_mean']:>9.3f}"
                  f"{x['arch_mean']:>9.3f}{x['delta']:>+8.3f}{fmt_p(x['p']):>8}"
                  f"{fmt_p(x['p_holm']):>8}{'  *' if x['significant_holm'] else ''}")
    if joint:
        print(f"\nfamily sensitivity: Holm over all {joint['n_tests']} tests "
              f"as ONE family")
        for t in joint["tests"]:
            if t["p_holm_split"] < ALPHA or t["p_holm_joint"] < ALPHA:
                print(f"  {t['label']:<28}p={fmt_p(t['p'])}"
                      f"  split={fmt_p(t['p_holm_split'])}"
                      f"  joint={fmt_p(t['p_holm_joint'])}")
        print("  verdicts that change: "
              + (", ".join(joint["verdicts_changed"]) or "none"))
    print(f"\nsaved {os.path.relpath(jp, HERE)}")


if __name__ == "__main__":
    main()
