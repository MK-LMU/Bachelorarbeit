# -*- coding: utf-8 -*-
"""Generate RESULTS_MULTISEED.md — one document with ALL multi-seed values
(mean±std) for both methods on all datasets, from results_multiseed_<ds>.json
plus the per-seed idc_out_<ds>_seed<k>.npz (for IDC's correlation faithfulness,
which the aggregator does not carry). Rerun after any campaign extension."""
import os, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["two_moons", "two_moons_tuned", "two_moons_best",
         "blobs", "blobs_tuned", "blobs_best",
         "iris", "iris_best", "breast_cancer", "breast_cancer_best",
         "digits", "digits_best", "har", "har_best",
         "cifar10", "cifar10_best", "mnist", "mnist_feats", "mnist_feats_best"]
NAMES = {"two_moons": "Two Moons", "blobs": "Gaussian Blobs", "iris": "Iris",
         "breast_cancer": "Breast Cancer", "digits": "Digits", "har": "HAR",
         "cifar10": "CIFAR-10 (ResNet feats)", "mnist": "MNIST",
         "mnist_feats": "MNIST (ResNet feats)",
         "two_moons_tuned": "Two Moons — IDC tuned (LGL=0.1)",
         "blobs_tuned": "Gaussian Blobs — IDC tuned (LGL=0.1)"}
for _ds in list(ORDER):
    if _ds.endswith("_best"):
        NAMES[_ds] = NAMES[_ds[:-5]] + " — IDC best (grid/silhouette)"

# Per-dataset rows, grouped by the question each group answers. The subtitle is
# printed under the group heading so the table reads without external notes.
GROUPS = [
    ("Clustering quality",
     "how well does the partition match the true classes (higher = better)",
     [("ACC", "ACC"), ("ARI", "ARI"), ("NMI", "NMI")]),
    ("Decomposition: reference quality vs. tree approximation",
     "SpEx only — is a weak result the reference's fault or the tree's? "
     "ARI(tree, ref) near 1 means the tree copies its reference faithfully",
     [("ref_ARI", "ARI(spectral ref, y)"), ("tree_vs_ref_ARI", "ARI(tree, ref)")]),
    ("Explanation granularity",
     "how individual are the explanations? nn-identical = share of nearest-"
     "neighbour pairs with an IDENTICAL explanation row (high = one template "
     "per region, low = per-sample); it is scale-invariant and carries the "
     "granularity claim, uniqueness/stability measure magnitude distances",
     [("nn_identical_frac", "nn-identical frac"),
      ("uniqueness", "uniqueness (raw)"),
      ("uniqueness_rownorm", "uniqueness (row-norm)"),
      ("stability_k5", "stability (k=5)")]),
    ("Faithfulness",
     "do the features called important actually carry the assignment? "
     "(higher = better; masking sets features to 0)",
     [("faithfulness_top1drop", "faithfulness top-1 drop"),
      ("faithfulness_aopc", "faithfulness AOPC")]),
    ("Diversity across classes",
     "do different classes get different features? the original metric is "
     "defective (both defects are spelled out below) — use diversity_fixed",
     [("diversity", "diversity (original, defective)")]),
    ("Generalizability",
     "can a simple linear model still separate held-out samples from "
     "data x explanation? (higher = better)",
     [("generalizability", "generalizability")]),
]


def cfg_label(ds, r):
    if r.get("idc_config_validated"):
        return "**validated**"
    if ds.endswith("_best"):
        return "tuned (grid, silhouette-selected)"
    return "tuned (LGL=0.1)" if ds.endswith("_tuned") else "default"


def _num(v):
    """3 decimals, but never round a nonzero value to 0.000 (e.g. HAR
    uniqueness 0.0003 must not read as exactly zero)."""
    if v != 0 and abs(v) < 0.0005:
        return f"{v:.1e}"
    return f"{v:.3f}"


def fmt(a):
    if a is None or a.get("mean") is None:
        return "nan"
    s = _num(a["mean"])
    if a.get("std") is not None:
        s += f" ±{_num(a['std'])}"
    n = a.get("n")
    if n is not None and n < 5:
        # Aggregated over FEWER than 5 seeds because the rest were nan. Marking
        # only n==1 (the old rule) hid the survivorship case: on iris_best three
        # IDC seeds collapsed to one cluster, so a 2-seed mean stood next to a
        # 5-seed SpEx mean and read like a fair comparison.
        s += f" ⁿ⁼{n}"
    return s


def idc_corr_faithfulness(ds, seeds):
    vals = []
    for s in seeds:
        from wpaths import idc_out
        p = idc_out(f"idc_out_{ds}_seed{s}.npz")
        if os.path.exists(p):
            d = np.load(p)
            if "faithfulness" in d:
                vals.append(float(d["faithfulness"]))
    ok = [v for v in vals if np.isfinite(v)]
    if not ok:
        return "nan"
    s = f"{np.mean(ok):.3f}"
    if len(ok) > 1:
        s += f" ±{np.std(ok, ddof=1):.3f}"
    return s + (f" (n={len(ok)}/{len(vals)})" if len(ok) < len(vals) else "")


def main():
    lines = [
        "# Multi-seed results — all metrics, all datasets (mean ± std)",
        "",
        "Both methods are scored by **IDC's own metric code, imported verbatim**",
        "(faithfulness, diversity, uniqueness, stability, generalizability), on",
        "identical data. Rows added by this work are marked as such below and",
        "follow the established literature: AOPC — Samek et al. 2017; deletion —",
        "Petsiuk et al. 2018; stability of explanations — Alvarez-Melis &",
        "Jaakkola 2018.",
        "",
        "Generated by `gen_results_table.py` from `results_multiseed_<ds>.json`",
        "(campaign: 5 seeds per dataset; SpEx = Spectral+tree pipeline,",
        "spectral `random_state` = seed; IDC retrained per seed, weights persisted).",
        "Every number is reproducible from the commands listed in",
        "`reproduce_all.py`.",
        "",
        "Notes:",
        "- `faithfulness (correlation)` is IDC's original metric. It is a Pearson",
        "  correlation between claimed importance and the accuracy drop while",
        "  features are removed one by one, and is therefore **structurally",
        "  undefined** whenever fewer than 2 features are used or the masking",
        "  curve is exactly constant — a cliff instead of a slope. That is a",
        "  property of the metric, not missing data; the drop-based rows",
        "  (this work's addition) are defined everywhere.",
        "- `diversity` (IDC's original) has two implementation defects and its",
        "  values are struck through for IDC. (1) The Jaccard matrix is summed",
        "  INCLUDING the diagonal (each set compared with itself = 1) while the",
        "  divisor covers only the off-diagonal pairs K(K−1) — as long as every",
        "  cluster's median set is non-empty the attainable maximum is therefore",
        "  100·(1−1/(K−1)), i.e. 88.9 at K=10, 80 at K=6 and 0 at K=2, and the",
        "  scale is not comparable across datasets with different K. Values ABOVE",
        "  that bound (e.g. IDC on HAR: 86.7 and 90.0 at K=6) arise only through",
        "  defect (2): an empty median set contributes 0 to the sum instead of 1.",
        "  (2) `sklearn.jaccard_score` returns 0 for two EMPTY sets, so a cluster",
        "  whose median gate vector is all-zero counts as maximally diverse —",
        "  saying nothing scores best. Per-seed evidence in the JSONs: MNIST IDC",
        "  scores 98.889 on one seed and 100.0 on the others — 98.889 is exactly",
        "  100·(1−1/90), the signature of a single non-empty set. `diversity_fixed`",
        "  repairs both defects: off-diagonal",
        "  pairs only, weighted Jaccard on mean gate vectors, empty -> nan.",
        "  SpEx's original values are legitimate but only comparable within",
        "  the same K.",
        "- IDC config is paper-validated **only for MNIST**; all other IDC columns",
        "  use a default config and are lower bounds under un-tuned settings.",
        "- SpEx rows: the Spectral+tree pipeline is seed-invariant in everything",
        "  that depends only on the tree — ACC/ARI/NMI, all gate metrics,",
        "  faithfulness and diversity have std 0.000 across the 5 spectral seeds",
        "  on every dataset (on CIFAR/MNIST the spectral reference itself moves",
        "  by ~1e-4 in ARI(ref, y); the tree output does not). generalizability",
        "  is the exception by design: its 70/30 split uses the seed, identically",
        "  for both methods. Rows are 5-seed aggregates unless marked ⁿ⁼k,",
        "  which means the value averages only k < 5 seeds because the others",
        "  returned nan — read those against the 5-seed value beside them with",
        "  care, since the dropped seeds are usually the degenerate ones.",
        "- `IDC best` rows: config selected per dataset from a LGL x epochs grid",
        "  by the UNSUPERVISED silhouette score (no label access; candidates'",
        "  ARI was logged but never used for selection — supervised tuning",
        "  would be leakage). Grid + winners: results/tuning/selection.json.",
        "- **uniqueness vs nn-identical can rank the methods differently**",
        "  (e.g. Digits): uniqueness measures gate *magnitude* distances and",
        "  shrinks when gates are weak (un-tuned IDC), nn-identical measures",
        "  *identity* and is config/scale-robust — base granularity claims on",
        "  nn-identical.",
        "- generalizability does not penalise non-selection: constant all-open",
        "  gates (tuned 2-D IDC) reduce it to plain-X separability.",
        "",
        "## Reading the per-dataset tables",
        "",
        "Every dataset block below groups its rows under these headings; the",
        "groups are explained here once.",
        "",
        "| group | what the numbers answer |",
        "|---|---|",
    ] + [f"| **{t}** | {sub} |" for t, sub, _ in GROUPS] + [
        "",
        "## Overview (headline metrics)",
        "",
        "| dataset | k-means ARI (baseline) | ARI SpEx | ARI IDC | nn-ident SpEx | nn-ident IDC | faith. top-1 drop SpEx | faith. top-1 drop IDC | IDC config |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    data = {}
    for ds in ORDER:
        from wpaths import results
        p = results(f"results_multiseed_{ds}.json")
        if not os.path.exists(p):
            continue
        data[ds] = json.load(open(p))
        r = data[ds]
        km = f"{r['kmeans_ari']:.3f}" if r.get("kmeans_ari") is not None else "—"
        lines.append(
            f"| {NAMES[ds]} | {km} | {fmt(r['spex']['ARI'])} | {fmt(r['idc']['ARI'])} "
            f"| {fmt(r['spex']['nn_identical_frac'])} | {fmt(r['idc']['nn_identical_frac'])} "
            f"| {fmt(r['spex']['faithfulness_top1drop'])} | {fmt(r['idc']['faithfulness_top1drop'])} "
            f"| {cfg_label(ds, r)} |")

    for ds, r in data.items():
        km_note = (f", k-means baseline ARI {r['kmeans_ari']:.3f}"
                   if r.get("kmeans_ari") is not None else "")
        lines += ["", f"## {NAMES[ds]}  (N={r['N']}, D={r['D']}, K={r['K']}, "
                      f"seeds={r['seeds']}, IDC config: {cfg_label(ds, r).strip('*')}"
                      f"{km_note})",
                  "", "| metric | SpEx (Spectral+tree, +Tree SHAP) | IDC |", "|---|---|---|"]

        zs, zi = r.get("spex_zero_gate_seeds", []), r.get("idc_zero_gate_seeds", [])
        if zs or zi:
            # A run whose gate matrix is identically zero produced NO explanation,
            # yet the distance metrics still return numbers (uniqueness 0,
            # nn-identical 1.0, diversity at the ceiling). Say so, or "said
            # nothing" reads as "maximally stable and maximally diverse".
            parts = []
            if zs:
                parts.append(f"SpEx seeds {zs}")
            if zi:
                parts.append(f"IDC seeds {zi}")
            lines += [f"> **All gates zero** in {' and '.join(parts)}: those runs "
                      "produced no explanation at all. The granularity and diversity "
                      "rows below still show numbers for them, but those numbers "
                      "describe an empty matrix, not a coarse one.", ""]

        corr = idc_corr_faithfulness(ds, r["seeds"])
        spex_corr = (fmt(r["spex"]["faithfulness_corr"])
                     if r["spex"].get("faithfulness_corr") else "not recomputed")
        if spex_corr == "nan":
            spex_corr = "undef.*"

        for title, _subtitle, rows in GROUPS:   # explained once in the legend above
            lines.append(f"| **{title}** | | |")
            for key, label in rows:
                s, i = fmt(r["spex"].get(key)), fmt(r["idc"].get(key))
                if key == "diversity":
                    # Mark by CONDITION, not by column: the diagonal offset hits
                    # both methods equally, so striking through only the IDC cell
                    # gave identical numbers opposite labels (two_moons_tuned:
                    # -100.000 plain on one side, struck through on the other).
                    # What is demonstrably degenerate is a value ABOVE the
                    # ceiling 100*(1-1/(K-1)) -- only an empty median set can
                    # produce that.
                    ceil = 100.0 * (1 - 1 / (r["K"] - 1)) if r["K"] > 1 else 0.0
                    for side, cell in (("spex", "s"), ("idc", "i")):
                        m = (r[side].get(key) or {}).get("mean")
                        if m is not None and m > ceil + 1e-9:
                            val = f"~~{s if cell == 's' else i}~~ (above the ceiling"
                            val += f" {ceil:.1f} -> empty feature sets)"
                            if cell == "s":
                                s = val
                            else:
                                i = val
                if key in ("ref_ARI", "tree_vs_ref_ARI"):
                    i = "—"
                lines.append(f"| {label} | {s} | {i} |")
            # special rows where they belong thematically
            if title.startswith("Explanation granularity") and "uniqueness_dedup" in r["spex"]:
                lines.append(f"| uniqueness (X dedup) | {fmt(r['spex']['uniqueness_dedup'])} "
                             f"| {fmt(r['idc']['uniqueness_dedup'])} |")
                lines.append(f"| stability k=5 (X dedup) | {fmt(r['spex']['stability_k5_dedup'])} "
                             f"| {fmt(r['idc']['stability_k5_dedup'])} |")
            if title.startswith("Faithfulness"):
                lines.append(f"| faithfulness (correlation) | {spex_corr} | {corr} |")
            if title.startswith("Diversity") and r["spex"].get("diversity_fixed"):
                lines.append(f"| diversity_fixed (weighted Jaccard) | "
                             f"{fmt(r['spex']['diversity_fixed'])} | "
                             f"{fmt(r['idc']['diversity_fixed'])} |")
    lines += ["", "\\* faithfulness (correlation) needs ≥2 used features, a "
              "non-constant masking curve AND a non-constant importance vector. "
              "For SpEx it is undefined wherever the tree asks so few questions "
              "that the accuracy curve is exactly constant (verified: std 0) — that "
              "is Breast Cancer, Iris, HAR and Two Moons in every variant, i.e. it "
              "does NOT depend on K. For IDC it is undefined where the gates leave "
              "too little to mask: Two Moons and Blobs under the default config use "
              "ZERO features (all gates closed), Iris-best 0–1, Breast Cancer 1–2, "
              "while the tuned 2-D runs have the opposite problem — both features "
              "open for every sample, so the importance vector is constant. With "
              "only D=2 maskable features a defined correlation "
              "is trivially ±1 (Blobs SpEx 1.000 — do not quote it as a result). "
              "The drop-based rows above are the defined-everywhere "
              "replacement.", ""]

    out = os.path.join(HERE, "RESULTS_MULTISEED.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {out} ({len(data)} datasets)")


if __name__ == "__main__":
    main()
