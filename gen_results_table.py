# -*- coding: utf-8 -*-
"""Generate RESULTS_MULTISEED.md — one document with ALL multi-seed values
(mean±std) for both methods on all datasets, from results_multiseed_<ds>.json
plus the per-seed idc_out_<ds>_seed<k>.npz (for IDC's correlation faithfulness,
which the aggregator does not carry). Rerun after any campaign extension."""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpaths import N_SEEDS, CAMPAIGN_SEEDS, SELECTION_SEED   # one source for the seeds
from check_ranges import label as range_label   # printed range == checked range

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
     "how individual are the explanations? **The four rows are not on one "
     "scale and do not point the same way.** nn-identical is a share in [0,1] "
     "-- the fraction of samples whose nearest neighbour gets an IDENTICAL "
     "explanation row -- so HIGH means one template per region. uniqueness and "
     "stability divide a gate distance by an input distance, are unbounded "
     "above, and HIGH means more per-sample variation. nn-identical is "
     "scale-free (for gate magnitudes above ~1e-7, where the comparison is "
     "purely relative) and carries the granularity claim; the other two also "
     "react to gate magnitude, which is why the row-normalised variant is "
     "reported beside the raw one",
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
     "**supervised separability of the TRUE class labels after gating** — a "
     "LinearSVC is trained on X x gates to predict y, so this is NOT a measure "
     "of whether the CLUSTERING generalises, and the explanation is never "
     "compared against the cluster assignment. A method that gates nothing "
     "keeps X intact and therefore scores highest; read every value against "
     "the ungated reference row, which is the ceiling both methods are "
     "measured against",
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
    if n is not None and n < N_SEEDS:
        # Aggregated over FEWER seeds than the campaign ran because the rest
        # were nan. Marking only n==1 (the old rule) hid the survivorship case:
        # on iris_best the IDC runs collapse to one cluster on most seeds, so a
        # part-campaign mean over the survivors stood next to a full-campaign
        # SpEx mean and read like a fair comparison. The surviving seeds are
        # the non-degenerate ones, so such a mean is biased upward.
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


def best_epoch_gap(data):
    """How far IDC's LOGGED best-epoch ARI sits above the final-label ARI.

    The npz carries `ari` = model.best_ari, chosen while looking at the true
    labels; the tables recompute ARI from the final labels because a tree has
    no epoch to select. Reciting the margin by hand is how the other notes in
    this file went stale, so it is read off the artifacts at generation time."""
    from wpaths import idc_out
    worst = None
    for ds, r in data.items():
        vals = [v for v in (r["idc"]["ARI"].get("values") or []) if v is not None]
        best = []
        for s in r.get("seeds", []):
            p = idc_out(f"idc_out_{ds}_seed{s}.npz")
            if os.path.exists(p):
                d = np.load(p)
                if "ari" in d and np.isfinite(float(d["ari"])):
                    best.append(float(d["ari"]))
        if vals and len(best) == len(vals):
            gap = float(np.mean(best) - np.mean(vals))
            if worst is None or gap > worst[1]:
                worst = (NAMES[ds], gap)
    return "" if worst is None else (f"The largest gap in this campaign is "
                                     f"{worst[0]}, {worst[1]:+.3f} ARI.")


def load_all():
    """Every results file that exists, in ORDER."""
    from wpaths import results
    data = {}
    for ds in ORDER:
        p = results(f"results_multiseed_{ds}.json")
        if os.path.exists(p):
            data[ds] = json.load(open(p))
    return data


def empty_set_evidence(data):
    """One concrete per-seed witness of the empty-median defect, from the data.

    Hard-coding the witness is how this note went stale once already: it said
    "one seed" while the campaign had grown to four."""
    r = data.get("mnist")
    if not r:
        return ""
    v = [x for x in (r["idc"]["diversity"].get("values") or []) if x is not None]
    if not v:
        return ""
    K = r["K"]
    sig = 100.0 * (1 - 1.0 / (K * (K - 1)))
    n_sig = sum(1 for x in v if abs(x - sig) < 1e-3)
    n_max = sum(1 for x in v if abs(x - 100.0) < 1e-9)
    return (f"Per-seed evidence in the JSONs: MNIST IDC scores {sig:.3f} on "
            f"{n_sig} of {len(v)} seeds and 100.0 on {n_max} - {sig:.3f} is "
            f"exactly 100*(1-1/{K*(K-1)}), the signature of a single non-empty "
            f"set.")


def above_ceiling_evidence(data):
    """The distinct above-ceiling diversity values actually present, per dataset."""
    for ds in ("har", "har_best"):
        r = data.get(ds)
        if not r:
            continue
        K = r["K"]
        ceil = 100.0 * (1 - 1.0 / (K - 1)) if K > 2 else 0.0
        v = sorted({round(x, 1) for x in (r["idc"]["diversity"].get("values") or [])
                    if x is not None and x > ceil + 1e-9})
        if v:
            return (f"IDC on {NAMES[ds].split(' -- ')[0]}: "
                    + ", ".join(f"{x:.1f}" for x in v) + f" at K={K}")
    return ""


def undefined_corr(data, method):
    """Variants where faithfulness (correlation) is undefined, read off the data.

    "<name> (all)" for a fully undefined variant, "<name> (k of n seeds)" where
    only some seeds are. Reciting this list by hand is how the previous note
    came to name a dataset that is in fact defined on every seed."""
    out = []
    for ds, r in data.items():
        vals = ((r[method].get("faithfulness_corr") or {}).get("values")) or []
        if not vals:
            continue
        n_undef = sum(1 for v in vals if v is None)
        if n_undef == len(vals):
            out.append(f"{NAMES[ds]} (all)")
        elif n_undef:
            out.append(f"{NAMES[ds]} ({n_undef} of {len(vals)} seeds)")
    return out


def main():
    data = load_all()
    lines = [
        "# Multi-seed results — all metrics, all datasets (mean ± std)",
        "",
        "Both methods are scored by **the same evaluation code**, on identical",
        "data, so no measured difference can originate in the evaluation. IDC's",
        "originals (faithfulness, diversity, uniqueness, stability,",
        "generalizability) are imported directly wherever they run unmodified;",
        "where a correction was unavoidable it is labelled at its definition —",
        "`faithfulness_k` passes the dataset's real K instead of IDC's hardcoded",
        "10, casts the tree's float predictions to int, and returns nan below 2",
        "usable features. Rows added by this work are marked as such below and",
        "follow the established literature: AOPC — Samek et al. 2017; deletion —",
        "Petsiuk et al. 2018; stability of explanations — Alvarez-Melis &",
        "Jaakkola 2018.",
        "",
        "Generated by `gen_results_table.py` from `results_multiseed_<ds>.json`",
        f"(campaign: {N_SEEDS} seeds per dataset, numbered "
        f"{CAMPAIGN_SEEDS[0]}-{CAMPAIGN_SEEDS[-1]}. Seed {SELECTION_SEED} is the",
        "configuration-search seed of tune_idc.py and is deliberately NOT among",
        "them: the run that wins the selection must not also be one of the runs",
        "whose spread the error bars describe. SpEx = Spectral+tree pipeline,",
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
        "  property of the metric, not missing data. The drop-based rows (this",
        "  work's addition) are defined FAR MORE OFTEN, not everywhere: they",
        "  still return nan where no feature has a positive gate at all, which",
        "  is Two Moons and Blobs under the default config (10 of 10 seeds each)",
        "  and Iris-best (6 of 10).",
        "- `diversity` (IDC's original) has two implementation defects. A cell",
        "  is reported as n/a (with the defective value in parentheses) wherever",
        "  the value sits ABOVE the ceiling below, on either method — that",
        "  condition, not the column, is what marks a",
        "  value as demonstrably degenerate. (1) The Jaccard matrix is summed",
        "  INCLUDING the diagonal (each set compared with itself = 1) while the",
        "  divisor covers only the off-diagonal pairs K(K−1) — as long as every",
        "  cluster's median set is non-empty the attainable maximum is therefore",
        "  100·(1−1/(K−1)), i.e. 88.9 at K=10, 80 at K=6 and 0 at K=2, and the",
        "  scale is not comparable across datasets with different K. Values ABOVE",
        f"  that bound ({above_ceiling_evidence(data)}) arise only through",
        "  defect (2): an empty median set contributes 0 to the sum instead of 1.",
        "  (2) `sklearn.jaccard_score` returns 0 for two EMPTY sets, so a cluster",
        "  whose median gate vector is all-zero counts as maximally diverse —",
        "  saying nothing scores best. " + empty_set_evidence(data),
        "  `diversity_fixed` repairs both defects: off-diagonal",
        "  pairs only, weighted Jaccard on mean gate vectors, empty -> nan.",
        "  SpEx's original values are legitimate but only comparable within",
        "  the same K.",
        "- `(= raw)` on a row-norm cell means the value equals the raw one,",
        "  because every gate row of that run already peaks at 1 and the",
        "  normalisation therefore changes nothing. Where the two differ, the",
        "  gap is the share of the raw number that was gate MAGNITUDE rather",
        "  than granularity — on the SpEx side a factor of 1.7 to 2.9, except",
        "  on the single-feature trees (Two Moons, Breast Cancer, Iris), where",
        "  row normalisation turns every row into the same indicator vector and",
        "  the row-normalised value is exactly 0.",
        "- IDC config is paper-validated **only for MNIST**; all other IDC columns",
        "  use a default config and are lower bounds under un-tuned settings.",
        "- IDC's ACC/ARI/NMI are recomputed here from the FINAL model's labels",
        "  via Munkres. They are NOT the best-epoch values IDC logs during",
        "  training and stores in the npz as `acc`/`ari`/`nmi`: those are picked",
        "  while looking at the true labels, and a tree has no epochs to pick",
        "  from, so using them would compare two protocols. The gap is",
        "  systematic and one-directional (the npz value is never lower). "
        + best_epoch_gap(data),
        "- `±` is the standard deviation over the retraining seeds — the spread",
        "  across runs, not a measurement uncertainty of a single run. `n/a`",
        "  cells name the reason a value does not exist; they are not zeros.",
        "- SpEx rows: the TREE is seed-invariant, and so is everything that",
        "  depends only on it — ACC/ARI/NMI, all gate metrics,",
        f"  faithfulness and diversity have std 0.000 across the {N_SEEDS} spectral seeds",
        "  on every dataset (on CIFAR/MNIST the spectral reference itself moves",
        "  by ~1e-4 in ARI(ref, y); the tree output does not). generalizability",
        "  is the exception by design: its 70/30 split uses the seed, identically",
        f"  for both methods. Rows are {N_SEEDS}-seed aggregates unless marked ⁿ⁼k,",
        f"  which means the value averages only k < {N_SEEDS} seeds because the others",
        f"  returned nan — read those against the {N_SEEDS}-seed value beside them with",
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
        "  gates (tuned 2-D IDC) reduce it to plain-X separability. Cells whose",
        "  per-seed values EQUAL the ungated reference carry the marker",
        "  `(= plain X)` — the gating changed nothing there.",
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
        "One row per dataset — its PRIMARY configuration (IDC `_best` where a",
        "grid winner exists; MNIST its paper-validated config). The 19 variants",
        "below are NOT 19 datasets: `_best`, `_tuned` and default share X, y",
        "and K, so counting statements refer to these 9 primary configurations.",
        "Every variant keeps its full detail block below.",
        "",
        "| dataset | k-means ARI (baseline) | ARI SpEx | ARI IDC | nn-ident SpEx | nn-ident IDC | faith. top-1 drop SpEx | faith. top-1 drop IDC | IDC config |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    PRIMARY = ["two_moons_best", "blobs_best", "iris_best", "breast_cancer_best",
               "digits_best", "har_best", "cifar10_best", "mnist",
               "mnist_feats_best"]
    for ds in PRIMARY:
        r = data.get(ds)
        if r is None:
            continue
        base = ds[:-5] if ds.endswith("_best") else ds
        km = f"{r['kmeans_ari']:.3f}" if r.get("kmeans_ari") is not None else "—"
        lines.append(
            f"| {NAMES[base]} | {km} | {fmt(r['spex']['ARI'])} | {fmt(r['idc']['ARI'])} "
            f"| {fmt(r['spex']['nn_identical_frac'])} | {fmt(r['idc']['nn_identical_frac'])} "
            f"| {fmt(r['spex']['faithfulness_top1drop'])} | {fmt(r['idc']['faithfulness_top1drop'])} "
            f"| {cfg_label(ds, r)} |")

    for ds, r in data.items():
        km_note = (f", k-means baseline ARI {r['kmeans_ari']:.3f}"
                   if r.get("kmeans_ari") is not None else "")
        lines += ["", f"## {NAMES[ds]}  (N={r['N']}, D={r['D']}, K={r['K']}, "
                      f"seeds={r['seeds']}, IDC config: {cfg_label(ds, r).strip('*')}"
                      f"{km_note})"]

        # Three degeneracies that all produce metric numbers describing nothing.
        # Each is announced ABOVE the table: a blockquote between the header and
        # the first row ends the table in Markdown, and every row after it then
        # renders as prose.
        for key, headline, what in (
                ("zero_gate_seeds", "All gates zero",
                 "those runs produced no explanation at all, so the granularity "
                 "and diversity numbers below describe an empty matrix rather "
                 "than a coarse one"),
                ("constant_gate_seeds", "One explanation for every sample",
                 "the gate matrix has a single distinct row, so uniqueness and "
                 "stability read 0 and nn-identical reads 1 -- that is maximal "
                 "coarseness, not stability. The clustering rows are unaffected "
                 "and can still look competitive"),
                ("empty_median_seeds", "Every median gate set empty",
                 "IDC's original diversity compares empty sets, scores them 0 "
                 "and therefore reports its maximum; read diversity_fixed "
                 "instead")):
            zs, zi = r.get(f"spex_{key}", []), r.get(f"idc_{key}", [])
            # An all-zero gate matrix is trivially constant and trivially has
            # empty median sets, so it would trigger all three notices. Report
            # each seed under the strongest category only; the JSON keeps all
            # three facts, this is a display decision.
            if key != "zero_gate_seeds":
                zs = [x for x in zs if x not in r.get("spex_zero_gate_seeds", [])]
                zi = [x for x in zi if x not in r.get("idc_zero_gate_seeds", [])]
            if not (zs or zi):
                continue
            parts = ([f"SpEx seeds {zs}"] if zs else []) +                     ([f"IDC seeds {zi}"] if zi else [])
            lines += ["", f"> **{headline}** in {' and '.join(parts)}: {what}."]

        lines += ["", "| metric | SpEx (Spectral+tree, +Tree SHAP) | IDC |", "|---|---|---|"]

        corr = idc_corr_faithfulness(ds, r["seeds"])
        spex_corr = (fmt(r["spex"]["faithfulness_corr"])
                     if r["spex"].get("faithfulness_corr") else "not recomputed")
        if spex_corr == "nan":
            spex_corr = "undef.*"

        for title, _subtitle, rows in GROUPS:   # explained once in the legend above
            lines.append(f"| **{title}** | | |")
            for key, label in rows:
                label = f"{label}  {range_label(key)}".rstrip()
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
                            val = (f"n/a (defective metric: {s if cell == 's' else i} "
                                   f"lies above the ceiling {ceil:.1f} -> empty "
                                   f"median sets; see diversity_fixed)")
                            if cell == "s":
                                s = val
                            else:
                                i = val
                if key == "uniqueness_rownorm":
                    # Row-normalisation divides each gate row by its max, so it
                    # does nothing when every row already peaks at 1 -- which is
                    # the case for IDC on several datasets. Saying so stops the
                    # two rows from implying two independent measurements.
                    raw = r["spex"].get("uniqueness"), r["idc"].get("uniqueness")
                    for side, cell in ((0, "s"), (1, "i")):
                        a = (raw[side] or {}).get("values")
                        b = ((r["spex"] if side == 0 else r["idc"]).get(key) or {}).get("values")
                        m = (raw[side] or {}).get("mean")
                        # Only where the equality is informative. A shared nan
                        # or a shared 0.000 is already accounted for by the
                        # degeneracy notice above the table.
                        if a and a == b and m is not None and abs(m) > 1e-12:
                            if cell == "s":
                                s += " (= raw)"
                            else:
                                i += " (= raw)"
                if key in ("nn_identical_frac", "uniqueness",
                           "uniqueness_rownorm", "stability_k5"):
                    # A number here can describe NOTHING: with every gate zero
                    # there is no explanation to measure, and with one shared
                    # row every gate distance is 0 by construction. Writing
                    # 0.000 (or a 1.000 nn-identical over an empty matrix)
                    # reads like a stable result -- the same misreading the
                    # diversity defect produces. So the cell itself names why
                    # the value does not exist; the notice above the table
                    # lists the seeds.
                    n_all = len(r["seeds"])
                    for side, cell in (("spex", "s"), ("idc", "i")):
                        zero = r.get(f"{side}_zero_gate_seeds") or []
                        const = [x for x in (r.get(f"{side}_constant_gate_seeds") or [])
                                 if x not in zero]
                        if len(zero) >= n_all:
                            val = "n/a (all gates zero -- no explanation to measure)"
                        elif (key != "nn_identical_frac"
                              and len(zero) + len(const) >= n_all):
                            val = ("n/a (one shared explanation row -- "
                                   "gate distance 0 by construction)")
                        else:
                            continue
                        if cell == "s":
                            s = val
                        else:
                            i = val
                if key == "generalizability":
                    # (= plain X): the per-seed values equal the ungated
                    # reference exactly -- the gating changed nothing, the
                    # number is plain-X separability, not a method result.
                    px = (r.get("generalizability_plain_x") or {}).get("values")
                    for side, cell in (("spex", "s"), ("idc", "i")):
                        gv = (r[side].get(key) or {}).get("values")
                        if px and gv == px:
                            if cell == "s":
                                s += " (= plain X)"
                            else:
                                i += " (= plain X)"
                if key in ("ref_ARI", "tree_vs_ref_ARI"):
                    i = "—"
                lines.append(f"| {label} | {s} | {i} |")
            # special rows where they belong thematically
            if title == "Clustering quality":
                # Where only SOME seeds collapse, mean±std averages a bimodal
                # distribution (SEED_VERGLEICH_5_VS_10.md section 7) -- show
                # the shape next to it instead of leaving it implied.
                zero = r.get("idc_zero_gate_seeds") or []
                if 0 < len(zero) < len(r["seeds"]):
                    v = [x for x in (r["idc"]["ARI"].get("values") or [])
                         if x is not None]
                    if v:
                        q1, med, q3 = np.percentile(v, [25, 50, 75])
                        lines.append(
                            "| ARI per-seed distribution (IDC) | — | "
                            f"median {med:.3f}, IQR [{q1:.3f}, {q3:.3f}]; "
                            f"{len(zero)}/{len(r['seeds'])} seeds collapse to "
                            "all-zero gates -- the mean±std above averages a "
                            "bimodal distribution |")
            if title.startswith("Explanation granularity"):
                # The raw failure counts behind the n/a cells and notices,
                # straight from the JSON fields (P8): how often did each
                # method produce no usable explanation at all?
                def _fail(side):
                    zero = r.get(f"{side}_zero_gate_seeds") or []
                    const = [x for x in (r.get(f"{side}_constant_gate_seeds") or [])
                             if x not in zero]
                    return (f"zero {len(zero)}/{len(r['seeds'])} · "
                            f"constant {len(const)}/{len(r['seeds'])}")
                lines.append("| gate failure rate (all-zero · single-row) "
                             f"| {_fail('spex')} | {_fail('idc')} |")
            if title.startswith("Explanation granularity") and "uniqueness_dedup" in r["spex"]:
                lines.append(f"| uniqueness (X dedup)  {range_label('uniqueness_dedup')} "
                             f"| {fmt(r['spex']['uniqueness_dedup'])} "
                             f"| {fmt(r['idc']['uniqueness_dedup'])} |")
                lines.append(f"| stability k=5 (X dedup)  {range_label('stability_k5_dedup')} "
                             f"| {fmt(r['spex']['stability_k5_dedup'])} "
                             f"| {fmt(r['idc']['stability_k5_dedup'])} |")
            if title.startswith("Faithfulness"):
                lines.append(f"| faithfulness (correlation)  "
                             f"{range_label('faithfulness_corr')} | {spex_corr} | {corr} |")
            if title.startswith("Generalizability") and r.get("generalizability_plain_x"):
                # The same number in both columns on purpose: it is one shared
                # ceiling, not a per-method result. Seeing it twice is what
                # makes "IDC generalises better" legible as "IDC's gates leave
                # more of X in place".
                px = fmt(r["generalizability_plain_x"])
                lines.append(f"| no gating, X unchanged (reference)  "
                             f"{range_label('generalizability')} | {px} | {px} |")
            if title.startswith("Diversity") and r["spex"].get("diversity_fixed"):
                lines.append(f"| diversity_fixed (weighted Jaccard)  "
                             f"{range_label('diversity_fixed')} | "
                             f"{fmt(r['spex']['diversity_fixed'])} | "
                             f"{fmt(r['idc']['diversity_fixed'])} |")
    lines += ["", "\\* faithfulness (correlation) needs ≥2 used features, a "
              "non-constant masking curve AND a non-constant importance vector. "
              "Where any of those fails the metric is undefined — a property "
              "of the metric, not missing data. The affected variants are listed "
              "below, read off the results rather than recited, because this note "
              "has gone stale before:", ""]
    lines += [f"  - {side}: " + (", ".join(entries) if entries else "none")
              for side, entries in (("SpEx", undefined_corr(data, "spex")),
                                    ("IDC", undefined_corr(data, "idc")))]
    lines += ["", "For SpEx the cause is always the same: the tree asks so few "
              "questions that the accuracy curve is exactly constant, which does "
              "NOT depend on K. For IDC there are two causes — too few open "
              "gates to mask (the default 2-D runs close all of them), or the "
              "opposite, every gate open for every sample so the importance "
              "vector is constant (the tuned 2-D runs). With only D=2 maskable "
              "features a defined correlation is trivially ±1 (Blobs SpEx "
              "1.000 — do not quote it as a result). The drop-based rows "
              "above are the defined-everywhere replacement.", ""]

    out = os.path.join(HERE, "RESULTS_MULTISEED.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {out} ({len(data)} datasets)")


if __name__ == "__main__":
    main()
