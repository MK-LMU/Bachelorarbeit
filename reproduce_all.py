# -*- coding: utf-8 -*-
"""Every program invocation of this thesis, in dependency order.

This file is first of all a LIST: it documents every run behind the numbers in
RESULTS_MULTISEED.md, the result JSONs in results/ and the figures of the
thesis in figures/.

    python reproduce_all.py --dry     # print the commands, compute nothing
    python reproduce_all.py           # run everything (~20 h, GPU + CPU)

Prerequisites (once, created by no script):
  python -m venv venv
  venv\\Scripts\\pip install -r requirements.txt
  python build_har.py   -> artifacts/data/har_data.npz (UCI HAR via OpenML,
      keys X = (10299, 561), y = labels 0..5; byte-identical to the campaign
      file, checked against DATA_CHECKSUMS.sha256)
  checksums of the three data artefacts (har_data, cifar_feats, mnist_feats):
      DATA_CHECKSUMS.sha256 — the ResNet features depend on GPU and torch
      version and are not bit-identically re-extractable; use the original
      files for exact CIFAR / MNIST-feats numbers

Runtime at 10 seeds, from the measured parts: the training loop below is
~5 h on its own, evaluate.py ~1.5 h, and the equivalence proof in metrics.py
several hours more (it rebuilds IDC's full N x N distance matrix per call).
This is the only place that number lives; the README points here rather than
repeating it.

On repeating: run_idc.py ALWAYS retrains. To fill in missing runs only, use
train_all.py, the resumable variant of the training loop below. tune_idc.py,
reselect_best.py and perturbation_stability.py skip existing results by
themselves.

Not part of this list: results/results_<ds>.json, left over from the first
single-seed comparisons. They do not feed RESULTS_MULTISEED.md and are kept
only as provenance.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import CAMPAIGN_SEEDS          # one seed count for the whole campaign

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")
DRY = "--dry" in sys.argv

DATASETS = ["two_moons", "blobs", "iris", "breast_cancer", "digits", "har", "cifar10"]


def run(*args):
    print("$ python " + " ".join(args), flush=True)
    if not DRY:
        subprocess.run([PY, *args], cwd=HERE, check=True)


def copy(src, dst):
    print(f"$ copy {src} -> {dst}", flush=True)
    if not DRY:
        shutil.copy2(os.path.join(HERE, src), os.path.join(HERE, dst))


# --------------------------------------------------------------------------
# 1. Correctness of the SpEx-tree -> SHAP converter  (needs no data)
# --------------------------------------------------------------------------
run("test_correctness.py")                    # correctness chain (5 tests) +
                                              # 3 edge cases
run("test_metrics.py")                        # property tests of the metric layer:
                                              # diversity offset/ceiling, K pass-through,
                                              # fast path == original (data-free)

# --------------------------------------------------------------------------
# 2. External data: CIFAR embeddings (downloads CIFAR-10 + ResNet18 itself)
# --------------------------------------------------------------------------
run("extract_cifar.py")                       # -> artifacts/data/cifar_feats.npz

# --------------------------------------------------------------------------
# 3. Base training: IDC with the default config, every campaign seed
#    (MNIST uses IDC's paper-validated config with 700 epochs and therefore
#    needs ~15 min per run instead of seconds to minutes)
# --------------------------------------------------------------------------
for ds in DATASETS + ["mnist"]:
    for seed in CAMPAIGN_SEEDS:
        run("run_idc.py", "--data", ds, "--seed", str(seed))

# --------------------------------------------------------------------------
# 4. The 2-D synthetics again with ONE knob turned (LGL=0.1) — shows that the
#    default collapse on Two Moons / Blobs is caused by the configuration
# --------------------------------------------------------------------------
for ds in ["two_moons", "blobs"]:
    for seed in CAMPAIGN_SEEDS:
        run("run_idc.py", "--data", ds, "--lgl", "0.1", "--tag", "_tuned",
            "--seed", str(seed))

# --------------------------------------------------------------------------
# 4b. Seed 0 with the default config, every dataset. SELECTION_SEED is kept
#     out of the reported seeds on purpose (see wpaths.py), but two later
#     steps need its run as a DATA baseline: extract_mnist_feats.py reads
#     idc_out_mnist_seed0.npz, and metrics.py (step 6) takes X from
#     idc_out_<ds>_seed0.npz for every dataset -- without these files the
#     former aborts and the latter silently skips all datasets. Found by the
#     clean-copy reproduction of 2026-08-31 (repro_check/VERIFICATION_REPORT.md);
#     the runs reproduce the historical files bit-exactly.
# --------------------------------------------------------------------------
for ds in DATASETS + ["mnist"]:
    run("run_idc.py", "--data", ds, "--seed", "0")

# --------------------------------------------------------------------------
# 5. MNIST on ResNet features (proposal: "MNIST features")
#    extract_mnist_feats.py reads X/y from idc_out_mnist_seed0.npz and must
#    therefore run AFTER step 3, so that both MNIST variants use the same
#    10,000 samples
# --------------------------------------------------------------------------
run("extract_mnist_feats.py")                 # -> artifacts/data/mnist_feats.npz
for seed in CAMPAIGN_SEEDS:
    run("run_idc.py", "--data", "mnist_feats", "--seed", str(seed))
run("run_idc.py", "--data", "mnist_feats", "--seed", "0")   # baseline for
                                              # metrics.py, same reason as 4b

# --------------------------------------------------------------------------
# 6. Equivalence proof: the fast metric path == IDC's original code (bit-exact),
#    on ALL 9 datasets (X from idc_out_<ds>_seed0.npz) with real IDC gates,
#    real SpEx |SHAP| gates, random gates and a tie worst case
#    -> results/equivalence_check.json
#    (needs the seed-0 runs from steps 3 and 5; ~1 h CPU)
# --------------------------------------------------------------------------
run("metrics.py")

# --------------------------------------------------------------------------
# 7. Label-free tuning: grid LGL x epochs, selection by silhouette
#    (ARI is logged but never used for selection = leakage protection)
# --------------------------------------------------------------------------
for ds in DATASETS:
    run("tune_idc.py", ds)                    # 6 grid candidates + _best on all seeds

run("tune_idc.py", "mnist_feats")             # not covered without an argument

run("reselect_best.py")                       # K-constrained selection rule; acts on
                                              # every entry in
                                              # results/tuning/selection.json,
                                              # hence after both tune calls

# --------------------------------------------------------------------------
# 7b. _best with seed 0 for the RQ2 datasets. perturbation_stability.py reads
#     idc_out_<ds>_best_seed0.npz as its clean baseline, but tune_idc.py and
#     reselect_best.py train _best for CAMPAIGN_SEEDS only. Same command as
#     reselect_best.train_best(), config = the rule-2 winner just selected.
#     (Clean-copy reproduction 2026-08-31: the runs are bit-identical to the
#     historical files.) In --dry mode before any tuning run, selection.json
#     may not exist yet -- then the winner is printed as a placeholder.
# --------------------------------------------------------------------------
_sel_path = os.path.join(HERE, "results", "tuning", "selection.json")
_sel = json.load(open(_sel_path)) if os.path.exists(_sel_path) else None
for ds in ["iris", "breast_cancer", "digits", "har"]:
    w = (_sel[ds]["winner"] if _sel else
         {"lgl": "<winner lgl>", "epochs": "<winner epochs>"})
    run("run_idc.py", "--data", ds, "--seed", "0", "--lgl", str(w["lgl"]),
        "--epochs", str(w["epochs"]), "--tag", "_best")

run("check_architecture.py")                  # is network size the limit on
                                              # Blobs/Digits? small/tiny, 3 seeds,
                                              # same LGL/epochs as the _best runs
                                              # (result: refuted)
#                                             # -> results/arch_check.json

# --------------------------------------------------------------------------
# 8. Reference symmetry: the same label-free protocol for SpEx's only knob
#    (spectral vs. k-means as the reference clusterer)
# --------------------------------------------------------------------------
run("reference_selection.py")                 # -> results/tuning/reference_selection.json

# --------------------------------------------------------------------------
# 9. RQ2 — stability
# --------------------------------------------------------------------------
run("perturbation_stability.py")              # noise sigma {0, .01, .05, .10}
                                              # + subsampling {90%, 80%}, 3 reps,
                                              # on iris/breast_cancer/digits/har;
                                              # sigma=0 is the retrain control
# HELD BACK:
# run("check_gener_scale.py")                 # generalizability raw vs. row-normalised
# HELD BACK:
# run("check_uniq_scale.py")                  # scale share of the uniqueness gap on
#                                             # Digits + homogeneity probe u(2g)=2u(g).
#                                             # NOTE: reads the legacy file
#                                             # idc_out_digits.npz, whose IDC gates
#                                             # predate the documented config and are
#                                             # NOT reproducible (ARI 0.212 vs 0.143).
#                                             # The SpEx numbers do reproduce; the same
#                                             # finding is reproducibly covered by the
#                                             # uniqueness_rownorm rows of the table.
# Seed 0 by design: compare_kprime.py reads the legacy name, and the copy is
# a provenance artefact of the first single-seed comparison. X is seed-invariant.
copy("artifacts/idc_out/idc_out_mnist_seed0.npz",
     "artifacts/idc_out/idc_out_mnist.npz")   # compare_kprime.py reads the legacy name;
                                              # checked field by field: X/y/K/gates/ari
                                              # of the seed-0 run are identical, the copy
                                              # only lacks keys added later
run("compare_kprime.py")                      # k' > k on MNIST (10/50/200 leaves):
                                              # more leaves improve the clustering but
                                              # do not close the granularity gap
                                              # (over-segmentation)
run("check_faithfulness_masking.py")          # faithfulness with mean instead of zero
                                              # masking, both methods; IDC from the
                                              # persisted .pt (no training)
                                              # -> results/faithfulness_masking_check.json

# --------------------------------------------------------------------------
# 10. Evaluation and deliverables
# --------------------------------------------------------------------------
run("test_correctness_evaluated.py")          # part 3 of the correctness chain:
                                              # equivalence, additivity, production
                                              # wiring and brute force on the 9 trees
                                              # actually evaluated
run("evaluate.py")                            # all metrics of both methods,
                                              # one pass per dataset (19 entries)
run("gen_results_table.py")                   # -> RESULTS_MULTISEED.md
run("significance.py")                        # t-tests of the ARI comparisons with
                                              # Holm correction + Welch tests of the
                                              # architecture spot check
                                              # -> results/significance.json
run("visualize_explanations.py")              # -> notes/figures/*.png
run("check_ranges.py")                        # every reported value against the
                                              # range its definition allows, plus
                                              # the cross-metric invariants;
                                              # exits non-zero on any violation
# seed_robustness.py always writes results/seed_robustness.json, so the two
# comparisons below must run in THIS order: the integrity check first (and be
# copied away), the canonical 5-vs-10 comparison last.
run("seed_robustness.py",                     # (a) integrity check against the interim
    "results/baselines/10seeds_0bis9")        # seeds-0-9 campaign: on the shared seeds
                                              # 1-9 every value must be identical, i.e.
                                              # the move to seeds 1-10 changed nothing
                                              # (SEED_VERGLEICH_5_VS_10.md, section 2)
copy("results/seed_robustness.json",
     "results/seed_robustness_integrity_seedshift.json")
run("seed_robustness.py")                     # (b) CANONICAL: 5-seed vs 10-seed campaign
                                              # (default baseline results/baselines/5seeds):
                                              # which conclusions did the seed count
                                              # move? This is the file that
                                              # SEED_VERGLEICH_5_VS_10.md reports
                                              # (14 series beyond 1 sigma, Iris verdict
                                              # resolved, Blobs CI wider). Until
                                              # 2026-08-31 the shipped JSON was the
                                              # output of (a), overwritten in place.

# --------------------------------------------------------------------------
# 11. Figures of the thesis  (figures/, named as in the LaTeX sources)
# --------------------------------------------------------------------------
run("figures/compute_faithfulness_curves.py") # deletion curves of both methods,
                                              # both fills, seeds 1-10, from the
                                              # checkpoints (no training)
                                              # -> results/faithfulness_curves.json
run("figures/make_faithfulness_figures.py")   # Figure 5.1 and the fill dependence
run("figures/make_violins.py")                # Figures 5.2 and A.3
run("figures/make_pipeline_figure.py")        # schematic of Chapter 4 (no data)
run("figures/collect_explanation_figures.py") # Figures 4.2, A.1, A.2: the three
                                              # per-sample figures written by
                                              # visualize_explanations.py above

# --------------------------------------------------------------------------
# HELD-BACK CHECK SCRIPTS
#
# Two scripts are commented out above. The thesis does not cite their
# findings, so they are not part of this repository:
#   check_gener_scale.py           generalizability raw vs. row-normalised
#   check_uniq_scale.py            scale share of the uniqueness gap
#
# Their result file (gener_scale_check.json) is therefore not shipped either.
# check_architecture.py and check_faithfulness_masking.py, whose results the
# thesis reports (capacity study, masking study), run above like every other
# step and ship with their JSONs (arch_check.json,
# faithfulness_masking_check.json).
# significance.py copes with that: without arch_check.json its architecture
# section is null and the head-to-head t-tests are unaffected.
# --------------------------------------------------------------------------

print("\ndone." if not DRY else "\n(dry run — nothing executed.)")
