# -*- coding: utf-8 -*-
"""Every program invocation of this thesis, in dependency order.

This file is first of all a LIST: it documents every run behind the numbers in
RESULTS_MULTISEED.md, the result JSONs in results/ and the figures in
notes/figures/.

    python reproduce_all.py --dry     # print the commands, compute nothing
    python reproduce_all.py           # run everything (~13 h, GPU + CPU)

Prerequisites (once, created by no script):
  python -m venv venv
  venv\\Scripts\\pip install -r requirements.txt
  build artifacts/data/har_data.npz from the UCI HAR dataset
      (keys X = (10299, 561), y = labels 0..5; train and test merged)
  checksums of the three data artefacts (har_data, cifar_feats, mnist_feats):
      DATA_CHECKSUMS.sha256 — the ResNet features depend on GPU and torch
      version and are not bit-identically re-extractable; use the original
      files for exact CIFAR / MNIST-feats numbers

On repeating: run_idc.py ALWAYS retrains. To fill in missing runs only, use
train_all.py, the resumable variant of the training loop below. tune_idc.py,
reselect_best.py and perturbation_stability.py skip existing results by
themselves.

Not part of this list: results/results_<ds>.json (legacy files of the first
single-seed comparisons, produced by archive/compare_final.py and others).
They do not feed RESULTS_MULTISEED.md and are kept only as provenance.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
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
# 3. Base training: IDC with the default config, 5 seeds per dataset
#    (MNIST uses IDC's paper-validated config with 700 epochs and therefore
#    needs ~15 min per run instead of seconds to minutes)
# --------------------------------------------------------------------------
for ds in DATASETS + ["mnist"]:
    for seed in range(5):
        run("run_idc.py", "--data", ds, "--seed", str(seed))

# --------------------------------------------------------------------------
# 4. The 2-D synthetics again with ONE knob turned (LGL=0.1) — shows that the
#    default collapse on Two Moons / Blobs is caused by the configuration
# --------------------------------------------------------------------------
for ds in ["two_moons", "blobs"]:
    for seed in range(5):
        run("run_idc.py", "--data", ds, "--lgl", "0.1", "--tag", "_tuned",
            "--seed", str(seed))

# --------------------------------------------------------------------------
# 5. MNIST on ResNet features (proposal: "MNIST features")
#    extract_mnist_feats.py reads X/y from idc_out_mnist_seed0.npz and must
#    therefore run AFTER step 3, so that both MNIST variants use the same
#    10,000 samples
# --------------------------------------------------------------------------
run("extract_mnist_feats.py")                 # -> artifacts/data/mnist_feats.npz
for seed in range(5):
    run("run_idc.py", "--data", "mnist_feats", "--seed", str(seed))

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
    run("tune_idc.py", ds)                    # 6 grid candidates + _best over 5 seeds

run("tune_idc.py", "mnist_feats")             # not covered without an argument

run("reselect_best.py")                       # K-constrained selection rule; acts on
                                              # every entry in
                                              # results/tuning/selection.json,
                                              # hence after both tune calls

# HELD BACK (see the note at the end of this file):
# run("check_architecture.py")                # is network size the limit on
#                                             # Blobs/Digits? small/tiny, 3 seeds,
#                                             # same LGL/epochs as the _best runs
#                                             # (result: refuted)
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
copy("artifacts/idc_out/idc_out_mnist_seed0.npz",
     "artifacts/idc_out/idc_out_mnist.npz")   # compare_kprime.py reads the legacy name;
                                              # checked field by field: X/y/K/gates/ari
                                              # of the seed-0 run are identical, the copy
                                              # only lacks keys added later
run("compare_kprime.py")                      # k' > k on MNIST (10/50/200 leaves):
                                              # more leaves improve the clustering but
                                              # do not close the granularity gap
                                              # (over-segmentation)
# HELD BACK:
# run("check_faithfulness_masking.py")        # faithfulness with mean instead of zero
#                                             # masking, both methods; IDC from the
#                                             # persisted .pt (no training)
#                                             # -> results/faithfulness_masking_check.json

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

# --------------------------------------------------------------------------
# HELD-BACK CHECK SCRIPTS
#
# Four scripts are commented out above; pending a discussion with the
# supervisor they live in notes/ausgelagert/ and are not part of this
# repository:
#   check_architecture.py          network size as the limit? -> refuted
#   check_gener_scale.py           generalizability raw vs. row-normalised
#   check_uniq_scale.py            scale share of the uniqueness gap
#   check_faithfulness_masking.py  mean instead of zero masking
#
# Their result files (arch_check.json, gener_scale_check.json,
# faithfulness_masking_check.json) are therefore not shipped either.
# significance.py copes with that: without arch_check.json its architecture
# section is null and the head-to-head t-tests are unaffected.
# --------------------------------------------------------------------------

print("\ndone." if not DRY else "\n(dry run — nothing executed.)")
