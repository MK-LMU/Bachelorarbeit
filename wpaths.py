# -*- coding: utf-8 -*-
"""Central workspace paths and campaign size — the single place that knows the
folder layout and how many seeds the campaign runs.

Layout (see README.md):
  artifacts/idc_out/   idc_out_<ds>[_tag]_seed<k>.npz  (IDC training outputs)
  artifacts/models/    idc_model_*.pt                  (persisted IDC weights)
  artifacts/data/      har_data.npz, cifar_feats.npz, mnist_data/, cifar_data/
  results/             results_*.json, results_multiseed_*.json
  logs/                console outputs of all runs
  notes/figures/       explanation visualizations (kept with the thesis, not
                       in the repository — see .gitignore)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
IDC_OUT = os.path.join(HERE, "artifacts", "idc_out")
MODELS = os.path.join(HERE, "artifacts", "models")
DATA = os.path.join(HERE, "artifacts", "data")
TUNING = os.path.join(HERE, "artifacts", "tuning")       # grid candidates (phase 1)
PERTURB = os.path.join(HERE, "artifacts", "perturbation")  # RQ2 perturbed runs
RESULTS = os.path.join(HERE, "results")
RESULTS_TUNING = os.path.join(RESULTS, "tuning")          # selection.json etc.
RESULTS_PERTURB = os.path.join(RESULTS, "perturbation")   # stability.json
LOGS = os.path.join(HERE, "logs")
# Figures and HTML reports live with the thesis, not in the repository
# (notes/ is gitignored), but the paths stay defined here so that scripts
# writing them do not each hard-code the location.
FIGURES = os.path.join(HERE, "notes", "figures")
DOCS = os.path.join(HERE, "notes", "erklaerungen")

for _d in (IDC_OUT, MODELS, DATA, TUNING, PERTURB, RESULTS, RESULTS_TUNING,
           RESULTS_PERTURB, LOGS, FIGURES, DOCS):
    os.makedirs(_d, exist_ok=True)

# How many seeds the campaign runs, and which. Every driver and the evaluation
# import this instead of writing range(...) themselves -- the count used to sit
# in eight places at once, and a table legend claiming a different number than
# the runs behind it is not a mistake anyone notices by reading.
#
# SELECTION_SEED is where tune_idc.py searches the config grid. It is kept OUT
# of the reported seeds: the run that wins the selection would otherwise also
# be one of the runs whose spread the error bars describe, i.e. in-sample
# against its own selection. Training is deterministic, so that run cannot be
# "retrained fresh" -- the only fix is not to report it.
N_SEEDS = 10
SELECTION_SEED = 0
CAMPAIGN_SEEDS = list(range(SELECTION_SEED + 1, SELECTION_SEED + 1 + N_SEEDS))


def idc_out(fname):
    return os.path.join(IDC_OUT, fname)


def results(fname):
    return os.path.join(RESULTS, fname)
