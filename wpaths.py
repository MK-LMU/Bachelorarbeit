# -*- coding: utf-8 -*-
"""Central workspace paths — the single place that knows the folder layout.

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


def idc_out(fname):
    return os.path.join(IDC_OUT, fname)


def results(fname):
    return os.path.join(RESULTS, fname)
