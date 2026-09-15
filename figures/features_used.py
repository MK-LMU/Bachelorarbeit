# -*- coding: utf-8 -*-
"""Table A.7 of the thesis: how many features each explanation uses.

For SpEx that is the number of split features of the tree, for IDC the number
of features with an open gate, per representation. Both are the `n_used`
entries of results/faithfulness_curves.json (zero fill): a feature counts as
used when its importance is positive for at least one sample, which for the
tree means it is split on and for IDC that its gate is open. SpEx is
seed-constant; for IDC the mean over seeds 1-10 and the range are reported.

Writes results/features_used.json and prints the table.

Usage (repo root):  python figures/features_used.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from wpaths import RESULTS  # noqa: E402

ORDER = [("two_moons_best", "Two Moons"), ("blobs_best", "Gaussian Blobs"), ("iris_best", "Iris"),
         ("breast_cancer_best", "Breast Cancer"), ("digits_best", "Digits"), ("har_best", "HAR"),
         ("cifar10_best", "CIFAR-10 (feat.)"), ("mnist_feats_best", "MNIST (feat.)"),
         ("mnist", "MNIST (pixels)")]

curves = json.load(open(os.path.join(RESULTS, "faithfulness_curves.json"), encoding="utf-8"))
out = {}
print("%-18s %5s %6s %s" % ("Dataset", "D", "SpEx", "IDC open gates, mean (min-max)"))
for key, name in ORDER:
    e = curves[key]
    spex = sorted({c["zero"]["n_used"] for c in e["spex"].values()})
    idc = [c["zero"]["n_used"] for c in e["idc"].values()]
    rec = {"name": name, "D": e["D"], "spex_split_features": spex,
           "idc_open_gates": {"per_seed": idc, "mean": float(np.mean(idc)),
                              "min": int(min(idc)), "max": int(max(idc))}}
    out[key] = rec
    sp = "/".join(str(s) for s in spex)
    g = rec["idc_open_gates"]
    rng = "" if g["min"] == g["max"] else "  (%d-%d)" % (g["min"], g["max"])
    print("%-18s %5d %6s %7.1f%s" % (name, e["D"], sp, g["mean"], rng))

dst = os.path.join(RESULTS, "features_used.json")
json.dump(out, open(dst, "w", encoding="utf-8"), indent=1)
print("written", os.path.relpath(dst, ROOT))
