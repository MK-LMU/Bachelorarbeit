# -*- coding: utf-8 -*-
"""Copy the three per-sample explanation figures of the thesis from the output
folder of visualize_explanations.py into this folder, under the names the
thesis uses:

    two_moons_importance_space.png     -> fig_two_moons_space.png   (Figure A.1)
    digits_pixel_importance_best.png   -> fig_digits_pixels.png     (Figure A.2, tuned IDC)
    mnist_pixel_importance_rows.png    -> fig_mnist_pixels.png      (Figure 4.2)

Run visualize_explanations.py first; it writes to wpaths.FIGURES.

Usage (repo root):  python figures/collect_explanation_figures.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from wpaths import FIGURES  # noqa: E402

PAIRS = [("two_moons_importance_space.png", "fig_two_moons_space.png"),
         ("digits_pixel_importance_best.png", "fig_digits_pixels.png"),
         ("mnist_pixel_importance_rows.png", "fig_mnist_pixels.png")]

for src, dst in PAIRS:
    s = os.path.join(FIGURES, src)
    if not os.path.exists(s):
        print("missing:", s, "(run visualize_explanations.py first)")
        continue
    shutil.copy2(s, os.path.join(HERE, dst))
    print("copied", src, "->", dst)
