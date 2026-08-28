# -*- coding: utf-8 -*-
"""One-shot driver for the proposal's 'MNIST features' dataset: extraction ->
default 5-seed runs -> tuning grid -> K-constrained reselection -> evaluation
-> results table. Every step is a subprocess and resumable (existing outputs
are skipped by the underlying scripts)."""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import DATA, idc_out

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")


def step(name, args):
    t = time.time()
    print(f"=== {name} ===", flush=True)
    r = subprocess.run([PY] + args, cwd=HERE)
    if r.returncode != 0:
        print(f"FAILED: {name} (rc={r.returncode})", flush=True)
        sys.exit(1)
    print(f"--- {name} ok ({(time.time()-t)/60:.1f} min)\n", flush=True)


if not os.path.exists(os.path.join(DATA, "mnist_feats.npz")):
    step("extract features", [os.path.join(HERE, "extract_mnist_feats.py")])

for s in range(5):
    if not os.path.exists(idc_out(f"idc_out_mnist_feats_seed{s}.npz")):
        step(f"default seed {s}", [os.path.join(HERE, "run_idc.py"),
                                   "--data", "mnist_feats", "--seed", str(s)])

step("tuning grid", [os.path.join(HERE, "tune_idc.py"), "mnist_feats"])
step("K-constrained reselection", [os.path.join(HERE, "reselect_best.py")])
step("evaluate", [os.path.join(HERE, "evaluate.py"),
                  "mnist_feats", "mnist_feats_best"])
step("results table", [os.path.join(HERE, "gen_results_table.py")])
print("CHAIN DONE", flush=True)
