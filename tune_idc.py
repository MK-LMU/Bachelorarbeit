# -*- coding: utf-8 -*-
"""Per-dataset IDC tuning campaign (GPU) — fills the last fairness gap:
IDC's hyperparameters were the one thing still left at their defaults while
SpEx had nothing to tune.

Phase 1 (selection): train a small grid of configs (LGL x epochs, seed 0) per
dataset and score each candidate UNSUPERVISED by silhouette on (X, predicted
labels). ARI is logged for transparency but NEVER used for selection — a
clustering method has no access to ground-truth labels. Collapsed solutions
(<2 used clusters) score -1 and cannot win.

Phase 2 (error bars): the winner is retrained with seeds 1-4 under the tag
`_best` into the canonical artifacts/idc_out/, so evaluate.py and the results
table pick the runs up like any other dataset variant.

Outputs: artifacts/tuning/ (grid candidates), results/tuning/selection.json
(grid table + winner), artifacts/idc_out|models/ (the `_best` seed runs).

Usage: tune_idc.py [dataset ...]     (default: all 7 non-MNIST datasets)
"""
import os, sys, json, shutil, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import TUNING, RESULTS_TUNING, IDC_OUT, MODELS

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")

DATASETS = ["two_moons", "blobs", "iris", "breast_cancer", "digits",
            "har", "cifar10"]
# LGL in decades below IDC's published default of 1.0 -- the gate penalty is the
# knob that collapses the 2-D synthetics; 120 epochs = run_idc.py's default
# budget, 300 = the largest budget affordable for all 7 datasets on one GPU.
# The grid is searched at SEED 0 ONLY; seeds 1-4 go into error bars for the
# winner (phase 2), not into the selection.
GRID = [(lgl, ep) for lgl in (1.0, 0.1, 0.01) for ep in (120, 300)]
SEEDS_PHASE2 = [1, 2, 3, 4]


def gtag(lgl, ep):
    return f"_g{lgl}e{ep}"


def train(ds, seed, lgl, ep, tag, outdir=None):
    """One run_idc subprocess; resumable via existence check.
    Returns (npz path, status) where status is "cached" or a runtime string."""
    base = outdir or IDC_OUT
    out = os.path.join(base, f"idc_out_{ds}{tag}_seed{seed}.npz")
    if os.path.exists(out):
        return out, "cached"
    cmd = [PY, os.path.join(HERE, "run_idc.py"), "--data", ds,
           "--seed", str(seed), "--lgl", str(lgl), "--epochs", str(ep),
           "--tag", tag]
    if outdir:
        cmd += ["--outdir", outdir]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if r.returncode != 0 or not os.path.exists(out):
        tail = "\n".join((r.stdout + "\n" + r.stderr).splitlines()[-10:])
        raise RuntimeError(f"{ds}{tag} seed{seed} failed:\n{tail}")
    return out, f"{(time.time()-t)/60:.1f}min"


def score(npz_path):
    """Unsupervised selection score + transparency info from a run's npz."""
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    d = np.load(npz_path)
    X, y = np.ascontiguousarray(d["X"], float), d["y_true"].astype(int)
    lp = d["labels_pred"].astype(int)
    n_used = int(len(np.unique(lp)))
    sil = -1.0 if n_used < 2 else float(silhouette_score(X, lp))
    return {"silhouette": round(sil, 4),
            "ari_logged_not_used": round(float(adjusted_rand_score(y, lp)), 4),
            "n_used_clusters": n_used,
            "open_gates_per_sample": round(float((d["gates"] > 0).sum(1).mean()), 2)}


def main():
    only = sys.argv[1:] or DATASETS
    sel_path = os.path.join(RESULTS_TUNING, "selection.json")
    selection = json.load(open(sel_path)) if os.path.exists(sel_path) else {}

    for ds in only:
        print("=" * 72); print(f"TUNING {ds}  (grid: LGL x epochs = {GRID})")
        print("=" * 72, flush=True)
        cands = []
        for lgl, ep in GRID:
            tag = gtag(lgl, ep)
            out, dt = train(ds, 0, lgl, ep, tag, outdir=TUNING)
            s = score(out)
            cands.append({"lgl": lgl, "epochs": ep, "tag": tag, **s})
            print(f"  LGL={lgl:<5} ep={ep:<4} ({dt:>8}): sil={s['silhouette']:+.3f} "
                  f"clusters={s['n_used_clusters']} gates/sample="
                  f"{s['open_gates_per_sample']} (ARI={s['ari_logged_not_used']:.3f}, "
                  f"logged only)", flush=True)

        best = max(cands, key=lambda c: c["silhouette"])
        print(f"WINNER {ds}: LGL={best['lgl']} epochs={best['epochs']} "
              f"(silhouette {best['silhouette']:+.3f})", flush=True)

        # phase 2: winner -> canonical idc_out under tag _best
        # seed 0 = copy of the grid candidate (identical run, no retrain)
        src_npz = os.path.join(TUNING, f"idc_out_{ds}{best['tag']}_seed0.npz")
        src_pt = os.path.join(TUNING, f"idc_model_{ds}{best['tag']}_seed0.pt")
        dst_npz = os.path.join(IDC_OUT, f"idc_out_{ds}_best_seed0.npz")
        dst_pt = os.path.join(MODELS, f"idc_model_{ds}_best_seed0.pt")
        if not os.path.exists(dst_npz):
            shutil.copy2(src_npz, dst_npz)
            if os.path.exists(src_pt):
                shutil.copy2(src_pt, dst_pt)
        for s_ in SEEDS_PHASE2:
            _, dt = train(ds, s_, best["lgl"], best["epochs"], "_best")
            print(f"  _best seed {s_}: {dt}", flush=True)

        selection[ds] = {"criterion": "silhouette (unsupervised; ARI logged only)",
                         "grid": cands, "winner": best}
        json.dump(selection, open(sel_path, "w"), indent=2)
        print(f"updated {os.path.relpath(sel_path, HERE)}\n", flush=True)

    print("TUNING DONE:", {d: selection[d]["winner"]["tag"] for d in selection
                           if d in only})


if __name__ == "__main__":
    main()
