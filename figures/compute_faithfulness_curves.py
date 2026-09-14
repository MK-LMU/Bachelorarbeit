# -*- coding: utf-8 -*-
"""Deletion curves (accuracy after cumulatively masking the used features in
importance order) for both methods, both fills, campaign seeds 1-10.

The thesis reports only two readings of this curve (top-1 drop = first step,
AOPC = mean drop over the used features; metrics.faithfulness_drop). The curve
itself is not stored anywhere, so it is recomputed here: SpEx is refit per seed
via spex_side() (deterministic given the spectral reference), IDC is rebuilt
from the persisted checkpoints exactly as check_faithfulness_masking.py does
(no training). Unlike that study, which used the legacy seeds 0-4, this one
uses wpaths.CAMPAIGN_SEEDS (1-10), the seeds behind every table of Chapter 5.

Needs the training artefacts (artifacts/idc_out, artifacts/models), i.e. the
campaign must have run first. The shipped results/faithfulness_curves.json is
the output of this script; the figure scripts next to it read that file.

Output: results/faithfulness_curves.json
  {entry: {"name", "seeds", "D", "K", "N",
           "spex": {seed: {"zero": curve, "mean": curve}}, "idc": {...},
           "consistency": [...]}}
  curve = {"baseline", "acc" [n_used], "order" [n_used], "n_used", "top1drop", "aopc"}
Consistency: for fill 0 the recomputed top1drop/aopc are compared per seed with
results/results_multiseed_<entry>.json (the canonical table values).

Usage (repo root):  python figures/compute_faithfulness_curves.py [entry ...]
"""
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

# reuse the checkpoint loader of the masking study (sets up sys.path for IDC/SpEx itself)
_spec = importlib.util.spec_from_file_location(
    "check_faithfulness_masking", os.path.join(ROOT, "check_faithfulness_masking.py"))
_cfm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfm)
load_idc_model = _cfm.load_idc_model

from wpaths import idc_out, results, MODELS, CAMPAIGN_SEEDS  # noqa: E402
from metrics import get_accuracy  # noqa: E402
from spex_pipeline import spex_side  # noqa: E402
from evaluate import base_of  # noqa: E402

# the nine table rows of Chapter 5
ENTRIES = [("two_moons_best", "Two Moons"), ("blobs_best", "Gaussian Blobs"), ("iris_best", "Iris"),
           ("breast_cancer_best", "Breast Cancer"), ("digits_best", "Digits"), ("har_best", "HAR"),
           ("cifar10_best", "CIFAR-10 (feat.)"), ("mnist_feats_best", "MNIST (feat.)"),
           ("mnist", "MNIST (pixels)")]
OUT = results("faithfulness_curves.json")


def curve(gates, X, infer, y, K, fill):
    """IDC's masking loop with a configurable fill; returns the whole curve.
    fill = 0 reproduces metrics.faithfulness_drop (and drop_curve) exactly."""
    imp = np.sum(gates > 0, axis=0)
    ind = np.where(imp > 0)[0]
    order = ind[np.argsort(-imp[imp > 0])]
    base = get_accuracy(np.asarray(infer(X)).astype(int), y, K)
    mask = np.ones(X.shape[1])
    acc = []
    for i in order:
        mask[i] = 0
        Xm = X * mask + (1.0 - mask) * fill
        acc.append(float(get_accuracy(np.asarray(infer(Xm)).astype(int), y, K)))
    acc_a = np.array(acc)
    return {"baseline": float(base), "acc": acc, "order": [int(i) for i in order], "n_used": int(len(order)),
            "top1drop": float(base - acc_a[0]) if len(acc) else None,
            "aopc": float(np.mean(base - acc_a)) if len(acc) else None}


def main():
    want = sys.argv[1:]
    entries = [(e, n) for e, n in ENTRIES if not want or e in want]
    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    spex_cache = {}
    t_all = time.time()
    for entry, name in entries:
        t0 = time.time()
        base = base_of(entry)
        seeds = [s for s in CAMPAIGN_SEEDS if os.path.exists(idc_out(f"idc_out_{entry}_seed{s}.npz"))
                 and os.path.exists(os.path.join(MODELS, f"idc_model_{entry}_seed{s}.pt"))]
        if not seeds:
            print(f"[{entry}] no npz/checkpoint, skipped", flush=True)
            continue
        d0 = np.load(idc_out(f"idc_out_{entry}_seed{seeds[0]}.npz"))
        X32 = np.ascontiguousarray(d0["X"], np.float32)
        X64 = np.ascontiguousarray(d0["X"], float)
        y, K = d0["y_true"].astype(int), int(d0["K"])
        mu = X64.mean(axis=0)
        res = json.load(open(results(f"results_multiseed_{entry}.json"), encoding="utf-8"))
        res_seeds = res.get("seeds", CAMPAIGN_SEEDS)
        print("=" * 72 + f"\n{entry} ({name}): N={len(y)} D={X64.shape[1]} K={K} seeds {seeds}", flush=True)
        e = {"name": name, "seeds": seeds, "D": int(X64.shape[1]), "K": K, "N": int(len(y)),
             "spex": {}, "idc": {}, "consistency": []}
        for s in seeds:
            if (base, s) not in spex_cache:
                tree, _, gates = spex_side(X64, K, y, seed=s)
                infer_t = lambda Xm, t=tree: t.predict(np.ascontiguousarray(Xm, float)).astype(int)
                spex_cache[(base, s)] = {"zero": curve(gates, X64, infer_t, y, K, 0.0),
                                         "mean": curve(gates, X64, infer_t, y, K, mu)}
            e["spex"][str(s)] = spex_cache[(base, s)]
            infer_i, gates_of = load_idc_model(entry, s, X32, y, K)
            g = gates_of(X32)
            e["idc"][str(s)] = {"zero": curve(g, X32, infer_i, y, K, 0.0),
                                "mean": curve(g, X32, infer_i, y, K, mu.astype(np.float32))}
            # consistency with the canonical table values (fill 0)
            c = {"seed": s}
            for side in ("spex", "idc"):
                for m, key in (("top1drop", "faithfulness_top1drop"), ("aopc", "faithfulness_aopc")):
                    ref = res[side][key]["values"][res_seeds.index(s)] if s in res_seeds else None
                    new = e[side][str(s)]["zero"][m]
                    c[f"{side}_{m}_absdiff"] = (None if ref is None or new is None
                                                else float(abs(float(ref) - float(new))))
            e["consistency"].append(c)
            worst = max([v for k, v in c.items() if k != "seed" and v is not None] or [0.0])
            print(f"  seed {s:2d}: SpEx n_used {e['spex'][str(s)]['zero']['n_used']:4d} top1 "
                  f"{e['spex'][str(s)]['zero']['top1drop']:+.3f}/{e['spex'][str(s)]['mean']['top1drop']:+.3f}"
                  f"  IDC n_used {e['idc'][str(s)]['zero']['n_used']:4d} top1 "
                  f"{(e['idc'][str(s)]['zero']['top1drop'] or float('nan')):+.3f}/"
                  f"{(e['idc'][str(s)]['mean']['top1drop'] or float('nan')):+.3f}"
                  f"  | max|diff| vs table {worst:.1e}", flush=True)
        out[entry] = e
        json.dump(out, open(OUT, "w", encoding="utf-8"))
        print(f"  {entry}: {time.time() - t0:.0f} s", flush=True)
    worst = max([v for e in out.values() for c in e["consistency"] for k, v in c.items()
                 if k != "seed" and v is not None] or [0.0])
    print(f"\nsaved {os.path.relpath(OUT, ROOT)}; max |diff| vs table values over all entries/seeds: "
          f"{worst:.1e}; total {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
