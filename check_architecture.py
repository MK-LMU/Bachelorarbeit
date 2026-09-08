# -*- coding: utf-8 -*-
"""On Blobs and Digits the configuration tuning (tune_idc.py) helped little.
The obvious conjecture was that the ARCHITECTURE, not LGL or the epoch count,
is the limit there. This spot check tests it. Result: refuted, no smaller
network helps (see the evaluation below and significance.py).

Design: the same tuned LGL/epochs as the _best run (from selection.json),
only the network is shrunk, so that exactly ONE factor varies.

    default : encdec [256,256,512,64,512,256,256], gates_hidden 256
    small   : encdec [64,64,128,16,128,64,64],     gates_hidden 64
    tiny    : encdec [16,16,32,8,32,16,16],        gates_hidden 16

Three seeds per combination, outputs under artifacts/arch/, compared against
the existing _best values. Result -> results/arch_check.json
"""
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

# Project root; sys.path, SpEx/, IDC/, venv/ and artifacts/ hang off HERE.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import idc_out, results, RESULTS_TUNING

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")
ARCH_DIR = os.path.join(HERE, "artifacts", "arch")
os.makedirs(ARCH_DIR, exist_ok=True)

DATASETS = ["blobs", "digits"]
ARCHS = ["small", "tiny"]
SEEDS = [0, 1, 2]


def ari_of(path):
    d = np.load(path)
    return float(adjusted_rand_score(d["y_true"].astype(int),
                                     d["labels_pred"].astype(int)))


def main():
    sel = json.load(open(os.path.join(RESULTS_TUNING, "selection.json")))
    out = {}
    for ds in DATASETS:
        w = sel[ds]["winner"]
        lgl, ep = str(w["lgl"]), str(w["epochs"])
        # Reference: the existing _best runs (default architecture)
        base = [ari_of(idc_out(f"idc_out_{ds}_best_seed{s}.npz"))
                for s in range(5)
                if os.path.exists(idc_out(f"idc_out_{ds}_best_seed{s}.npz"))]
        out[ds] = {"winner_lgl": w["lgl"], "winner_epochs": w["epochs"],
                   "default": {"mean": round(float(np.mean(base)), 4),
                               "std": round(float(np.std(base, ddof=1)), 4),
                               "values": [round(v, 4) for v in base]}}
        print(f"\n=== {ds}  (LGL={lgl}, epochs={ep}) ===")
        print(f"  default-Architektur : {out[ds]['default']['mean']:.3f} "
              f"±{out[ds]['default']['std']:.3f}", flush=True)

        for arch in ARCHS:
            vals = []
            for s in SEEDS:
                tag = f"_arch{arch}"
                npz = os.path.join(ARCH_DIR, f"idc_out_{ds}{tag}_seed{s}.npz")
                if not os.path.exists(npz):
                    r = subprocess.run(
                        [PY, os.path.join(HERE, "run_idc.py"), "--data", ds,
                         "--seed", str(s), "--lgl", lgl, "--epochs", ep,
                         "--arch", arch, "--tag", tag, "--outdir", ARCH_DIR],
                        capture_output=True, text=True, cwd=HERE)
                    if r.returncode != 0 or not os.path.exists(npz):
                        print("\n".join((r.stdout + r.stderr).splitlines()[-6:]))
                        raise SystemExit(f"Lauf fehlgeschlagen: {ds} {arch} seed{s}")
                vals.append(ari_of(npz))
            out[ds][arch] = {"mean": round(float(np.mean(vals)), 4),
                             "std": round(float(np.std(vals, ddof=1)), 4),
                             "values": [round(v, 4) for v in vals]}
            delta = out[ds][arch]["mean"] - out[ds]["default"]["mean"]
            print(f"  {arch:<19}: {out[ds][arch]['mean']:.3f} "
                  f"±{out[ds][arch]['std']:.3f}   Delta {delta:+.3f}", flush=True)
        json.dump(out, open(results("arch_check.json"), "w"), indent=2)

    print("\n" + "=" * 62)
    for ds in out:
        best_alt = max(ARCHS, key=lambda a: out[ds][a]["mean"])
        d = out[ds][best_alt]["mean"] - out[ds]["default"]["mean"]
        verdict = ("architecture IS the limit (a smaller network helps clearly)"
                   if d > 0.05 else
                   "no relevant gain -> network size is NOT the limit")
        print(f"{ds:<8} best smaller network: {best_alt} "
              f"({out[ds][best_alt]['mean']:.3f}, delta {d:+.3f}) -> {verdict}")
    print(f"\nsaved: {results('arch_check.json')}")


if __name__ == "__main__":
    main()
