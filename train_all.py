# -*- coding: utf-8 -*-
"""Multi-seed IDC training campaign (GPU) — the resumable driver.

Runs run_idc.py for every (variant, seed) pair as a SUBPROCESS (fresh CUDA
context per run) and skips pairs whose npz already exists, so the campaign can
be interrupted and continued. Small datasets first, MNIST last: it uses IDC's
validated config with 700 epochs and takes ~15 min per run against seconds for
the rest, so a configuration error surfaces long before it costs an hour.

Covers all three variant families the results table reports:
  default   run_idc.py's own defaults
  _tuned    the 2-D synthetics with LGL=0.1, the one-knob counter-example
  _best     the tuned winner per dataset, read from results/tuning/selection.json

Reading the winner rather than re-deriving it matters: tune_idc.py would rerun
the grid and overwrite selection.json's `winner` with the pure-silhouette
choice, undoing the K-constrained rule that reselect_best.py applied. Adding
seeds must not touch model selection.

Seed count comes from wpaths.N_SEEDS. After this, run evaluate.py and
gen_results_table.py.
"""
import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import idc_out, CAMPAIGN_SEEDS, RESULTS_TUNING

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")

# small -> large, so failures surface cheaply; MNIST is appended last
DATASETS = ["two_moons", "blobs", "iris", "breast_cancer", "digits",
            "har", "cifar10", "mnist_feats"]
TUNED_LGL01 = ["two_moons", "blobs"]


def variants():
    """(tag, dataset, extra run_idc args) for every variant in the table."""
    out = [("", ds, []) for ds in DATASETS]
    out += [("_tuned", ds, ["--lgl", "0.1"]) for ds in TUNED_LGL01]
    sel_path = os.path.join(RESULTS_TUNING, "selection.json")
    if os.path.exists(sel_path):
        for ds, entry in json.load(open(sel_path)).items():
            w = entry["winner"]
            out.append(("_best", ds, ["--lgl", str(w["lgl"]),
                                      "--epochs", str(w["epochs"])]))
    else:
        print("WARNING: no selection.json -- _best variants skipped. "
              "Run tune_idc.py + reselect_best.py first.", flush=True)
    out.append(("", "mnist", []))
    return out


def main():
    runs = [(tag, ds, extra, s) for tag, ds, extra in variants()
            for s in CAMPAIGN_SEEDS]
    t0 = time.time()
    done, failed = [], []
    for i, (tag, ds, extra, seed) in enumerate(runs, 1):
        out = idc_out(f"idc_out_{ds}{tag}_seed{seed}.npz")
        label = f"[{i}/{len(runs)}] {ds}{tag} seed={seed}"
        if os.path.exists(out):
            print(f"{label}: exists, skipped", flush=True)
            done.append((ds + tag, seed, "cached"))
            continue
        cmd = [PY, os.path.join(HERE, "run_idc.py"), "--data", ds,
               "--seed", str(seed)] + extra
        if tag:
            cmd += ["--tag", tag]
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        dt = time.time() - t
        if r.returncode == 0 and os.path.exists(out):
            print(f"{label}: OK ({dt/60:.1f} min)", flush=True)
            done.append((ds + tag, seed, f"{dt/60:.1f}min"))
        else:
            print(f"{label}: FAILED rc={r.returncode} ({dt/60:.1f} min)", flush=True)
            print("\n".join((r.stdout + "\n" + r.stderr).splitlines()[-15:]), flush=True)
            failed.append((ds + tag, seed))
    print(f"\nCAMPAIGN DONE in {(time.time()-t0)/3600:.2f} h: "
          f"{len(done)} ok, {len(failed)} failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
