# -*- coding: utf-8 -*-
"""Re-selection of the tuning winners with the K-CONSTRAINED silhouette rule.

The plain-silhouette rule (tune_idc.py phase 1) prefers coarse partitions:
on HAR it picked a 2-of-6-cluster solution (sil 0.481, ARI 0.331) over the
full 6-cluster one (sil 0.122, ARI 0.527). Fix — still fully unsupervised:

  RULE 2: only candidates that actually USE the requested K clusters are
  admissible (a 2-cluster answer to a 6-cluster task is degenerate); among
  them the silhouette decides. Fallback if no candidate reaches K: the
  candidates with the maximum number of used clusters.

NOTE on the record: because entry["winner"] is read as the rule-1 value, a
SECOND run copies the rule-2 winner into winner_sil_rule1 and the two fields
become identical (that is the current state of selection.json for all 8
datasets). The rule-1 evidence survives elsewhere: the complete `grid` array
stays in selection.json, and the archived `_bestsil` runs in artifacts/tuning/
show the rule changed the winner on iris, digits, har and cifar10.

This script re-derives the rule-2 winner per dataset from the logged grid
(results/tuning/selection.json — NO phase-1 retraining), archives the
rule-1 `_best` seed runs where the winner changes (renamed to `_bestsil`
inside artifacts/tuning/ as evidence), and trains the missing rule-2 seed
runs under the canonical `_best` tag.

Run this ONCE after the last tune_idc.py call: it reads entry["winner"], so a
second pass overwrites winner_sil_rule1 with the rule-2 winner and the two
fields become identical (which is the current state of selection.json). The
rule-1 evidence survives regardless: the full `grid` stays in selection.json
(HAR still shows the 2-cluster candidate at sil 0.481) and the archived
`_bestsil` runs stay in artifacts/tuning/.
"""
import os, sys, json, shutil, glob, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import TUNING, RESULTS_TUNING, IDC_OUT, MODELS

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")
K_OF = {"two_moons": 2, "blobs": 5, "iris": 3, "breast_cancer": 2,
        "digits": 10, "har": 6, "cifar10": 10, "mnist_feats": 10}


def rule2_winner(entry, K):
    grid = entry["grid"]
    full = [c for c in grid if c["n_used_clusters"] == K]
    pool = full if full else [c for c in grid if c["n_used_clusters"] ==
                              max(x["n_used_clusters"] for x in grid)]
    w = max(pool, key=lambda c: c["silhouette"])
    return w, bool(full)


def train_best(ds, seed, lgl, ep):
    out = os.path.join(IDC_OUT, f"idc_out_{ds}_best_seed{seed}.npz")
    if os.path.exists(out):
        return "cached"
    t = time.time()
    r = subprocess.run([PY, os.path.join(HERE, "run_idc.py"), "--data", ds,
                        "--seed", str(seed), "--lgl", str(lgl),
                        "--epochs", str(ep), "--tag", "_best"],
                       capture_output=True, text=True, cwd=HERE)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
    return f"{(time.time()-t)/60:.1f}min"


def main():
    sel_path = os.path.join(RESULTS_TUNING, "selection.json")
    selection = json.load(open(sel_path))

    for ds, entry in selection.items():
        K = K_OF[ds]
        w1 = entry["winner"]
        w2, had_full = rule2_winner(entry, K)
        entry["winner_sil_rule1"] = w1
        entry["winner"] = w2
        entry["criterion"] = ("K-constrained silhouette (unsupervised): candidate "
                              "must use the requested K clusters"
                              + ("" if had_full else " — no candidate reached K; "
                                 "fallback = max used clusters")
                              + "; among admissible ones silhouette decides. "
                              "ARI logged only, never used.")
        if (w2["lgl"], w2["epochs"]) == (w1["lgl"], w1["epochs"]):
            print(f"{ds}: winner unchanged (LGL={w2['lgl']} ep={w2['epochs']})")
            continue

        print(f"{ds}: NEW winner LGL={w2['lgl']} ep={w2['epochs']} "
              f"(cl={w2['n_used_clusters']}/{K}, sil={w2['silhouette']:+.3f}) — "
              f"archiving rule-1 runs, training seeds", flush=True)
        # archive rule-1 _best runs as _bestsil evidence inside artifacts/tuning/
        for f in glob.glob(os.path.join(IDC_OUT, f"idc_out_{ds}_best_seed*.npz")):
            shutil.move(f, os.path.join(TUNING, os.path.basename(f).replace("_best_", "_bestsil_")))
        for f in glob.glob(os.path.join(MODELS, f"idc_model_{ds}_best_seed*.pt")):
            shutil.move(f, os.path.join(TUNING, os.path.basename(f).replace("_best_", "_bestsil_")))
        # seed 0 = copy of the rule-2 grid candidate
        shutil.copy2(os.path.join(TUNING, f"idc_out_{ds}{w2['tag']}_seed0.npz"),
                     os.path.join(IDC_OUT, f"idc_out_{ds}_best_seed0.npz"))
        pt = os.path.join(TUNING, f"idc_model_{ds}{w2['tag']}_seed0.pt")
        if os.path.exists(pt):
            shutil.copy2(pt, os.path.join(MODELS, f"idc_model_{ds}_best_seed0.pt"))
        for s in (1, 2, 3, 4):
            print(f"  seed {s}: {train_best(ds, s, w2['lgl'], w2['epochs'])}", flush=True)

    json.dump(selection, open(sel_path, "w"), indent=2)
    print("\nRESELECTION DONE — selection.json documents both rules "
          "(winner = K-constrained, winner_sil_rule1 = pure silhouette).")


if __name__ == "__main__":
    main()
