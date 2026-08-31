# -*- coding: utf-8 -*-
"""Re-selection of the tuning winners with the K-CONSTRAINED silhouette rule.

Plain silhouette (tune_idc.py phase 1) prefers coarse partitions: on HAR it
picked a 2-of-6-cluster solution (sil 0.481) over the full 6-cluster one
(sil 0.122). The fix stays fully unsupervised -- only candidates that USE the
requested K clusters are admissible, and among those the silhouette decides;
if none reaches K, those with the most used clusters. K is part of the task,
not a label.

The rule-2 winner is re-derived from the logged grid in
results/tuning/selection.json -- no phase-1 retraining. Where the winner
changes, the rule-1 `_best` runs are archived as `_bestsil` and the missing
rule-2 seeds are trained.

Run this ONCE after the last tune_idc.py call: it reads entry["winner"] as the
rule-1 value, so a second pass copies the rule-2 winner into winner_sil_rule1
and the two become identical (the current state for all 8 datasets). The rule-1
evidence survives in the full `grid` and in the archived `_bestsil` runs.
"""
import os, sys, json, shutil, glob, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wpaths import TUNING, RESULTS_TUNING, IDC_OUT, MODELS, CAMPAIGN_SEEDS

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")
K_OF = {"two_moons": 2, "blobs": 5, "iris": 3, "breast_cancer": 2,
        "digits": 10, "har": 6, "cifar10": 10, "mnist_feats": 10}


def both_winners(entry, K):
    """Both selection rules, ALWAYS derived from the untouched `grid`.

    rule 1 = plain silhouette. rule 2 = silhouette among the candidates that
    actually use the requested K; K is the task, not a label, so this stays
    label-free. If no candidate reaches K the pool falls back to those with the
    most clusters found -- otherwise a dataset where every config collapses
    would have no winner.

    Deriving both from `grid` is what makes this file safe to run twice. The
    earlier version read rule 1 out of entry["winner"], which on a second run
    is already the rule-2 winner -- so rerunning silently overwrote the rule-1
    record with the rule-2 one. It did, on iris, digits, har and cifar10."""
    grid = entry["grid"]
    r1 = max(grid, key=lambda c: c["silhouette"])
    full = [c for c in grid if c["n_used_clusters"] == K]
    pool = full if full else [c for c in grid if c["n_used_clusters"] ==
                              max(x["n_used_clusters"] for x in grid)]
    r2 = max(pool, key=lambda c: c["silhouette"])
    return r1, r2, bool(full)


def train_best(ds, seed, lgl, ep):
    out = os.path.join(IDC_OUT, f"idc_out_{ds}_best_seed{seed}.npz")
    if os.path.exists(out):
        # The filename carries no config, so a leftover run from a previous
        # winner would be accepted under the new winner's name and the variant
        # would mix two configs across its seeds. Check the checkpoint instead
        # of trusting the name.
        pt = os.path.join(MODELS, f"idc_model_{ds}_best_seed{seed}.pt")
        if os.path.exists(pt):
            import torch
            ck = torch.load(pt, map_location="cpu", weights_only=False)
            if (ck.get("lgl"), ck.get("epochs")) != (lgl, ep):
                raise SystemExit(
                    f"[{ds} seed {seed}] cached _best run has lgl={ck.get('lgl')}, "
                    f"epochs={ck.get('epochs')}, but lgl={lgl}, epochs={ep} is "
                    f"requested. Delete artifacts/idc_out/idc_out_{ds}_best_seed*"
                    f" and artifacts/models/idc_model_{ds}_best_seed* first.")
        return "cached"
    t = time.time()
    r = subprocess.run([PY, os.path.join(HERE, "run_idc.py"), "--data", ds,
                        "--seed", str(seed), "--lgl", str(lgl),
                        "--epochs", str(ep), "--tag", "_best"],
                       capture_output=True, text=True, cwd=HERE)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
    return f"{(time.time()-t)/60:.1f}min"


def already_rule2(ds, w2):
    """Do the canonical _best runs already carry the K-constrained config?

    The second half of the idempotency guard. Without it a rerun would archive
    the rule-2 runs as `_bestsil` (they are the ones in place) and retrain from
    scratch -- destroying exactly the evidence the archive is meant to hold.
    run_idc.py stores lgl/epochs in the checkpoint, so the question is
    answerable without retraining."""
    pt = os.path.join(MODELS, f"idc_model_{ds}_best_seed1.pt")
    if not os.path.exists(pt):
        return False
    import torch
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    return (ck.get("lgl"), ck.get("epochs")) == (w2["lgl"], w2["epochs"])


def main():
    sel_path = os.path.join(RESULTS_TUNING, "selection.json")
    selection = json.load(open(sel_path))

    for ds, entry in selection.items():
        K = K_OF[ds]
        w1, w2, had_full = both_winners(entry, K)
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
        if already_rule2(ds, w2):
            print(f"{ds}: _best already holds the K-constrained winner "
                  f"(LGL={w2['lgl']} ep={w2['epochs']}), nothing to move")
            continue

        print(f"{ds}: NEW winner LGL={w2['lgl']} ep={w2['epochs']} "
              f"(cl={w2['n_used_clusters']}/{K}, sil={w2['silhouette']:+.3f}) — "
              f"archiving rule-1 runs, training seeds", flush=True)
        # archive rule-1 _best runs as _bestsil evidence inside artifacts/tuning/
        for f in glob.glob(os.path.join(IDC_OUT, f"idc_out_{ds}_best_seed*.npz")):
            shutil.move(f, os.path.join(TUNING, os.path.basename(f).replace("_best_", "_bestsil_")))
        for f in glob.glob(os.path.join(MODELS, f"idc_model_{ds}_best_seed*.pt")):
            shutil.move(f, os.path.join(TUNING, os.path.basename(f).replace("_best_", "_bestsil_")))
        # No copy of the grid run: it won the selection, so it must not also
        # be one of the reported seeds (see wpaths.SELECTION_SEED).
        for s in CAMPAIGN_SEEDS:
            print(f"  seed {s}: {train_best(ds, s, w2['lgl'], w2['epochs'])}", flush=True)

    json.dump(selection, open(sel_path, "w"), indent=2)
    print("\nRESELECTION DONE — selection.json documents both rules "
          "(winner = K-constrained, winner_sil_rule1 = pure silhouette).")


if __name__ == "__main__":
    main()
