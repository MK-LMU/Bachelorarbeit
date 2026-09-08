# -*- coding: utf-8 -*-
"""Robustness check on the masking asymmetry: faithfulness under MEAN masking
vs the original ZERO masking, for both methods.

IDC's faithfulness masks with `x * mask`, i.e. sets a feature to 0 = the
feature MINIMUM after MinMax scaling. For IDC that is in-distribution (the
gated model is trained on `x * gates`); for the SpEx tree it is out-of-
distribution (every sample is pushed to the '<=' side of that split). This
script recomputes the drop-based faithfulness (top-1 drop, AOPC — the rows in
RESULTS_MULTISEED.md) with the masked feature set to its COLUMN MEAN instead,
and reports whether the SpEx-vs-IDC gap keeps its sign.

IDC inference comes from the persisted checkpoints (artifacts/models/*.pt):
the model is rebuilt exactly as run_idc.py builds it (same cfg, same dataset
injection), weights loaded, no training. As a by-product the zero-masking
values are recomputed from the reloaded model and compared with the npz
values written at training time — the first direct proof that the persisted
weights reproduce the artifacts. SpEx is rebuilt per seed via spex_side().

Output: results/faithfulness_masking_check.json + console table.
Usage: check_faithfulness_masking.py [entry ...]   (default: all 19 table entries)
"""
import os, sys, json, warnings
import numpy as np

for _n, _v in [("infty", np.inf), ("NaN", np.nan), ("float", float),
               ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _n):
        setattr(np, _n, _v)

# Project root; sys.path, SpEx/, IDC/, venv/ and artifacts/ hang off HERE.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "IDC"))
sys.path.insert(0, os.path.join(HERE, "SpEx"))
warnings.filterwarnings("ignore")

import torch
from omegaconf import OmegaConf

import dataset as idc_dataset
from dataset import ClusteringDataset
import train_evaluate

from wpaths import idc_out, results, MODELS
from metrics import get_accuracy
from spex_pipeline import spex_side
from evaluate import DATASETS, base_of, agg

SEEDS = [0, 1, 2, 3, 4]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ IDC reload
def load_idc_model(entry, seed, X32, y, K):
    ck = torch.load(os.path.join(MODELS, f"idc_model_{entry}_seed{seed}.pt"),
                    map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    if str(cfg.dataset).startswith("SHARED_"):
        class _Shared(ClusteringDataset):                 # same injection as run_idc.py
            @classmethod
            def setup(cls, _cfg):
                return cls(X32, y, K)
        setattr(idc_dataset, str(cfg.dataset), _Shared)
    model = train_evaluate.BaseModule(cfg)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    model.to(DEV)

    def infer(Xm):                                        # == run_idc.idc_infer
        with torch.no_grad():
            xt = torch.tensor(np.asarray(Xm, np.float32), device=DEV)
            g = model.gating_net.get_gates(xt)
            emb = model.encdec.encoder(xt * g)
            return model.clustering_head(emb).argmax(-1).cpu().numpy()

    def gates_of(Xm):
        with torch.no_grad():
            xt = torch.tensor(np.asarray(Xm, np.float32), device=DEV)
            return model.gating_net.get_gates(xt).cpu().numpy()

    return infer, gates_of


# ------------------------------------------------------------------ masking curve
def drop_curve(gates, X, infer, y, K, fill):
    """IDC's masking loop (importance = #samples with gate>0, cumulative masking
    in descending order) with a configurable fill value for masked features.
    fill = 0 reproduces metrics.faithfulness_drop bit for bit."""
    imp = np.sum(gates > 0, axis=0)
    ind = np.where(imp > 0)[0]
    order = ind[np.argsort(-imp[imp > 0])]
    base = get_accuracy(np.asarray(infer(X)).astype(int), y, K)
    mask = np.ones(X.shape[1])
    acc = []
    for i in order:
        mask[i] = 0
        Xm = X * mask + (1.0 - mask) * fill
        acc.append(get_accuracy(np.asarray(infer(Xm)).astype(int), y, K))
    acc = np.array(acc)
    top1 = float(base - acc[0]) if len(acc) else float("nan")
    aopc = float(np.mean(base - acc)) if len(acc) else float("nan")
    return {"top1drop": top1, "aopc": aopc}


def main():
    entries = sys.argv[1:] or DATASETS
    out_path = results("faithfulness_masking_check.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    spex_cache = {}                                       # (base, seed) -> rows

    for entry in entries:
        base = base_of(entry)
        paths = {s: idc_out(f"idc_out_{entry}_seed{s}.npz") for s in SEEDS}
        have = [s for s in SEEDS if os.path.exists(paths[s])
                and os.path.exists(os.path.join(MODELS, f"idc_model_{entry}_seed{s}.pt"))]
        if not have:
            print(f"[{entry}] no npz/model, skipped"); continue
        print("=" * 72); print(f"{entry}  (seeds {have}, device {DEV})"); print("=" * 72, flush=True)
        d0 = np.load(paths[have[0]])
        X32 = np.ascontiguousarray(d0["X"], np.float32)   # what the IDC model saw
        X64 = np.ascontiguousarray(d0["X"], float)        # what evaluate.py gives SpEx
        y, K = d0["y_true"].astype(int), int(d0["K"])
        mu = X64.mean(axis=0)
        res_json = json.load(open(results(f"results_multiseed_{entry}.json")))

        spex_rows, idc_rows, cons = [], [], []
        for s in have:
            # ---- SpEx (cached per base dataset: same X for base/_tuned/_best) ----
            if (base, s) not in spex_cache:
                tree, _, gates = spex_side(X64, K, y, seed=s)
                infer_t = lambda Xm, t=tree: t.predict(np.ascontiguousarray(Xm, float)).astype(int)
                spex_cache[(base, s)] = {
                    "zero": drop_curve(gates, X64, infer_t, y, K, 0.0),
                    "mean": drop_curve(gates, X64, infer_t, y, K, mu)}
            spex_rows.append(spex_cache[(base, s)])

            # ---- IDC from the checkpoint ----
            d = np.load(paths[s])
            infer_i, gates_of = load_idc_model(entry, s, X32, y, K)
            g_re = gates_of(X32)
            g_npz = np.asarray(d["gates"], np.float32)
            zero = drop_curve(g_re, X32, infer_i, y, K, 0.0)
            mean = drop_curve(g_re, X32, infer_i, y, K, mu.astype(np.float32))
            idc_rows.append({"zero": zero, "mean": mean})
            def absdiff(a, b):            # None when both undefined (collapsed gates)
                v = abs(float(a) - float(b))
                return None if np.isnan(v) else float(v)
            c = {"seed": s,
                 "gates_maxabsdiff": float(np.max(np.abs(g_re - g_npz))),
                 "labels_mismatch": int((infer_i(X32) != d["labels_pred"].astype(int)).sum()),
                 "top1drop_zero_vs_npz": absdiff(zero["top1drop"], d["faithfulness_top1drop"]),
                 "aopc_zero_vs_npz": absdiff(zero["aopc"], d["faithfulness_aopc"])}
            # SpEx zero-masking must equal the canonical table value for this seed
            sj = res_json["spex"]["faithfulness_top1drop"]["values"][have.index(s)]
            c["spex_top1drop_zero_vs_json"] = absdiff(spex_rows[-1]["zero"]["top1drop"], sj) if sj is not None else None
            cons.append(c)
            print(f"  seed {s}: SpEx top1 {spex_rows[-1]['zero']['top1drop']:+.3f}->{spex_rows[-1]['mean']['top1drop']:+.3f}"
                  f"  IDC top1 {zero['top1drop']:+.3f}->{mean['top1drop']:+.3f}"
                  f"  | reload: gates dmax {c['gates_maxabsdiff']:.1e}, labels mism {c['labels_mismatch']}, "
                  f"top1 dnpz {c['top1drop_zero_vs_npz'] if c['top1drop_zero_vs_npz'] is None else format(c['top1drop_zero_vs_npz'], '.1e')}", flush=True)

        def flat(rows, fill):
            return [{"top1drop": r[fill]["top1drop"], "aopc": r[fill]["aopc"]} for r in rows]
        e = {"seeds": have,
             "spex": {"zero": agg(flat(spex_rows, "zero")), "mean": agg(flat(spex_rows, "mean"))},
             "idc": {"zero": agg(flat(idc_rows, "zero")), "mean": agg(flat(idc_rows, "mean"))},
             "reload_consistency": cons}
        for m in ("top1drop", "aopc"):
            def gap(fill):
                a, b = e["idc"][fill][m]["mean"], e["spex"][fill][m]["mean"]
                return None if a is None or b is None else round(a - b, 4)
            gz, gm = gap("zero"), gap("mean")
            e[f"gap_zero_{m}"], e[f"gap_mean_{m}"] = gz, gm
            e[f"sign_flip_{m}"] = (bool(np.sign(gz) != np.sign(gm))
                                   if gz not in (None, 0) and gm not in (None, 0) else None)
        out[entry] = e
        json.dump(out, open(out_path, "w"), indent=2)

    # ---- summary table ----
    print(f"\n{'entry':<20}{'SpEx top1 0->mean':>20}{'IDC top1 0->mean':>20}{'gap 0->mean':>18}  flip"
          f"{'SpEx AOPC 0->mean':>20}{'IDC AOPC 0->mean':>20}")
    print("-" * 126)
    nflip = nent = 0
    fm = lambda v, w=9, sign=False: ("nan".rjust(w) if v is None else
                                     format(v, f"{'+' if sign else ''}{w}.3f"))
    for entry in [d for d in DATASETS if d in out]:
        e = out[entry]
        f = lambda side, fill, m: e[side][fill][m]["mean"]
        nent += 1; nflip += bool(e["sign_flip_top1drop"])
        print(f"{entry:<20}{fm(f('spex','zero','top1drop'))}->{fm(f('spex','mean','top1drop')).strip():<9}"
              f"{fm(f('idc','zero','top1drop'))}->{fm(f('idc','mean','top1drop')).strip():<9}"
              f"{fm(e['gap_zero_top1drop'], 8, True)}->{fm(e['gap_mean_top1drop'], 8, True).strip():<8}"
              f"  {'YES' if e['sign_flip_top1drop'] else ('-' if e['sign_flip_top1drop'] is None else 'no')}"
              f"{fm(f('spex','zero','aopc'))}->{fm(f('spex','mean','aopc')).strip():<9}"
              f"{fm(f('idc','zero','aopc'))}->{fm(f('idc','mean','aopc')).strip():<9}")
    allc = [c for e in out.values() for c in e["reload_consistency"]]
    worst_g = max(c["gates_maxabsdiff"] for c in allc)
    worst_t = max([c["top1drop_zero_vs_npz"] for c in allc if c["top1drop_zero_vs_npz"] is not None] or [0.0])
    worst_l = max(c["labels_mismatch"] for c in allc)
    print(f"\nsign flips of the top-1-drop gap under mean masking: {nflip}/{nent} entries")
    print(f"checkpoint reload consistency over all models: gates max|diff| {worst_g:.1e}, "
          f"labels mismatches max {worst_l}, zero-masking top1drop vs npz max|diff| {worst_t:.1e}")
    print(f"saved {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
