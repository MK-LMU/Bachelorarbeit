# -*- coding: utf-8 -*-
"""RQ2: are the explanations stable under DATA perturbations -- for SpEx,
are the same tree splits selected; for IDC, are the same gates learned?

Four datasets (iris, breast_cancer, digits, har) under noise sigma in
{0.01, 0.05, 0.10} and subsampling {90%, 80%}, 3 deterministic repetitions
each. Each method is rebuilt on the perturbed data and compared with its own
baseline on two levels: does it still produce the same PARTITION (ARI), and
does it still point at the same FEATURES (split sets and thresholds for SpEx,
global and per-sample gates for IDC). Both ARIs are measured on the clean
points, so the columns stay comparable -- see the symmetry note at the IDC
branch.

IDC is retrained with its fixed _best config and a fixed seed, so its numbers
are read against idc_seed_baseline: the pairwise ARI of the _best seed
runs, i.e. how much IDC already moves under its own training noise.

Outputs: results/perturbation/stability.json,
         notes/figures/perturbation_stability.png
"""
import os, sys, json, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SpEx"))

from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(HERE, "IDC"))
# Same NumPy-2 shim as run_idc.py/metrics.py: the vendored IDC code still
# uses aliases NumPy >= 2 removed, and train_evaluate is imported below.
for _n, _v in [("infty", np.inf), ("NaN", np.nan), ("float", float),
               ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _n):
        setattr(np, _n, _v)
import torch
from omegaconf import OmegaConf
import dataset as idc_dataset
from dataset import ClusteringDataset
import train_evaluate

from wpaths import (idc_out, PERTURB, RESULTS_PERTURB, RESULTS_TUNING,
                    FIGURES, CAMPAIGN_SEEDS)
from perturbations import perturb, tag_of
from spex_pipeline import spex_side

DEV = "cuda" if torch.cuda.is_available() else "cpu"

PY = os.path.join(HERE, "venv", "Scripts", "python.exe")
DATASETS = ["iris", "breast_cancer", "digits", "har"]
NOISE = [0.0, 0.01, 0.05, 0.10]   # 0.0 is the retrain control: same data, fresh
                                   # run. IDC reproduced it EXACTLY (ARI 1.000 in
                                   # all 12 control runs), so every deviation at
                                   # sigma > 0 is a data effect, not training
                                   # noise. SpEx is deterministic by construction.
SUBS = [0.9, 0.8]
REPS = [0, 1, 2]


# ---------------------------------------------------------------- measures
def split_set(tree):
    """(feature -> mean threshold) over the tree's internal nodes."""
    out = {}

    def walk(c):
        if c.left is None:
            return
        out.setdefault(int(c.coordinate), []).append(float(c.threshold))
        walk(c.left); walk(c.right)

    walk(tree.tree.root)
    return {f: float(np.mean(t)) for f, t in out.items()}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def topk(v, k=15):
    o = np.argsort(v)[::-1][:k]
    return o[v[o] > 0]


# ---------------------------------------------------------------- IDC side
def idc_perturbed(ds, spec, lgl, epochs):
    out = os.path.join(PERTURB, f"idc_out_{ds}{tag_of(spec)}_seed0.npz")
    ckpt = os.path.join(PERTURB, f"idc_model_{ds}{tag_of(spec)}_seed0.pt")
    if os.path.exists(out) and os.path.exists(ckpt):
        # tag_of() encodes only kind/strength/repetition, NOT the config. If the
        # tuning winner changed since this run was cached, reusing it would
        # compare a baseline trained with one config against a perturbed run
        # trained with another -- and the "instability" would be bookkeeping,
        # not a data effect. So verify instead of hoping.
        got = torch.load(ckpt, map_location="cpu", weights_only=False)
        if (got.get("lgl"), got.get("epochs")) != (lgl, epochs):
            raise SystemExit(
                f"[{ds} {spec}] cached run was trained with lgl={got.get('lgl')}, "
                f"epochs={got.get('epochs')}, but lgl={lgl}, epochs={epochs} is "
                f"requested. The cache key does not include the config -- delete "
                f"artifacts/perturbation/*{tag_of(spec)}* and rerun.")
    if not os.path.exists(out):
        r = subprocess.run([PY, os.path.join(HERE, "run_idc.py"), "--data", ds,
                            "--seed", "0", "--lgl", str(lgl),
                            "--epochs", str(epochs), "--tag", tag_of(spec),
                            "--outdir", PERTURB, "--perturb", spec],
                           capture_output=True, text=True, cwd=HERE)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
    return np.load(out)


def check_cached_config(ds, spec, lgl, epochs):
    """Does a cached perturbation run carry the config we would train with now?

    Same question idc_perturbed() asks, but reachable on the cached path. A
    mismatch means the stored numbers compare a baseline trained with one
    config against a perturbed run trained with another.
    """
    ckpt = os.path.join(PERTURB, f"idc_model_{ds}{tag_of(spec)}_seed0.pt")
    if not os.path.exists(ckpt):
        return
    got = torch.load(ckpt, map_location="cpu", weights_only=False)
    if (got.get("lgl"), got.get("epochs")) != (lgl, epochs):
        raise SystemExit(
            f"[{ds} {spec}] CACHED run was trained with lgl={got.get('lgl')}, "
            f"epochs={got.get('epochs')}, but the current winner is lgl={lgl}, "
            f"epochs={epochs}. Delete artifacts/perturbation/*{tag_of(spec)}* "
            f"and results/perturbation/stability.json, then rerun.")


def idc_labels_on(ds, spec, X_eval, y, K):
    """Rebuild the perturbed IDC model from its persisted checkpoint and label
    X_eval with it. No training -- the .pt files in artifacts/perturbation/ hold
    the weights. Needed because the npz stores labels for X' only, while the
    comparison below has to score BOTH methods on the same input."""
    ck = torch.load(os.path.join(PERTURB, f"idc_model_{ds}{tag_of(spec)}_seed0.pt"),
                    map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    X32 = np.asarray(X_eval, np.float32)
    if str(cfg.dataset).startswith("SHARED_"):
        class _Shared(ClusteringDataset):            # same injection as run_idc.py
            @classmethod
            def setup(cls, _cfg):
                return cls(X32, y, K)
        setattr(idc_dataset, str(cfg.dataset), _Shared)
    model = train_evaluate.BaseModule(cfg)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval(); model.to(DEV)
    with torch.no_grad():
        xt = torch.tensor(X32, device=DEV)
        g = model.gating_net.get_gates(xt)
        emb = model.encdec.encoder(xt * g)
        return model.clustering_head(emb).argmax(-1).cpu().numpy()


def main():
    sel = json.load(open(os.path.join(RESULTS_TUNING, "selection.json")))
    res_path = os.path.join(RESULTS_PERTURB, "stability.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}

    for ds in (sys.argv[1:] or DATASETS):
        print("=" * 72); print(f"PERTURBATION {ds}"); print("=" * 72, flush=True)
        # Seed 0 on purpose, and it stays. RQ2 measures a DIFFERENCE at a
        # fixed seed -- clean run against perturbed run -- so whatever the
        # selection seed carries cancels between the two sides. Moving to a
        # reported seed would invalidate 72 cached perturbation runs to change
        # nothing that the measure can see.
        d0 = np.load(idc_out(f"idc_out_{ds}_best_seed0.npz"))
        X = np.ascontiguousarray(d0["X"], float)
        y, K = d0["y_true"].astype(int), int(d0["K"])
        w = sel[ds]["winner"]

        # baselines
        tree_b, labels_b_tree, _ = spex_side(X, K, y, seed=0)
        splits_b = split_set(tree_b)
        gates_b = np.ascontiguousarray(d0["gates"], float)
        labels_b_idc = d0["labels_pred"].astype(int)
        # IDC seed-noise yardstick: pairwise ARI of the _best seed labelings
        seed_labels = []
        for s in CAMPAIGN_SEEDS:
            p = idc_out(f"idc_out_{ds}_best_seed{s}.npz")
            if os.path.exists(p):
                seed_labels.append(np.load(p)["labels_pred"].astype(int))
        pair = [adjusted_rand_score(seed_labels[i], seed_labels[j])
                for i in range(len(seed_labels)) for j in range(i + 1, len(seed_labels))]
        entry = res.get(ds) or {}          # resume: keep finished conditions
        if len(pair) < 3:
            # <3 pairwise values means _best seed runs are missing. np.mean([])
            # would give nan, json would write NaN, and make_figure's axhspan
            # would silently draw nothing -- the panel would then read as if
            # IDC had no seed noise at all, inverting its message.
            raise SystemExit(f"[{ds}] only {len(seed_labels)} _best seed runs found; "
                             "the seed-noise band needs at least 3. Run tune_idc.py "
                             "/ reselect_best.py first.")
        entry["idc_seed_baseline_ari"] = round(float(np.mean(pair)), 4)
        entry["winner_config"] = {"lgl": w["lgl"], "epochs": w["epochs"]}
        entry.setdefault("conditions", {})
        # Write the yardstick back NOW, not only when a condition is computed.
        # It is derived from the _best seed runs, so it moves whenever their
        # number does -- while the perturbation conditions stay cached. Storing
        # it inside the loop below meant a fully-cached rerun recomputed the
        # value, printed it, drew it into the figure and then discarded it,
        # leaving JSON and figure disagreeing.
        res[ds] = entry
        json.dump(res, open(res_path, "w"), indent=2)

        for kind, params in (("noise", NOISE), ("subsample", SUBS)):
            for p in params:
                if f"{kind}:{p}" in entry["conditions"]:
                    # Verify the cached condition even though nothing is
                    # recomputed. The guard inside idc_perturbed() only runs on
                    # the path that trains, so a cached run whose config no
                    # longer matches the tuning winner would sail through --
                    # exactly the case the guard exists for.
                    check_cached_config(ds, f"{kind}:{p}:0", w["lgl"], w["epochs"])
                    continue
                reps = []
                for r_ in REPS:
                    spec = f"{kind}:{p}:{r_}"
                    Xp, yp, idx = perturb(X, y, spec)
                    # --- SpEx ---
                    tree_p, _, _ = spex_side(Xp, K, yp, seed=0)
                    splits_p = split_set(tree_p)
                    shared = set(splits_b) & set(splits_p)
                    m = {"spex_split_jaccard": jaccard(splits_b, splits_p),
                         "spex_thr_drift": (float(np.mean([abs(splits_b[f] - splits_p[f])
                                                           for f in shared]))
                                            if shared else float("nan")),
                         "spex_ari_stability": float(adjusted_rand_score(
                             tree_p.predict(X[idx]).astype(int),
                             labels_b_tree[idx]))}
                    # --- IDC ---
                    dp = idc_perturbed(ds, spec, w["lgl"], w["epochs"])
                    g_p = np.ascontiguousarray(dp["gates"], float)
                    l_p = dp["labels_pred"].astype(int)
                    m["idc_gate_cos_global"] = cos(gates_b[idx].mean(0), g_p.mean(0))
                    m["idc_top15_jaccard"] = jaccard(topk(gates_b[idx].mean(0)),
                                                     topk(g_p.mean(0)))
                    percos = [cos(gates_b[idx][i], g_p[i]) for i in range(len(idx))]
                    m["idc_gate_cos_local"] = float(np.nanmean(percos))
                    # SYMMETRY: the SpEx row above scores tree_p on the CLEAN
                    # X[idx], i.e. pure model drift. Scoring IDC on l_p (its
                    # labels for X') would mix model drift with input drift and
                    # make the two columns incomparable under noise. So relabel
                    # the clean points with the perturbed model. Under
                    # subsampling X[idx] == X', so both agree there by
                    # construction. The old, asymmetric value is kept alongside.
                    l_p_clean = idc_labels_on(ds, spec, X[idx], y[idx], K)
                    m["idc_ari_stability"] = float(adjusted_rand_score(
                        l_p_clean, labels_b_idc[idx]))
                    m["idc_ari_stability_on_perturbed"] = float(
                        adjusted_rand_score(l_p, labels_b_idc[idx]))
                    reps.append(m)
                    print(f"  {spec:<18} SpEx J={m['spex_split_jaccard']:.2f} "
                          f"ARI={m['spex_ari_stability']:.3f} | IDC "
                          f"cos={m['idc_gate_cos_global']:.3f} "
                          f"ARI={m['idc_ari_stability']:.3f}", flush=True)
                agg = {k: {"mean": round(float(np.nanmean([r[k] for r in reps])), 4),
                           "std": round(float(np.nanstd([r[k] for r in reps], ddof=1)), 4)}
                       for k in reps[0]}
                entry["conditions"][f"{kind}:{p}"] = {"reps": reps, "agg": agg}
                res[ds] = entry
                json.dump(res, open(res_path, "w"), indent=2)
        print(f"[{ds}] done (seed-baseline ARI {entry['idc_seed_baseline_ari']})\n",
              flush=True)

    make_figure(res)


# ---------------------------------------------------------------- figure
def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE = "#2a78d6", "#eb6834"
    SURFACE, INK, INK2, MUTED, GRID_C, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                               "#898781", "#e1e0d9", "#c3c2b7")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
        "font.size": 9})

    dss = [d for d in DATASETS if d in res]
    fig, axes = plt.subplots(2, len(dss), figsize=(3.0 * len(dss), 5.8),
                             constrained_layout=True, sharex=True)
    axes = np.asarray(axes).reshape(2, len(dss))
    pos = list(range(len(NOISE)))                 # categorical x: equal spacing
    xlabels = ["0\ncontrol"] + [f"{p:g}" for p in NOISE[1:]]

    for col, ds in enumerate(dss):
        e = res[ds]
        get = lambda m: [e["conditions"][f"noise:{p}"]["agg"][m]["mean"] for p in NOISE]
        err = lambda m: [e["conditions"][f"noise:{p}"]["agg"][m]["std"] for p in NOISE]

        ax = axes[0, col]
        ax.axhspan(e["idc_seed_baseline_ari"], 1.0, color=GRID_C, alpha=0.55,
                   lw=0, label="IDC's variability across\nseeds (reference band)"
                   if col == 0 else None)
        ax.errorbar(pos, get("spex_ari_stability"), yerr=err("spex_ari_stability"),
                    color=BLUE, lw=2, marker="o", ms=4, capsize=2, label="SpEx")
        ax.errorbar(pos, get("idc_ari_stability"), yerr=err("idc_ari_stability"),
                    color=ORANGE, lw=2, marker="s", ms=4, capsize=2, label="IDC")
        if col == 0:
            ax.annotate("both exactly 1.0\n(retrain reproduces)", xy=(0, 1.0),
                        xytext=(0.35, 0.62), fontsize=7, color=INK2,
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
        ax.set_title(ds, fontsize=10, color=INK)
        ax.set_ylim(-0.05, 1.07)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID_C, lw=0.6)

        ax = axes[1, col]
        # Bounded and coarsely quantised at n=3: on Iris the tree uses two
        # features, so the split Jaccard can only be 0, 0.5 or 1, and D=4 < k=15
        # makes IDC's top-15 Jaccard purely 0 or 1. mean±std bars would suggest
        # values that cannot occur (and reach past the [0,1] bounds), so show the
        # INDIVIDUAL repetitions plus a thin mean line for the trend.
        for m_, color, dx, mk in (("spex_split_jaccard", BLUE, -0.09, "o"),
                                  ("idc_top15_jaccard", ORANGE, 0.09, "s")):
            ax.plot(pos, get(m_), color=color, lw=1.6, alpha=0.55, zorder=2)
            for i_, p_ in enumerate(NOISE):
                vals = [r[m_] for r in e["conditions"][f"noise:{p_}"]["reps"]]
                ax.scatter([pos[i_] + dx] * len(vals), vals, s=16, marker=mk,
                           color=color, edgecolors=SURFACE, lw=0.5, zorder=3)
        ax.set_ylim(-0.05, 1.07)
        ax.set_xlabel("data noise level σ", fontsize=8)
        ax.set_xticks(pos)
        ax.set_xticklabels(xlabels, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID_C, lw=0.6)

    axes[0, 0].set_ylabel("Does the PARTITION\nstay the same? (ARI, 1 = identical)",
                          fontsize=8.5, color=INK2)
    axes[1, 0].set_ylabel("Do the EXPLANATION FEATURES\nstay the same? (Jaccard)",
                          fontsize=8.5, color=INK2)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("RQ2 — what happens when the data is slightly perturbed and "
                 "the methods rerun? (top: partition; bottom: selected "
                 "features — SpEx splits resp. IDC top-15 gates)",
                 fontsize=10, color=INK)
    out = os.path.join(FIGURES, "perturbation_stability.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
