# -*- coding: utf-8 -*-
"""Shared data-perturbation function for the RQ2 stability experiment.

Imported by BOTH perturbation_stability.py (SpEx side, CPU) and run_idc.py
(IDC side, GPU) so both methods see EXACTLY the same perturbed data.

spec format: "<kind>:<param>:<rep>"
  noise:0.05:2      -> X' = clip(X + N(0, 0.05^2), 0, 1)      (MinMax domain)
  subsample:0.9:1   -> keep a random 90% of the samples (without replacement)

Deterministic: the RNG seed is derived from (kind, param, rep), so the same
spec always produces the same X' in every process.
"""
import numpy as np

KINDS = ("noise", "subsample")


def parse_spec(spec):
    kind, param, rep = spec.split(":")
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation kind: {kind!r}")
    return kind, float(param), int(rep)


def _rng(kind, param, rep):
    return np.random.default_rng([KINDS.index(kind), int(round(param * 1000)), rep])


def perturb(X, y, spec):
    """Returns (X', y', idx) with idx = indices of X' rows in the ORIGINAL X
    (identity for noise). X stays within [0, 1] (MinMax domain)."""
    kind, param, rep = parse_spec(spec)
    rng = _rng(kind, param, rep)
    X = np.asarray(X, float)
    if kind == "noise":
        Xp = np.clip(X + rng.normal(0.0, param, size=X.shape), 0.0, 1.0)
        return Xp, np.asarray(y), np.arange(len(X))
    n_keep = int(round(param * len(X)))
    idx = np.sort(rng.permutation(len(X))[:n_keep])
    return X[idx].copy(), np.asarray(y)[idx].copy(), idx


def tag_of(spec):
    kind, param, rep = parse_spec(spec)
    return f"_p{kind}{param}r{rep}"
