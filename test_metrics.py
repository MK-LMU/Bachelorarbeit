"""Property tests for the metric layer (metrics.py) — data-free, runs in seconds,
exit code 1 on any failure. Pins down the facts the thesis relies on:

  diversity offset    IDC's original `diversity` sums the Jaccard matrix
      INCLUDING the diagonal but divides by K(K-1): code == (published A.7
      formula) - 100/(K-1) whenever every cluster's median set is non-empty;
      its ceiling is 100*(1-1/(K-1)). With an EMPTY median set sklearn's
      jaccard_score(0,0) == 0 and the value can exceed that ceiling (the
      empty-set pathology).
  diversity repaired  `diversity_fixed`: range [0, 100], 100 for disjoint
      feature use, 0 for identical, invariant under joint rescaling, nan for
      an all-zero cluster.
  K pass-through      `faithfulness_k` is IDC's faithfulness with n_clusters
      passed through: identical output at K = 10; the original raises
      IndexError at K != 10.
  fast path           the neighbour-structure fast path (uniqueness_pre /
      stability_pre) is bit-exact vs IDC's originals on random data and on an
      integer tie grid; stability(k=2) == uniqueness(k=2); per sample
      stability_k5 >= uniqueness_k2.
  normalisation       `row_normalize`: every non-zero row has max 1, all-zero
      rows stay 0; nn_identical_frac (sklearn) == nn_identical_frac_pre (IDC
      argsort) without ties.
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from sklearn.metrics import jaccard_score
from metrics import (faithfulness, diversity, uniqueness, stability,
                     faithfulness_k, diversity_fixed, row_normalize, precompute_nn,
                     uniqueness_pre, stability_pre, nn_identical_frac,
                     nn_identical_frac_pre, distance_matrix)

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def median_sets(y, gates, K):
    return [set(np.where(np.median(gates[y == c], axis=0) > 0)[0]) for c in range(K)]


def ref_diversity(sets):
    """Published A.7 reading: mean Jaccard over pairs i != j, scaled to [0, 100]."""
    K = len(sets)
    sims = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            for i in range(K) for j in range(K) if i != j]
    return 100.0 * (1.0 - np.mean(sims))


def nan_equal(a, b):
    return (np.isnan(a) and np.isnan(b)) or a == b


rng = np.random.default_rng(0)

# ------------------------------------------------- diversity offset
print("[diversity offset] diagonal, ceiling, empty-set pathology")
for K in (2, 6, 10):
    N, D = 40 * K, 30
    y = np.repeat(np.arange(K), 40)
    g = np.zeros((N, D))
    for c in range(K):
        feats = rng.choice(D, size=rng.integers(3, 10), replace=False)
        g[np.ix_(np.where(y == c)[0], feats)] = 1.0
    code = float(diversity(y, g, num_clusters=K, num_features=D))
    ref = ref_diversity(median_sets(y, g, K))
    check(f"K={K}: code == A.7 formula - 100/(K-1)", np.isclose(code, ref - 100.0 / (K - 1)),
          f"code={code:.6f} ref={ref:.6f}")
    gd = np.zeros((20 * K, 5 * K))
    yd = np.repeat(np.arange(K), 20)
    for c in range(K):
        gd[np.ix_(np.where(yd == c)[0], range(5 * c, 5 * c + 5))] = 1.0
    ceil = float(diversity(yd, gd, num_clusters=K, num_features=5 * K))
    check(f"K={K}: disjoint sets hit the ceiling 100(1-1/(K-1))",
          np.isclose(ceil, 100.0 * (1 - 1 / (K - 1))), f"{ceil:.4f}")
check("jaccard_score(empty, empty) == 0 (sklearn default zero_division='warn')",
      jaccard_score(np.zeros(8), np.zeros(8)) == 0.0)
K = 3
y3 = np.repeat(np.arange(K), 20)
g3 = np.zeros((60, 9))
# clusters 1 and 2 get disjoint feature sets; cluster 0 is left untouched, so
# its median gate vector is all-zero -> empty median set
g3[np.ix_(np.where(y3 == 1)[0], [0, 1, 2])] = 1.0
g3[np.ix_(np.where(y3 == 2)[0], [3, 4, 5])] = 1.0
v = float(diversity(y3, g3, num_clusters=K, num_features=9))
check("empty median set pushes the value ABOVE the ceiling (pathology)",
      v > 100.0 * (1 - 1 / (K - 1)) + 1e-9, f"{v:.4f} > {100.0 * (1 - 1 / (K - 1)):.4f}")

# ----------------------------------------------- diversity repaired
print("[diversity repaired] properties of diversity_fixed")
K = 4
y4 = np.repeat(np.arange(K), 25)
dis = np.zeros((100, 8))
for c in range(K):
    dis[np.ix_(np.where(y4 == c)[0], [2 * c, 2 * c + 1])] = rng.random((25, 2)) + 0.1
check("disjoint feature use -> 100", np.isclose(diversity_fixed(y4, dis, K), 100.0))
same = np.tile(rng.random((1, 8)) + 0.1, (100, 1))
check("identical feature use -> 0", np.isclose(diversity_fixed(y4, same, K), 0.0))
mix = rng.random((100, 8)) * (rng.random((100, 8)) < 0.5)
mix[np.arange(100), rng.integers(0, 8, 100)] += 0.5               # every cluster non-empty
a, b = diversity_fixed(y4, mix, K), diversity_fixed(y4, 7.3 * mix, K)
check("invariant under joint rescaling", np.isclose(a, b), f"{a:.6f} vs {b:.6f}")
check("range [0, 100]", 0.0 <= a <= 100.0, f"{a:.4f}")
empty = mix.copy()
empty[y4 == 2] = 0.0
check("all-zero cluster mean -> nan (not 'maximally diverse')",
      np.isnan(diversity_fixed(y4, empty, K)))
# same feature SET, different magnitudes per cluster -> must be 0, not diverse.
# This is the case joint-rescaling invariance cannot catch.
onefeat = np.zeros((100, 8))
for c in range(K):
    onefeat[np.where(y4 == c)[0], 3] = 0.2 + 0.3 * c      # one shared feature
check("same feature set, different magnitudes -> 0",
      np.isclose(diversity_fixed(y4, onefeat, K), 0.0),
      f"{diversity_fixed(y4, onefeat, K):.4f}")

# ---------------------------------------------------- K pass-through
print("[K pass-through] faithfulness_k vs IDC faithfulness")
D = 6
X10, y10 = rng.random((200, D)), rng.integers(0, 10, 200)
g10 = (rng.random((200, D)) < 0.5) * rng.random((200, D))
inf10 = lambda Xm: (np.abs(Xm).sum(1) * 13).astype(int) % 10
orig = float(faithfulness(g10, X10, inf10, y10, num_features=D))
ours = faithfulness_k(g10, X10, inf10, y10, num_features=D, n_clusters=10)
check("K=10: identical value", nan_equal(orig, ours), f"{orig!r} vs {ours!r}")
X3, y3b = rng.random((60, D)), rng.integers(0, 3, 60)
g3b = (rng.random((60, D)) < 0.4) * rng.random((60, D))
inf3 = lambda Xm: (np.abs(Xm).sum(1) * 7).astype(int) % 3
raised = False
try:
    faithfulness(g3b, X3, inf3, y3b, num_features=D)
except IndexError:
    raised = True
check("K=3: IDC original raises IndexError (hardcoded n_clusters=10)", raised)
check("K=3: faithfulness_k runs", np.isfinite(faithfulness_k(g3b, X3, inf3, y3b,
                                                              num_features=D, n_clusters=3)))
g1 = np.zeros((60, D)); g1[:, 2] = 1.0
check("<2 used features -> nan (documented degeneracy)",
      np.isnan(faithfulness_k(g1, X3, inf3, y3b, num_features=D, n_clusters=3)))

# --------------------------------------------------------- fast path
print("[fast path] == IDC originals (bit-exact), k=2/k=5 relations")
cases = {"random": rng.random((300, 5)),
         "integer tie grid": rng.integers(0, 10, size=(300, 4)).astype(float)}
for name, X in cases.items():
    Nn, Dd = X.shape
    gates_list = {"dense": rng.random((Nn, Dd)),
                  "sparse": rng.random((Nn, Dd)) * (rng.random((Nn, Dd)) < 0.1),
                  "leaf-constant": np.repeat(rng.random((6, Dd)), 50, axis=0)}
    nn_d, nn_i = precompute_nn(X, kmax=5)
    for gname, g in gates_list.items():
        ok = True
        # subset_size=Nn is required, not a tuning choice: IDC's originals loop
        # over range(subset_size) with a default of 10000, which would run off
        # the end of these 300-sample fixtures. Passing the full N also makes the
        # bit-exactness claim cover every sample. k=2/5 are the two values
        # all_metrics actually reports (uniqueness at 2, stability at 5).
        for k in (2, 5):
            for slow, fast in ((uniqueness, uniqueness_pre), (stability, stability_pre)):
                a = float(slow(X, g, k=k, subset_size=Nn))
                b = float(fast(g, nn_d, nn_i, k=k))
                ok &= nan_equal(a, b)
        check(f"{name}/{gname}: uniqueness_pre/stability_pre == originals (k=2,5)", ok)
    g = gates_list["dense"]
    u2 = float(uniqueness(X, g, k=2, subset_size=Nn))
    s2 = float(stability(X, g, k=2, subset_size=Nn))
    check(f"{name}: stability(k=2) == uniqueness(k=2)", nan_equal(u2, s2), f"{u2!r} vs {s2!r}")
    per_u2, per_s5 = [], []
    for i in range(Nn):
        r5 = distance_matrix(g[nn_i[i, :5]], g[nn_i[i, :5]])[0][1:] / nn_d[i, :5][1:]
        per_u2.append(r5[0]); per_s5.append(max(r5))
    per_u2, per_s5 = np.array(per_u2), np.array(per_s5)
    fin = np.isfinite(per_u2) & np.isfinite(per_s5)
    check(f"{name}: per-sample stability_k5 >= uniqueness_k2", bool(np.all(per_s5[fin] >= per_u2[fin])),
          f"{int(fin.sum())} finite samples")

# ----------------------------------------------------- normalisation
print("[normalisation] row_normalize and nn_identical_frac variants")
g = rng.random((50, 7)); g[3] = 0.0; g[10] *= 1e-3
rn = row_normalize(g)
check("row max == 1 for non-zero rows", np.allclose(rn[rn.max(1) > 0].max(1), 1.0))
check("all-zero row stays 0", np.all(rn[3] == 0.0))
X = rng.random((200, 4))
g = np.repeat(rng.random((8, 4)), 25, axis=0)
_, nn_i = precompute_nn(X, kmax=2)
check("nn_identical_frac (sklearn NN) == nn_identical_frac_pre (IDC argsort), no ties",
      nn_identical_frac(X, g) == nn_identical_frac_pre(g, nn_i))

print("=" * 60)
print(f"RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
