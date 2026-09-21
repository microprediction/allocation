"""What shrinkage is HRP equivalent to?

Every allocator is minimum variance on SOME covariance, so the question is
which structured family the implied covariance lies in. Three candidates,
each fitted to reproduce the HRP weights as closely as it can:

  flat      S(l) = l*S + (1-l)*diag(S)          linear shrinkage to the diagonal
  taper     S(t) = S o T,  T_ij = t_{lca(i,j)}  one multiplier per tree level
  scalar    S(a) = S + a*I                      ridge

Fit quality is L1 weight distance, against the yardstick that HRP itself sits
||w_hrp - w_ivp||_1 away from inverse variance.
"""
import numpy as np
from scipy.optimize import minimize
from confound import true_sigma, hrp, min_var
from allocation._schur.bridge import bisection_tree, tree_leaves
from allocation._schur.seriation import seriate


def lca_depth(tree, n):
    """L[i,j] = depth of the lowest common ancestor of i and j; L[i,i] = 0."""
    L = np.zeros((n, n), dtype=int)
    def rec(t, depth):
        if isinstance(t, np.ndarray):
            return list(t)
        a, b = rec(t[0], depth + 1), rec(t[1], depth + 1)
        for i in a:
            for j in b:
                L[i, j] = L[j, i] = depth + 1
        return a + b
    rec(tree, 0)
    return L


def fit_flat(S, w):
    D = np.diag(np.diag(S))
    f = lambda l: np.abs(min_var(l[0]*S + (1-l[0])*D, 1e-10) - w).sum()
    g = np.linspace(0, 1, 201)
    v = [f([x]) for x in g]
    return g[int(np.argmin(v))], min(v)


def fit_taper(S, w, L):
    lv = sorted(set(L[L > 0].tolist()))
    def make(t):
        T = np.ones_like(S)
        for k, l in enumerate(lv):
            T[L == l] = t[k]
        return S * T
    f = lambda t: np.abs(min_var(make(np.clip(t, 0, 1.5)), 1e-10) - w).sum()
    best = None
    for x0 in (np.full(len(lv), .5), np.linspace(.1, .9, len(lv)), np.full(len(lv), .05)):
        r = minimize(f, x0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return np.clip(best.x, 0, 1.5), best.fun, lv


def fit_ridge(S, w):
    tr = np.trace(S)/S.shape[0]
    g = np.geomspace(1e-6, 1e3, 200)*tr
    v = [np.abs(min_var(S + a*np.eye(S.shape[0]), 0.0) - w).sum() for a in g]
    return g[int(np.argmin(v))]/tr, min(v)


rng = np.random.default_rng(101)
n = 40
res = {"flat": [], "taper": [], "ridge": [], "ivp": []}
tapers = []
for _ in range(120):
    Sig, _ = true_sigma(rng, 5, 8, 0.7, 0.15)
    X = rng.normal(size=(120, n)) @ np.linalg.cholesky(Sig).T
    S = np.cov(X, rowvar=False)
    tree = bisection_tree(seriate(S)[0], leaf_size=1)
    L = lca_depth(tree, n)
    w = hrp(S)
    ivp = 1/np.diag(S); ivp /= ivp.sum()
    res["ivp"].append(np.abs(w - ivp).sum())
    lam, d1 = fit_flat(S, w);       res["flat"].append((lam, d1))
    t, d2, lv = fit_taper(S, w, L); res["taper"].append(d2); tapers.append((lv, t))
    a, d3 = fit_ridge(S, w);        res["ridge"].append((a, d3))

print(f"HRP weights, n = {n}, L1 distances (total weight is 1.0)\n")
print(f"  distance from inverse variance          {np.median(res['ivp']):.4f}")
fl = np.array([r[1] for r in res["flat"]]); lam = np.array([r[0] for r in res["flat"]])
rd = np.array([r[1] for r in res["ridge"]]); aa = np.array([r[0] for r in res["ridge"]])
print(f"  best flat shrinkage to the diagonal     {np.median(fl):.4f}   at lambda = {np.median(lam):.3f}")
print(f"  best ridge                              {np.median(rd):.4f}   at a/tr = {np.median(aa):.3g}")
print(f"  best tree taper, one dial per level     {np.median(res['taper']):.4f}")
lv0 = tapers[0][0]
M = np.array([t for _, t in tapers])
print(f"\nfitted taper multiplier by tree level (level 1 = the top split)")
for k, l in enumerate(lv0):
    q = np.percentile(M[:, k], [25, 50, 75])
    print(f"  level {l}   median {q[1]:6.3f}   IQR [{q[0]:.3f}, {q[2]:.3f}]")
