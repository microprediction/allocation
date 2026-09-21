"""The shrinkage that most resembles HRP, run properly, beats HRP.

The two-level taper: on HRP's own tree, zero every correlation between
assets that separate above depth d, keep a fraction phi of the rest. Fitted
to HRP's weights it lands near d = 5, phi = 0.5 and is the closest sensible
shrinkage to HRP. Here (d, phi) are FIXED IN ADVANCE, identical on every
draw, with no reference to HRP's output, and the covariance is then handed
to a plain long-only minimum-variance solve.

If most of this crude surface beats HRP then the recursion is not paying for
itself: the same information, shrunk the same way and solved properly, does
better.
"""
import numpy as np
import cvxpy as cp
from _markets import blocks, one_factor, wishart
from confound import hrp, min_var, tree, quasi_diag
from decompose import lca_depth

n = 40



_w = cp.Variable(n); _P = cp.Parameter((n, n), PSD=True)
_prob = cp.Problem(cp.Minimize(cp.quad_form(_w, _P)), [cp.sum(_w) == 1, _w >= 0])

def mv_lo(S):
    ev, V = np.linalg.eigh((S + S.T)/2)
    _P.value = V @ np.diag(np.maximum(ev, 1e-10)) @ V.T
    try:
        _prob.solve(solver=cp.CLARABEL)
        if _w.value is not None:
            w = np.maximum(_w.value, 0); return w/w.sum()
    except Exception:
        pass
    w = np.maximum(min_var(S, 1e-6), 0); return w/max(w.sum(), 1e-12)

def taper(S, L, d, phi):
    T = np.ones((n, n))
    T[(L <= d) & (L > 0)] = 0.0
    T[L > d] = phi
    np.fill_diagonal(T, 1.0)
    return S*T

DS = (1, 3, 5)
PHIS = (0.0, 0.25, 0.5, 0.75, 1.0)

def run(gen, T, draws=150, seed=0):
    rng = np.random.default_rng(seed)
    acc = {(d, p): [] for d in DS for p in PHIS}
    h = []
    for _ in range(draws):
        Sig = gen(rng)
        X = rng.normal(size=(T, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        Zl, _ = tree(S); L = lca_depth(quasi_diag(Zl, n), n)
        w = hrp(S); h.append(float(w @ Sig @ w))
        for d in DS:
            for p in PHIS:
                wt = mv_lo(taper(S, L, d, p))
                acc[(d, p)].append(float(wt @ Sig @ wt))
    return acc, np.array(h)

if __name__ == "__main__":
    for name, gen in (("five blocks", blocks), ("one factor", one_factor)):
        for T in (80, 400):
            acc, h = run(gen, T, seed=T + len(name))
            print(f"\n=== {name}, T/n = {T/n:.0f}.  HRP = {np.median(h):.4f} ===")
            print(f"{'depth':>7s}" + "".join(f"{('phi='+str(p)):>14s}" for p in PHIS))
            for d in DS:
                cells = []
                for p in PHIS:
                    v = np.array(acc[(d, p)])
                    cells.append(f"{np.median(v):.4f} ({np.mean(v < h):.0%})")
                print(f"{d:7d}" + "".join(f"{c:>14s}" for c in cells))
            print("   cell = median realized variance (share of draws beating HRP)")
