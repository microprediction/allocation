"""Split HRP's penalty into what it DISCARDS and what it DOES.

HRP differs from minimum variance in two ways at once:

  (1) the information set.  Its recursion never reads the covariance between
      the two halves of the top split, so it optimises on a block-diagonal
      view of the data.
  (2) the rule.  On what it does keep it uses inverse-variance splits rather
      than a minimum-variance solve.

Z_d(S) is S with every entry zeroed whose two assets separate at depth <= d
in the bisection tree: block diagonal with 2^d blocks, each block the exact
sub-covariance.  Z_0 = S, and at full depth Z = diag(S).  min-var on Z_d is
therefore the BEST use of the information HRP retains at that depth.

   information cost = var(minvar(Z_1)) - var(minvar(S))
   rule cost        = var(HRP)         - var(minvar(Z_1))

If the rule cost dominates, HRP is a poor allocator and not an estimation
choice at all.  If the information cost dominates, and is negative, then
discarding is the whole of HRP's contribution and it is an estimator.
"""
import numpy as np
from confound import true_sigma, hrp, min_var, tree, quasi_diag


def lca_depth(order, n):
    L = np.zeros((n, n), dtype=int)
    def rec(idx, depth):
        if len(idx) <= 1:
            return
        h = len(idx) // 2
        l, r = idx[:h], idx[h:]
        for i in l:
            for j in r:
                L[i, j] = L[j, i] = depth + 1
        rec(l, depth + 1); rec(r, depth + 1)
    rec(list(order), 0)
    return L


def Z(S, L, d):
    """S with every pair separating at depth <= d set to zero."""
    M = S.copy()
    M[(L <= d) & (L > 0)] = 0.0
    return M


def run(T, n=40, draws=250, seed=0):
    rng = np.random.default_rng(seed)
    dmax = 6
    out = {("Z", d): [] for d in range(dmax + 1)}
    out[("hrp", 0)] = []
    for _ in range(draws):
        Sig, _ = true_sigma(rng, 5, 8, 0.7, 0.15)
        X = rng.normal(size=(T, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        Zl, _ = tree(S); order = quasi_diag(Zl, n)
        L = lca_depth(order, n)
        w = hrp(S); out[("hrp", 0)].append(float(w @ Sig @ w))
        for d in range(dmax + 1):
            M = Z(S, L, d)
            wv = min_var(M, 1e-10)
            out[("Z", d)].append(float(wv @ Sig @ wv))
    return {k: np.median(v) for k, v in out.items()}


if __name__ == "__main__":
    print("true-covariance variance, median over 250 draws, n = 40, five blocks of eight")
    print(f"{'T/n':>6s} {'minvar(S)':>11s} {'minvar(Z1)':>12s} {'minvar(Z2)':>12s}"
          f" {'minvar(Z3)':>12s} {'inv var':>9s} {'HRP':>9s}")
    rows = {}
    for T in (20, 40, 60, 120, 400):
        r = run(T, seed=T)
        rows[T] = r
        print(f"{T/40:6.2f} {r[('Z',0)]:11.4f} {r[('Z',1)]:12.4f} {r[('Z',2)]:12.4f}"
              f" {r[('Z',3)]:12.4f} {r[('Z',6)]:9.4f} {r[('hrp',0)]:9.4f}")
    print()
    print("decomposition of HRP's penalty relative to minimum variance on the raw estimate")
    print(f"{'T/n':>6s} {'total':>10s} {'information':>13s} {'rule':>10s} {'rule share':>12s}")
    for T, r in rows.items():
        total = r[("hrp", 0)] - r[("Z", 0)]
        info = r[("Z", 1)] - r[("Z", 0)]
        rule = r[("hrp", 0)] - r[("Z", 1)]
        share = rule / total if total != 0 else float("nan")
        print(f"{T/40:6.2f} {total:+10.4f} {info:+13.4f} {rule:+10.4f} {share:11.0%}")
