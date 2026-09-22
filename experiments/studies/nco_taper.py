"""The same question for nested clustered optimization.

NCO solves minimum variance inside each cluster, builds a reduced covariance
over the cluster portfolios, solves minimum variance across them, and
multiplies. It uses a proper solve at both levels, so unlike HRP it should
not behave like inverse variance.

The matching shrinkage family is a flat one-parameter taper on the SAME
clusters: keep within-cluster covariance, scale every cross-cluster entry by
phi, hand it to one long-only minimum-variance solve. phi = 0 is the
block-diagonal estimate, phi = 1 is the raw estimate. Where does NCO sit?
"""
import numpy as np
import cvxpy as cp
from scipy.cluster.hierarchy import fcluster
from _markets import blocks, one_factor, wishart
from confound import hrp, min_var, tree

n = 40



_cache = {}
def mv_lo(S):
    m = S.shape[0]
    if m == 1:
        return np.ones(1)
    if m not in _cache:
        w = cp.Variable(m); P = cp.Parameter((m, m), PSD=True)
        _cache[m] = (w, P, cp.Problem(cp.Minimize(cp.quad_form(w, P)),
                                      [cp.sum(w) == 1, w >= 0]))
    w, P, prob = _cache[m]
    ev, V = np.linalg.eigh((S + S.T)/2)
    P.value = V @ np.diag(np.maximum(ev, 1e-10)) @ V.T
    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is not None:
            x = np.maximum(w.value, 0); return x/x.sum()
    except Exception:
        pass
    x = np.maximum(min_var(S, 1e-6), 0); return x/max(x.sum(), 1e-12)

def clusters(S, k=5):
    Z, _ = tree(S)
    lab = fcluster(Z, t=k, criterion="maxclust")
    return [np.where(lab == g)[0] for g in np.unique(lab)]

def nco(S, cl):
    inner = [mv_lo(S[np.ix_(c, c)]) for c in cl]
    K = len(cl)
    R = np.empty((K, K))
    for a in range(K):
        for b in range(K):
            R[a, b] = inner[a] @ S[np.ix_(cl[a], cl[b])] @ inner[b]
    v = mv_lo(R)
    w = np.zeros(n)
    for a, c in enumerate(cl):
        w[c] = v[a]*inner[a]
    return w/w.sum()

def flat_taper(S, cl, phi):
    T = np.full((n, n), phi)
    for c in cl:
        T[np.ix_(c, c)] = 1.0
    np.fill_diagonal(T, 1.0)
    return S*T

PHIS = (0.0, 0.25, 0.5, 0.75, 1.0)

def run(gen, T, draws=150, seed=0):
    rng = np.random.default_rng(seed)
    acc = {p: [] for p in PHIS}
    ncov, hrpv = [], []
    for _ in range(draws):
        Sig = gen(rng)
        X = rng.normal(size=(T, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        cl = clusters(S, 5)
        w = nco(S, cl); ncov.append(float(w @ Sig @ w))
        wh = hrp(S); hrpv.append(float(wh @ Sig @ wh))
        for p in PHIS:
            wt = mv_lo(flat_taper(S, cl, p)); acc[p].append(float(wt @ Sig @ wt))
    return acc, np.array(ncov), np.array(hrpv)

if __name__ == "__main__":
    for name, gen in (("five blocks", blocks), ("one factor", one_factor)):
        for T in (80, 400):
            acc, ncov, hrpv = run(gen, T, seed=T + len(name))
            print(f"\n=== {name}, T/n = {T/n:.0f} ===")
            print(f"  NCO {np.median(ncov):.4f}    HRP {np.median(hrpv):.4f}")
            print(f"{'phi':>7s}{'variance':>11s}{'beats NCO':>12s}{'beats HRP':>12s}")
            for p in PHIS:
                v = np.array(acc[p])
                print(f"{p:7.2f}{np.median(v):11.4f}{np.mean(v < ncov):11.0%}{np.mean(v < hrpv):12.0%}")
