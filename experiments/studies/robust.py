"""Robustness across simulated dependence structures, not three hand-picks.

Every draw samples a fresh structure: the family, the number and sizes of
clusters, correlation levels, extra common factors, volatility dispersion,
the sample ratio and the tail weight. Nothing is tuned per draw. The
shrinkage rule is a SINGLE fixed setting, phi = 0.5 on cross-cluster entries
of a five-cluster cut, so it has no advantage of hindsight. The oracle-best
phi is reported separately for reference only.
"""
import numpy as np
import cvxpy as cp
from scipy.cluster.hierarchy import fcluster
from confound import hrp, min_var, tree

PHIS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIXED_PHI = 0.5
KCUT = 5


def random_structure(rng, n):
    """A correlation matrix from one of six families, with random parameters."""
    fam = rng.choice(["blocks", "blocks+factor", "factor", "equicorr",
                      "wishart", "nested"])
    if fam in ("blocks", "blocks+factor"):
        K = int(rng.integers(2, 11))
        cuts = np.sort(rng.choice(np.arange(1, n), size=K - 1, replace=False))
        sizes = np.diff(np.concatenate([[0], cuts, [n]]))
        rho_out = float(rng.uniform(0.0, 0.35))
        C = np.full((n, n), rho_out)
        i = 0
        for s in sizes:
            rin = float(rng.uniform(max(rho_out + 0.05, 0.2), 0.92))
            C[i:i + s, i:i + s] = rin
            i += s
        np.fill_diagonal(C, 1.0)
        if fam == "blocks+factor":
            nf = int(rng.integers(1, 4))
            B = rng.normal(0, rng.uniform(0.2, 0.6), size=(n, nf))
            C = C + B @ B.T
    elif fam == "factor":
        nf = int(rng.integers(1, 5))
        B = rng.normal(rng.uniform(0, 1.0), rng.uniform(0.2, 0.8), size=(n, nf))
        C = B @ B.T + np.diag(np.exp(rng.normal(0, rng.uniform(0.2, 0.8), n)))
    elif fam == "equicorr":
        r = float(rng.uniform(0.05, 0.85))
        C = np.full((n, n), r); np.fill_diagonal(C, 1.0)
    elif fam == "wishart":
        df = int(rng.integers(n + 2, 4 * n))
        A = rng.normal(size=(n, df)); C = A @ A.T / df
    else:                                        # nested hierarchy
        C = np.full((n, n), float(rng.uniform(0.0, 0.2)))
        for depth, lev in enumerate((2, 4, 8)):
            step = max(n // lev, 1)
            add = float(rng.uniform(0.05, 0.3))
            for b in range(lev):
                s = slice(b * step, min((b + 1) * step, n))
                C[s, s] += add
        np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh((C + C.T) / 2)
    C = V @ np.diag(np.maximum(ev, 1e-6)) @ V.T
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    vol = np.exp(rng.normal(0, float(rng.uniform(0.0, 0.8)), n))
    return vol[:, None] * C * vol[None, :], fam


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
    ev, V = np.linalg.eigh((S + S.T) / 2)
    P.value = V @ np.diag(np.maximum(ev, 1e-10)) @ V.T
    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is not None:
            x = np.maximum(w.value, 0)
            if x.sum() > 0:
                return x / x.sum()
    except Exception:
        pass
    x = np.maximum(min_var(S, 1e-6), 0)
    return x / max(x.sum(), 1e-12)


def clusters(S, k):
    Z, _ = tree(S)
    lab = fcluster(Z, t=k, criterion="maxclust")
    return [np.where(lab == g)[0] for g in np.unique(lab)]


def nco(S, cl, n):
    inner = [mv_lo(S[np.ix_(c, c)]) for c in cl]
    K = len(cl)
    R = np.empty((K, K))
    for a in range(K):
        for b in range(K):
            R[a, b] = inner[a] @ S[np.ix_(cl[a], cl[b])] @ inner[b]
    v = mv_lo(R)
    w = np.zeros(n)
    for a, c in enumerate(cl):
        w[c] = v[a] * inner[a]
    return w / w.sum()


def taper(S, cl, phi, n):
    T = np.full((n, n), phi)
    for c in cl:
        T[np.ix_(c, c)] = 1.0
    np.fill_diagonal(T, 1.0)
    return S * T


def main(draws=500, seed=17):
    rng = np.random.default_rng(seed)
    rec = []
    for _ in range(draws):
        n = int(rng.choice([20, 40, 60]))
        Sig, fam = random_structure(rng, n)
        ratio = float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0]))
        T = max(int(ratio * n), 5)
        L = np.linalg.cholesky(Sig)
        if rng.random() < 0.3:                       # fat tails sometimes
            nu = float(rng.uniform(4, 12))
            Zt = rng.standard_t(nu, size=(T, n)) / np.sqrt(nu / (nu - 2))
        else:
            Zt = rng.normal(size=(T, n))
        X = Zt @ L.T
        S = np.cov(X, rowvar=False)
        cl = clusters(S, min(KCUT, n))
        v = lambda w: float(w @ Sig @ w)
        row = {"fam": fam, "n": n, "ratio": ratio,
               "hrp": v(hrp(S)), "nco": v(nco(S, cl, n)),
               "fixed": v(mv_lo(taper(S, cl, FIXED_PHI, n))),
               "raw": v(mv_lo(S))}
        row["oracle"] = min(v(mv_lo(taper(S, cl, p, n))) for p in PHIS)
        rec.append(row)
    return rec


if __name__ == "__main__":
    rec = main()
    import collections
    def rate(a, b):
        return np.mean([r[a] < r[b] for r in rec])
    def ratio(a, b):
        return np.median([r[a] / r[b] for r in rec])
    print(f"{len(rec)} random structures: family, cluster count and sizes, correlation")
    print("levels, extra factors, vol dispersion, n, T/n and tail weight all resampled.\n")
    print(f"fixed phi=0.5 taper beats HRP   {rate('fixed','hrp'):.0%}   "
          f"median variance ratio {ratio('fixed','hrp'):.3f}")
    print(f"fixed phi=0.5 taper beats NCO   {rate('fixed','nco'):.0%}   "
          f"median variance ratio {ratio('fixed','nco'):.3f}")
    print(f"NCO beats HRP                   {rate('nco','hrp'):.0%}   "
          f"median variance ratio {ratio('nco','hrp'):.3f}")
    print(f"raw long-only min-var beats HRP {rate('raw','hrp'):.0%}   "
          f"median variance ratio {ratio('raw','hrp'):.3f}")
    print("\nfixed-taper win rate over HRP, by family")
    by = collections.defaultdict(list)
    for r in rec:
        by[r["fam"]].append(r["fixed"] < r["hrp"])
    for k in sorted(by):
        print(f"  {k:16s} {np.mean(by[k]):5.0%}   (n={len(by[k])})")
    print("\nfixed-taper win rate over HRP, by sample ratio")
    by = collections.defaultdict(list)
    for r in rec:
        by[r["ratio"]].append(r["fixed"] < r["hrp"])
    for k in sorted(by):
        print(f"  T/n = {k:5.1f}      {np.mean(by[k]):5.0%}   (n={len(by[k])})")
