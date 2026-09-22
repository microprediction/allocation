"""Why no equivalent shrinkage turns up: a dimension count, not a bad search.

A shrinkage family with k parameters traces, at a FIXED S, a k-dimensional
set of covariance matrices, hence at most a k-dimensional set of
minimum-variance weight vectors. The HRP weights are a point in the budget
simplex with n-1 free coordinates. If k < n-1 the point generically misses
the set, however cleverly the family is chosen and however the parameter is
tuned per market.

Test: taper families S o T with T = 1 + B theta, where B is a random
orthonormal basis of k directions in the n(n-1)/2 upper-triangle space. This
is the most generous small family available, since the basis is unstructured
and theta is refitted for every market. Sweep k and watch where the error
falls off a cliff.
"""
import numpy as np
from scipy.optimize import least_squares
from confound import true_sigma, hrp, min_var

n = 40
iu = np.triu_indices(n, 1)
P = len(iu[0])


def fit_k(S, w, B):
    def resid(th):
        T = np.ones((n, n))
        T[iu] = 1.0 + B @ th
        T = np.triu(T, 1)
        T = T + T.T + np.eye(n)
        return min_var(S * T, 1e-10) - w
    r = least_squares(resid, np.zeros(B.shape[1]), method="trf",
                      xtol=1e-13, ftol=1e-13, max_nfev=120)
    return float(np.abs(resid(r.x)).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    KS = [1, 2, 4, 8, 16, 24, 32, 36, 38, 39, 40, 44, 56, 80]
    acc = {k: [] for k in KS}
    for _ in range(8):
        Sig, _ = true_sigma(rng, 5, 8, 0.7, 0.15)
        X = rng.normal(size=(120, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        w = hrp(S)
        G = rng.normal(size=(P, max(KS)))
        Q, _ = np.linalg.qr(G)                 # shared nested basis
        for k in KS:
            acc[k].append(fit_k(S, w, Q[:, :k]))
    print(f"n = {n}, so the weight vector has n - 1 = {n-1} free coordinates")
    print(f"{'k parameters':>14s}{'median L1 miss':>18s}")
    for k in KS:
        m = np.median(acc[k])
        flag = "   <-- n-1" if k == n - 1 else ""
        print(f"{k:14d}{m:18.2e}{flag}")
