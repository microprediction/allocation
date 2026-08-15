"""k-factor Thurstone races: forward, calibration, and derivatives.

Upgrades the one-factor engine of :mod:`calibrate` to the general factor
model X_i = a_i + v_i . f + sqrt(D_i) eps_i with f ~ N(0, I_k), following
the kinetics factor-probit work (Cotton 2026, "Scalable Probit
Calibration"; algorithms originate in the `winning`/`thurstone` lattice
transform, SIAM J. Financial Mathematics 2021).

What is new relative to the old fixed-point calibrator:

* Log-domain lattice kernel throughout (log Phi directly; forming
  1 - Phi(z) by subtraction rounds to zero near z ~ 8.3).
* Damped Jacobi quasi-Newton inversion with analytic own-location slopes
  and an independent-inverse warm start (about 60x faster than the damped
  Picard fixed point at N=1000 in the kinetics benchmarks).
* Jacobian-vector products in O(QNL), both the continuum (weighted graph
  Laplacian) form and the exact frozen-grid form.
* Contrast-space factor fitting: choices depend on (a, Sigma) only through
  P a and P Sigma P with P = I - 11'/N, so factor rank must not be spent on
  the choice-irrelevant common component.

Why the old preconditioner worked (the observation recorded in
experiments/preconditioner.py that the choice-space Jacobian has condition
number near 1 even when cond(C) ~ 800): the Jacobian is minus a weighted
graph Laplacian whose edge weight between assets i and j is the
photo-finish density (the probability density of a tie for the win). Its
conditioning is governed by how evenly photo-finish mass spreads across
pairs, not by the spectrum of C. Well-diffused win probabilities give a
well-conditioned Laplacian regardless of the covariance's pathology.

Convention: minimum performance wins; smaller ability = stronger.
Optional acceleration: if the `fastrace` extension (kinetics rust/fastrace)
is importable, the forward and slope passes use it transparently.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

__all__ = [
    "winprobs_factor",
    "calibrate_factor",
    "jacobian_vector_product",
    "factor_model_contrast",
]

try:                                     # optional compiled kernels
    import fastrace as _fastrace
except ImportError:                      # pragma: no cover
    _fastrace = None

_PFLOOR = 1e-300


def _forward_and_slopes(a, V, D, F, W, points=1501):
    """Unnormalized min-wins win integrals and own-location slopes."""
    a = np.asarray(a, dtype=float)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    D = np.asarray(D, dtype=float)
    if _fastrace is not None:
        p, slope, total = _fastrace.forward_and_slopes(
            a, V, D, np.ascontiguousarray(F, dtype=float),
            np.ascontiguousarray(W, dtype=float), points)
        return p * total, slope, total
    sd = np.sqrt(D)
    N = len(a)
    M_all = a[None, :] + F @ V.T
    lo = M_all.min() - 8.0 * sd.max()
    hi = M_all.max() + 8.0 * sd.max()
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    p = np.zeros(N)
    slope = np.zeros(N)
    chunk = max(1, int(5e6 / (N * points)))
    for a0 in range(0, len(F), chunk):
        M = M_all[a0:a0 + chunk]
        Wc = W[a0:a0 + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        f = np.exp(-0.5 * z**2) / (sd[None, :, None] * np.sqrt(2.0 * np.pi))
        logS = log_ndtr(-z)
        rest = np.exp(np.clip(logS.sum(axis=1)[:, None, :] - logS, -745.0, 0.0))
        p += Wc @ (np.sum(f * rest, axis=2) * dx)
        slope += Wc @ (np.sum(z * f / sd[None, :, None] * rest, axis=2) * dx)
    return p, slope, p.sum()


def winprobs_factor(ability, V, D, F, W, points: int = 1501) -> np.ndarray:
    """Normalized min-wins winning probabilities under the k-factor model."""
    p, _, total = _forward_and_slopes(ability, V, D, F, W, points)
    return p / total


def calibrate_factor(target, V, D, F, W, n_iter: int = 50, tol: float = 1e-6,
                     return_info: bool = False):
    """Abilities reproducing `target` win probabilities: damped Jacobi
    quasi-Newton with independent-inverse warm start. Globally well-posed:
    strictly positive targets determine abilities uniquely up to a common
    shift (the photo-finish Laplacian theorem)."""
    p = np.asarray(target, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    p = p / p.sum()
    logp = np.log(p)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    D = np.asarray(D, dtype=float)
    sd = np.sqrt(D)
    N = len(p)
    floor = max(1e-9, 1e-4 / N)
    ident = p > floor
    if V.shape[1] >= 1 and np.any(V != 0.0):
        sd_tot2 = D + np.sum(V**2, axis=1)
        mu = calibrate_factor(p, np.zeros((N, 1)), sd_tot2,
                              np.zeros((1, 1)), np.ones(1),
                              n_iter=n_iter, tol=tol)
    else:
        mu = (logp - logp.mean()) / 2.0
    step_cap = 1.0 * np.sqrt(D + np.sum(V**2, axis=1))
    prev_res = np.inf
    damp = 1.0
    res = np.inf
    it = 0
    for it in range(n_iter):
        praw, slope, total = _forward_and_slopes(mu, V, D, F, W)
        phat = np.maximum(praw / total, _PFLOOR)
        resid = np.log(phat) - logp
        res = np.abs(resid[ident]).max() if np.any(ident) else np.abs(resid).max()
        if res < tol:
            break
        if res > prev_res * 1.2:
            damp = max(0.25, damp * 0.5)
        prev_res = res
        dlogp = (slope / total) / phat
        dlogp = np.minimum(dlogp, -1e-3 / (sd + 1e-9))
        mu = mu - np.clip(damp * resid / dlogp, -step_cap, step_cap)
        mu -= mu.mean()
    if return_info:
        return mu, {"iterations": it + 1, "residual": float(res),
                    "converged": bool(res < tol)}
    return mu


def jacobian_vector_product(a, V, D, F, W, h, points: int = 3001,
                            form: str = "ibp") -> np.ndarray:
    """(J h) for J = d p / d a of the min-wins race, O(QNL).

    form="ibp": continuum weighted-Laplacian derivative (symmetric, and
    minus it is PSD with the ones vector as null space). form="grid": exact
    derivative of the frozen-grid rectangle sum.
    """
    if _fastrace is not None:
        return _fastrace.jacobian_vector_product(
            np.asarray(a, dtype=float), np.atleast_2d(np.asarray(V, float)),
            np.asarray(D, dtype=float),
            np.ascontiguousarray(F, dtype=float),
            np.ascontiguousarray(W, dtype=float),
            np.asarray(h, dtype=float), points, form)
    a = np.asarray(a, dtype=float)
    h = np.asarray(h, dtype=float)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    D = np.asarray(D, dtype=float)
    sd = np.sqrt(D)
    N = len(a)
    M_all = a[None, :] + F @ V.T
    x = np.linspace(M_all.min() - 8 * sd.max(), M_all.max() + 8 * sd.max(),
                    points)
    dx = x[1] - x[0]
    out = np.zeros(N)
    log_norm = np.log(sd * np.sqrt(2 * np.pi))
    for c in range(len(F)):
        z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
        logS = log_ndtr(-z)
        logg = -0.5 * z**2 - log_norm[:, None]
        haz = np.exp(logg - logS)
        A = (h[:, None] * haz).sum(0)
        gR = np.exp(np.clip(logg + logS.sum(0)[None, :] - logS, -745.0, 700.0))
        if form == "grid":
            own = h[:, None] * (x[None, :] - M_all[c][:, None]) / D[:, None]
            integ = gR * (own + A[None, :] - h[:, None] * haz)
        else:
            integ = gR * (A[None, :] - h[:, None] * haz.sum(0)[None, :])
        out += W[c] * (integ.sum(1) * dx)
    return out


def factor_model_contrast(C, k, n_iter: int = 200, tol: float = 1e-10):
    """Factor fit in the choice-relevant quotient space: iterative
    principal-factor heuristic on P C P (P = I - 11'/N), D from the
    quotient fit (a common addition to D is choice-relevant; only the
    common factor direction is not), loadings centered and
    SVD-canonicalized."""
    C = np.asarray(C, dtype=float)
    n = len(C)
    P = np.eye(n) - np.ones((n, n)) / n
    CP = P @ C @ P
    D = np.full(n, 0.5 * float(np.mean(np.diag(CP))) + 1e-3)
    V = np.zeros((n, k))
    for _ in range(n_iter):
        lam, U = np.linalg.eigh(CP - np.diag(D))
        idx = np.argsort(lam)[::-1][:k]
        V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        D_new = np.clip(np.diag(CP) - np.sum(V**2, axis=1), 1e-3, None)
        if np.abs(D_new - D).max() < tol:
            D = D_new
            break
        D = D_new
    V = P @ V
    A, sv, _ = np.linalg.svd(V, full_matrices=False)
    V = A[:, :k] * sv[:k]
    for j in range(V.shape[1]):
        i0 = np.argmax(np.abs(V[:, j]))
        if V[i0, j] < 0:
            V[:, j] = -V[:, j]
    return V, D


def hermite_nodes(k: int, Q: int = 15, prune: float = 1e-7):
    """Pruned product Gauss-Hermite rule for E over N(0, I_k)."""
    x, w = np.polynomial.hermite_e.hermegauss(Q)
    w = w / np.sqrt(2.0 * np.pi)
    if k == 1:
        return x[:, None], w
    grids = np.meshgrid(*([x] * k), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for d in range(k):
        W *= w[np.searchsorted(x, F[:, d])]
    keep = W > prune * W.max()
    return F[keep], W[keep]
