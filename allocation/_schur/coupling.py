"""Schur-complementary allocation: the gamma cross-block coupling + recursive bisection.

Faithful port of the Schur Complementary algorithm (Peter Cotton, 2022; in skfolio
as ``SchurComplementary``, BSD-3, derived from ``precise``). Given a seriation
(asset order) and a covariance, recursive median bisection allocates between halves
in inverse proportion to each half's variance -- but each half's covariance is first
*augmented* by the Schur complement of the other half, scaled by ``gamma``:
``gamma = 0`` ignores the cross-block (Hierarchical Risk Parity); ``gamma -> 1`` tends
to the minimum-variance portfolio.

Only the *order* comes from clustering; everything here is a continuous function of
the covariance, so a smooth order (Fiedler seriation) yields smooth weights.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["compute_weights", "compute_monotonic_weights", "schur_augmentation"]


def _inverse_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a^-1 @ b via a solve (more accurate than forming the inverse)."""
    return np.linalg.solve(a, b)


def _ridge_solve(a: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    """``(a + ridge * scale * I)^-1 @ b`` -- a robust block solve.

    The Schur coupling inverts the *other* block, which is rank-deficient at the
    top of the recursion on a large universe. A ridge proportional to the block's
    own scale (``trace/n``) keeps the solve well-posed for any ``a`` and makes the
    coupling self-tempering: as a block becomes less estimable the ridge dominates,
    the cross-block term shrinks, and Schur degrades gracefully toward HRP.
    ``ridge=0`` is the exact solve (with a least-squares safety net).
    """
    if ridge > 0.0:
        n = a.shape[0]
        scale = float(np.trace(a)) / n if n else 1.0
        a = a + ridge * scale * np.eye(n)
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def _multiply_by_inverse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a @ b^-1."""
    return np.linalg.solve(b.T, a.T).T


def _symmetrize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _is_spd(a: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False


def _nearest_spd(a: np.ndarray) -> np.ndarray:
    a = _symmetrize(a)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, 1e-10, None)
    return _symmetrize((vecs * vals) @ vecs.T)


def _symmetric_step_up_matrix(n1: int, n2: int) -> np.ndarray:
    """M with ``M @ ones(n2) == ones(n1)`` (|n1 - n2| <= 1)."""
    if n1 == n2:
        return np.eye(n1)
    if n1 < n2:
        return _symmetric_step_up_matrix(n2, n1).T * n1 / n2
    m = np.zeros((n1, n2))
    j_row = np.ones((1, n2)) / n2
    e = np.eye(n2)
    for j in range(n1):
        mj = np.concatenate([e[:j], j_row, e[j:]], axis=0)
        m += mj / n1
    return m


def schur_augmentation(
    a: np.ndarray, b: np.ndarray, d: np.ndarray, gamma: float, ridge: float = 0.0
) -> np.ndarray:
    """Augment block ``a`` with the gamma-scaled Schur complement of ``d``.

    ``ridge > 0`` regularizes the (rank-deficient at scale) inverse of the other
    block ``d`` -- see :func:`_ridge_solve`.
    """
    n_a, n_d = a.shape[0], d.shape[0]
    if gamma == 0 or n_a == 1 or n_d == 1:
        return a
    db = _ridge_solve(d, b.T, ridge)  # d^-1 b^T (n_d, n_a); d symmetric so b d^-1 == db^T
    a_aug = a - gamma * b @ db
    m = _symmetric_step_up_matrix(n1=n_a, n2=n_d)
    r = np.eye(n_a) - gamma * db.T @ m.T
    return _symmetrize(_ridge_solve(r, a_aug, ridge))


def _bisection(items):
    for e in items:
        n = len(e)
        if n > 1:
            mid = n // 2
            yield e[:mid], e[mid:]


def _naive_portfolio_variance(cov: np.ndarray) -> float:
    w = 1.0 / np.diag(cov)
    w = w / w.sum()
    return float(w @ cov @ w.T)


def compute_weights(
    order: np.ndarray, covariance: np.ndarray, gamma: float, force_spd: bool = True,
    ridge: float = 0.0,
) -> np.ndarray:
    """Schur-complementary weights for a given seriation ``order`` and covariance.

    Weights are returned in the original asset order (not the seriation order),
    are long-only, and sum to one. ``ridge > 0`` regularizes the cross-block
    solves so ``gamma > 0`` stays well-posed when blocks are rank-deficient.
    """
    cov = np.array(covariance, dtype=float, copy=True)
    n = len(cov)
    weights = np.ones(n)
    items = [np.asarray(order, dtype=int)]
    while items:
        new_items = []
        for left, right in _bisection(items):
            new_items += [left, right]
            a = cov[np.ix_(left, left)]
            d = cov[np.ix_(right, right)]
            if len(left) <= 1 or len(right) <= 1:
                a_aug, d_aug = a, d
            else:
                b = cov[np.ix_(left, right)]
                a_aug = schur_augmentation(a, b, d, gamma=gamma, ridge=ridge)
                d_aug = schur_augmentation(d, b.T, a, gamma=gamma, ridge=ridge)
                if force_spd:
                    if not _is_spd(a_aug):
                        a_aug = _nearest_spd(a_aug)
                    if not _is_spd(d_aug):
                        d_aug = _nearest_spd(d_aug)
                elif not _is_spd(a_aug) or not _is_spd(d_aug):
                    return None  # signal infeasible gamma to the monotonic sweep
                cov[np.ix_(left, left)] = a_aug
                cov[np.ix_(right, right)] = d_aug
            lv = _naive_portfolio_variance(a_aug)
            rv = _naive_portfolio_variance(d_aug)
            alpha = 1.0 - lv / (lv + rv)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
        items = new_items
    s = weights.sum()
    return weights / s if s > 0 else np.full(n, 1.0 / n)


def _binary_search(objective, low_gamma, high_gamma, low_variance, tol=1e-4):
    """Locate the gamma turning point where portfolio variance stops decreasing."""
    max_iter = math.ceil(math.log2(max(high_gamma - low_gamma, tol) / tol) * 2 + 1)
    is_decreasing = False
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (low_gamma + high_gamma)
        variance, weights = objective(mid)
        variance_h = objective(mid - tol)[0]
        if variance <= low_variance and variance <= variance_h:
            is_decreasing = True
            low_gamma, low_variance, best = mid, variance, weights
        else:
            high_gamma = mid
        if is_decreasing and best is not None and (high_gamma - low_gamma) <= tol:
            return best, low_gamma
    raise RuntimeError("no permissible gamma with monotonically decreasing variance")


def compute_monotonic_weights(
    order: np.ndarray,
    covariance: np.ndarray,
    max_gamma: float,
    step: float = 0.1,
    tol: float = 1e-4,
    ridge: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Weights at the largest gamma (<= ``max_gamma``) for which variance still falls.

    Caps gamma at the turning point ``effective_gamma`` so that
    ``variance(Schur) <= variance(HRP)`` even for ill-conditioned covariances
    (skfolio's ``keep_monotonic=True`` behaviour). Returns ``(weights, effective_gamma)``.
    """
    if max_gamma == 0:
        return compute_weights(order, covariance, 0.0, force_spd=True, ridge=ridge), 0.0
    cov = np.asarray(covariance, dtype=float)

    def objective(x):
        w = compute_weights(order, cov, x, force_spd=False, ridge=ridge)
        risk = np.inf if w is None else float(w @ cov @ w.T)
        return risk, w

    n = int(np.ceil(max_gamma / step)) + 1
    gammas = np.linspace(0.0, max_gamma, n)
    variances = np.full(n, np.nan)
    variance, weights_0 = objective(gammas[0])
    variances[0] = variance
    weights = weights_0
    for i in range(1, n):
        variance, weights = objective(gammas[i])
        variances[i] = variance
        if variance >= variances[i - 1]:
            lo = gammas[0] if i == 1 else gammas[i - 2]
            lo_var = variances[0] if i == 1 else variances[i - 2]
            try:
                return _binary_search(objective, lo, gammas[i], lo_var, tol)
            except RuntimeError:
                return weights_0, 0.0
    # monotonically decreasing up to max_gamma
    if variance <= objective(max_gamma - tol)[0]:
        return weights, max_gamma
    try:
        return _binary_search(objective, gammas[-2], max_gamma, variances[-2], tol)
    except RuntimeError:
        return weights, max_gamma
