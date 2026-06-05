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

import numpy as np

__all__ = ["compute_weights", "schur_augmentation"]


def _inverse_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a^-1 @ b via a solve (more accurate than forming the inverse)."""
    return np.linalg.solve(a, b)


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


def schur_augmentation(a: np.ndarray, b: np.ndarray, d: np.ndarray, gamma: float) -> np.ndarray:
    """Augment block ``a`` with the gamma-scaled Schur complement of ``d``."""
    n_a, n_d = a.shape[0], d.shape[0]
    if gamma == 0 or n_a == 1 or n_d == 1:
        return a
    a_aug = a - gamma * b @ _inverse_multiply(d, b.T)
    m = _symmetric_step_up_matrix(n1=n_a, n2=n_d)
    r = np.eye(n_a) - gamma * _multiply_by_inverse(b, d) @ m.T
    return _symmetrize(_inverse_multiply(r, a_aug))


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
    order: np.ndarray, covariance: np.ndarray, gamma: float, force_spd: bool = True
) -> np.ndarray:
    """Schur-complementary weights for a given seriation ``order`` and covariance.

    Weights are returned in the original asset order (not the seriation order),
    are long-only, and sum to one.
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
                a_aug = schur_augmentation(a, b, d, gamma=gamma)
                d_aug = schur_augmentation(d, b.T, a, gamma=gamma)
                if force_spd:
                    if not _is_spd(a_aug):
                        a_aug = _nearest_spd(a_aug)
                    if not _is_spd(d_aug):
                        d_aug = _nearest_spd(d_aug)
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
