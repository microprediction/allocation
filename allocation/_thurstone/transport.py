"""Common-seed transport: realize the correlated race with fixed seeds.

The portfolio weights are the win frequencies of ``N(ability, C_tilt)``. We
evaluate them by Monte Carlo with a *fixed* ensemble of standard-normal seeds and
the symmetric square root of the correlation, so that as the correlation drifts
the weights move smoothly (low turnover): see the smoothness theorem in the
accompanying paper.
"""

from __future__ import annotations

import numpy as np

from .covariance import cov_to_corr, nearest_correlation

__all__ = ["symmetric_sqrt", "blend_correlation", "transport_weights"]


def symmetric_sqrt(C: np.ndarray) -> np.ndarray:
    """Symmetric (PCA) square root of a correlation matrix.

    Smoother in ``C`` than a Cholesky factor (continuous except at eigenvalue
    crossings), which is what keeps the recoloured paths -- and hence the
    weights -- smooth as ``C`` drifts.
    """
    vals, vecs = np.linalg.eigh(C)
    return (vecs * np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T


def blend_correlation(C_calib: np.ndarray, cov: np.ndarray, phi: float) -> np.ndarray:
    """Tilt correlation ``C_tilt = nearest_corr((1-phi) C_calib + phi corr(cov))``.

    ``phi = 0`` recovers the reference (and so reproduces the target); ``phi = 1``
    uses the full estimated correlation.
    """
    Ct = (1.0 - phi) * np.asarray(C_calib, dtype=float) + phi * cov_to_corr(cov)
    return nearest_correlation(Ct)


def transport_weights(ability: np.ndarray, C_tilt: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Win frequencies of ``N(ability, C_tilt)`` using the fixed ``seeds``.

    ``seeds`` has shape ``(M, n)``; the same seeds across calls is what makes the
    weights move smoothly with ``C_tilt`` (turnover tracks correlation change,
    not sampling noise). The minimum performance wins.
    """
    a = np.asarray(ability, dtype=float)
    S = symmetric_sqrt(C_tilt)
    X = a + seeds @ S  # rows ~ N(a, C_tilt)
    winners = np.argmin(X, axis=1)
    counts = np.bincount(winners, minlength=len(a)).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else np.full(len(a), 1.0 / len(a))
