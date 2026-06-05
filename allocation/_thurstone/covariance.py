"""Covariance / correlation utilities for the ability tilt.

Nothing here knows about Thurstone; these are the matrix-hygiene helpers the
calibration and tilt steps need: turn a covariance estimate into a valid
correlation matrix, scale correlation, extract one-factor (market) loadings, and
reconstruct the implied one-factor correlation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cov_to_corr",
    "scale_off_diagonal",
    "nearest_correlation",
    "market_betas",
    "one_factor_corr",
]


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix from a covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def scale_off_diagonal(corr: np.ndarray, phi: float) -> np.ndarray:
    """Scale every off-diagonal correlation by ``phi`` (0 -> identity)."""
    c = np.array(corr, dtype=float, copy=True)
    off = ~np.eye(c.shape[0], dtype=bool)
    c[off] *= phi
    return c


def nearest_correlation(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to the nearest valid correlation matrix.

    Eigenvalue clipping plus diagonal renormalisation: guarantees
    positive-definiteness and unit diagonal without requiring the input to be
    well-conditioned. Used before sampling.
    """
    c = np.asarray(corr, dtype=float)
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, eps, None)
    c2 = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(c2), 1e-300, None))
    c2 = c2 / np.outer(d, d)
    np.fill_diagonal(c2, 1.0)
    return np.clip(c2, -1.0, 1.0)


def market_betas(
    cov: np.ndarray, weights: np.ndarray | None = None, b_max: float = 0.99
) -> np.ndarray:
    """One-factor loadings: each asset's correlation with the market portfolio.

    The "market" is the ``weights``-weighted portfolio of the assets (the index
    we calibrate to, by default). The loading ``b_i = corr(asset_i, market)`` is
    the natural single-factor coefficient: under ``X_i = a_i + b_i Z + ...`` the
    common factor ``Z`` is the (standardised) market. Returns values in
    ``[-b_max, b_max]``.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    sig = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    cov_im = cov @ w  # covariance of each asset with the market portfolio
    var_m = float(w @ cov @ w)
    sig_m = np.sqrt(max(var_m, 1e-300))
    b = cov_im / (sig * sig_m)
    return np.clip(b, -b_max, b_max)


def one_factor_corr(betas: np.ndarray) -> np.ndarray:
    """Correlation implied by a single common factor with the given loadings.

    ``C_ij = b_i b_j`` off-diagonal, ``1`` on the diagonal. This is the reference
    correlation ``C_calib`` used by the one-factor calibration; the tilt then
    re-races under the *actual* correlation, so the deviation from the target is
    driven by how far the real correlation departs from this one-factor model.
    """
    b = np.asarray(betas, dtype=float)
    c = np.outer(b, b)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0)
