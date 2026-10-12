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
    "factor_decompose",
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


def factor_decompose(corr: np.ndarray, k: int, oversample: int = 5, seed: int = 0):
    """Top-``k`` factor approximation ``corr ~= B B^T + diag(d)`` (unit diagonal).

    Returns ``(B, d)`` with ``B`` of shape ``(n, k)`` and ``d`` the idiosyncratic
    residual variances, so ``B B^T + diag(d)`` has unit diagonal. Uses a
    block-Krylov range finder (a few ``corr @ Q`` matvecs, thin QRs, a tiny
    eigendecomposition) -- ``O(n^2 k)`` rather than the ``O(n^3)`` of a full eigh.
    The result feeds :func:`transport_weights_lowrank`.

    The starting block is built from each asset's own row of ``corr`` (row means
    of its elementwise powers), not drawn positionally, so relabelling the assets
    relabels the factors and leaves ``B B^T + diag(d)`` unchanged (issue #79): a
    positional Gaussian sketch is regenerated in the new order rather than
    permuted with the assets, and changed implied correlations by up to 0.8. The
    start is a smooth function of ``corr``, so the factors still vary smoothly
    with it. ``seed`` no longer affects the result and is kept for compatibility.

    A genuine factor covariance estimator would supply ``(B, d)`` directly in
    ``O(n k)`` and remove the ``O(n^2)`` floor here; this keeps the tilt usable to
    a few thousand names without one.
    """
    c = np.asarray(corr, dtype=float)
    n = c.shape[0]
    k = max(1, min(int(k), n))
    p = min(n, k + oversample)
    # permutation-equivariant start: row moments of corr
    cj = c.copy()
    start = [cj.mean(axis=1), np.abs(c).mean(axis=1)]
    for _ in range(3):
        cj = cj * c
        start.append(cj.mean(axis=1))
    q, _ = np.linalg.qr(np.column_stack(start))
    basis = q
    while basis.shape[1] < p:               # grow the Krylov block to p columns
        y = c @ q
        y -= basis @ (basis.T @ y)
        q, _ = np.linalg.qr(y)
        basis = np.hstack([basis, q])
    q, _ = np.linalg.qr(c @ c @ basis[:, :p])  # one power step sharpens the subspace
    small = q.T @ c @ q                     # (p, p)
    vals, vecs = np.linalg.eigh(small)
    order = np.argsort(vals)[::-1][:k]
    lam = np.clip(vals[order], 0.0, None)
    B = (q @ vecs[:, order]) * np.sqrt(lam)
    d = np.clip(1.0 - np.sum(B * B, axis=1), 1e-6, None)
    return B, d


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
