"""The baseline (variance-only) portfolio the ability tilt starts from."""

from __future__ import annotations

import numpy as np

__all__ = ["diagonal_portfolio"]


def diagonal_portfolio(cov: np.ndarray) -> np.ndarray:
    """Long-only inverse-variance weights.

    This is the minimum-variance portfolio when correlations are ignored: each
    asset is weighted by the reciprocal of its variance, normalised to sum to
    one. It is the starting point the tilt then ``polishes`` by reintroducing
    correlation through a Thurstonian race.
    """
    cov = np.asarray(cov, dtype=float)
    var = np.diag(cov).astype(float)
    inv = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 0.0)
    total = inv.sum()
    if total <= 0:
        # Degenerate covariance: fall back to equal weight.
        n = cov.shape[0]
        return np.full(n, 1.0 / n)
    return inv / total
