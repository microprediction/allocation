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
    if np.any(var < 0):
        raise ValueError("diagonal_portfolio: negative variance on the diagonal")
    n = var.size
    riskless = var == 0
    if riskless.any():
        # The exact minimum-variance limit. As an asset's variance goes to
        # zero its inverse-variance weight dominates, so the portfolio sits on
        # the riskless assets; with several, the limit depends on their
        # relative rates and the symmetric convention is an equal split. This
        # keeps the allocation continuous at zero (a 1e-12 variance already
        # takes essentially all the weight) instead of excluding the one
        # asset that minimises risk. Note the streaming inverse-variance path
        # uses a positive variance floor instead; that convention is separate.
        w = np.zeros(n)
        w[riskless] = 1.0 / riskless.sum()
        return w
    inv = 1.0 / var
    return inv / inv.sum()
