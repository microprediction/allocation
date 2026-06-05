"""Standard baseline portfolios as online estimators -- kept smooth.

These are the textbook allocations every backtest is measured against. Each is a
:class:`BaseOnlinePortfolio` so it shares the streaming interface (``fit`` /
``partial_fit`` / ``weights_`` / ``score``) and the pluggable online covariance.

The point of putting them here is not novelty but *smoothness*: each is written
so that ``partial_fit`` over a drifting covariance moves weights only as much as
the covariance moved, the same low-turnover property the Thurstone and Schur
estimators have.

* ``EqualWeight`` -- ``1/n``; constant, hence trivially smooth.
* ``InverseVariance`` -- ``w propto 1/sigma_i^2``; a smooth function of the
  covariance diagonal (naive risk parity).
* ``RiskParity`` -- equal-risk-contribution (ERC). The ERC portfolio is the
  unique strictly-positive solution of a convex problem, so it is a smooth
  function of the covariance with no active-set kinks; we solve it by
  cyclical coordinate descent **warm-started from the previous weights**, which
  both accelerates convergence and tracks the smooth solution across updates.

Deliberately omitted: long-only *constrained* minimum-variance / mean-variance
via a generic QP. Those are smooth in the interior but kink whenever an asset
hits the zero bound -- exactly the turnover this package exists to avoid. The
smooth route to minimum variance here is ``SchurComplementary`` as ``gamma->1``.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOnlinePortfolio
from ._thurstone.diagonal import diagonal_portfolio

__all__ = ["risk_parity_weights", "EqualWeight", "InverseVariance", "RiskParity"]


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


def risk_parity_weights(
    covariance: np.ndarray,
    budgets=None,
    x0=None,
    n_iter: int = 500,
    tol: float = 1e-10,
):
    """Equal-risk-contribution (risk-parity) weights by coordinate descent.

    Solves the convex problem whose first-order condition is
    ``x_i (Sigma x)_i = b_i`` (risk budget ``b_i``), giving the closed-form
    coordinate update ``x_i = (-c_i + sqrt(c_i^2 + 4 sigma_ii b_i)) / (2 sigma_ii)``
    with ``c_i = sum_{j != i} sigma_ij x_j`` (Spinu 2013; Griveau-Billion et al.
    2013). The iterate stays strictly positive, so the solution is interior and a
    smooth function of the covariance.

    Parameters
    ----------
    covariance : array (n, n)
    budgets : array (n,) or None
        Risk budgets (normalised internally). None = equal budgets ``1/n``.
    x0 : array (n,) or None
        Warm start for the unnormalised iterate. Non-finite or non-positive
        entries are replaced by an inverse-volatility default, so a partial warm
        start (e.g. continuing names + one new name) is fine.

    Returns
    -------
    (weights, x) : the long-only simplex weights and the unnormalised iterate
        (the latter carries scale and is the warm start for the next update).
    """
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else _normalize(budgets)
    var = np.diag(cov).astype(float)
    vol = np.sqrt(np.where(var > 0, var, 1.0))
    default = 1.0 / vol

    if x0 is None:
        x = default.copy()
    else:
        x = np.asarray(x0, dtype=float).copy()
        bad = ~np.isfinite(x) | (x <= 0)
        x[bad] = default[bad]
    x = np.clip(x, 1e-12, None)

    diag = np.where(var > 0, var, 1e-12)
    for _ in range(n_iter):
        x_old = x.copy()
        for i in range(n):
            c_i = cov[i] @ x - diag[i] * x[i]  # sum_{j != i} sigma_ij x_j
            x[i] = (-c_i + np.sqrt(c_i * c_i + 4.0 * diag[i] * b[i])) / (2.0 * diag[i])
        if np.max(np.abs(x - x_old)) <= tol * (np.max(np.abs(x)) + 1e-12):
            break
    return _normalize(x), x


class EqualWeight(BaseOnlinePortfolio):
    """Equal-weight (``1/n``) portfolio. Constant, hence perfectly smooth."""

    def __init__(self, *, covariance_estimator=None, halflife: float = 60.0):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)

    def _cold_start(self, cov: np.ndarray) -> None:
        n = cov.shape[0]
        self._weights = np.full(n, 1.0 / n)

    def _online_update(self, cov: np.ndarray) -> None:
        self._cold_start(cov)


class InverseVariance(BaseOnlinePortfolio):
    """Inverse-variance (naive risk-parity) portfolio, ``w propto 1/sigma_i^2``.

    A smooth function of the covariance diagonal; the minimum-variance portfolio
    when correlations are ignored.
    """

    def __init__(self, *, covariance_estimator=None, halflife: float = 60.0):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)

    def _cold_start(self, cov: np.ndarray) -> None:
        self._weights = diagonal_portfolio(cov)

    def _online_update(self, cov: np.ndarray) -> None:
        self._weights = diagonal_portfolio(cov)


class RiskParity(BaseOnlinePortfolio):
    """Equal-risk-contribution (ERC) portfolio, warm-started for smoothness.

    Parameters
    ----------
    budgets : array (n,) or None, default None
        Risk budgets (None = equal). Held fixed across updates.
    covariance_estimator : estimator or None, default None
        Online covariance (``partial_fit`` / ``covariance_``). None uses EWMA.
    halflife : float, default 60.0
        EWMA halflife when ``covariance_estimator is None``.
    """

    def __init__(self, *, budgets=None, covariance_estimator=None, halflife: float = 60.0):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.budgets = budgets
        self._x = None  # unnormalised ERC iterate, carried as the warm start

    def _cold_start(self, cov: np.ndarray) -> None:
        self._weights, self._x = risk_parity_weights(cov, budgets=self.budgets)

    def _online_update(self, cov: np.ndarray) -> None:
        self._weights, self._x = risk_parity_weights(cov, budgets=self.budgets, x0=self._x)
