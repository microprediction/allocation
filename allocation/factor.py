"""Factor-aware optimization: well-posed minimum-variance at any dimension.

When the covariance is a factor model ``Sigma = B B^T + diag(psi)`` (``B`` is
``n x k`` loadings, ``psi`` the idiosyncratic variances), its inverse is cheap and
well-conditioned via the **Woodbury identity**

    Sigma^{-1} = D^{-1} - D^{-1} B (I_k + B^T D^{-1} B)^{-1} B^T D^{-1},  D = diag(psi),

costing ``O(n k^2 + k^3)`` and never forming or inverting an ``n x n`` matrix. With
a strictly-positive idiosyncratic floor the condition number is bounded by
``lambda_max(Sigma) / floor``, so minimum-variance is **always well-posed** -- the
fix for the rank-deficient regime where dense ``Sigma^{-1}`` does not exist.

The factor model is read straight from a factor covariance estimator that exposes
``loadings_`` / ``idiosyncratic_`` (e.g. an adapted ``precise.FactorCovariance``,
``O(n k)`` end to end), or built from a dense covariance with a randomized
``factor_decompose`` when none is supplied (``O(n^2 k)`` -- still well-posed, just
bounded by the dense covariance).
"""

from __future__ import annotations

import numpy as np

from .base import BaseOnlinePortfolio
from ._thurstone.covariance import cov_to_corr, factor_decompose

__all__ = [
    "woodbury_solve",
    "covariance_to_factors",
    "factor_min_variance_weights",
    "factor_max_diversification_weights",
    "FactorMinimumVariance",
    "FactorMaximumDiversification",
]


def woodbury_solve(loadings, idiosyncratic, rhs) -> np.ndarray:
    """``(B B^T + diag(psi))^{-1} @ rhs`` via Woodbury -- ``O(n k^2 + k^3)``."""
    B = np.asarray(loadings, dtype=float)
    psi = np.asarray(idiosyncratic, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    d_inv = 1.0 / psi
    bt_dinv = B.T * d_inv  # (k, n)
    k = B.shape[1]
    cap = np.eye(k) + bt_dinv @ B  # (k, k) capacitance matrix
    z = np.linalg.solve(cap, bt_dinv @ rhs)
    return d_inv * rhs - d_inv * (B @ z)


def covariance_to_factors(cov: np.ndarray, k: int, floor: float = 1e-3):
    """Factor model ``(B, psi)`` of a covariance, with a positive idiosyncratic floor.

    Factor the *correlation* (unit diagonal) and rescale by volatilities, so
    ``B B^T + diag(psi) ~= Sigma``. ``psi`` is floored at ``floor * mean variance``
    so the implied covariance is strictly positive-definite (hence invertible).
    """
    cov = np.asarray(cov, dtype=float)
    s = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    Bc, dc = factor_decompose(cov_to_corr(cov), max(1, min(int(k), cov.shape[0])))
    B = s[:, None] * Bc
    psi = s**2 * dc
    return B, np.clip(psi, floor * float(np.mean(s**2)), None)


def _normalize_signed(w: np.ndarray) -> np.ndarray:
    s = float(w.sum())
    return w / s if abs(s) > 1e-12 else np.full(len(w), 1.0 / len(w))


def factor_min_variance_weights(loadings, idiosyncratic) -> np.ndarray:
    """Minimum-variance ``Sigma^{-1} 1`` for a factor covariance (sum one, signed)."""
    n = len(idiosyncratic)
    return _normalize_signed(woodbury_solve(loadings, idiosyncratic, np.ones(n)))


def factor_max_diversification_weights(loadings, idiosyncratic) -> np.ndarray:
    """Maximum-diversification ``Sigma^{-1} sigma`` for a factor covariance."""
    B = np.asarray(loadings, dtype=float)
    psi = np.asarray(idiosyncratic, dtype=float)
    sigma = np.sqrt(np.sum(B * B, axis=1) + psi)  # diag(Sigma) ** 0.5
    return _normalize_signed(woodbury_solve(B, psi, sigma))


class _FactorOptimizer(BaseOnlinePortfolio):
    """Base for factor-covariance optimizers; always well-posed (psi floored > 0)."""

    def __init__(self, *, factors: int = 5, floor: float = 1e-3,
                 covariance_estimator=None, halflife: float = 60.0):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.factors = factors
        self.floor = floor

    def _factor_model(self, cov: np.ndarray):
        # fast path: a factor covariance estimator exposes the model directly (O(n k))
        B = getattr(self._cov_estimator, "loadings_", None)
        psi = getattr(self._cov_estimator, "idiosyncratic_", None)
        if B is not None and psi is not None:
            B = np.asarray(B, dtype=float)
            psi = np.clip(np.asarray(psi, dtype=float), 1e-300, None)
            return B, psi
        return covariance_to_factors(cov, self.factors, self.floor)

    def _weights_from(self, B, psi) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _cold_start(self, cov: np.ndarray) -> None:
        self._weights = self._weights_from(*self._factor_model(cov))

    def _online_update(self, cov: np.ndarray) -> None:
        self._cold_start(cov)


class FactorMinimumVariance(_FactorOptimizer):
    """Minimum-variance via a factor covariance + Woodbury -- well-posed at any ``n``.

    Parameters
    ----------
    factors : int, default 5
        Number of factors ``k`` when a dense covariance must be factored.
    floor : float, default 1e-3
        Idiosyncratic-variance floor (fraction of mean variance) that bounds the
        condition number, so the inverse always exists. Plug a factor covariance
        (``loadings_`` / ``idiosyncratic_``) via ``covariance_estimator`` for the
        ``O(n k)`` path.
    """

    def _weights_from(self, B, psi) -> np.ndarray:
        return factor_min_variance_weights(B, psi)


class FactorMaximumDiversification(_FactorOptimizer):
    """Maximum-diversification via a factor covariance + Woodbury (well-posed)."""

    def _weights_from(self, B, psi) -> np.ndarray:
        return factor_max_diversification_weights(B, psi)
