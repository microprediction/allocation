"""Closed-form convex allocators -- smooth, but signed.

Minimum-variance (``w propto Sigma^{-1} 1``) and maximum-diversification
(``w propto Sigma^{-1} sigma``) are *rational* functions of the covariance, so
they are already smooth in ``Sigma``: their turnover is exactly the turnover of
the covariance estimate feeding them. There is no clustering, no sampling, no
active set -- so unlike HRP/Thurstone there is nothing extra to smooth. Pair them
with a smooth online covariance (the default EWMA, a shrinkage estimator, or a
``precise`` skater) and the weights inherit that smoothness.

Two things still threaten smoothness and are handled here:

* **Conditioning.** ``Sigma^{-1}`` blows up as an eigenvalue approaches zero, so a
  small covariance change can swing the weights. The ``shrinkage`` parameter
  blends ``Sigma`` toward a scaled identity (``delta=1`` -> equal weight for
  min-variance), which both conditions the inverse and gives a smooth
  min-variance <-> equal-weight family.
* **The long-only constraint.** A hard long-only QP kinks whenever an asset hits
  the zero bound -- the turnover this package exists to avoid. So these
  estimators are deliberately *unconstrained* and may go short (weights sum to
  one but can be negative). The smooth long-only route to minimum variance is
  :class:`allocation.SchurComplementary` as ``gamma -> 1``.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOnlinePortfolio
from ._thurstone.covariance import cov_to_corr

__all__ = [
    "min_variance_weights",
    "max_diversification_weights",
    "max_decorrelation_weights",
    "mean_variance_weights",
    "is_singular",
    "MinimumVariance",
    "MaximumDiversification",
    "MaximumDecorrelation",
    "MeanVariance",
]


def _shrink(cov: np.ndarray, delta: float) -> np.ndarray:
    """Blend ``cov`` toward a scaled identity by ``delta`` in [0, 1]."""
    if delta <= 0.0:
        return cov
    d = np.diag(cov)
    mu = float(np.mean(d)) if d.size else 1.0
    n = cov.shape[0]
    return (1.0 - delta) * cov + delta * mu * np.eye(n)


def _solve(cov: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``cov^{-1} rhs`` via a solve, falling back to least squares if singular."""
    try:
        return np.linalg.solve(cov, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(cov, rhs, rcond=None)[0]


def is_singular(cov: np.ndarray, cond_tol: float = 1e10) -> bool:
    """Whether ``cov`` is rank-deficient or worse-conditioned than ``cond_tol``.

    The inversion-based allocators here are only well-posed on a full-rank,
    decently-conditioned covariance; with an EWMA covariance that fails whenever
    fewer observations than assets have been seen (rank <= #obs). When this is
    true ``Sigma^{-1}`` does not meaningfully exist and minimum-variance is
    ill-posed -- shrinkage, or a method that needs no full inverse
    (InverseVariance / HierarchicalRiskParity / SchurComplementary), is required.
    """
    ev = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    lo, hi = float(ev[0]), float(ev[-1])
    return lo <= 1e-14 * max(hi, 1e-30) or hi / max(lo, 1e-300) > cond_tol


def _guard(name: str, cov: np.ndarray, shrinkage: float, strict: bool) -> bool:
    """Set/return the singular flag for an inversion estimator; raise if ``strict``."""
    sing = is_singular(_shrink(np.asarray(cov, dtype=float), shrinkage))
    if strict and sing:
        raise np.linalg.LinAlgError(
            f"{name}: the covariance is rank-deficient / ill-conditioned, so this "
            "inversion-based allocation is undefined. Add shrinkage, or use a method "
            "that needs no full inverse (InverseVariance, HierarchicalRiskParity, "
            "SchurComplementary)."
        )
    return sing


def _normalize_signed(w: np.ndarray) -> np.ndarray:
    """Scale to sum one; fall back to equal weight if the sum vanishes."""
    s = float(w.sum())
    if abs(s) < 1e-12:
        return np.full(len(w), 1.0 / len(w))
    return w / s


def min_variance_weights(covariance: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """Unconstrained minimum-variance weights ``Sigma^{-1} 1`` (sum one, signed)."""
    cov = _shrink(np.asarray(covariance, dtype=float), shrinkage)
    return _normalize_signed(_solve(cov, np.ones(cov.shape[0])))


def max_diversification_weights(covariance: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """Unconstrained maximum-diversification weights ``Sigma^{-1} sigma`` (sum one, signed)."""
    cov = _shrink(np.asarray(covariance, dtype=float), shrinkage)
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return _normalize_signed(_solve(cov, sigma))


def max_decorrelation_weights(covariance: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """Maximum-decorrelation weights ``C^{-1} 1`` (min-variance on the *correlation*).

    Minimum variance ignoring individual volatilities -- the allocation that
    minimises average pairwise portfolio correlation. ``shrinkage`` blends the
    correlation toward the identity (``1.0`` -> equal weight).
    """
    C = _shrink(cov_to_corr(np.asarray(covariance, dtype=float)), shrinkage)
    return _normalize_signed(_solve(C, np.ones(C.shape[0])))


def mean_variance_weights(covariance: np.ndarray, mean, shrinkage: float = 0.0) -> np.ndarray:
    """Tangency / mean-variance weights ``Sigma^{-1} mu`` (sum one, signed).

    Smooth in *both* ``Sigma`` and ``mu``; the catch is statistical, not in the
    map -- the expected-return estimate ``mu`` is the noisy input, so smoothness
    of the weights is only as good as the smoothness of whatever supplies ``mu``.
    """
    cov = _shrink(np.asarray(covariance, dtype=float), shrinkage)
    return _normalize_signed(_solve(cov, np.asarray(mean, dtype=float)))


class MinimumVariance(BaseOnlinePortfolio):
    """Unconstrained minimum-variance portfolio (smooth, may short).

    Parameters
    ----------
    shrinkage : float in [0, 1], default 0.0
        Blend the covariance toward a scaled identity before inverting. Improves
        conditioning (hence smoothness); ``1.0`` recovers equal weight.
    strict : bool, default False
        If True, raise when the (shrunk) covariance is rank-deficient /
        ill-conditioned -- i.e. when minimum-variance is genuinely ill-posed --
        instead of returning a least-squares pseudo-solution. Either way the
        ``singular_`` attribute records whether that happened.
    covariance_estimator, halflife : see :class:`BaseOnlinePortfolio`.
    """

    def __init__(self, *, shrinkage: float = 0.0, strict: bool = False,
                 covariance_estimator=None, halflife: float = 60.0):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.shrinkage = shrinkage
        self.strict = strict
        self.singular_ = False

    def _cold_start(self, cov: np.ndarray) -> None:
        self.singular_ = _guard(type(self).__name__, cov, self.shrinkage, self.strict)
        self._weights = min_variance_weights(cov, self.shrinkage)

    def _online_update(self, cov: np.ndarray) -> None:
        self._cold_start(cov)


class MaximumDiversification(BaseOnlinePortfolio):
    """Unconstrained maximum-diversification portfolio (smooth, may short).

    Maximises the diversification ratio ``(w . sigma) / sqrt(w' Sigma w)``; the
    unconstrained optimum is ``w propto Sigma^{-1} sigma``. Parameters as in
    :class:`MinimumVariance` (including ``strict`` / ``singular_``).
    """

    def __init__(self, *, shrinkage: float = 0.0, strict: bool = False,
                 covariance_estimator=None, halflife: float = 60.0):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.shrinkage = shrinkage
        self.strict = strict
        self.singular_ = False

    def _cold_start(self, cov: np.ndarray) -> None:
        self.singular_ = _guard(type(self).__name__, cov, self.shrinkage, self.strict)
        self._weights = max_diversification_weights(cov, self.shrinkage)

    def _online_update(self, cov: np.ndarray) -> None:
        self._cold_start(cov)


class MaximumDecorrelation(BaseOnlinePortfolio):
    """Maximum-decorrelation portfolio ``C^{-1} 1`` (min-variance on the correlation).

    Minimum variance with individual volatilities stripped out. Parameters as in
    :class:`MinimumVariance` (including ``strict`` / ``singular_``).
    """

    def __init__(self, *, shrinkage: float = 0.0, strict: bool = False,
                 covariance_estimator=None, halflife: float = 60.0):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.shrinkage = shrinkage
        self.strict = strict
        self.singular_ = False

    def _cold_start(self, cov: np.ndarray) -> None:
        self.singular_ = _guard(type(self).__name__, cov, self.shrinkage, self.strict)
        self._weights = max_decorrelation_weights(cov, self.shrinkage)

    def _online_update(self, cov: np.ndarray) -> None:
        self._cold_start(cov)


class MeanVariance(BaseOnlinePortfolio):
    """Tangency / mean-variance portfolio ``Sigma^{-1} mu`` (smooth, may short).

    Parameters
    ----------
    expected_returns : array (n,) or None, default None
        The expected-return vector ``mu``. ``None`` uses the covariance
        estimator's running EWMA mean (a smooth path), so the estimator is
        self-contained; note the statistical quality of ``mu`` is on you.
    shrinkage : float in [0, 1], default 0.0
        Conditioning of the covariance before inversion (see
        :class:`MinimumVariance`).
    strict : bool, default False
        Raise on a rank-deficient / ill-conditioned covariance instead of
        pseudo-solving; ``singular_`` records it either way.
    covariance_estimator, halflife : see :class:`BaseOnlinePortfolio`.
    """

    def __init__(
        self, *, expected_returns=None, shrinkage: float = 0.0, strict: bool = False,
        covariance_estimator=None, halflife: float = 60.0
    ):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.expected_returns = expected_returns
        self.shrinkage = shrinkage
        self.strict = strict
        self.singular_ = False

    def _mu(self, n: int) -> np.ndarray:
        if self.expected_returns is not None:
            mu = np.asarray(self.expected_returns, dtype=float)
            if len(mu) != n:
                raise ValueError(f"expected_returns has length {len(mu)}, expected {n}")
            return mu
        mu = getattr(self._cov_estimator, "mean_", None)
        return np.asarray(mu, dtype=float) if mu is not None else np.ones(n)

    def _cold_start(self, cov: np.ndarray) -> None:
        self.singular_ = _guard(type(self).__name__, cov, self.shrinkage, self.strict)
        self._weights = mean_variance_weights(cov, self._mu(cov.shape[0]), self.shrinkage)

    def _online_update(self, cov: np.ndarray) -> None:
        self._cold_start(cov)
