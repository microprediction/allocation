"""Schur-complementary portfolio as an online estimator.

Your Schur Complementary allocation (the ``gamma`` cross-block coupling that
interpolates HRP at ``gamma=0`` and minimum variance as ``gamma->1``), but with
the asset order coming from **Fiedler seriation** instead of agglomerative
linkage. Because the Fiedler order is a smooth function of the covariance, the
whole allocation is smooth -- so ``partial_fit`` over a drifting covariance gives
low-turnover updates, the same streaming property the Thurstone estimator has.

The seriation sign is carried across updates so the order stays stable; the
similarity graph can be kNN-sparsified and blended with a prior (sectors /
fundamentals / LLM) for scale and cold-start placement.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOnlinePortfolio
from ._schur.coupling import compute_monotonic_weights, compute_weights
from ._schur.seriation import seriate

__all__ = ["SchurComplementary", "HierarchicalRiskParity"]


class SchurComplementary(BaseOnlinePortfolio):
    """Online Schur-complementary portfolio with Fiedler seriation.

    Parameters
    ----------
    gamma : float in [0, 1], default 0.5
        Cross-block coupling. 0 = HRP (block-diagonal); ->1 = minimum variance.
    knn : int or None, default None
        If set, sparsify the similarity graph to k nearest neighbours (for scale).
    prior : array (n, n) or None, default None
        Optional prior similarity blended into the seriation graph.
    prior_weight : float in [0, 1], default 0.0
        Weight on ``prior`` in the similarity blend.
    covariance_estimator : estimator or None, default None
        Online covariance (``partial_fit`` / ``covariance_``). None uses EWMA.
    halflife : float, default 60.0
        EWMA halflife when ``covariance_estimator is None``.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.5,
        keep_monotonic: bool = True,
        knn: int | None = None,
        prior=None,
        prior_weight: float = 0.0,
        covariance_estimator=None,
        halflife: float = 60.0,
    ):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.gamma = gamma
        self.keep_monotonic = keep_monotonic
        self.knn = knn
        self.prior = prior
        self.prior_weight = prior_weight
        self._fiedler = None
        self._order = None
        self.effective_gamma_ = None

    def _seriate(self, cov: np.ndarray):
        order, v = seriate(
            cov,
            previous=self._fiedler,
            knn=self.knn,
            prior=self.prior,
            prior_weight=self.prior_weight,
        )
        self._fiedler = v
        self._order = order
        return order

    def _allocate(self, order, cov) -> None:
        if self.keep_monotonic:
            self._weights, self.effective_gamma_ = compute_monotonic_weights(
                order, cov, self.gamma
            )
        else:
            self._weights = compute_weights(order, cov, self.gamma)
            self.effective_gamma_ = self.gamma

    def _cold_start(self, cov: np.ndarray) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        self._allocate(self._seriate(cov), cov)

    def _online_update(self, cov: np.ndarray) -> None:
        self._allocate(self._seriate(cov), cov)  # warm-started sign keeps order stable

    @property
    def order_(self) -> np.ndarray:
        return self._order

    @property
    def fiedler_(self) -> np.ndarray:
        return self._fiedler


class HierarchicalRiskParity(SchurComplementary):
    """Dynamic HRP: recursive-bisection risk parity over a *smooth* order.

    Classic HRP (Lopez de Prado, 2016) takes the asset order from agglomerative
    clustering, whose dendrogram reorders discontinuously as the covariance
    drifts -- a source of turnover. This estimator is exactly
    :class:`SchurComplementary` at ``gamma=0`` (no cross-block coupling, so the
    recursion is the plain inverse-variance recursive bisection of HRP), but with
    the smooth Fiedler seriation in place of the dendrogram, so ``partial_fit``
    gives low-turnover updates.

    Provided as a named estimator for recognisability and for benchmark tables;
    it is the ``gamma=0`` special case of the Schur construction. ``knn`` / a
    seriation ``prior`` are exposed as on the parent.
    """

    def __init__(
        self,
        *,
        knn: int | None = None,
        prior=None,
        prior_weight: float = 0.0,
        covariance_estimator=None,
        halflife: float = 60.0,
    ):
        super().__init__(
            gamma=0.0,
            keep_monotonic=False,
            knn=knn,
            prior=prior,
            prior_weight=prior_weight,
            covariance_estimator=covariance_estimator,
            halflife=halflife,
        )
