"""``SchurBridge``: the one estimator the others are corners of.

A partition of the assets, two damping dials, a companion vector and a budget
rule (see :mod:`allocation._schur.bridge`). The partition comes from the smooth
Fiedler seriation (a bisection tree, or the order cut into ``n_clusters``
contiguous blocks) or from a fixed label vector such as sectors. Everything
downstream of the partition is closed form and continuous in the covariance,
and at ``(gamma, eta) = (1, 1)`` the weights do not depend on the partition at
all, so reclustering churn is damped by the same dials that damp estimation
noise.

Corners, with the defaults ``conditioning='tree'``, ``fitness='held'``,
``outer='stack'``::

    SchurBridge(gamma=0, eta=0, n_clusters=k)      HERC on k clusters
    SchurBridge(gamma=0, eta=1, n_clusters=k)      cluster min-var, inverse-variance budgets
    SchurBridge(gamma=0, eta=1, n_clusters=k, outer='optimize')   NCO
    SchurBridge(gamma=g, eta=1, n_clusters=k, outer='optimize', conditioning='knots')   the NCO bridge
    SchurBridge(gamma=0, fitness='naive')          HRP over the Fiedler order
    SchurBridge(gamma=1, eta=1)                    minimum variance (any partition)
    SchurBridge(gamma=1, eta=1, companion='vol')   maximum diversification
    SchurBridge(gamma=1, eta=1, companion='mean')  tangency
    SchurBridge(n_clusters=1, eta=e)               inverse variance -> min-var along Stevens' path
"""

from __future__ import annotations

import math

import numpy as np

from .base import BaseOnlinePortfolio
from ._schur.bridge import bisection_tree, bridge_weights, tree_leaves
from ._schur.seriation import seriate

__all__ = ["SchurBridge"]


def _companion_vector(companion, cov, mean):
    if companion is None or (isinstance(companion, str) and companion in ("ones", "vol")):
        return companion
    if isinstance(companion, str) and companion == "mean":
        if mean is None:
            raise ValueError("companion='mean' needs a covariance estimator exposing mean_")
        return np.asarray(mean, dtype=float)
    return np.asarray(companion, dtype=float)


def _partition_from_labels(labels, conditioning):
    labels = np.asarray(labels)
    ids = [np.where(labels == g)[0] for g in np.unique(labels)]
    if conditioning == "tree":
        return _tree_over(ids)
    return ids


def _tree_over(leaves):
    if len(leaves) == 1:
        return np.asarray(leaves[0], dtype=int)
    mid = len(leaves) // 2
    return (_tree_over(leaves[:mid]), _tree_over(leaves[mid:]))


class SchurBridge(BaseOnlinePortfolio):
    """The Schur bridge on a smooth partition: HERC, HRP, NCO, min-variance,
    maximum diversification and tangency are all settings of this estimator.

    Parameters
    ----------
    gamma : float in [0, 1], default 0.5
        Conditioning of each cluster on the assets outside it.
    eta : float in [0, 1], default 1.0
        Conditioning of each asset on its cluster mates. ``0`` holds inverse
        variance inside a cluster, ``1`` the cluster's minimum-variance direction.
    n_clusters : int or None, default None
        Cut the Fiedler order into this many contiguous clusters. ``None``
        bisects down to single assets (the HRP / Schur tree).
    clusters : array (n,) of labels or None, default None
        A fixed partition (sectors, a prior clustering). Overrides seriation.
    conditioning : {'tree', 'flat', 'knots'}, default 'tree'
        How a cluster is conditioned on the outside: composed down the
        bisection tree, on every other asset, or on one factor-mimicking
        portfolio per other cluster (the scalable choice; exact under a block
        one-factor model of cross-cluster dependence).
    fitness : {'held', 'naive'}, default 'held'
        Budget rule between clusters. ``'held'`` (inverse variance of what the
        cluster holds, on its conditioned pair) is exact at ``gamma = 1``;
        ``'naive'`` is HRP's split, exact HRP at ``gamma = 0``.
    outer : {'stack', 'optimize'}, default 'stack'
        Stack the clusters' inverse-fitness vectors, or run NCO's outer
        minimum-variance step over the cluster directions.
    companion : {'ones', 'vol', 'mean'} or array (n,), default 'ones'
        The vector ``u`` of ``Sigma^{-1} u`` at full coupling.
    long_only : bool, default False
        Cap ``eta`` at each cluster's long-only frontier.
    ridge : float, default 0.0
        Regularization of every block solve, as in :class:`SchurComplementary`.
    knn, prior, prior_weight : seriation options, as in :class:`SchurComplementary`.
    covariance_estimator, halflife : see :class:`BaseOnlinePortfolio`.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.5,
        eta: float = 1.0,
        n_clusters: int | None = None,
        clusters=None,
        conditioning: str = "tree",
        fitness: str = "held",
        outer: str = "stack",
        companion="ones",
        long_only: bool = False,
        ridge: float = 0.0,
        knn: int | None = None,
        prior=None,
        prior_weight: float = 0.0,
        covariance_estimator=None,
        halflife: float = 60.0,
    ):
        super().__init__(covariance_estimator=covariance_estimator, halflife=halflife)
        self.gamma = gamma
        self.eta = eta
        self.n_clusters = n_clusters
        self.clusters = clusters
        self.conditioning = conditioning
        self.fitness = fitness
        self.outer = outer
        self.companion = companion
        self.long_only = long_only
        self.ridge = ridge
        self.knn = knn
        self.prior = prior
        self.prior_weight = prior_weight
        self._fiedler = None
        self._order = None
        self._clusters = None
        self.eta_effective_ = None

    # ------------------------------------------------------------ partition
    def _partition(self, cov: np.ndarray):
        n = cov.shape[0]
        if self.clusters is not None:
            self._order = np.arange(n)
            part = _partition_from_labels(self.clusters, self.conditioning)
        else:
            order, v = seriate(
                cov, previous=self._fiedler, knn=self.knn, prior=self.prior,
                prior_weight=self.prior_weight,
            )
            self._fiedler = v
            self._order = order
            leaf = 1 if self.n_clusters is None else int(math.ceil(n / max(1, int(self.n_clusters))))
            part = bisection_tree(order, leaf_size=leaf)
            if self.conditioning != "tree":
                part = tree_leaves(part)
        self._clusters = tree_leaves(part) if isinstance(part, tuple) else part
        return part

    def _allocate(self, cov: np.ndarray) -> None:
        mean = getattr(self._cov_estimator, "mean_", None)
        u = _companion_vector(self.companion, cov, mean)
        part = self._partition(cov)
        w, info = bridge_weights(
            cov, part, gamma=self.gamma, eta=self.eta, conditioning=self.conditioning,
            fitness=self.fitness, outer=self.outer, companion=u, ridge=self.ridge,
            long_only=self.long_only, return_info=True,
        )
        self._weights = w
        self.eta_effective_ = info["eta_effective"] or None

    def _cold_start(self, cov: np.ndarray) -> None:
        if not (0.0 <= self.gamma <= 1.0 and 0.0 <= self.eta <= 1.0):
            raise ValueError("gamma and eta must lie in [0, 1].")
        self._allocate(cov)

    def _online_update(self, cov: np.ndarray) -> None:
        self._allocate(cov)

    @property
    def order_(self) -> np.ndarray:
        return self._order

    @property
    def fiedler_(self) -> np.ndarray:
        return self._fiedler

    @property
    def clusters_(self) -> list:
        return self._clusters
