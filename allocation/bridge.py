"""``SchurBridge``: bridges named by their endpoints, one engine underneath.

Schur conditioning is the technique. A *bridge* is what it connects: a
heuristic at the near end and an optimizer at the far end, both exact. The
named constructors say which:

    SchurBridge.hrp_to_min_variance(gamma)
    SchurBridge.hmv_to_min_variance(gamma)
    SchurBridge.herc_to_min_variance(gamma, eta, n_clusters)
    SchurBridge.nco_to_min_variance(gamma, n_clusters)
    SchurBridge.inverse_variance_to_min_variance(eta)

and ``companion='vol'`` or ``'mean'`` on any of them lands on maximum
diversification or the tangency portfolio instead. ``endpoints_`` reports the
pair for whatever settings are in force.

The partition comes from the smooth Fiedler seriation (a bisection tree, or
the order cut into ``n_clusters`` contiguous blocks) or from a fixed label
vector such as sectors. Everything downstream of the partition is closed form
and continuous in the covariance, and at the far end the weights do not
depend on the partition at all, so reclustering churn is damped by the same
dials that damp estimation noise. See :mod:`allocation._schur.bridge` for the
split rules and paths, and the taxonomy page at schur.microprediction.org for
which published "Schur" is which.
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
    if conditioning == "siblings":
        return _tree_over(ids)
    return ids


def _tree_over(leaves):
    if len(leaves) == 1:
        return np.asarray(leaves[0], dtype=int)
    mid = len(leaves) // 2
    return (_tree_over(leaves[:mid]), _tree_over(leaves[mid:]))


class SchurBridge(BaseOnlinePortfolio):
    """A Schur bridge on a smooth partition. HERC, HRP, HMV, NCO, minimum
    variance, maximum diversification and tangency are all settings of it.

    Parameters
    ----------
    gamma : float in [0, 1], default 0.5
        Conditioning of each cluster on the assets outside it.
    eta : float in [0, 1], default 1.0
        Conditioning of each asset on its cluster mates. ``0`` holds inverse
        variance inside a cluster, ``1`` the cluster's minimum-variance direction.
    n_clusters : int or None, default None
        Cut the Fiedler order into this many contiguous clusters. ``None``
        bisects down to single assets (the HRP / HMV trees).
    clusters : array (n,) of labels or None, default None
        A fixed partition (sectors, a prior clustering). Overrides seriation.
    conditioning : {'siblings', 'all', 'factor'}, default 'siblings'
        The path: condition a cluster on its sibling block down the tree, on
        every other asset, or on one factor-mimicking portfolio per other
        cluster (the scalable choice; exact under a block one-factor model).
    split : {'dial', 'minvar', 'hrp'}, default 'dial'
        Budget rule between siblings and clusters. ``'dial'`` starts at HRP,
        ``'minvar'`` at hierarchical minimum variance; both are exact at the
        far end. ``'hrp'`` is the collapsed skfolio rule, not exact there.
    outer : {'stack', 'optimize'}, default 'stack'
        Stack the clusters' inverse-fitness vectors, or run NCO's outer
        minimum-variance step over the cluster directions.
    companion : {'ones', 'vol', 'mean'} or array (n,), default 'ones'
        The vector ``u`` of ``Sigma^{-1} u`` at the far end.
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
        conditioning: str = "siblings",
        split: str = "dial",
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
        self.split = split
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

    # ---------------------------------------------------- named by endpoints
    @classmethod
    def hrp_to_min_variance(cls, gamma: float = 0.5, **kw):
        """HRP at ``gamma = 0``, minimum variance at ``gamma = 1`` (both exact)."""
        return cls(gamma=gamma, eta=gamma, n_clusters=None, conditioning="siblings", split="dial", **kw)

    @classmethod
    def hmv_to_min_variance(cls, gamma: float = 0.5, **kw):
        """Hierarchical minimum variance (Cotton 2024) at ``gamma = 0``, minimum variance at ``1``."""
        return cls(gamma=gamma, eta=1.0, n_clusters=None, conditioning="siblings", split="minvar", **kw)

    @classmethod
    def herc_to_min_variance(cls, gamma: float = 0.5, eta: float = 0.5, n_clusters: int = 8, **kw):
        """HERC at ``(0, 0)``, minimum variance at ``(1, 1)``: the square.

        Exact at both ends with the default sibling conditioning (or ``'all'``);
        ``conditioning='factor'`` is cheaper and exact only under a block
        one-factor model of cross-cluster dependence."""
        kw.setdefault("conditioning", "siblings")
        return cls(gamma=gamma, eta=eta, n_clusters=n_clusters, outer="stack", **kw)

    @classmethod
    def nco_to_min_variance(cls, gamma: float = 0.5, n_clusters: int = 8, **kw):
        """NCO at ``gamma = 0``, minimum variance at ``gamma = 1``.

        Exact at both ends with the default sibling conditioning (or ``'all'``);
        ``conditioning='factor'`` is the scalable path, exact only under a block
        one-factor model."""
        kw.setdefault("conditioning", "siblings")
        return cls(gamma=gamma, eta=1.0, n_clusters=n_clusters, outer="optimize", **kw)

    @classmethod
    def inverse_variance_to_min_variance(cls, eta: float = 0.5, **kw):
        """Inverse variance at ``eta = 0``, minimum variance at ``eta = 1``, along Stevens' path."""
        return cls(gamma=0.0, eta=eta, n_clusters=1, conditioning="all", **kw)

    @property
    def endpoints_(self) -> tuple[str, str]:
        """``(near end, far end)`` of the bridge implied by the settings: what the
        dials at 0 and at 1 give, as method names. The far end is marked
        approximate when the split or the path does not reach it exactly."""
        far = {"ones": "minimum variance", "vol": "maximum diversification", "mean": "tangency"}.get(
            self.companion if isinstance(self.companion, str) else "array", "Sigma^{-1} u"
        )
        if self.split == "hrp" or self.conditioning == "factor":
            far += " (approximate)"
        if self.n_clusters == 1 and self.clusters is None:
            near = "inverse variance"
        elif self.outer == "optimize":
            near = "NCO"
        elif self.n_clusters is None and self.clusters is None:
            near = {"dial": "HRP", "minvar": "hierarchical minimum variance", "hrp": "HRP"}[self.split]
        else:
            near = "HERC"
        return near, far

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
            if self.conditioning != "siblings":
                part = tree_leaves(part)
        self._clusters = tree_leaves(part) if isinstance(part, tuple) else part
        return part

    def _allocate(self, cov: np.ndarray) -> None:
        mean = getattr(self._cov_estimator, "mean_", None)
        u = _companion_vector(self.companion, cov, mean)
        part = self._partition(cov)
        w, info = bridge_weights(
            cov, part, gamma=self.gamma, eta=self.eta, conditioning=self.conditioning,
            split=self.split, outer=self.outer, companion=u, ridge=self.ridge,
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
