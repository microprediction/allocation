"""Every allocator under one interface, with the facts that make a comparison fair.

Each entry records whether it is long-only, because comparing a long-only
heuristic against an unconstrained optimizer measures the constraint as much as
the rule, and whether it comes from the package or is an outside baseline.

A method returns weights or raises. The protocol records failures as missing
rather than substituting equal weight, so a method that cannot fit at a sample
size is visible instead of being quietly flattered.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Method:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    long_only: bool
    source: str


def _pkg(cls_name, **kw):
    import allocation as A
    cls = getattr(A, cls_name)

    def run(X):
        return np.asarray(cls(**kw).fit(X).weights_, dtype=float)
    return run


def _ledoit_wolf_min_var(X):
    from sklearn.covariance import LedoitWolf
    return _min_var_long_only(LedoitWolf(assume_centered=False).fit(X).covariance_)


def _min_var_long_only(S):
    import cvxpy as cp
    m = S.shape[0]
    ev, V = np.linalg.eigh((S + S.T) / 2)
    P = V @ np.diag(np.maximum(ev, 1e-10)) @ V.T
    w = cp.Variable(m)
    cp.Problem(cp.Minimize(cp.quad_form(w, P)), [cp.sum(w) == 1, w >= 0]).solve(
        solver=cp.CLARABEL)
    if w.value is None:
        raise RuntimeError("long-only minimum variance did not solve")
    x = np.maximum(w.value, 0)
    return x / x.sum()


def _sample_min_var_long_only(X):
    return _min_var_long_only(np.cov(X, rowvar=False))


def _clustered_taper(phi=0.5, k=5):
    """Minimum variance on the sample covariance with cross-cluster entries
    scaled by phi, on clusters from the same linkage a hierarchical rule uses.
    The estimator-side counterpart of a coupling dial."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    def run(X):
        S = np.cov(X, rowvar=False)
        n = S.shape[0]
        d = np.sqrt(np.diag(S))
        R = np.clip(S / np.outer(d, d), -1, 1)
        D = np.sqrt(np.maximum(0.5 * (1 - R), 0)); np.fill_diagonal(D, 0)
        lab = fcluster(linkage(squareform(D, checks=False), method="single"),
                       t=min(k, n), criterion="maxclust")
        T = np.full((n, n), phi)
        for g in np.unique(lab):
            idx = np.where(lab == g)[0]
            T[np.ix_(idx, idx)] = 1.0
        np.fill_diagonal(T, 1.0)
        return _min_var_long_only(S * T)
    return run


def registry(include_slow=True):
    M = [
        Method("equal weight", _pkg("EqualWeight"), True, "package"),
        Method("inverse variance", _pkg("InverseVariance"), True, "package"),
        Method("risk parity", _pkg("RiskParity"), True, "package"),
        Method("HRP", _pkg("HierarchicalRiskParity"), True, "package"),
        Method("SchurComplementary", _pkg("SchurComplementary"), True, "package"),
        Method("SchurBridge g=0.5", _pkg("SchurBridge", gamma=0.5), False, "package"),
        Method("SchurBridge g=1", _pkg("SchurBridge", gamma=1.0), False, "package"),
        Method("minimum variance", _pkg("MinimumVariance"), False, "package"),
        Method("max diversification", _pkg("MaximumDiversification"), False, "package"),
        Method("max decorrelation", _pkg("MaximumDecorrelation"), False, "package"),
        Method("factor min variance", _pkg("FactorMinimumVariance"), False, "package"),
        Method("min-var long-only", _sample_min_var_long_only, True, "reference"),
        Method("min-var Ledoit-Wolf", _ledoit_wolf_min_var, True, "reference"),
        Method("clustered taper 0.5", _clustered_taper(0.5), True, "reference"),
    ]
    if include_slow:
        M.append(Method("Thurstone", _pkg("ThurstonePortfolio", n_paths=4096),
                        True, "package"))
    return M
