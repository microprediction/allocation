"""Fiedler (spectral) seriation -- a smooth ordering of assets.

Standard HRP/Schur take the asset order from agglomerative clustering, whose
dendrogram reorders discontinuously as the covariance drifts -- the source of
turnover. We replace it with the Fiedler vector (the eigenvector of the second
smallest Laplacian eigenvalue of the similarity graph): it varies *continuously*
with the covariance, so the order changes only when two assets' Fiedler
coordinates cross. Feeding this order into the Schur recursion makes the whole
allocation a smooth function of the covariance.

Scale: the similarity graph can be sparsified to k nearest neighbours so the
Fiedler vector is a single eigenpair of a sparse Laplacian (the dense ``eigh``
here is fine to a few thousand assets; an iterative, warm-started solver is the
drop-in for larger universes). The graph is a blend of realized correlation and
an optional prior (sectors / fundamentals / an LLM similarity) -- the seam that
lets a prior place cold-start names the realized correlation cannot.
"""

from __future__ import annotations

import numpy as np

from .._thurstone.covariance import cov_to_corr

__all__ = ["affinity", "fiedler_vector", "seriate"]


def affinity(
    cov: np.ndarray,
    knn: int | None = None,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> np.ndarray:
    """Similarity (affinity) graph in [0, 1], optionally kNN-sparsified.

    Realized similarity is ``(corr + 1) / 2``; blended with an optional ``prior``
    similarity by ``prior_weight``. With ``knn`` set, each row keeps only its k
    largest off-diagonal affinities (symmetrized).
    """
    A = 0.5 * (cov_to_corr(cov) + 1.0)
    if prior is not None and prior_weight > 0.0:
        A = (1.0 - prior_weight) * A + prior_weight * np.asarray(prior, dtype=float)
    np.fill_diagonal(A, 0.0)
    if knn is not None and knn < A.shape[0] - 1:
        n = A.shape[0]
        mask = np.zeros_like(A, dtype=bool)
        keep = np.argsort(-A, axis=1)[:, :knn]
        rows = np.repeat(np.arange(n), knn)
        mask[rows, keep.ravel()] = True
        mask |= mask.T  # symmetric kNN graph
        A = np.where(mask, A, 0.0)
    return A


def fiedler_vector(A: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    """Fiedler vector of the graph Laplacian ``L = D - A``.

    Sign is fixed by alignment with ``previous`` (the Fiedler vector is defined up
    to sign; without this the order would flip arbitrarily between updates). A
    dense symmetric eigendecomposition is used; only the 2nd-smallest eigenpair
    matters, so this is the place to swap in an iterative warm-started solver.
    """
    L = np.diag(A.sum(axis=1)) - A
    vals, vecs = np.linalg.eigh(L)
    v = vecs[:, 1]  # second smallest eigenvalue -> Fiedler vector
    if previous is not None and len(previous) == len(v):
        if float(v @ previous) < 0.0:
            v = -v
    return v


def seriate(
    cov: np.ndarray,
    previous: np.ndarray | None = None,
    knn: int | None = None,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(order, fiedler)``: the seriation order and its Fiedler vector.

    ``previous`` is last period's Fiedler vector, used to keep the sign (and hence
    the order) stable across streaming updates.
    """
    A = affinity(cov, knn=knn, prior=prior, prior_weight=prior_weight)
    v = fiedler_vector(A, previous=previous)
    order = np.argsort(v)
    return order, v
