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


def _fiedler_dense(A: np.ndarray) -> np.ndarray:
    L = np.diag(A.sum(axis=1)) - A
    _, vecs = np.linalg.eigh(L)
    return vecs[:, 1]  # 2nd-smallest eigenvalue -> Fiedler vector


def _fiedler_sparse(A: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Fiedler vector via LOBPCG on a sparse Laplacian (only one eigenpair).

    We deflate the constant null vector (`Y = 1`, the eigenvalue-0 eigenvector of a
    connected Laplacian) and ask for the smallest remaining eigenpair -- the
    Fiedler vector. Warm-started from ``previous``. This is what scales: an
    `O(nnz)` matvec on a kNN-sparsified graph rather than a dense `O(n^3)` eigh.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = A.shape[0]
    Asp = sp.csr_matrix(A)
    deg = np.asarray(Asp.sum(axis=1)).ravel()
    L = sp.diags(deg) - Asp
    if previous is not None and len(previous) == n:
        X = previous.reshape(n, 1).copy()
    else:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 1))
    Y = np.ones((n, 1))  # deflate the constant (lambda=0) eigenvector
    _, vecs = spla.lobpcg(L, X, Y=Y, largest=False, maxiter=200, tol=1e-7)
    return np.asarray(vecs)[:, 0]


def fiedler_vector(
    A: np.ndarray, previous: np.ndarray | None = None, sparse: bool = False
) -> np.ndarray:
    """Fiedler vector of the graph Laplacian ``L = D - A``.

    With ``sparse=True`` an iterative single-eigenpair LOBPCG solve is used
    (scales to large kNN graphs); otherwise a dense ``eigh``. The sign is fixed by
    alignment with ``previous`` -- the Fiedler vector is defined up to sign, and
    without this the order would flip arbitrarily between streaming updates.
    """
    if sparse and A.shape[0] > 3:
        try:
            v = _fiedler_sparse(A, previous)
        except Exception:
            v = _fiedler_dense(A)
    else:
        v = _fiedler_dense(A)
    if previous is not None and len(previous) == len(v) and float(v @ previous) < 0.0:
        v = -v
    return v


def seriate(
    cov: np.ndarray,
    previous: np.ndarray | None = None,
    knn: int | None = None,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
    sparse: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(order, fiedler)``: the seriation order and its Fiedler vector.

    ``previous`` is last period's Fiedler vector, used to keep the sign (and hence
    the order) stable across streaming updates. ``sparse`` selects the iterative
    LOBPCG solver; if ``None`` it defaults to sparse whenever ``knn`` is set.
    """
    A = affinity(cov, knn=knn, prior=prior, prior_weight=prior_weight)
    if sparse is None:
        sparse = knn is not None
    v = fiedler_vector(A, previous=previous, sparse=sparse)
    order = np.argsort(v)
    return order, v
