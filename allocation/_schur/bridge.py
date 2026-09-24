"""Schur bridges in pair form: one engine, named by endpoints.

Schur conditioning is a technique. What distinguishes the objects built with it
is where they start, where they land, and the path between. Take a partition
of the assets into clusters. Condition each cluster on the assets outside it
(damped by ``gamma``), condition each asset on its cluster mates (damped by
``eta``), and budget each cluster by the inverse of its variance. Block
inversion says this lands on the global direction ``Sigma^{-1} u`` exactly at
``(gamma, eta) = (1, 1)`` for *any* partition, and the near ends are the
familiar heuristics:

===================================  ====================================
near end (dials at 0)                far end (dials at 1)
===================================  ====================================
HRP (bisection tree, split='dial')   minimum variance
HMV (bisection tree, split='minvar') minimum variance
HERC (clusters, both dials)          minimum variance
NCO (clusters, outer='optimize')     minimum variance
inverse variance (one cluster)       minimum variance, along Stevens' path
any of the above with u = sigma, mu  maximum diversification, tangency
===================================  ====================================

Only a rule that is exact at *both* ends deserves the plain name of a bridge.
The split rule decides the near end on a tree:

* ``'dial'``   -- the variance of the child's ``eta``-damped naive portfolio on
  its conditioned pair. HRP exactly at the near end, exact at the far end.
  The default.
* ``'minvar'`` -- the child pair's minimum variance ``1 / (b^T Q^{-1} b)``.
  Hierarchical minimum variance (Cotton 2024) at the near end, exact at the
  far end.
* ``'hrp'``    -- HRP's naive split, the variance of the inverse-variance
  portfolio of the child block. HRP at the near end, *not* exact at the far
  end. This is the rule of the collapsed skfolio encoding; kept for comparison.

On a flat partition the same rule budgets the clusters: ``'dial'`` is HERC's
rule, the variance of what the cluster holds, and ``'minvar'`` budgets by the
cluster's minimum variance; they agree at ``eta = 1``.

The objects are *pairs* ``(Q, b)``: an SPD block and its companion vector.
Conditioning a subset ``I`` on ``J`` inside a pair, damped by ``g``::

    Q_I <- Q_II - g Q_IJ Q_JJ^{-1} Q_JI,    b_I <- b_I - g Q_IJ Q_JJ^{-1} b_J

Everything is closed form and continuous in the covariance for a fixed
partition; a change of partition, at a Fiedler coordinate crossing, moves the weights by
a finite amount that vanishes at ``(1, 1)``.
At ``(1, 1)`` the weights do not depend on the partition at all, so the cost of
a membership change shrinks as the dials approach the far end.

The path is the conditioning set of the outer dial:

* ``'siblings'`` -- composed down a bisection tree, each block on its sibling
  (exact at ``gamma = 1`` because Schur complements compose; a solve of half
  size at the root);
* ``'all'``      -- each cluster on every other asset (exact; one solve of
  size ``n - |C|`` per cluster);
* ``'factor'``   -- each cluster on one factor-mimicking portfolio per other
  cluster (exact under a block one-factor model of cross-cluster dependence;
  cluster-sized solves and one ``k x k`` solve; the scalable, smooth choice).

References: Cotton (2024) arXiv:2411.05807; the NCO bridge note (SSRN 7480738);
the HERC square note; the taxonomy page at schur.microprediction.org.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "condition_pair",
    "leaf_direction",
    "leaf_frontier",
    "bisection_tree",
    "contiguous_partition",
    "one_factor",
    "knot_portfolios",
    "bridge_weights",
]


# ----------------------------------------------------------------- pairs
def _ridge_solve(a: np.ndarray, rhs: np.ndarray, ridge: float) -> np.ndarray:
    """``(a + ridge * scale * I)^{-1} rhs``; least squares if singular."""
    if ridge > 0.0:
        n = a.shape[0]
        scale = float(np.trace(a)) / n if n else 1.0
        a = a + ridge * scale * np.eye(n)
    try:
        return np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, rhs, rcond=None)[0]


def condition_pair(
    Q: np.ndarray, b: np.ndarray, I, J, g: float, ridge: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Pair of the subset ``I`` conditioned on ``J`` inside ``(Q, b)``, damped by ``g``."""
    I = np.asarray(I, dtype=int)
    J = np.asarray(J, dtype=int)
    A = Q[np.ix_(I, I)]
    bI = b[I]
    if g == 0.0 or len(J) == 0:
        return A.copy(), bI.copy()
    B = Q[np.ix_(I, J)]
    D = Q[np.ix_(J, J)]
    # X = D^{-1} [B^T, b_J]  in one solve
    rhs = np.column_stack([B.T, b[J]])
    X = _ridge_solve(D, rhs, ridge)
    return A - g * B @ X[:, :-1], bI - g * B @ X[:, -1]


# ------------------------------------------------------------------ leaves
def _leaf_quantities(Q: np.ndarray, b: np.ndarray, ridge: float = 0.0):
    """``(q, s)``: residual variances and companion residuals of each asset on
    its leaf mates, from one inverse of the leaf block (Stevens' quantities)."""
    m = Q.shape[0]
    if m == 1:
        return np.array([Q[0, 0]]), np.array([b[0]])
    if ridge > 0.0:
        Q = Q + ridge * (float(np.trace(Q)) / m) * np.eye(m)
    try:
        Qi = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        Qi = np.linalg.pinv(Q)
    d = np.diag(Qi)
    d = np.where(d > 0, d, np.inf)
    q = 1.0 / d
    s = (Qi @ b) / d
    return q, s


def leaf_direction(Q: np.ndarray, b: np.ndarray, eta: float, ridge: float = 0.0) -> np.ndarray:
    """Unnormalized leaf weights ``z_i = [(1-eta) b_i + eta s_i] / [(1-eta) Q_ii + eta q_i]``.

    ``eta = 0`` is ``b_i / Q_ii`` (inverse variance with the companion in the
    numerator); ``eta = 1`` is ``(Q^{-1} b)_i`` (the leaf's minimum-variance
    direction). Both numerator and denominator are affine in ``eta``.
    """
    b = np.asarray(b, dtype=float)
    if eta == 0.0:
        num, den = b, np.diag(Q).astype(float)
    else:
        q, s = _leaf_quantities(Q, b, ridge)
        num = (1.0 - eta) * b + eta * s
        den = (1.0 - eta) * np.diag(Q) + eta * q
    # A zero conditioned variance makes the minimum-variance direction
    # unbounded: the division returns inf, then nan once normalised. Two
    # cases. No variance information in the leaf at all (the cold start,
    # Q = 0): the direction is the prior b, at every eta -- at eta = 1 the
    # numerator Q^{-1} b is itself zero there, so the denominator alone
    # cannot be guarded. Zero-variance assets among positive ones: floor
    # them at the smallest positive conditioned variance in the leaf, so
    # they take the largest finite weight, the positive-floor convention
    # the streaming path already uses.
    pos = den > 0
    if not pos.any():
        return b.copy()
    return num / np.where(pos, den, den[pos].min())


def leaf_frontier(Q: np.ndarray, b: np.ndarray, ridge: float = 0.0) -> float:
    """Largest ``eta`` keeping every leaf weight nonnegative (``b > 0`` assumed).

    ``eta_+ = min_{i: s_i < 0} b_i / (b_i - s_i)``, and ``1`` when no ``s_i`` is
    negative. Beyond it the asset whose hedge sum first exceeds its companion
    entry goes short.
    """
    _, s = _leaf_quantities(Q, b, ridge)
    neg = s < 0
    if not np.any(neg):
        return 1.0
    return float(np.min(b[neg] / (b[neg] - s[neg])))


# --------------------------------------------------------------- partitions
def bisection_tree(order, leaf_size: int = 1):
    """Balanced bisection of ``order`` into a nested tuple whose leaves are
    contiguous index arrays of size at most ``leaf_size``."""
    order = np.asarray(order, dtype=int)
    if len(order) <= max(1, leaf_size):
        return order
    mid = len(order) // 2
    return (bisection_tree(order[:mid], leaf_size), bisection_tree(order[mid:], leaf_size))


def tree_leaves(tree) -> list:
    if isinstance(tree, np.ndarray):
        return [tree]
    return tree_leaves(tree[0]) + tree_leaves(tree[1])


def contiguous_partition(order, n_clusters: int) -> list:
    """Cut a seriation ``order`` into ``n_clusters`` contiguous blocks of near-equal size.

    Membership changes only when two assets cross in the order, the same events
    at which the bisection tree reorders.
    """
    order = np.asarray(order, dtype=int)
    k = max(1, min(int(n_clusters), len(order)))
    return [np.asarray(c, dtype=int) for c in np.array_split(order, k)]


# -------------------------------------------------------------------- knots
def one_factor(S: np.ndarray, n_iter: int = 100, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Principal-axis fit ``S ~= beta beta^T + diag(psi)`` of a cluster block.

    Alternates the leading eigenvector of ``S - diag(psi)`` with
    ``psi = diag(S) - beta^2``. When the block really is one factor plus
    idiosyncratic noise this converges to the exact ``beta``; otherwise it is
    the usual one-factor approximation. Continuous in ``S`` away from
    eigenvalue crossings.
    """
    S = 0.5 * (S + S.T)
    n = S.shape[0]
    d = np.diag(S)
    psi = np.zeros(n)
    beta = np.zeros(n)
    floor = 1e-10 * float(d.max()) if n else 0.0
    for _ in range(n_iter):
        vals, vecs = np.linalg.eigh(S - np.diag(psi))
        beta = vecs[:, -1] * np.sqrt(max(float(vals[-1]), 0.0))
        psi_new = np.maximum(d - beta * beta, floor)
        if np.max(np.abs(psi_new - psi)) <= tol * max(float(d.max()), 1e-300):
            psi = psi_new
            break
        psi = psi_new
    return beta, psi


def knot_portfolios(cov: np.ndarray, clusters, ridge: float = 0.0) -> list:
    """One factor-mimicking portfolio per cluster, ``a_C = Sigma_CC^{-1} beta_C``,
    with ``beta_C`` from a one-factor fit of the cluster block.

    If cross-cluster dependence passes through one latent factor per cluster,
    so that each cross block ``Sigma_{C C'}`` is rank one in the direction of
    the cluster factors, conditioning a cluster on the other clusters' ``a_C``
    portfolios equals conditioning on all their assets: the mimicking
    portfolio is the sufficient statistic. With a real knot asset ``p`` it is
    ``e_p`` exactly. The sign of ``a_C`` is irrelevant to conditioning.
    """
    out = []
    for C in clusters:
        S = cov[np.ix_(C, C)]
        if len(C) == 1:
            out.append(np.array([1.0]))
            continue
        beta, _ = one_factor(S)
        out.append(_ridge_solve(S, beta, ridge))
    return out


def _knot_pairs(cov, u, clusters, gamma, ridge):
    """Pairs of every cluster conditioned on the other clusters' knot portfolios."""
    n = cov.shape[0]
    k = len(clusters)
    a = knot_portfolios(cov, clusters, ridge)
    A = np.zeros((n, k))                      # block-diagonal matrix of knot portfolios
    for j, Cj in enumerate(clusters):
        A[Cj, j] = a[j]
    X = cov @ A                               # (n, k): Cov(asset, knot portfolio)
    K = A.T @ X                               # (k, k): covariance of the knot portfolios
    u_p = A.T @ u                             # companion entries of the knot portfolios
    pairs = []
    for i, Ci in enumerate(clusters):
        Aii = cov[np.ix_(Ci, Ci)]
        bI = u[Ci]
        if gamma == 0.0 or k == 1:
            pairs.append((Aii.copy(), bI.copy()))
            continue
        others = np.array([j for j in range(k) if j != i])
        B = X[np.ix_(Ci, others)]
        D = K[np.ix_(others, others)]
        Y = _ridge_solve(D, np.column_stack([B.T, u_p[others]]), ridge)
        pairs.append((Aii - gamma * B @ Y[:, :-1], bI - gamma * B @ Y[:, -1]))
    return pairs


# ------------------------------------------------------------------ engine
def _naive_variance(Q: np.ndarray) -> float:
    w = 1.0 / np.diag(Q)
    w = w / w.sum()
    return float(w @ Q @ w)


def _leaf_vector(Q, b, eta, ridge, long_only, info):
    """Unnormalized vector of a leaf pair, with the long-only frontier applied."""
    e = eta
    if long_only and eta > 0.0 and Q.shape[0] > 1:
        e = min(eta, leaf_frontier(Q, b, ridge))
        info["eta_effective"].append(e)
    return leaf_direction(Q, b, e, ridge)


def _rescale(z, Q, b, split, eta, ridge=0.0):
    """Scale a child's unnormalized vector so that stacking children is the
    inverse-fitness budget rule ``w / nu`` with ``b^T w = 1``.

    ``'dial'``   : ``nu`` = variance of the ``eta``-damped naive portfolio of the
                   child pair (HRP at the near end, exact at the far end).
    ``'minvar'`` : ``nu = 1 / (b^T Q^{-1} b)`` (hierarchical minimum variance
                   at the near end, exact at the far end).
    ``'hrp'``    : ``nu`` = variance of the inverse-variance portfolio of the
                   child block (HRP at the near end, not exact at the far end).
    The scaling is invariant to the scale of ``z`` and has no singularity when
    ``b^T z`` crosses zero, except for ``'hrp'`` which normalizes by cash.
    """
    zQz = float(z @ Q @ z)
    if zQz <= 0.0:
        return np.zeros_like(z)
    if split == "dial":
        v = leaf_direction(Q, b, eta, ridge)
        bv = float(b @ v)
        vQv = float(v @ Q @ v)
        if vQv <= 0.0 or bv == 0.0:
            return np.zeros_like(z)
        nu = vQv / (bv * bv)
        bz = float(b @ z)
        return z * (1.0 / (bz * nu)) if bz != 0.0 else np.zeros_like(z)
    if split == "minvar":
        q = _ridge_solve(Q, b, ridge)
        nu = 1.0 / float(b @ q)
        bz = float(b @ z)
        return z * (1.0 / (bz * nu)) if bz != 0.0 else np.zeros_like(z)
    if split == "hrp":
        s = float(z.sum())
        if abs(s) < 1e-300:
            return np.zeros_like(z)
        return (z / s) / _naive_variance(Q)
    raise ValueError("split must be 'dial', 'minvar' or 'hrp'")


def _recurse_tree(Q, b, tree, gamma, eta, ridge, split, long_only, info):
    """Unnormalized vector on the local indices of the pair, conditioning
    composed down ``tree`` (leaves are index arrays into the *local* pair)."""
    if isinstance(tree, np.ndarray):
        return _leaf_vector(Q, b, eta, ridge, long_only, info)
    L, R = tree
    nL = sum(len(t) for t in tree_leaves(L))
    n = Q.shape[0]
    I = np.arange(nL)
    J = np.arange(nL, n)
    QL, bL = condition_pair(Q, b, I, J, gamma, ridge)
    QR, bR = condition_pair(Q, b, J, I, gamma, ridge)
    zL = _recurse_tree(QL, bL, _relabel(L, 0), gamma, eta, ridge, split, long_only, info)
    zR = _recurse_tree(QR, bR, _relabel(R, nL), gamma, eta, ridge, split, long_only, info)
    return np.concatenate([_rescale(zL, QL, bL, split, eta, ridge), _rescale(zR, QR, bR, split, eta, ridge)])


def _relabel(tree, offset):
    """Shift a subtree's leaf indices so they index the child's local pair."""
    if isinstance(tree, np.ndarray):
        return tree - offset
    return (_relabel(tree[0], offset), _relabel(tree[1], offset))


def _localize(tree):
    """Replace a tree over asset indices by one over positions 0..n-1 in leaf
    order, returning ``(local_tree, order)``."""
    leaves = tree_leaves(tree)
    order = np.concatenate(leaves)
    pos = {int(a): i for i, a in enumerate(order)}

    def rec(t):
        if isinstance(t, np.ndarray):
            return np.array([pos[int(a)] for a in t], dtype=int)
        return (rec(t[0]), rec(t[1]))

    return rec(tree), order


def bridge_weights(
    covariance: np.ndarray,
    partition,
    *,
    gamma: float = 1.0,
    eta: float = 1.0,
    conditioning: str = "siblings",
    split: str = "dial",
    outer: str = "stack",
    companion=None,
    ridge: float = 0.0,
    long_only: bool = False,
    return_info: bool = False,
):
    """Weights of the Schur bridge at ``(gamma, eta)`` on a partition.

    Parameters
    ----------
    covariance : (n, n) array
    partition : list of index arrays, or a nested tuple (bisection tree) whose
        leaves are index arrays. A tree is used by ``conditioning='siblings'``
        (one is built over the clusters if a list is given); ``'all'`` and
        ``'factor'`` use only the leaves.
    gamma : float in [0, 1]
        Damping of each cluster's conditioning on the outside.
    eta : float in [0, 1]
        Damping of each asset's conditioning on its cluster mates.
    conditioning : {'siblings', 'all', 'factor'}
        The path: the conditioning set of the outer dial (see the module docstring).
    split : {'dial', 'minvar', 'hrp'}
        Budget rule between siblings on a tree and between clusters on a flat
        partition.
        ``'dial'`` and ``'minvar'`` are exact at the far end; ``'hrp'`` is not.
    outer : {'stack', 'optimize'}
        Combine clusters by stacking their inverse-fitness vectors (HERC / the
        tree recursion) or by a minimum-variance optimizer over the cluster
        directions (NCO's outer step; ignores ``split``).
    companion : None, 'ones', 'vol', or (n,) array
        The vector ``u``: ``None``/'ones' gives ``Sigma^{-1} 1`` at full
        coupling, ``'vol'`` gives ``Sigma^{-1} sigma`` (maximum diversification),
        an array gives ``Sigma^{-1} u`` (``u = mu`` is the tangency direction).
    ridge : float
        Regularizes every solve (see :mod:`allocation._schur.coupling`).
    long_only : bool
        Cap ``eta`` at each leaf's long-only frontier. Weights are then
        nonnegative whenever every companion entry is (always at ``gamma = 0``).
    return_info : bool
        Also return a dict with ``eta_effective`` per leaf and the clusters.
    """
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    if companion is None or (isinstance(companion, str) and companion == "ones"):
        u = np.ones(n)
    elif isinstance(companion, str) and companion == "vol":
        u = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        u = np.asarray(companion, dtype=float)
    if not (0.0 <= gamma <= 1.0 and 0.0 <= eta <= 1.0):
        raise ValueError("gamma and eta must lie in [0, 1].")
    info = {"eta_effective": []}

    is_tree = isinstance(partition, tuple)
    clusters = tree_leaves(partition) if is_tree else [np.asarray(c, dtype=int) for c in partition]
    info["clusters"] = clusters

    if conditioning == "siblings":
        if not is_tree:
            partition = bisection_tree(np.concatenate(clusters), leaf_size=1) if len(clusters) == n \
                else _balanced_tree_over(clusters)
        local, order = _localize(partition)
        Q = cov[np.ix_(order, order)]
        b = u[order]
        if outer == "optimize":
            pairs = _tree_pairs(Q, b, local, gamma, ridge)
            D = []
            for leaf, (QC, bC) in pairs:
                zc = _leaf_vector(QC, bC, eta, ridge, long_only, info)
                col = np.zeros(n)
                col[order[leaf]] = zc
                D.append(col)
            w = _outer_optimize(cov, u, D)
        else:
            zl = _recurse_tree(Q, b, local, gamma, eta, ridge, split, long_only, info)
            w = np.zeros(n)
            w[order] = zl
    elif conditioning in ("all", "factor"):
        if conditioning == "all":
            pairs = []
            for C in clusters:
                J = np.setdiff1d(np.arange(n), C)
                pairs.append(condition_pair(cov, u, C, J, gamma, ridge))
        else:
            pairs = _knot_pairs(cov, u, clusters, gamma, ridge)
        D = []
        w = np.zeros(n)
        for C, (QC, bC) in zip(clusters, pairs):
            zc = _leaf_vector(QC, bC, eta, ridge, long_only, info)
            if outer == "optimize":
                col = np.zeros(n)
                col[C] = zc
                D.append(col)
            else:
                w[C] = _rescale(zc, QC, bC, split, eta, ridge)
        if outer == "optimize":
            w = _outer_optimize(cov, u, D)
    else:
        raise ValueError("conditioning must be 'siblings', 'all' or 'factor'")

    s = float(w.sum())
    if abs(s) > 1e-300:
        w = w / s
        degenerate = False
    else:
        # The recursion produced a zero-sum book, which happens on a singular
        # covariance. Returning equal weight quietly made a failure read as a
        # result: in a comparison the bridge appeared to match equal weight at
        # low sample sizes, which is a wrong conclusion about the method rather
        # than a visible failure. It is still returned, because a portfolio is
        # more useful than an exception here, but it is now announced and the
        # caller can read `degenerate` from the info dict.
        w = np.full(n, 1.0 / n)
        degenerate = True
        warnings.warn(
            "the Schur recursion produced a zero-sum portfolio, which means the "
            "covariance is degenerate at this gamma; returning equal weight. "
            "Pass return_info=True and read info['degenerate'] to detect this.",
            RuntimeWarning, stacklevel=2)
    if return_info:
        info = dict(info or {})
        info["degenerate"] = degenerate
        return w, info
    return w


def _balanced_tree_over(clusters):
    """Bisection tree whose leaves are the given clusters, in the given order."""
    if len(clusters) == 1:
        return clusters[0]
    mid = len(clusters) // 2
    return (_balanced_tree_over(clusters[:mid]), _balanced_tree_over(clusters[mid:]))


def _tree_pairs(Q, b, tree, gamma, ridge, offset=0):
    """Leaf pairs after conditioning composed down the tree (no budgeting)."""
    if isinstance(tree, np.ndarray):
        return [(tree + offset, (Q, b))]
    L, R = tree
    nL = sum(len(t) for t in tree_leaves(L))
    n = Q.shape[0]
    I = np.arange(nL)
    J = np.arange(nL, n)
    QL, bL = condition_pair(Q, b, I, J, gamma, ridge)
    QR, bR = condition_pair(Q, b, J, I, gamma, ridge)
    return (_tree_pairs(QL, bL, _relabel(L, 0), gamma, ridge, offset)
            + _tree_pairs(QR, bR, _relabel(R, nL), gamma, ridge, offset + nL))


def _outer_optimize(cov, u, columns):
    """NCO's outer step on the span of the cluster directions:
    ``w = D (D^T Sigma D)^{-1} D^T u`` (unnormalized; scale-invariant in the columns)."""
    D = np.column_stack([c for c in columns if np.any(c != 0.0)])
    if D.shape[1] == 0:
        return np.ones(cov.shape[0])
    M = D.T @ cov @ D
    try:
        a = np.linalg.solve(M, D.T @ u)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(M, D.T @ u, rcond=None)[0]
    return D @ a
