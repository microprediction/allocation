"""Bug hunting for the bridge engine by means other than the corners it was built for.

1. Differential tests against an independent exact-rational implementation and
   against skfolio's own HERC and NCO.
2. Algebraic invariants the engine never targeted: scale, permutation,
   block-diagonal covariance, an in-sample variance floor, continuity.
3. Streaming versus batch agreement on a fixed universe.
4. Degenerate inputs.
"""

from fractions import Fraction as Fr

import numpy as np
import pytest

from allocation import SchurBridge, StreamingSchurBridge
from allocation._schur.bridge import bisection_tree, bridge_weights, contiguous_partition
from allocation._schur.coupling import compute_weights
from allocation.convex import min_variance_weights


def _cov(n, rng, cond=1.0):
    G = rng.standard_normal((n, 3 * n))
    return G @ G.T / (3 * n) + cond * np.diag(rng.uniform(0.2, 0.6, n))


def _close(a, b, tol=1e-9):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) < tol


# ------------------------------------------------ 1a. exact rational reference
def _fr_inv(M):
    n = len(M)
    A = [list(r) + [Fr(int(i == j)) for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [r[n:] for r in A]


def _fr_square(S, clusters, gamma, eta):
    """Flat HERC square in exact arithmetic (the paper's certificate, condensed)."""
    n = len(S)
    u = [Fr(1)] * n
    w = [Fr(0)] * n

    def cond(Q, b, I, J, g):
        A = [[Q[i][j] for j in I] for i in I]
        bI = [b[i] for i in I]
        if g == 0 or not J:
            return A, bI
        D = _fr_inv([[Q[i][j] for j in J] for i in J])
        B = [[Q[i][j] for j in J] for i in I]
        BD = [[sum(B[r][k] * D[k][c] for k in range(len(J))) for c in range(len(J))] for r in range(len(I))]
        Qc = [[A[r][c] - g * sum(BD[r][k] * B[c][k] for k in range(len(J))) for c in range(len(I))] for r in range(len(I))]
        bc = [bI[r] - g * sum(BD[r][k] * b[J[k]] for k in range(len(J))) for r in range(len(I))]
        return Qc, bc

    for C in clusters:
        J = [j for j in range(n) if j not in C]
        Q, b = cond(S, u, list(C), J, gamma)
        m = len(C)
        z = []
        for i in range(m):
            Ji = [j for j in range(m) if j != i]
            Qc, bc = cond(Q, b, [i], Ji, eta) if Ji else ([[Q[0][0]]], [b[0]])
            z.append(bc[0] / Qc[0][0])
        s = sum(bi * zi for bi, zi in zip(b, z))
        v = [zi / s for zi in z]
        nu = sum(v[r] * Q[r][c] * v[c] for r in range(m) for c in range(m))
        for k, i in enumerate(C):
            w[i] = v[k] / nu
    t = sum(w)
    return [x / t for x in w]


def test_engine_matches_the_exact_rational_certificate():
    rng = np.random.default_rng(100)
    for _ in range(5):
        n = 7
        G = rng.integers(-3, 4, size=(n, n))
        S_int = G @ G.T + n * np.eye(n, dtype=int)
        S_fr = [[Fr(int(x)) for x in row] for row in S_int]
        clusters = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6])]
        for gamma, eta in ((Fr(0), Fr(0)), (Fr(1, 3), Fr(2, 3)), (Fr(1), Fr(1)), (Fr(1), Fr(0))):
            exact = np.array([float(x) for x in _fr_square(S_fr, clusters, gamma, eta)])
            got = bridge_weights(S_int.astype(float), clusters, gamma=float(gamma), eta=float(eta), conditioning="all")
            assert _close(got, exact, 1e-10)


# ------------------------------------------------ 1b. skfolio's HERC and NCO
def test_zero_corner_matches_skfolio_herc_on_two_clusters():
    pytest.importorskip("skfolio")
    from skfolio.cluster import HierarchicalClustering
    from skfolio.optimization import HierarchicalEqualRiskContribution

    rng = np.random.default_rng(101)
    X = rng.standard_normal((500, 8)) @ np.diag(rng.uniform(0.5, 2.0, 8))
    X[:, 4:] += 0.5 * X[:, :4]  # some structure
    est = HierarchicalEqualRiskContribution(
        hierarchical_clustering_estimator=HierarchicalClustering(max_clusters=2)
    ).fit(X)
    labels = est.hierarchical_clustering_estimator_.labels_
    cov = est.prior_estimator_.return_distribution_.covariance
    clusters = [np.where(labels == g)[0] for g in np.unique(labels)]
    assert len(clusters) == 2
    ours = bridge_weights(cov, clusters, gamma=0.0, eta=0.0, conditioning="all")
    assert _close(ours, est.weights_, 1e-8)


def test_nco_corner_matches_skfolio_nco_without_cv():
    pytest.importorskip("skfolio")
    from skfolio.cluster import HierarchicalClustering
    from skfolio.optimization import MeanRisk, NestedClustersOptimization

    rng = np.random.default_rng(102)
    X = rng.standard_normal((600, 9)) @ np.diag(rng.uniform(0.5, 2.0, 9))
    X[:, 3:6] += 0.4 * X[:, :3]
    inner = MeanRisk(min_weights=-10.0, max_weights=10.0)
    outer = MeanRisk(min_weights=-10.0, max_weights=10.0)
    est = NestedClustersOptimization(
        inner_estimator=inner, outer_estimator=outer, cv="ignore",
        clustering_estimator=HierarchicalClustering(max_clusters=3),
    ).fit(X)
    labels = est.clustering_estimator_.labels_
    clusters = [np.where(labels == g)[0] for g in np.unique(labels)]
    cov = np.cov(X, rowvar=False)
    ours = bridge_weights(cov, clusters, gamma=0.0, eta=1.0, conditioning="all", outer="optimize")
    assert np.max(np.abs(ours - est.weights_)) < 1e-4  # solver tolerance


# ------------------------------------------------ 2. invariants
@pytest.mark.parametrize("conditioning", ["siblings", "all", "factor"])
def test_scale_invariance_of_covariance_and_companion(conditioning):
    rng = np.random.default_rng(103)
    cov = _cov(10, rng)
    part = contiguous_partition(rng.permutation(10), 3)
    for g, e in ((0.3, 0.7), (1.0, 0.5), (0.0, 0.0)):
        w = bridge_weights(cov, part, gamma=g, eta=e, conditioning=conditioning)
        assert _close(w, bridge_weights(7.3 * cov, part, gamma=g, eta=e, conditioning=conditioning))
        u = rng.uniform(0.5, 1.5, 10)
        wu = bridge_weights(cov, part, gamma=g, eta=e, conditioning=conditioning, companion=u)
        assert _close(wu, bridge_weights(cov, part, gamma=g, eta=e, conditioning=conditioning, companion=2.5 * u))


@pytest.mark.parametrize("conditioning", ["siblings", "all", "factor"])
def test_permutation_equivariance(conditioning):
    rng = np.random.default_rng(104)
    cov = _cov(11, rng)
    order = rng.permutation(11)
    tree = bisection_tree(order, leaf_size=3)
    w = bridge_weights(cov, tree, gamma=0.6, eta=0.4, conditioning=conditioning)
    perm = rng.permutation(11)
    inv = np.argsort(perm)
    cov_p = cov[np.ix_(perm, perm)]          # asset i of the new problem is asset perm[i] of the old

    def relabel(t):
        return inv[t] if isinstance(t, np.ndarray) else (relabel(t[0]), relabel(t[1]))

    w_p = bridge_weights(cov_p, relabel(tree), gamma=0.6, eta=0.4, conditioning=conditioning)
    assert _close(w_p, w[perm])


def test_block_diagonal_covariance_makes_gamma_idle_and_eta_exact():
    rng = np.random.default_rng(105)
    blocks = [_cov(3, rng), _cov(4, rng), _cov(2, rng)]
    n = 9
    cov = np.zeros((n, n))
    parts, s = [], 0
    for B in blocks:
        m = B.shape[0]
        cov[s:s + m, s:s + m] = B
        parts.append(np.arange(s, s + m))
        s += m
    for conditioning in ("siblings", "all", "factor"):
        base = bridge_weights(cov, parts, gamma=0.0, eta=0.5, conditioning=conditioning)
        for g in (0.3, 1.0):
            assert _close(bridge_weights(cov, parts, gamma=g, eta=0.5, conditioning=conditioning), base)
        assert _close(bridge_weights(cov, parts, gamma=0.0, eta=1.0, conditioning=conditioning), min_variance_weights(cov))


def test_in_sample_variance_never_beats_the_far_end():
    rng = np.random.default_rng(106)
    cov = _cov(12, rng)
    gmv = min_variance_weights(cov)
    floor = gmv @ cov @ gmv
    for k in (1, 3, 12):
        part = contiguous_partition(rng.permutation(12), k)
        for conditioning in ("siblings", "all", "factor"):
            for g in (0.0, 0.5, 1.0):
                for e in (0.0, 0.5, 1.0):
                    w = bridge_weights(cov, part, gamma=g, eta=e, conditioning=conditioning)
                    assert w @ cov @ w >= floor - 1e-12


def test_equal_volatility_equicorrelated_cluster_makes_eta_idle():
    rho = 0.4
    cov = np.full((6, 6), rho) + (1 - rho) * np.eye(6)
    part = [np.arange(3), np.arange(3, 6)]
    base = bridge_weights(cov, part, gamma=0.5, eta=0.0, conditioning="all")
    for e in (0.3, 1.0):
        assert _close(bridge_weights(cov, part, gamma=0.5, eta=e, conditioning="all"), base)


@pytest.mark.parametrize("conditioning", ["siblings", "all", "factor"])
def test_continuity_in_the_dials_and_the_covariance(conditioning):
    rng = np.random.default_rng(107)
    cov = _cov(10, rng)
    part = contiguous_partition(rng.permutation(10), 3)
    grid = np.linspace(0, 1, 41)
    prev = None
    for t in grid:
        w = bridge_weights(cov, part, gamma=float(t), eta=float(1 - t), conditioning=conditioning)
        if prev is not None:
            assert np.abs(w - prev).max() < 0.05
        prev = w
    E = rng.standard_normal((10, 10))
    E = 1e-6 * (E + E.T)
    w0 = bridge_weights(cov, part, gamma=0.5, eta=0.5, conditioning=conditioning)
    w1 = bridge_weights(cov + E, part, gamma=0.5, eta=0.5, conditioning=conditioning)
    assert np.abs(w1 - w0).max() < 1e-4


# ------------------------------------------------ 3. streaming vs batch
def test_streaming_equals_batch_on_a_fixed_universe():
    rng = np.random.default_rng(108)
    X = rng.standard_normal((120, 6)) @ np.diag(rng.uniform(0.5, 2.0, 6))
    ids = list("abcdef")
    for kwargs in (dict(gamma=0.5, eta=0.5, n_clusters=2, conditioning="all"),
                   dict(gamma=0.7, eta=1.0, n_clusters=None, conditioning="siblings"),
                   dict(gamma=0.4, eta=0.6, n_clusters=3, conditioning="factor", outer="optimize")):
        batch = SchurBridge(halflife=30.0, **kwargs)
        stream = StreamingSchurBridge(halflife=30.0, min_obs=1, **kwargs)
        for t in range(120):
            row = X[t]
            batch.fit(row) if t == 0 else batch.partial_fit(row)
            stream.learn_one(dict(zip(ids, row)))
        w_stream = np.array([stream.predict_one()[k] for k in ids])
        assert _close(w_stream, batch.weights_, 1e-9)


# ------------------------------------------------ 4. degenerate inputs
def test_tiny_universes_and_singleton_clusters():
    rng = np.random.default_rng(109)
    for n in (2, 3):
        cov = _cov(n, rng)
        gmv = min_variance_weights(cov)
        for conditioning in ("siblings", "all", "factor"):
            w = bridge_weights(cov, bisection_tree(np.arange(n)), gamma=1.0, eta=1.0, conditioning=conditioning)
            assert _close(w, gmv)
            w = bridge_weights(cov, [np.array([i]) for i in range(n)], gamma=0.0, eta=0.0, conditioning=conditioning)
            ivp = 1 / np.diag(cov)
            if conditioning == "siblings":
                # on a tree the near end is HRP, which is inverse variance only for two assets
                assert _close(w, compute_weights(np.arange(n), cov, 0.0))
            else:
                assert _close(w, ivp / ivp.sum())
    cov = _cov(5, rng)
    mixed = [np.array([0]), np.array([1, 2, 3]), np.array([4])]
    for conditioning in ("siblings", "all"):  # factor knots are exact here only under a one-factor model
        assert _close(bridge_weights(cov, mixed, gamma=1.0, eta=1.0, conditioning=conditioning), min_variance_weights(cov))


def test_duplicate_asset_is_handled_with_ridge_and_the_pair_shares_weight():
    rng = np.random.default_rng(110)
    cov = _cov(6, rng)
    cov7 = np.zeros((7, 7))
    cov7[:6, :6] = cov
    cov7[6, :6] = cov[5, :]
    cov7[:6, 6] = cov[:, 5]
    cov7[6, 6] = cov[5, 5]  # asset 6 duplicates asset 5 exactly (singular)
    part = [np.arange(3), np.arange(3, 7)]
    w = bridge_weights(cov7, part, gamma=0.8, eta=0.8, conditioning="all", ridge=1e-6)
    assert np.all(np.isfinite(w)) and abs(w.sum() - 1) < 1e-12
    assert abs(w[5] - w[6]) < 1e-3


def test_companion_with_zero_entries_gives_zero_weight_there_at_the_far_end():
    rng = np.random.default_rng(111)
    cov = _cov(6, rng)
    u = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    w = bridge_weights(cov, contiguous_partition(np.arange(6), 2), gamma=1.0, eta=1.0, conditioning="all", companion=u)
    ref = np.linalg.solve(cov, u)
    assert _close(w, ref / ref.sum())


def test_streaming_survives_a_constant_asset_and_a_singleton_warm_set():
    est = StreamingSchurBridge(gamma=0.5, eta=0.5, n_clusters=2, min_obs=3)
    rng = np.random.default_rng(112)
    for t in range(20):
        x = {"a": float(rng.standard_normal()), "b": float(rng.standard_normal()), "c": 0.0}
        if t < 10:
            x.pop("b")  # only one warm asset for a while
        est.learn_one(x)
        w = est.predict_one()
        assert all(np.isfinite(v) for v in w.values())
        if w:
            assert abs(sum(w.values()) - 1) < 1e-9
