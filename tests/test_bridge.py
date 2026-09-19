"""The Schur bridge engine: exact corners, nesting of the other allocators,
partition independence at full coupling, and the batch / streaming estimators."""

import numpy as np
import pytest

from allocation import SchurBridge, StreamingSchurBridge
from allocation._schur.bridge import (
    bisection_tree,
    bridge_weights,
    contiguous_partition,
    leaf_direction,
    leaf_frontier,
    one_factor,
)
from allocation._schur.coupling import compute_weights
from allocation.convex import max_diversification_weights, mean_variance_weights, min_variance_weights


def _cov(n, rng):
    G = rng.standard_normal((n, 3 * n))
    return G @ G.T / (3 * n) + np.diag(rng.uniform(0.2, 0.6, n))


def _one_factor_blocks(rng, k=3, m=5):
    Cf = np.array([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]])[:k, :k]
    N = k * m
    beta = rng.uniform(0.5, 1.5, N)
    psi = rng.uniform(0.2, 1.0, N)
    lab = np.repeat(np.arange(k), m)
    S = np.outer(beta, beta) * Cf[np.ix_(lab, lab)] + np.diag(psi)
    return S, [np.where(lab == j)[0] for j in range(k)], beta, psi


def _close(a, b, tol=1e-10):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) < tol


# ------------------------------------------------------------------ corners
@pytest.mark.parametrize("conditioning", ["siblings", "all"])
@pytest.mark.parametrize("outer", ["stack", "optimize"])
def test_full_coupling_is_minimum_variance_for_any_partition(conditioning, outer):
    rng = np.random.default_rng(0)
    cov = _cov(12, rng)
    gmv = min_variance_weights(cov)
    for k in (1, 3, 5, 12):
        part = contiguous_partition(rng.permutation(12), k)
        w = bridge_weights(cov, part, gamma=1.0, eta=1.0, conditioning=conditioning, outer=outer)
        assert _close(w, gmv)


def test_full_coupling_does_not_depend_on_the_partition():
    rng = np.random.default_rng(1)
    cov = _cov(10, rng)
    ws = [bridge_weights(cov, contiguous_partition(rng.permutation(10), k), gamma=1.0, eta=1.0,
                         conditioning="all") for k in (2, 3, 4)]
    assert _close(ws[0], ws[1]) and _close(ws[1], ws[2])


def test_hrp_is_gamma_zero_with_naive_fitness():
    rng = np.random.default_rng(2)
    cov = _cov(13, rng)
    order = rng.permutation(13)
    w = bridge_weights(cov, bisection_tree(order), gamma=0.0, eta=1.0, split="hrp")
    assert _close(w, compute_weights(order, cov, 0.0))


def test_herc_is_the_zero_corner():
    rng = np.random.default_rng(3)
    cov = _cov(11, rng)
    part = contiguous_partition(rng.permutation(11), 3)
    ref = np.zeros(11)
    for C in part:
        A = cov[np.ix_(C, C)]
        v = 1.0 / np.diag(A)
        v /= v.sum()
        ref[C] = v / float(v @ A @ v)  # inverse variance inside, inverse cluster variance across
    ref /= ref.sum()
    assert _close(bridge_weights(cov, part, gamma=0.0, eta=0.0, conditioning="all"), ref)


def test_nco_is_gamma_zero_with_the_outer_optimizer():
    rng = np.random.default_rng(4)
    cov = _cov(12, rng)
    part = contiguous_partition(rng.permutation(12), 3)
    V = np.zeros((12, 3))
    for j, C in enumerate(part):
        v = np.linalg.solve(cov[np.ix_(C, C)], np.ones(len(C)))
        V[C, j] = v / v.sum()
    a = np.linalg.solve(V.T @ cov @ V, np.ones(3))
    ref = V @ a
    ref /= ref.sum()
    assert _close(bridge_weights(cov, part, gamma=0.0, eta=1.0, conditioning="all", outer="optimize"), ref)


def test_stevens_path_from_inverse_variance_to_minimum_variance():
    rng = np.random.default_rng(5)
    cov = _cov(9, rng)
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    one = [np.arange(9)]
    assert _close(bridge_weights(cov, one, gamma=0.0, eta=0.0, conditioning="all"), ivp)
    assert _close(bridge_weights(cov, one, gamma=0.0, eta=1.0, conditioning="all"), min_variance_weights(cov))
    single = [np.array([i]) for i in range(9)]
    assert _close(bridge_weights(cov, single, gamma=0.0, eta=0.0, conditioning="all"), ivp)
    assert _close(bridge_weights(cov, single, gamma=1.0, eta=0.0, conditioning="all"), min_variance_weights(cov))


def test_companion_vectors_give_max_diversification_and_tangency():
    rng = np.random.default_rng(6)
    cov = _cov(10, rng)
    tree = bisection_tree(rng.permutation(10))
    assert _close(bridge_weights(cov, tree, gamma=1.0, eta=1.0, companion="vol"), max_diversification_weights(cov))
    mu = rng.uniform(0.5, 1.5, 10)
    assert _close(bridge_weights(cov, tree, gamma=1.0, eta=1.0, companion=mu), mean_variance_weights(cov, mu))


# ------------------------------------------------------------------- leaves
def test_leaf_direction_is_affine_between_naive_and_stevens_quantities():
    rng = np.random.default_rng(7)
    Q = _cov(6, rng)
    b = rng.uniform(0.5, 1.5, 6)
    Qi = np.linalg.inv(Q)
    for eta in (0.0, 0.3, 1.0):
        z = leaf_direction(Q, b, eta)
        q = 1.0 / np.diag(Qi)
        s = (Qi @ b) / np.diag(Qi)
        assert _close(z, ((1 - eta) * b + eta * s) / ((1 - eta) * np.diag(Q) + eta * q))
    assert _close(leaf_direction(Q, b, 1.0), Qi @ b)


def test_leaf_frontier_bounds_the_long_only_region():
    rng = np.random.default_rng(8)
    # a strongly correlated block with heterogeneous vols so min-var shorts
    sig = np.array([1.0, 2.0, 3.0, 5.0])
    R = 0.35 * np.ones((4, 4)) + 0.65 * np.eye(4)
    Q = np.outer(sig, sig) * R
    b = np.ones(4)
    assert np.any(leaf_direction(Q, b, 1.0) < 0)
    eta_plus = leaf_frontier(Q, b)
    assert 0.0 < eta_plus < 1.0
    assert np.all(leaf_direction(Q, b, eta_plus) >= -1e-12)
    assert np.any(leaf_direction(Q, b, min(1.0, eta_plus + 0.05)) < 0)


def test_long_only_option_caps_eta_and_keeps_weights_nonnegative():
    sig = np.array([1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 2.0, 4.0])
    R = 0.35 * np.ones((8, 8)) + 0.65 * np.eye(8)
    cov = np.outer(sig, sig) * R
    part = [np.arange(4), np.arange(4, 8)]
    w, info = bridge_weights(cov, part, gamma=0.0, eta=1.0, conditioning="all", long_only=True, return_info=True)
    assert np.all(w >= -1e-12)
    assert all(e <= 1.0 for e in info["eta_effective"]) and any(e < 1.0 for e in info["eta_effective"])


# -------------------------------------------------------------------- knots
def test_one_factor_fit_recovers_the_factor():
    rng = np.random.default_rng(9)
    S, clusters, beta, psi = _one_factor_blocks(rng)
    for C in clusters:
        b, p = one_factor(S[np.ix_(C, C)])
        assert _close(np.abs(b), beta[C], 1e-7)
        assert _close(p, psi[C], 1e-7)


@pytest.mark.parametrize("outer", ["stack", "optimize"])
def test_knots_are_exact_under_a_block_one_factor_model(outer):
    rng = np.random.default_rng(10)
    S, clusters, _, _ = _one_factor_blocks(rng)
    w = bridge_weights(S, clusters, gamma=1.0, eta=1.0, conditioning="factor", outer=outer)
    assert _close(w, min_variance_weights(S), 1e-8)


def test_knots_are_close_on_a_generic_covariance_and_continuous_in_gamma():
    rng = np.random.default_rng(11)
    cov = _cov(12, rng)
    part = contiguous_partition(rng.permutation(12), 3)
    w1 = bridge_weights(cov, part, gamma=1.0, eta=1.0, conditioning="factor")
    assert np.max(np.abs(w1 - min_variance_weights(cov))) < 0.1
    prev = None
    for g in np.linspace(0, 1, 11):
        w = bridge_weights(cov, part, gamma=float(g), eta=1.0, conditioning="factor")
        if prev is not None:
            assert np.max(np.abs(w - prev)) < 0.1
        prev = w


# --------------------------------------------------------------- estimators
def test_estimator_nests_named_methods():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((300, 8)) @ np.diag(rng.uniform(0.5, 2.0, 8))
    est = SchurBridge(gamma=1.0, eta=1.0, n_clusters=3).fit(X)
    cov = est._cov_estimator.covariance_
    assert _close(est.weights_, min_variance_weights(cov))
    est_mdp = SchurBridge(gamma=1.0, eta=1.0, companion="vol").fit(X)
    assert _close(est_mdp.weights_, max_diversification_weights(cov))
    est_tan = SchurBridge(gamma=1.0, eta=1.0, companion="mean").fit(X)
    assert _close(est_tan.weights_, mean_variance_weights(cov, est_tan._cov_estimator.mean_))
    hrp = SchurBridge(gamma=0.0, split="hrp").fit(X)
    assert _close(hrp.weights_, compute_weights(hrp.order_, cov, 0.0))
    assert len(SchurBridge(gamma=0.0, eta=0.0, n_clusters=4).fit(X).clusters_) == 4


def test_estimator_fixed_labels_and_partial_fit_are_smooth():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((400, 10))
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    est = SchurBridge(gamma=0.5, eta=0.5, clusters=labels, conditioning="factor", halflife=100.0)
    est.fit(X[:200])
    w0 = est.weights_.copy()
    est.partial_fit(X[200:201])
    assert np.sum(np.abs(est.weights_ - w0)) < 0.05
    assert abs(est.weights_.sum() - 1.0) < 1e-12


def test_estimator_rejects_bad_dials():
    with pytest.raises(ValueError):
        SchurBridge(gamma=1.5).fit(np.random.default_rng(0).standard_normal((50, 4)))


def test_streaming_bridge_changing_universe():
    rng = np.random.default_rng(14)
    est = StreamingSchurBridge(gamma=0.5, eta=1.0, n_clusters=2, min_obs=5, halflife=30.0)
    ids = ["a", "b", "c", "d"]
    for t in range(60):
        x = {k: float(rng.standard_normal()) for k in ids}
        if t >= 30:
            x["e"] = float(rng.standard_normal())  # a listing
        if t >= 45:
            x.pop("a")  # a delisting
        est.learn_one(x)
        w = est.predict_one()
        if w:
            assert abs(sum(w.values()) - 1.0) < 1e-9
    w = est.predict_one()
    assert "a" not in w and "e" in w


def test_streaming_bridge_fixed_clusters_and_mean_companion():
    rng = np.random.default_rng(15)
    est = StreamingSchurBridge(gamma=1.0, eta=1.0, clusters={"a": "x", "b": "x", "c": "y", "d": "y"},
                               conditioning="factor", companion="mean", min_obs=5)
    for _ in range(40):
        est.learn_one({k: float(rng.standard_normal()) for k in "abcd"})
    w = est.predict_one()
    assert set(w) == set("abcd") and abs(sum(w.values()) - 1.0) < 1e-9


# ------------------------------------------------------ named by endpoints
def _hmv_reference(order, cov):
    """Hierarchical minimum variance (Cotton 2024 at gamma = 0): recursive
    bisection with the min-variance variance of each child block as fitness."""
    w = np.ones(len(cov))

    def rec(idx, budget):
        if len(idx) == 1:
            w[idx[0]] = budget
            return
        mid = len(idx) // 2
        L, R = idx[:mid], idx[mid:]
        nuL = 1.0 / float(np.ones(len(L)) @ np.linalg.solve(cov[np.ix_(L, L)], np.ones(len(L))))
        nuR = 1.0 / float(np.ones(len(R)) @ np.linalg.solve(cov[np.ix_(R, R)], np.ones(len(R))))
        aL = (1 / nuL) / (1 / nuL + 1 / nuR)
        rec(L, budget * aL)
        rec(R, budget * (1 - aL))

    rec(np.asarray(order), 1.0)
    return w / w.sum()


@pytest.mark.parametrize("split, near", [("dial", "hrp"), ("minvar", "hmv"), ("hrp", "hrp")])
def test_split_rules_fix_the_near_end_and_decide_the_far_end(split, near):
    rng = np.random.default_rng(20)
    cov = _cov(16, rng)
    order = rng.permutation(16)
    tree = bisection_tree(order)
    ref = compute_weights(order, cov, 0.0) if near == "hrp" else _hmv_reference(order, cov)
    assert _close(bridge_weights(cov, tree, gamma=0.0, eta=0.0, split=split), ref)
    far = bridge_weights(cov, tree, gamma=1.0, eta=1.0, split=split)
    if split == "hrp":
        assert not _close(far, min_variance_weights(cov), 1e-6)  # the collapsed rule misses the far end
    else:
        assert _close(far, min_variance_weights(cov))


def test_hrp_to_min_variance_is_exact_at_both_ends_with_one_dial():
    rng = np.random.default_rng(21)
    X = rng.standard_normal((400, 12)) @ np.diag(rng.uniform(0.5, 2.0, 12))
    near = SchurBridge.hrp_to_min_variance(0.0).fit(X)
    far = SchurBridge.hrp_to_min_variance(1.0).fit(X)
    cov = far._cov_estimator.covariance_
    assert _close(near.weights_, compute_weights(near.order_, cov, 0.0))
    assert _close(far.weights_, min_variance_weights(cov))
    assert near.endpoints_ == ("HRP", "minimum variance")


def test_named_constructors_report_their_endpoints():
    assert SchurBridge.hmv_to_min_variance().endpoints_ == ("hierarchical minimum variance", "minimum variance")
    assert SchurBridge.herc_to_min_variance(0.0, 0.0, 4).endpoints_ == ("HERC", "minimum variance")
    assert SchurBridge.nco_to_min_variance(0.3, 4).endpoints_ == ("NCO", "minimum variance")
    assert SchurBridge.inverse_variance_to_min_variance(0.5).endpoints_ == ("inverse variance", "minimum variance")
    assert SchurBridge.hrp_to_min_variance(0.5, companion="vol").endpoints_ == ("HRP", "maximum diversification")
    assert SchurBridge(split="hrp").endpoints_[1].endswith("(approximate)")


def test_on_a_flat_partition_the_split_is_the_cluster_budget_rule():
    rng = np.random.default_rng(22)
    cov = _cov(12, rng)
    part = contiguous_partition(rng.permutation(12), 3)
    # at eta = 1 the cluster holds its min-variance direction, so 'dial' and 'minvar' agree
    ws = [bridge_weights(cov, part, gamma=0.4, eta=1.0, conditioning="all", split=s) for s in ("dial", "minvar")]
    assert _close(ws[0], ws[1])
    # at eta < 1 they differ: 'dial' budgets by the variance of what is held (HERC's rule)
    ws = [bridge_weights(cov, part, gamma=0.4, eta=0.5, conditioning="all", split=s) for s in ("dial", "minvar")]
    assert not _close(ws[0], ws[1], 1e-6)


def test_named_constructors_are_exact_at_the_far_end_by_default():
    rng = np.random.default_rng(23)
    X = rng.standard_normal((400, 12)) @ np.diag(rng.uniform(0.5, 2.0, 12))
    for make in (lambda: SchurBridge.herc_to_min_variance(1.0, 1.0, n_clusters=3),
                 lambda: SchurBridge.nco_to_min_variance(1.0, n_clusters=3),
                 lambda: SchurBridge.hmv_to_min_variance(1.0),
                 lambda: SchurBridge.inverse_variance_to_min_variance(1.0)):
        est = make().fit(X)
        assert _close(est.weights_, min_variance_weights(est._cov_estimator.covariance_))
        assert not est.endpoints_[1].endswith("(approximate)")
    assert SchurBridge.herc_to_min_variance(0.5, 0.5, 3, conditioning="factor").endpoints_[1].endswith("(approximate)")
    assert SchurBridge.herc_to_min_variance(1.0, 1.0, 3).endpoints_ == ("HERC", "minimum variance")
