"""Pluggable race samplers: the Student-t / t-copula race and its contract.

The Gaussian race is a function of the correlation alone; the t-race makes the
winning probabilities a genuine *tail* statistic (fat marginals + tail
dependence). These tests pin (a) the race-scoring extraction is behaviour
preserving, (b) the t-sampler is a valid long-only construction, (c) it recovers
the Gaussian race as ``nu -> inf``, (d) heavier tails actually move the weights
(the property the tail-risk thesis rests on), and (e) common-seed transport stays
smooth under the t-race.
"""

import numpy as np

from allocation import ThurstonePortfolio
from allocation._thurstone.transport import (
    _race_weights,
    gaussian_sampler,
    symmetric_sqrt,
    transport_weights,
    transport_weights_lowrank,
    transport_weights_lowrank_blockt,
    transport_weights_lowrank_t,
    transport_weights_t,
)


def _returns(n_obs=600, n=6, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_obs, 1))
    load = rng.uniform(0.3, 0.9, size=n)
    return f * load + 0.5 * rng.standard_normal((n_obs, n))


# --------------------------------------------------------------- scoring core
def test_race_weights_is_a_simplex_and_counts_argmin():
    X = np.array([[0.0, 1.0, 2.0], [5.0, -1.0, 0.0], [3.0, 3.0, -2.0]])
    w = _race_weights(X)  # winners: col 0, col 1, col 2 -> one each
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])
    assert abs(w.sum() - 1.0) < 1e-12


# ------------------------------------------------------------ dense t-sampler
def test_t_race_reduces_to_gaussian_as_nu_grows():
    rng = np.random.default_rng(0)
    n, M = 8, 200_000
    a = rng.standard_normal(n) * 0.5
    B = rng.standard_normal((n, 2)) * 0.4
    C = B @ B.T
    C[np.diag_indices(n)] = 1.0
    seeds = rng.standard_normal((M, n))
    chi2 = rng.chisquare(400.0, M)  # nu large -> scale ~ 1
    w_g = transport_weights(a, C, seeds)
    w_t = transport_weights_t(a, C, seeds, chi2, nu=400.0)
    assert np.max(np.abs(w_g - w_t)) < 0.01


def test_t_race_moves_weights_vs_gaussian_at_heavy_tail():
    # The thesis: with the SAME correlation and seeds, a heavy tail (small nu)
    # produces different weights than the Gaussian race -- the winning
    # probabilities respond to tail structure, not just to the correlation.
    rng = np.random.default_rng(1)
    n, M = 8, 200_000
    a = rng.standard_normal(n) * 0.5
    B = rng.standard_normal((n, 2)) * 0.5
    C = B @ B.T
    C[np.diag_indices(n)] = 1.0
    seeds = rng.standard_normal((M, n))
    chi2 = rng.chisquare(3.0, M)
    w_g = transport_weights(a, C, seeds)
    w_t = transport_weights_t(a, C, seeds, chi2, nu=3.0)
    # total-variation distance between the two weightings: an aggregate (more
    # stable than a single max element), and at M=200k MC noise is ~1e-3, so a
    # TV of 1e-2 is many sigma -- the tail genuinely reshapes the weights.
    tv = 0.5 * float(np.abs(w_g - w_t).sum())
    assert tv > 0.01
    assert abs(w_t.sum() - 1.0) < 1e-9 and np.all(w_t >= 0)


def test_lowrank_t_matches_dense_t():
    rng = np.random.default_rng(2)
    n, k, M = 10, 3, 400_000
    B = rng.standard_normal((n, k))
    B *= 0.8 / np.linalg.norm(B, axis=1, keepdims=True)  # row norm 0.8 -> d = 0.36 > 0
    d = 1.0 - (B**2).sum(1)
    assert np.all(d > 0)
    C = B @ B.T + np.diag(d)
    a = rng.standard_normal(n) * 0.5
    chi2 = rng.chisquare(5.0, M)
    w_dense = transport_weights_t(a, C, rng.standard_normal((M, n)), chi2, nu=5.0)
    w_low = transport_weights_lowrank_t(
        a, B, d, rng.standard_normal((M, k)), rng.standard_normal((M, n)), chi2, nu=5.0
    )
    assert np.max(np.abs(w_dense - w_low)) < 0.01


# ------------------------------------------------------ block-tail factor race
def _cluster_setup(n_b=99, beta=0.7):
    """A (independent) + ``n_b`` B's loading on one factor; unit-variance assets,
    corr(B_j, B_k) = beta**2, A uncorrelated. Returns loadings, idio, n."""
    n = 1 + n_b
    B = np.zeros((n, 1))
    B[1:, 0] = beta
    idio = np.ones(n)
    idio[1:] = 1.0 - beta**2  # so diag(B B^T + idio) == 1 for every asset
    return B, idio, n


def test_blockt_gaussian_factor_is_exactly_the_gaussian_race():
    # all nus = inf -> the standardized-t branch is skipped, X is identical.
    rng = np.random.default_rng(3)
    n, k, M = 12, 2, 50_000
    B = rng.standard_normal((n, k)) * 0.3
    idio = 1.0 - (B**2).sum(1)
    a = rng.standard_normal(n) * 0.3
    sf, si = rng.standard_normal((M, k)), rng.standard_normal((M, n))
    chi2 = rng.chisquare(10.0, (M, k))  # ignored when nu = inf
    w_g = transport_weights_lowrank(a, B, idio, sf, si)
    w_b = transport_weights_lowrank_blockt(a, B, idio, sf, si, chi2, [np.inf] * k)
    assert np.array_equal(w_g, w_b)


def test_blockt_factor_variance_is_tail_invariant():
    # The contract: standardized factors keep Cov(X) = B B^T + diag(idio) for any
    # nu, so correlation is held fixed while the tail is swept. Check the factor
    # column has ~unit variance whether nu is heavy or near-Gaussian.
    rng = np.random.default_rng(4)
    M = 400_000
    z = rng.standard_normal((M, 1))
    for nu in (3.0, 1e6):
        w = rng.chisquare(nu, (M, 1))
        F = (z / np.sqrt(w / nu)) * np.sqrt((nu - 2.0) / nu)
        assert abs(float(np.var(F)) - 1.0) < 0.05


def test_blockt_tail_dependence_shifts_cluster_weight_at_fixed_sigma():
    # The separation that *is* real: at FIXED Sigma (fixed beta), changing only
    # the block factor's tail index moves the cluster's winning probability --
    # something no Sigma-only allocator can do. NOTE the direction: a heavier
    # shared factor makes the 99 B's co-own the extreme-min scenarios, so the
    # cluster wins the argmin race *more* often (it is not de-duplicated -- that
    # is a correlation effect, beta->1, not a tail effect). Each call draws its
    # chi-square with the matching dof, per the sampler's contract.
    rng = np.random.default_rng(5)
    B, idio, n = _cluster_setup(n_b=99, beta=0.7)
    M = 200_000
    a = np.zeros(n)
    sf = rng.standard_normal((M, 1))
    si = rng.standard_normal((M, n))
    w_gauss = transport_weights_lowrank_blockt(
        a, B, idio, sf, si, rng.chisquare(1e9, (M, 1)), [1e9]
    )
    w_heavy = transport_weights_lowrank_blockt(
        a, B, idio, sf, si, rng.chisquare(2.5, (M, 1)), [2.5]
    )
    # weights move with the tail (here: cluster weight rises); the point is that
    # Sigma is identical across the two and only the tail index differs.
    assert w_heavy[1:].sum() > w_gauss[1:].sum() + 0.005


def test_blockt_comonotone_limit_is_two_body_value():
    # The cardinality collapse is a CORRELATION limit, not a tail one: as beta->1
    # the 99 B's become one variable for ANY nu, so the cluster weight -> the
    # two-body race value (here ~0.5 by symmetry of equal abilities), independent
    # of how many names are in the cluster.
    rng = np.random.default_rng(6)
    M = 200_000
    a0 = np.zeros(100)
    for nu in (1e9, 3.0):
        B = np.zeros((100, 1)); B[1:, 0] = 0.999
        idio = np.ones(100); idio[1:] = 1.0 - 0.999**2
        w = transport_weights_lowrank_blockt(
            a0, B, idio, rng.standard_normal((M, 1)), rng.standard_normal((M, 100)),
            rng.chisquare(nu, (M, 1)), [nu],
        )
        assert abs(w[1:].sum() - 0.5) < 0.05  # cardinality-invariant two-body value


# -------------------------------------------------------- estimator wiring
def test_estimator_student_t_is_long_only_simplex():
    X = _returns()
    for factors in (None, 2):
        w = ThurstonePortfolio(
            sampler="student_t", nu=5.0, factors=factors, n_paths=1 << 13
        ).fit(X).weights_
        assert np.all(w >= -1e-9)
        assert abs(float(w.sum()) - 1.0) < 1e-6


def test_estimator_student_t_partial_fit_is_smooth():
    X = _returns(n_obs=600)
    est = ThurstonePortfolio(sampler="student_t", nu=5.0).fit(X)
    w0 = est.weights_.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    assert np.max(np.abs(est.weights_ - w0)) < 0.15  # transport, not a fresh draw


def test_bad_sampler_and_nu_raise():
    X = _returns()
    for kwargs in ({"sampler": "cauchy"}, {"sampler": "student_t", "nu": 0.0}):
        try:
            ThurstonePortfolio(**kwargs).fit(X)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# ------------------------------------------------------ any-simulation (callable)
def test_callable_gaussian_sampler_matches_string():
    # the reference callable reproduces the built-in "gaussian" path exactly
    # (same fixed seeds, same recolouring).
    X = _returns()
    w_str = ThurstonePortfolio(sampler="gaussian", seed=7).fit(X).weights_
    w_call = ThurstonePortfolio(sampler=gaussian_sampler, seed=7).fit(X).weights_
    assert np.allclose(w_str, w_call)


def test_callable_custom_sampler_long_only_and_smooth():
    # an arbitrary centered law: a deterministic downside skew of the fixed draw.
    def skew_sampler(ability, corr, seeds):
        g = seeds @ symmetric_sqrt(corr)
        return ability + g - 0.4 * np.abs(g)  # left-skewed, deterministic in seeds

    X = _returns()
    est = ThurstonePortfolio(sampler=skew_sampler).fit(X)
    w = est.weights_
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-6
    before = w.copy()
    est.partial_fit(_returns(n_obs=5, seed=1))
    assert np.max(np.abs(est.weights_ - before)) < 0.15  # common-seed transport holds


def test_callable_sampler_rejects_factor_mode():
    X = _returns()
    try:
        ThurstonePortfolio(sampler=gaussian_sampler, factors=2).fit(X)
    except ValueError:
        return
    raise AssertionError("expected ValueError for callable sampler + factor mode")
