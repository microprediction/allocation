"""Guard the dependency this package now relies on completely.

`allocation` calibrates every Thurstone portfolio through `winning`'s front
door, so behaviour drift there changes results here silently. `winning` ships
a seeded golden check for exactly this purpose; `winning/contract.py` exists so
downstream repos can catch it in CI.

Also pins the two properties the port depends on, since neither is guaranteed
by the contract: that the one-factor spelling supplies the idiosyncratic
variance, and that a zero weight does not produce a non-finite ability.
"""
import numpy as np
import pytest


def test_winning_contract_holds():
    verify = pytest.importorskip("winning.contract").verify
    verify()


def test_one_factor_matches_the_documented_model():
    """X_i = a_i + b_i Z + sqrt(1 - b_i^2) eps_i, min wins.

    Passing the loadings without the idiosyncratic variances leaves the total
    variance inflated and costs about 6e-2, which no test caught.
    """
    from allocation._thurstone.calibrate import winprobs_one_factor
    rng = np.random.default_rng(11)
    n, M = 12, 400_000
    a = rng.normal(0, 0.5, n)
    b = rng.uniform(0.1, 0.8, n)
    Z = rng.normal(size=(M, 1))
    E = rng.normal(size=(M, n))
    truth = np.bincount((a + b * Z + np.sqrt(1 - b ** 2) * E).argmin(1), minlength=n) / M
    assert np.abs(winprobs_one_factor(a, b) - truth).max() < 5e-3


def test_zero_weights_give_finite_abilities():
    from allocation._thurstone.ability import state_price_implied_ability
    a = state_price_implied_ability(np.array([0.4, 0.3, 0.3, 0.0, 0.0]))
    assert np.isfinite(a).all()
    assert a[3] > a[0] and a[4] > a[0]      # min-wins: zero weight is the weakest
