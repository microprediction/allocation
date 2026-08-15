"""Tests for the k-factor Thurstone engine (allocation._thurstone.factor)."""

import numpy as np
import pytest

from allocation._thurstone.factor import (
    calibrate_factor,
    factor_model_contrast,
    hermite_nodes,
    jacobian_vector_product,
    winprobs_factor,
)

RNG = np.random.default_rng(7)


def test_forward_matches_monte_carlo():
    n, k = 8, 2
    a = RNG.normal(0, 0.6, n)
    V = 0.5 * RNG.standard_normal((n, k))
    D = RNG.uniform(0.4, 1.0, n)
    F, W = hermite_nodes(k)
    p = winprobs_factor(a, V, D, F, W)
    C = V @ V.T + np.diag(D)
    L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
    X = a[None, :] + RNG.standard_normal((2_000_000, n)) @ L.T
    ref = np.bincount(np.argmin(X, axis=1), minlength=n) / 2e6
    assert np.abs(p - ref).max() < 3e-3


def test_calibration_roundtrip():
    n = 40
    a_true = RNG.normal(0, 0.8, n)
    a_true -= a_true.mean()
    V = 0.4 * RNG.standard_normal((n, 2))
    D = RNG.uniform(0.5, 1.2, n)
    F, W = hermite_nodes(2)
    target = winprobs_factor(a_true, V, D, F, W)
    a_hat, info = calibrate_factor(target, V, D, F, W, return_info=True)
    assert info["converged"]
    res = target > 1e-3
    assert np.abs(a_hat - a_true)[res].max() < 1e-3


def test_jvp_is_laplacian_on_quotient():
    n = 9
    a = RNG.normal(0, 0.6, n)
    V = 0.4 * RNG.standard_normal((n, 2))
    D = RNG.uniform(0.5, 1.1, n)
    F, W = hermite_nodes(2)
    h = RNG.normal(0, 1, n); h -= h.mean()
    k2 = RNG.normal(0, 1, n); k2 -= k2.mean()
    Jh = jacobian_vector_product(a, V, D, F, W, h)
    Jk = jacobian_vector_product(a, V, D, F, W, k2)
    assert abs(h @ Jk - k2 @ Jh) < 1e-10       # symmetric
    assert h @ Jh < 0                           # minus a Laplacian (min-wins)
    assert abs(Jh.sum()) < 1e-11                # ones in the null space


def test_grid_jvp_matches_finite_differences():
    n = 8
    a = RNG.normal(0, 0.7, n)
    V = 0.4 * RNG.standard_normal((n, 2))
    D = RNG.uniform(0.5, 1.2, n)
    F, W = hermite_nodes(2)
    h = RNG.normal(0, 1, n); h -= h.mean()
    eps = 1e-5
    fd = (winprobs_factor(a + eps * h, V, D, F, W)
          - winprobs_factor(a - eps * h, V, D, F, W)) / (2 * eps)
    jv = jacobian_vector_product(a, V, D, F, W, h, form="grid")
    assert np.abs(jv - fd).max() < 1e-6


def test_contrast_fit_ignores_common_factor():
    n = 12
    b = RNG.normal(0, 1, n)
    Sig = 9.0 * np.ones((n, n)) + np.outer(b, b) + np.diag(RNG.uniform(0.5, 1, n))
    P = np.eye(n) - np.ones((n, n)) / n
    V, D = factor_model_contrast(Sig, 1)
    err = np.linalg.norm(P @ (Sig - V @ V.T - np.diag(D)) @ P)
    assert err < 0.1 * np.linalg.norm(P @ Sig @ P)
    assert np.abs(V.sum(axis=0)).max() < 1e-8
