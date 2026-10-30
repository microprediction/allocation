"""Regression coverage for issue #138: row scaling of multiple right-hand sides."""

import numpy as np
import pytest

from allocation.factor import woodbury_solve


@pytest.mark.parametrize("n_columns", [None, 1, 2, 3, 5])
@pytest.mark.parametrize("n_factors", [0, 1, 2])
def test_woodbury_vector_and_matrix_rhs_match_dense(n_columns, n_factors):
    B = np.array([[1.0, -0.2], [2.0, 0.7], [0.5, 1.3]])[:, :n_factors]
    psi = np.array([0.5, 2.0, 4.0])
    covariance = B @ B.T + np.diag(psi)
    if n_columns is None:
        rhs = np.array([1.0, -2.0, 3.0])
    elif n_columns == 3:
        rhs = np.eye(3)  # previously returned a plausible but incorrect inverse
    else:
        rhs = np.arange(3 * n_columns, dtype=float).reshape(3, n_columns) + 1

    actual = woodbury_solve(B, psi, rhs)

    assert actual.shape == rhs.shape
    np.testing.assert_allclose(actual, np.linalg.solve(covariance, rhs), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(covariance @ actual, rhs, rtol=1e-12, atol=1e-12)
    if n_columns is not None:
        separate = np.column_stack([woodbury_solve(B, psi, col) for col in rhs.T])
        np.testing.assert_allclose(actual, separate, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rhs", [1.0, np.ones((3, 3, 3))])
def test_woodbury_rejects_rhs_outside_vector_matrix_contract(rhs):
    with pytest.raises(ValueError, match="rhs must have shape"):
        woodbury_solve(np.ones((3, 1)), np.ones(3), rhs)
