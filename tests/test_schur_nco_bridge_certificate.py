"""Run the certificate for papers/schur-nco-bridge under pytest."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "papers" / "schur-nco-bridge" / "verify_schur_nco_bridge.py"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="schur-nco-bridge paper not present")


@pytest.fixture(scope="module")
def cert():
    spec = importlib.util.spec_from_file_location("verify_schur_nco_bridge", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_knots_are_sufficient(cert, rng):
    assert cert.check_sufficiency(rng, trials=30)["pair"] < cert.TOL


def test_right_end_is_global_optimum(cert, rng):
    r = cert.check_bridge_right(rng, trials=30)
    assert r["stacked"] < cert.TOL and r["recipe"] < cert.TOL


def test_left_end_is_nco(cert, rng):
    assert cert.check_bridge_left(rng, trials=30)["nco"] < cert.TOL


def test_exactness_fails_without_the_model(cert, rng):
    r = cert.check_violation(rng, trials=10)
    assert r["pair"] > 1e-3 and r["stacked"] > 1e-4 and r["recipe"] > 1e-4


def test_other_knot_regression_false_accepts_but_R_does_not(cert, rng):
    r = cert.check_diagnostic(rng)
    assert r["model"]["R"] < cert.TOL
    assert r["violated"]["other_knot_coef"] < cert.TOL
    assert r["violated"]["R"] > 1e-2


def test_covariance_only_rule_needs_mapping_back(cert, rng):
    r = cert.check_change_of_vars(rng, trials=30)
    assert r["mapped_back"] < cert.TOL
    assert np.allclose(r["unmapped_example"], [0.2, 0.8])


def test_outer_covariance_identity(cert, rng):
    assert cert.check_proxy(rng, trials=30)["identity"] < cert.TOL


def test_symmetric_factor_residual_formula(cert, rng):
    assert cert.check_symmetric(rng, trials=20)["formula"] < cert.TOL


def test_complements_compose(cert, rng):
    assert cert.check_quotient(rng, trials=20)["compose"] < cert.TOL
