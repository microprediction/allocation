"""DownsideSemicovariance: the co-lower-partial-moment estimator and its use as a
tail-consistent driver for the Thurstone race."""

import numpy as np

from allocation import DownsideSemicovariance, EwmaCovariance, ThurstonePortfolio
from allocation._thurstone.covariance import cov_to_corr


def _crash_together(rng, T=5000, n_b=6):
    """One independent asset A + a cluster that drops together but rallies
    independently (downside co-movement >> full co-movement), numpy only."""
    z = rng.standard_normal((T, 1))                       # common factor
    idio = rng.standard_normal((T, n_b))
    B = np.where(z < 0, z, idio)                          # crash together, rally apart
    A = rng.standard_normal((T, 1))
    return np.concatenate([A, B], axis=1) * 0.01


def _avg_cluster_corr(C):
    Cc = C[1:, 1:]
    return float(np.mean(Cc[np.triu_indices(Cc.shape[0], 1)]))


def test_psd_and_positive_diagonal():
    rng = np.random.default_rng(0)
    S = DownsideSemicovariance(halflife=120).fit(_crash_together(rng)).covariance_
    assert np.allclose(S, S.T)
    assert np.min(np.linalg.eigvalsh(S)) >= -1e-12      # PSD
    assert np.all(np.diag(S) > 0)


def test_downside_corr_exceeds_full_for_crash_cluster():
    rng = np.random.default_rng(1)
    R = _crash_together(rng)
    full = cov_to_corr(EwmaCovariance(halflife=120).fit(R).covariance_)
    down = cov_to_corr(DownsideSemicovariance(halflife=120).fit(R).covariance_)
    # the cluster crashes together: downside correlation sees it, full averages it away
    assert _avg_cluster_corr(down) > _avg_cluster_corr(full) + 0.1


def test_drives_thurstone_to_a_simplex():
    rng = np.random.default_rng(2)
    R = _crash_together(rng)
    w = ThurstonePortfolio(calib="diagonal", target="equal",
                           covariance_estimator=DownsideSemicovariance()).fit(R).weights_
    assert np.all(w >= -1e-9) and abs(float(w.sum()) - 1.0) < 1e-6


def test_downside_lifts_the_hedge_vs_full():
    # tail consistency end-to-end (diagonal calibration): driving the race with the
    # downside covariance de-weights the co-crashing cluster and lifts the
    # decorrelated hedge A, relative to the full-covariance race.
    rng = np.random.default_rng(3)
    R = _crash_together(rng)
    common = dict(calib="diagonal", target="equal", phi=1.0, seed=7)
    w_full = ThurstonePortfolio(covariance_estimator=EwmaCovariance(halflife=120),
                                **common).fit(R).weights_
    w_down = ThurstonePortfolio(covariance_estimator=DownsideSemicovariance(halflife=120),
                                **common).fit(R).weights_
    assert w_down[0] > w_full[0] + 0.02      # hedge weight rises under the downside race
