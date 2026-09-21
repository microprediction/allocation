"""Does the repair still hurt minimum-variance when the covariance is
KNOWN, not estimated?

The real-data check found min-variance's harm from the repair vanishes as
its own covariance shrinkage rises (41% win rate unshrunk -> 59.5% at
shrinkage=0.9). Two different explanations are consistent with that:

(a) ESTIMATION NOISE: the raw sample covariance is a bad, overfit estimate,
    the repair's re-race uses the SAME bad estimate, and the harm is really
    "two noisy things disagreeing," not a property of being at an optimum.
(b) SECOND-ORDER OPTIMALITY: min-variance sits at (approximately) a critical
    point of its own objective on the simplex tangent space, so ANY nonzero
    perturbation increases that objective to second order (a real, provable
    fact about constrained optima, independent of estimation quality) --
    heavier shrinkage just makes the base portfolio LESS optimal for the
    TRUE covariance, giving the repair legitimate room to improve it.

Real data can't cleanly separate these, since noise is always present. This
uses a synthetic k-factor market with a KNOWN true covariance and a large
enough in-sample window that the sample estimate is very close to the truth
(estimation noise made deliberately small) -- if the repair still hurts
unshrunk minimum-variance there, explanation (b) is confirmed as real and
distinct from (a), not merely a proxy for noise.
"""
from __future__ import annotations

import numpy as np

from schur_thurstone_repair import _nodes, es95

from allocation import BoxConstrained, MinimumVariance
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights


def synthetic_market(rng, n, k, T):
    """k-factor synthetic returns with a KNOWN true correlation C_true."""
    B = rng.standard_normal((n, k)) * 0.35
    idio = rng.uniform(0.5, 1.0, n)
    cov_true = B @ B.T + np.diag(idio)
    d = np.sqrt(np.diag(cov_true))
    corr_true = cov_true / d[:, None] / d[None, :]
    F = rng.standard_normal((T, k))
    R = F @ B.T + np.sqrt(idio) * rng.standard_normal((T, n))
    return R, corr_true


def independent_repair(w_base, corr_target, seeds, points=251):
    n = len(w_base)
    V = np.zeros((n, 1))
    D = np.ones(n)
    Fq, Wq = _nodes(1)
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=points)
    return transport_weights(theta, corr_target, seeds)


def run_trial(rng, n, k, T_in, T_out):
    Rin, corr_true = synthetic_market(rng, n, k, T_in)
    Rout, _ = synthetic_market(rng, n, k, T_out)  # fresh draw, SAME true C

    corr_sample = np.corrcoef(Rin, rowvar=False)
    sample_err = np.abs(corr_sample - corr_true).mean()

    est = BoxConstrained(MinimumVariance(shrinkage=0.0))
    est.fit(Rin)
    w_base = np.clip(est.weights_, 0.0, None)
    w_base = w_base / w_base.sum() if w_base.sum() > 0 else np.full(n, 1 / n)

    seeds = rng.standard_normal((1 << 14, n))
    # repair under the SAMPLE correlation (what you'd actually do)
    w_rep_sample = independent_repair(w_base, corr_sample, seeds)
    # repair under the TRUE correlation (the noise-free oracle case)
    w_rep_true = independent_repair(w_base, corr_true, seeds)

    r_base = Rout @ w_base
    base_es = es95(r_base)
    gain_sample = es95(Rout @ w_rep_sample) - base_es
    gain_true = es95(Rout @ w_rep_true) - base_es

    # in-sample variance under the TRUE covariance (the exact object the
    # "second-order optimality" argument is about)
    var_base = w_base @ corr_true @ w_base
    var_rep_true = w_rep_true @ corr_true @ w_rep_true

    return sample_err, gain_sample, gain_true, var_rep_true - var_base


def main(n_trials=100, n=20, k=4, T_out=2000, seed=0):
    rng_master = np.random.default_rng(seed)
    print(f"{'T_in':>7}{'sample_err':>12}{'gain(sample corr)':>20}"
          f"{'gain(TRUE corr)':>18}{'d(true variance)':>18}", flush=True)
    for T_in in (300, 1000, 5000, 30000):
        errs, g_samp, g_true, dvar = [], [], [], []
        for _ in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            e, gs, gt, dv = run_trial(rng, n, k, T_in, T_out)
            errs.append(e); g_samp.append(gs); g_true.append(gt); dvar.append(dv)
        errs, g_samp, g_true, dvar = map(np.array, (errs, g_samp, g_true, dvar))
        print(f"{T_in:>7}{errs.mean():>12.5f}"
              f"{g_samp.mean():>13.5f} ({100*(g_samp>0).mean():4.0f}%)"
              f"{g_true.mean():>11.5f} ({100*(g_true>0).mean():4.0f}%)"
              f"{dvar.mean():>18.6f}", flush=True)


if __name__ == "__main__":
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n_trials)
