"""Per-entry, noise-aware correlation shrinkage as the repair TARGET.

The MinimumVariance/MaxDecorrelation-family result (naive_baseline_repair_
check.py) looked like "the repair hurts already-correlation-aware methods."
A follow-up check complicated that story: MinimumVariance's OWN `shrinkage`
parameter defaults to 0.0 (no shrinkage), and sweeping it from 0.0 to 0.9
flips the repair from clearly harmful (41% win rate) to clearly beneficial
(59.5%). But that `shrinkage` is one scalar blending the WHOLE matrix toward
a scaled identity, uniformly -- every entry gets the same treatment
regardless of how reliably it was actually estimated (allocation/convex.py's
`_shrink`).

This builds a genuinely different, entrywise object: for a sample
correlation rho_ij estimated from T observations, the classical
(Fisher-z / delta-method) sampling variance is approximately
Var(rho_ij) ~ (1 - rho_ij^2)^2 / T -- strongly correlated pairs are
estimated MUCH more precisely than near-zero pairs, not just "the same
percentage noisier." A Bayesian-shrinkage weight toward zero,
lambda_ij = T / (T + tau0*(1-rho_ij^2)^2), then shrinks weakly-estimated
(near-zero, unreliable) entries hard while leaving well-estimated
(strongly correlated, reliable) entries almost untouched -- the opposite of
what a uniform scalar shrink does.

This is used as C_full for the repair (the base allocator -- deliberately
left UNSHRUNK, the worst case from the earlier check -- is untouched): does
denoising the repair's OWN target, without touching the base method at all,
turn harm back into benefit?
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95
from overnight_repair_validation import UNIVERSES, load_universe

from allocation import BoxConstrained, HierarchicalRiskParity, InverseVariance, MinimumVariance
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights

M_SEEDS = 1 << 13
POINTS = 251

METHODS = {
    "min_variance_lo (unshrunk)": lambda: BoxConstrained(MinimumVariance(shrinkage=0.0)),
    "inverse_variance": lambda: InverseVariance(),
    "hrp": lambda: HierarchicalRiskParity(),
}


def noise_aware_corr(corr, T, tau0):
    """Shrink each off-diagonal entry toward 0 by its OWN sampling
    precision: lambda_ij = T / (T + tau0*(1-rho_ij^2)^2)."""
    n = corr.shape[0]
    var_proxy = (1.0 - corr**2) ** 2
    lam = T / (T + tau0 * var_proxy)
    out = lam * corr
    np.fill_diagonal(out, 1.0)
    return out


def independent_repair(w_base, corr_target, seeds):
    n = len(w_base)
    V = np.zeros((n, 1))
    D = np.ones(n)
    Fq, Wq = _nodes(1)
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
    return transport_weights(theta, corr_target, seeds)


def run_trial(rng, panels, tau0):
    universe = rng.choice(list(UNIVERSES))
    R_full, cols = panels[universe]
    T_total, N_total = R_full.shape
    max_n = min(25, N_total)
    n_assets = int(rng.integers(12, max_n + 1))
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    T_in = int(rng.integers(250, 450))
    T_out = 125
    max_start = T_total - T_in - T_out
    if max_start <= 0:
        return None
    start = int(rng.integers(0, max_start))
    Rin = R_full[start:start + T_in][:, asset_idx]
    Rout = R_full[start + T_in:start + T_in + T_out][:, asset_idx]
    seeds = rng.standard_normal((M_SEEDS, n_assets))

    out = {}
    for name, ctor in METHODS.items():
        est = ctor()
        est.fit(Rin)
        w_base = np.clip(est.weights_, 0.0, None)
        w_base = w_base / w_base.sum() if w_base.sum() > 0 else np.full(n_assets, 1 / n_assets)
        inner = getattr(est, "estimator", est)
        corr_raw = cov_to_corr(np.asarray(inner._cov_estimator.covariance_, dtype=float))
        corr_denoised = noise_aware_corr(corr_raw, T_in, tau0)

        r_base = Rout @ w_base
        base_es = es95(r_base)
        try:
            w_raw = independent_repair(w_base, corr_raw, seeds)
            w_dn = independent_repair(w_base, corr_denoised, seeds)
        except Exception:
            out[name] = None
            continue
        out[name] = (es95(Rout @ w_raw) - base_es, es95(Rout @ w_dn) - base_es)
    return out


def main(n_trials=300, tau0=150.0, seed=0):
    panels = {name: load_universe(name) for name in UNIVERSES}
    results = {name: {"raw": [], "denoised": []} for name in METHODS}
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        out = run_trial(rng, panels, tau0)
        if out is None:
            continue
        for name, val in out.items():
            if val is not None:
                raw_gain, dn_gain = val
                results[name]["raw"].append(raw_gain)
                results[name]["denoised"].append(dn_gain)
        if i % 25 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s (tau0={tau0})\n")
    print(f"{'method':<28}{'raw mean':>10}{'raw win%':>10}"
          f"{'denoised mean':>15}{'denoised win%':>15}{'n':>6}")
    for name in METHODS:
        raw = np.array(results[name]["raw"])
        dn = np.array(results[name]["denoised"])
        print(f"{name:<28}{raw.mean():>10.5f}{100*(raw>0).mean():>10.1f}"
              f"{dn.mean():>15.5f}{100*(dn>0).mean():>15.1f}{len(raw):>6}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tau0 = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
    main(n, tau0)
