"""Does the simple (F=0) repair help the allocators people actually use --
equal-weight and inverse-variance -- not just HRP/Schur?

The gamma-blend result only matters for Schur at high gamma, and almost
nobody runs Schur at gamma near 1; most practitioners use plain HRP (gamma=0,
where the uncomplicated F=0 repair already worked cleanly, no blend needed)
or something simpler still. This asks the more broadly relevant question:
take the correlation-blind baselines equal-weight and inverse-variance
(``1/n`` and ``1/sigma_i^2`` -- neither ever looks at a correlation, so
there is no tree to read a reference off of, and none is needed), calibrate
under pure independence (F=0's trivial case: V=0, D=1), and re-race under
the real correlation. Does the SAME race-reallocation effect that dominated
the HRP/Schur result show up here too?

Equal-risk-contribution (RiskParity) is included as a contrast case: unlike
equal-weight and inverse-variance, it already uses the full covariance (its
risk contributions are correlation-aware), so by the same logic that broke
F=0 for high-gamma Schur, the independent-reference repair should help it
LESS, or not at all -- a prediction this script also checks.

Real S&P 500 / REIT sub-portfolios, same design as overnight_repair_validation
.py. Also logs mean return alongside ES95, to check the earlier finding (the
repair moves volatility/tail risk, not expected return) generalizes to these
baselines too.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95
from overnight_repair_validation import UNIVERSES, load_universe

from allocation import (
    BoxConstrained, EqualWeight, FactorMaximumDiversification, FactorMinimumVariance,
    HierarchicalRiskParity, InverseVariance, MaximumDecorrelation, MaximumDiversification,
    MinimumVariance, RiskParity, SchurComplementary,
)
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights

M_SEEDS = 1 << 14
POINTS = 251

# MinimumVariance / MaximumDiversification / MaximumDecorrelation / the two
# factor variants all short by default (weights can be negative), which the
# repair cannot consume -- w must lie in the open simplex, interpreted as
# win probabilities. Box-constrain them long-only first; this changes what
# they optimize (a constrained problem, not the same portfolio), so their
# "base" numbers below are the constrained versions, not the textbook ones.
METHODS = {
    "equal_weight": lambda: EqualWeight(),
    "inverse_variance": lambda: InverseVariance(),
    "risk_parity": lambda: RiskParity(),
    "hrp": lambda: HierarchicalRiskParity(),
    "schur_0.5": lambda: SchurComplementary(gamma=0.5, keep_monotonic=False),
    "min_variance_lo": lambda: BoxConstrained(MinimumVariance()),
    "max_diversification_lo": lambda: BoxConstrained(MaximumDiversification()),
    "max_decorrelation_lo": lambda: BoxConstrained(MaximumDecorrelation()),
    "factor_min_variance_lo": lambda: BoxConstrained(FactorMinimumVariance()),
    "factor_max_diversification_lo": lambda: BoxConstrained(FactorMaximumDiversification()),
}


def _cov_of(est):
    """covariance_ lives on the wrapped estimator for BoxConstrained."""
    inner = getattr(est, "estimator", est)
    return np.asarray(inner._cov_estimator.covariance_, dtype=float)


def independent_repair(w_base, corr_full, seeds):
    n = len(w_base)
    V = np.zeros((n, 1))
    D = np.ones(n)
    Fq, Wq = _nodes(1)
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=POINTS)
    return transport_weights(theta, corr_full, seeds)


def run_trial(rng, panels):
    universe = rng.choice(list(UNIVERSES))
    R_full, cols = panels[universe]
    T_total, N_total = R_full.shape
    max_n = min(30, N_total)
    n_assets = int(rng.integers(12, max_n + 1))
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    T_in = int(rng.integers(250, 750))
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
        corr_in = cov_to_corr(_cov_of(est))
        r_base = Rout @ w_base
        base_es = es95(r_base)
        base_mean = float(r_base.mean())
        try:
            w_rep = independent_repair(w_base, corr_in, seeds)
            r_rep = Rout @ w_rep
            rep_es = es95(r_rep)
            rep_mean = float(r_rep.mean())
            out[name] = (rep_es - base_es, rep_mean - base_mean)
        except Exception:
            out[name] = None
    return out


def main(n_trials=300, seed=0):
    panels = {name: load_universe(name) for name in UNIVERSES}
    results = {name: {"es_gain": [], "mean_gain": []} for name in METHODS}
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        out = run_trial(rng, panels)
        if out is None:
            continue
        for name, val in out.items():
            if val is not None:
                es_gain, mean_gain = val
                results[name]["es_gain"].append(es_gain)
                results[name]["mean_gain"].append(mean_gain)
        if i % 25 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s\n")
    print(f"{'method':<18}{'ES95 mean gain':>16}{'ES95 win%':>12}"
          f"{'return win%':>14}{'n':>6}")
    for name in METHODS:
        es_g = np.array(results[name]["es_gain"])
        mn_g = np.array(results[name]["mean_gain"])
        print(f"{name:<18}{es_g.mean():>16.5f}{100*(es_g>0).mean():>12.1f}"
              f"{100*(mn_g>0).mean():>14.1f}{len(es_g):>6}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
