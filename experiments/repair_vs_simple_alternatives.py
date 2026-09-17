"""Does the Thurstone race repair beat something MUCH simpler that also
"adds correlation information to a naive base"?

Every result so far compared repaired-HRP against base-HRP. That establishes
the repair helps HRP, but not that the specific race machinery is doing
anything a simpler correlation-aware construction couldn't do just as well.
This runs, head to head, on the SAME real trials:

  hrp             -- the naive base
  hrp_repaired    -- HRP + the Thurstone F=0 correlation repair (the thing
                     tested all session)
  minvar_shrunk   -- MinimumVariance(shrinkage=0.7), long-only, run DIRECTLY
                     (no race, no repair -- just a sensibly regularized
                     optimizer, established earlier to be the base-method
                     fix that works)
  risk_parity     -- equal-risk-contribution, run DIRECTLY
  linear_blend    -- w = 0.5*w_hrp + 0.5*w_minvar_shrunk, the simplest
                     possible "repair": just average the two portfolios'
                     WEIGHTS, no race, no calibration, nothing clever

If hrp_repaired doesn't beat minvar_shrunk / risk_parity / linear_blend on
the same trials, the honest conclusion is that the race machinery isn't
earning its complexity here -- a naive practitioner would do just as well
switching to a regularized optimizer or even just averaging two portfolios.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95
from overnight_repair_validation import UNIVERSES, load_universe

from allocation import BoxConstrained, HierarchicalRiskParity, MinimumVariance, RiskParity
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights
from precise import LedoitWolfCovariance

M_SEEDS = 1 << 14
POINTS = 251


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

    hrp = HierarchicalRiskParity(); hrp.fit(Rin)
    w_hrp = hrp.weights_
    corr_in = cov_to_corr(np.asarray(hrp._cov_estimator.covariance_, dtype=float))
    seeds = rng.standard_normal((M_SEEDS, n_assets))
    try:
        w_repaired = independent_repair(w_hrp, corr_in, seeds)
    except Exception:
        return None

    mv = BoxConstrained(MinimumVariance(shrinkage=0.7))
    mv.fit(Rin)
    w_mv = np.clip(mv.weights_, 0.0, None)
    w_mv = w_mv / w_mv.sum() if w_mv.sum() > 0 else np.full(n_assets, 1 / n_assets)

    mv_lw = BoxConstrained(MinimumVariance(covariance_estimator=LedoitWolfCovariance()))
    mv_lw.fit(Rin)
    w_mv_lw = np.clip(mv_lw.weights_, 0.0, None)
    w_mv_lw = w_mv_lw / w_mv_lw.sum() if w_mv_lw.sum() > 0 else np.full(n_assets, 1 / n_assets)

    rp = RiskParity(); rp.fit(Rin)
    w_rp = rp.weights_

    w_blend = 0.5 * w_hrp + 0.5 * w_mv
    w_blend = w_blend / w_blend.sum()

    portfolios = {
        "hrp": w_hrp, "hrp_repaired": w_repaired,
        "minvar_shrunk": w_mv, "minvar_ledoitwolf": w_mv_lw,
        "risk_parity": w_rp, "linear_blend": w_blend,
    }
    es = {name: es95(Rout @ w) for name, w in portfolios.items()}
    turnover_from_hrp = {name: float(np.abs(w - w_hrp).sum()) for name, w in portfolios.items()}
    return es, turnover_from_hrp


def main(n_trials=800, seed=0):
    panels = {name: load_universe(name) for name in UNIVERSES}
    names = ["hrp", "hrp_repaired", "minvar_shrunk", "minvar_ledoitwolf",
             "risk_parity", "linear_blend"]
    es_vals = {name: [] for name in names}
    turnover_vals = {name: [] for name in names}
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        out = run_trial(rng, panels)
        if out is not None:
            es, turnover = out
            for name in names:
                es_vals[name].append(es[name])
                turnover_vals[name].append(turnover[name])
        if i % 100 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    es_vals = {name: np.array(v) for name, v in es_vals.items()}
    turnover_vals = {name: np.array(v) for name, v in turnover_vals.items()}
    n = len(es_vals["hrp"])
    print(f"\ndone in {time.time()-t0:.0f}s, n={n}\n")
    print(f"{'method':<18}{'mean ES95':>12}{'win% vs HRP':>14}{'win% vs repaired':>18}"
          f"{'L1 from HRP':>14}")
    base_hrp = es_vals["hrp"]
    base_rep = es_vals["hrp_repaired"]
    for name in names:
        v = es_vals[name]
        w_hrp = 100 * (v > base_hrp).mean()
        w_rep = 100 * (v > base_rep).mean()
        print(f"{name:<18}{v.mean():>12.5f}{w_hrp:>14.1f}{w_rep:>18.1f}"
              f"{turnover_vals[name].mean():>14.4f}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 800)
