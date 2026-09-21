"""Re-run the real-data tail-dependence repair (bootstrap-resampled
historical returns as the target -- no parametric tail assumption, the
"estimate the tail structure directly" approach) with the fix the synthetic
oracle check just found: evaluate on a DEEP-tail statistic (ES99 / joint
multi-name crash probability), not ES95.

overnight_repair_validation.py already found REIT rewards the bootstrap
("hist") repair at ES95 (62-63% win rate) while S&P 500 loses (33-39%).
That used HRP/Schur as the base, correctly (heterogeneous abilities, not
equal-weight -- the earlier degeneracy). The one thing untested was whether
ES95 was even the right instrument; the synthetic check showed the same
setup can look like a null at ES95 and a clear win at ES99. Same real
panels, same repair mechanism, sharper metric.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95, nested_tree_factor
from overnight_repair_validation import UNIVERSES, load_universe

from allocation import HierarchicalRiskParity
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import race_weights, transport_weights


def es_q(returns, q):
    thresh = np.quantile(returns, q)
    tail = returns[returns <= thresh]
    return float(tail.mean()) if len(tail) else float(thresh)


def joint_crash_prob(r_assets, q):
    return float(np.all(r_assets < -q, axis=1).mean())


def run_trial(rng, panels, universe, F):
    R_full, cols = panels[universe]
    T_total, N_total = R_full.shape
    max_n = min(30, N_total)
    n_assets = int(rng.integers(12, max_n + 1))
    asset_idx = rng.choice(N_total, size=n_assets, replace=False)
    T_in = int(rng.integers(250, 750))
    T_out = 250
    max_start = T_total - T_in - T_out
    if max_start <= 0:
        return None
    start = int(rng.integers(0, max_start))
    Rin = R_full[start:start + T_in][:, asset_idx]
    Rout = R_full[start + T_in:start + T_in + T_out][:, asset_idx]

    est = HierarchicalRiskParity(); est.fit(Rin)
    w_base = est.weights_
    corr_in = cov_to_corr(np.asarray(est._cov_estimator.covariance_, dtype=float))
    V, D = nested_tree_factor(corr_in, est.order_, F)
    Fq, Wq = _nodes(V.shape[1])
    theta = calibrate_factor(w_base, V, D, Fq, Wq, points=251)

    med = np.median(Rin, axis=0, keepdims=True)
    mad = np.median(np.abs(Rin - med), axis=0) * 1.4826
    mad = np.maximum(mad, 1e-8)
    Lstd_in = -(Rin - med) / mad
    boot_idx = rng.integers(0, T_in, 1 << 14)
    seeds = rng.standard_normal((1 << 14, n_assets))

    w_gauss = transport_weights(theta, corr_in, seeds)
    w_hist = race_weights(theta[None, :] + Lstd_in[boot_idx])

    r_base, r_gauss, r_hist = Rout @ w_base, Rout @ w_gauss, Rout @ w_hist
    out = {
        "es95_gauss": es95(r_gauss) - es95(r_base),
        "es95_hist": es95(r_hist) - es95(r_base),
        "es99_gauss": es_q(r_gauss, 0.01) - es_q(r_base, 0.01),
        "es99_hist": es_q(r_hist, 0.01) - es_q(r_base, 0.01),
    }
    return out


def main(n_trials=400, F=0, seed=0):
    panels = {name: load_universe(name) for name in UNIVERSES}
    results = {uni: {k: [] for k in ("es95_gauss", "es95_hist", "es99_gauss", "es99_hist")}
              for uni in UNIVERSES}
    rng_master = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(n_trials):
        for uni in UNIVERSES:
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            out = run_trial(rng, panels, uni, F)
            if out is not None:
                for k, v in out.items():
                    results[uni][k].append(v)
        if i % 50 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s (F={F})\n")
    print(f"{'universe':<8}{'metric':<12}{'target':<8}{'mean':>10}{'win%':>8}{'n':>6}")
    for uni in UNIVERSES:
        for metric in ("es95", "es99"):
            for target in ("gauss", "hist"):
                arr = np.array(results[uni][f"{metric}_{target}"])
                print(f"{uni:<8}{metric:<12}{target:<8}{arr.mean():>10.5f}"
                      f"{100*(arr>0).mean():>8.1f}{len(arr):>6}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
