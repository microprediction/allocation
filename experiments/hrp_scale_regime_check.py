"""The genuine n >> T regime: does HRP beat equal-weight, and does the
repair still help HRP, at real scale (thousands of names, a few hundred
observations)?

Every earlier check here used real S&P 500 / REIT data, capped at n<=40 by
the size of the universes available -- nowhere near the regime HRP and this
whole line of work are actually motivated by (companion paper: "the sample
covariance is rank-deficient... dense inversion allocators are undefined").
This pushes there directly with a synthetic k-factor market (known true
loadings, so the "out of sample" evaluation is exact, not itself noisy),
n up to several thousand, T_in a few hundred -- genuinely n >> T.

Nothing dense is ever formed at this scale. HRP needs a covariance array to
slice into blocks (O(n^2) total across the recursion, fine), but the
repair's re-race uses transport_weights_lowrank (O(M n k), not the dense
O(M n^2)) throughout, exactly the machinery this package already built for
"thousands of names" (allocation/_thurstone/transport.py's own docstring).
Calibration is F=0 (independent reference): the trivial one-node case, so
its cost does not depend on the tree/factor structure of the universe at
all, only on n itself, O(n) per calibration step.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95

from allocation import EqualWeight, HierarchicalRiskParity
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights_lowrank

POINTS = 251
M_SEEDS = 1 << 13


def make_universe(rng, n, k_true):
    B = rng.standard_normal((n, k_true)) * 0.30
    psi = rng.uniform(0.5, 1.2, n)
    return B, psi


def draw_returns(rng, B, psi, T):
    n, k = B.shape
    F = rng.standard_normal((T, k))
    return F @ B.T + np.sqrt(psi) * rng.standard_normal((T, n))


def independent_repair_lowrank(w_base, B_true, psi_true, seeds_factor, seeds_idio):
    """F=0 calibration (trivial, no tree), re-race with the KNOWN true
    low-rank structure -- never forms a dense n x n matrix."""
    n = len(w_base)
    Vc = np.zeros((n, 1))
    Dc = np.ones(n)
    Fq, Wq = _nodes(1)
    theta = calibrate_factor(w_base, Vc, Dc, Fq, Wq, points=POINTS)
    return transport_weights_lowrank(theta, B_true, psi_true, seeds_factor, seeds_idio)


def run_trial(rng, n, k_true, T_in, T_out, knn):
    B, psi = make_universe(rng, n, k_true)
    Rin = draw_returns(rng, B, psi, T_in)
    Rout = draw_returns(rng, B, psi, T_out)  # fresh draw, same true market

    eq = EqualWeight(); eq.fit(Rin)
    w_eq = eq.weights_
    es_eq = es95(Rout @ w_eq)

    hrp = HierarchicalRiskParity(knn=knn)
    t0 = time.time()
    hrp.fit(Rin)
    fit_time = time.time() - t0
    w_hrp = hrp.weights_
    es_hrp = es95(Rout @ w_hrp)

    seeds_factor = rng.standard_normal((M_SEEDS, k_true))
    seeds_idio = rng.standard_normal((M_SEEDS, n))
    t0 = time.time()
    w_rep = independent_repair_lowrank(w_hrp, B, psi, seeds_factor, seeds_idio)
    repair_time = time.time() - t0
    es_rep = es95(Rout @ w_rep)

    return es_hrp - es_eq, es_rep - es_hrp, fit_time, repair_time


def main(n_trials=20, seed=0):
    grid = [
        (200, 5, 250, 400, None),
        (1000, 5, 250, 100, 30),
        (2000, 5, 250, 100, 30),
        (5000, 8, 250, 60, 30),
    ]
    print(f"{'n':>6}{'T_in':>6}{'T/n':>7}{'hrp-vs-eq mean':>16}{'hrp-vs-eq win%':>16}"
          f"{'repair mean':>13}{'repair win%':>13}{'fit_s':>8}{'repair_s':>10}", flush=True)
    for n, k_true, T_in, n_trials_here, knn in grid:
        hrp_vs_eq, rep_eff, fit_times, rep_times = [], [], [], []
        rng_master = np.random.default_rng(seed)
        t0 = time.time()
        for _ in range(n_trials_here):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            a, b, ft, rt = run_trial(rng, n, k_true, T_in, 125, knn)
            hrp_vs_eq.append(a); rep_eff.append(b)
            fit_times.append(ft); rep_times.append(rt)
        hrp_vs_eq = np.array(hrp_vs_eq); rep_eff = np.array(rep_eff)
        ratio = T_in / n
        print(f"{n:>6}{T_in:>6}{ratio:>7.2f}{hrp_vs_eq.mean():>16.5f}"
              f"{100*(hrp_vs_eq>0).mean():>16.1f}{rep_eff.mean():>13.5f}"
              f"{100*(rep_eff>0).mean():>13.1f}{np.mean(fit_times):>8.2f}"
              f"{np.mean(rep_times):>10.2f}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
