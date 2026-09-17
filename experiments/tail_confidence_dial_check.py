"""Does the tail-view confidence dial behave the way the theory says it
must -- bounded, controlled, roughly-linear cost/benefit in phi -- and does
it degrade gracefully (not catastrophically) when the belief is WRONG?

Two things are provable about a confidence-weighted view, from the
companion paper's own machinery, without needing to know if the view is
correct: phi=0 reproduces the no-view anchor EXACTLY (the fixed-point
property), and the cost of a wrong view is bounded and grows with phi (the
smoothness theorem's Lipschitz bound + the Bregman-divergence local
expansion give a linear-in-phi departure and a quadratic-in-phi cost). What
is NOT provable in general is that raising phi improves realized
performance -- that depends on whether the view is actually right.

This tests both halves directly. w_phi = normalize((1-phi)*w_gauss +
phi*w_tail), the same style of linear confidence blend the paper already
uses for its correlation dial (blend_correlation). Two markets:
  TRUE tail market (nu=2.2, the oracle case that worked) -- does raising
    phi toward a CORRECT belief improve ES99 roughly monotonically?
  FALSE tail market (nu=inf, actually Gaussian, no real tail dependence
    at all) -- does over-confidently dialing in a WRONG belief (phi
    rising toward 1 with nu_believed=2.2) cost something small and
    controlled, or does it blow up?
A separate sweep checks robustness to the believed nu itself being only
approximately right (not the exact true value), at phi=1 (full confidence).
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from tail_dependence_win_check import (
    draw_returns, es_q, independent_repair_target, make_market,
)

from allocation import HierarchicalRiskParity
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.transport import transport_weights, transport_weights_lowrank_blockt


def run_trial(rng, n_clusters, per_cluster, loading, idio_lo, idio_hi,
             T_in, T_out, nu_true_market, nu_believed, phis, M_seeds):
    n = n_clusters * per_cluster
    B, idio = make_market(rng, n_clusters, per_cluster, loading, idio_lo, idio_hi)
    nus_market = np.full(n_clusters, nu_true_market)   # what actually happens
    nus_belief = np.full(n_clusters, nu_believed)       # what the investor believes

    Rin = draw_returns(rng, B, idio, T_in, nus_market)
    Rout = draw_returns(rng, B, idio, T_out, nus_market)

    hrp = HierarchicalRiskParity(); hrp.fit(Rin)
    w_base = hrp.weights_
    corr_hat = np.corrcoef(Rin, rowvar=False)
    theta = independent_repair_target(w_base)

    seeds = rng.standard_normal((M_seeds, n))
    w_gauss = transport_weights(theta, corr_hat, seeds)

    seeds_factor = rng.standard_normal((M_seeds, n_clusters))
    seeds_idio = rng.standard_normal((M_seeds, n))
    seeds_chi2 = rng.chisquare(nu_believed, size=(M_seeds, n_clusters))
    w_tail = transport_weights_lowrank_blockt(
        theta, B, idio, seeds_factor, seeds_idio, seeds_chi2, nus_belief)

    es_anchor = es_q(Rout @ w_gauss, 0.01)  # phi=0 reference point
    gains = {}
    for phi in phis:
        w_phi = (1 - phi) * w_gauss + phi * w_tail
        w_phi = np.clip(w_phi, 0.0, None)
        w_phi = w_phi / w_phi.sum()
        gains[phi] = es_q(Rout @ w_phi, 0.01) - es_anchor
    return gains


def sweep_phi(n_trials, nu_true_market, nu_believed, label, seed):
    phis = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng_master = np.random.default_rng(seed)
    rows = {phi: [] for phi in phis}
    t0 = time.time()
    for _ in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        gains = run_trial(rng, 3, 6, 1.1, 0.4, 1.6, 400, 5000,
                          nu_true_market, nu_believed, phis, 1 << 14)
        for phi in phis:
            rows[phi].append(gains[phi])
    print(f"{label} (n={n_trials}, {time.time()-t0:.0f}s)")
    print(f"  {'phi':>6}{'mean gain vs phi=0':>20}{'win% vs phi=0':>16}")
    for phi in phis:
        arr = np.array(rows[phi])
        print(f"  {phi:>6.2f}{arr.mean():>20.5f}{100*(arr>0).mean():>16.1f}")
    print()


def sweep_believed_nu(n_trials, nu_true_market, believed_grid, seed):
    rng_master = np.random.default_rng(seed)
    print(f"believed-nu robustness at phi=1, true market nu={nu_true_market} "
          f"(n={n_trials})")
    print(f"  {'nu_believed':>12}{'mean gain vs gauss':>20}{'win%':>8}")
    for nu_believed in believed_grid:
        rows = []
        rng2 = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        for _ in range(n_trials):
            rng = np.random.default_rng(rng2.integers(0, 2**31 - 1))
            gains = run_trial(rng, 3, 6, 1.1, 0.4, 1.6, 400, 5000,
                              nu_true_market, nu_believed, [1.0], 1 << 14)
            rows.append(gains[1.0])
        arr = np.array(rows)
        print(f"  {nu_believed:>12.1f}{arr.mean():>20.5f}{100*(arr>0).mean():>8.1f}")
    print()


def main(n_trials=200, seed=0):
    sweep_phi(n_trials, nu_true_market=2.2, nu_believed=2.2,
             label="TRUE tail market, correct belief (nu=2.2)", seed=seed)
    sweep_phi(n_trials, nu_true_market=np.inf, nu_believed=2.2,
             label="FALSE tail belief (market is actually Gaussian)", seed=seed + 1)
    sweep_believed_nu(n_trials, nu_true_market=2.2,
                      believed_grid=[2.2, 3.0, 5.0, 10.0, 30.0], seed=seed + 2)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
