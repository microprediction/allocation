"""Find a genuine case where tail-dependence repair beats correlation-only
repair -- cleanly, not confounded.

Two earlier attempts both had a real design flaw. The bowling-lab market
(schur_thurstone_repair.py) conflates linear correlation and tail
contagion: tight spatial clusters cause both together, so ordinary
correlation already captures most of what the tail race could add. The
real-data bootstrap target (overnight_repair_validation.py's "hist" arm)
injects genuine tail structure, but a bootstrap resample of a few hundred
rows is itself too noisy to give a clean signal.

A THIRD attempt (kept in git history / earlier in this session, not
reproduced here) used a common scale-mixture Student-t (every asset shares
one mixing draw per scenario) with an equal-weight base -- and found
EXACTLY ZERO difference between the Gaussian and tail repair, bit for bit.
That is not a bug: with an equal-weight base, independent-reference
calibration gives theta=0 for every asset, and scaling every asset by the
SAME positive scalar (the shared mixing variable) never changes which one
has the smallest value in a scenario. A common-scale Student-t can only
matter when abilities are genuinely heterogeneous -- worth recording, since
it is a real, provable degeneracy, not a corner case.

This version isolates the effect properly with the tool built for exactly
this (allocation._thurstone.transport.transport_weights_lowrank_blockt):
a k-factor market, one factor per cluster, EACH factor carrying its own
tail index nu. Because factors are standardized, the LINEAR correlation
contribution is IDENTICAL regardless of nu (the docstring's own claim,
checked below) -- so a Gaussian repair using the correct linear correlation
sees exactly the same dependence structure whether nu is 3 or infinite.
Only the tail-aware repair, using the true per-factor nus, sees the extra
cluster-specific co-crash risk. Base allocator is HRP (the paper's actual
protagonist), which gives genuinely heterogeneous abilities from
heterogeneous idiosyncratic variances even under an otherwise symmetric
cluster design, so the earlier degeneracy does not recur.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from schur_thurstone_repair import _nodes, es95

from allocation import HierarchicalRiskParity
from allocation._thurstone.factor import calibrate_factor
from allocation._thurstone.transport import transport_weights, transport_weights_lowrank_blockt


def make_market(rng, n_clusters, per_cluster, loading, idio_lo, idio_hi):
    n = n_clusters * per_cluster
    B = np.zeros((n, n_clusters))
    for c in range(n_clusters):
        B[c * per_cluster:(c + 1) * per_cluster, c] = loading
    idio = rng.uniform(idio_lo, idio_hi, n)
    return B, idio


def draw_returns(rng, B, idio, T, nus):
    """nus: length-k array, finite = Student-t factor (that many df),
    np.inf = Gaussian factor. Factors are variance-standardized regardless
    of nu, so the LINEAR covariance contribution B @ B.T is identical for
    any nus -- only the joint tail differs."""
    n, k = B.shape
    Z = rng.standard_normal((T, k))
    finite = np.isfinite(nus)
    if np.any(finite):
        W = rng.chisquare(nus[finite], size=(T, finite.sum()))
        Z[:, finite] = Z[:, finite] * np.sqrt(nus[finite] / W) * np.sqrt(
            (nus[finite] - 2.0) / nus[finite])
    return Z @ B.T + np.sqrt(idio) * rng.standard_normal((T, n))


def independent_repair_target(w_base, points=251):
    n = len(w_base)
    V = np.zeros((n, 1))
    D = np.ones(n)
    Fq, Wq = _nodes(1)
    return calibrate_factor(w_base, V, D, Fq, Wq, points=points)


def joint_crash_prob(r_assets, q):
    return float(np.all(r_assets < -q, axis=1).mean())


def es_q(returns, q):
    thresh = np.quantile(returns, q)
    tail = returns[returns <= thresh]
    return float(tail.mean()) if len(tail) else float(thresh)


def run_trial(rng, n_clusters, per_cluster, loading, idio_lo, idio_hi,
             T_in, T_out, nu_tail, M_seeds):
    n = n_clusters * per_cluster
    B, idio = make_market(rng, n_clusters, per_cluster, loading, idio_lo, idio_hi)
    nus_true = np.full(n_clusters, nu_tail)          # true market: fat-tailed clusters
    nus_gauss = np.full(n_clusters, np.inf)           # what the Gaussian repair assumes

    Rin = draw_returns(rng, B, idio, T_in, nus_true)
    Rout = draw_returns(rng, B, idio, T_out, nus_true)

    hrp = HierarchicalRiskParity(); hrp.fit(Rin)
    w_base = hrp.weights_
    corr_hat = np.corrcoef(Rin, rowvar=False)

    theta = independent_repair_target(w_base)
    seeds = rng.standard_normal((M_seeds, n))
    w_gauss = transport_weights(theta, corr_hat, seeds)

    seeds_factor = rng.standard_normal((M_seeds, n_clusters))
    seeds_idio = rng.standard_normal((M_seeds, n))
    seeds_chi2 = rng.chisquare(nu_tail, size=(M_seeds, n_clusters))
    seeds_idio_scale = np.sqrt(idio)
    w_tail = transport_weights_lowrank_blockt(
        theta, B, idio, seeds_factor, seeds_idio, seeds_chi2, nus_true)

    r_base, r_gauss, r_tail = Rout @ w_base, Rout @ w_gauss, Rout @ w_tail
    es99_base, es99_gauss, es99_tail = es_q(r_base, 0.01), es_q(r_gauss, 0.01), es_q(r_tail, 0.01)
    return (es99_gauss - es99_base, es99_tail - es99_base, es99_tail - es99_gauss)


def main(n_trials=200, n_clusters=3, per_cluster=6, loading=1.1,
        idio_lo=0.4, idio_hi=1.6, T_in=400, T_out=3000, nu_tail=3.0,
        M_seeds=1 << 14, seed=0):
    rng_master = np.random.default_rng(seed)
    gauss_gain, tail_gain, tail_vs_gauss = [], [], []
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        g, t, tvg = run_trial(rng, n_clusters, per_cluster, loading, idio_lo, idio_hi,
                              T_in, T_out, nu_tail, M_seeds)
        gauss_gain.append(g); tail_gain.append(t); tail_vs_gauss.append(tvg)
        if i % 50 == 0:
            print(f"{i}/{n_trials} ({time.time()-t0:.0f}s)", flush=True)
    gauss_gain, tail_gain, tail_vs_gauss = map(np.array, (gauss_gain, tail_gain, tail_vs_gauss))
    n = n_clusters * per_cluster
    print(f"\ndone in {time.time()-t0:.0f}s  (n={n}, {n_clusters} clusters x {per_cluster}, "
          f"T_in={T_in}, nu_tail={nu_tail})\n")
    print(f"gauss repair vs base:  mean={gauss_gain.mean():.5f}  win%={100*(gauss_gain>0).mean():.1f}")
    print(f"tail  repair vs base:  mean={tail_gain.mean():.5f}  win%={100*(tail_gain>0).mean():.1f}")
    print(f"tail repair vs GAUSS repair (the incremental tail-dependence effect):")
    print(f"                       mean={tail_vs_gauss.mean():.5f}  win%={100*(tail_vs_gauss>0).mean():.1f}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
