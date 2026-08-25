"""Thurstone repair of hierarchical portfolios: known-truth test.

HRP / Schur build a portfolio from a covariance, but the recursive-bisection
construction only ever consults *within-current-block* covariance: cross-block
covariance is discarded at (gamma=0) or partially retained through
(gamma>0) the Schur complement. This asks whether recovering that discarded
dependence -- and, more sharply, dependence a covariance can never see at all
(tail contagion) -- through a Thurstone repair actually helps, out of sample.

w_base   = HRP or Schur weights (built from the in-sample covariance only).
C_used   = a reconstruction of "the dependence the recursion consulted":
           a NESTED (ultrametric) factor model implied by the same
           median-bisection tree HRP/Schur used, with one latent factor per
           retained internal split and `max_factors` controlling how much of
           the tree survives. max_factors=0 is the independent reference
           (the naive "polish from scratch" baseline); max_factors=big
           approaches the full tree.
theta    = calibrate_factor(w_base, C_used)      -- invert under what was used
w_repair = race(theta, target law)               -- re-run under more

Two repair targets isolate two different things:
  "gauss"   -- race under the full in-sample LINEAR correlation (still no
              tail dependence). Tests whether completing discarded ordinary
              correlation alone helps.
  "bowling" -- race under the true in-sample bowling returns (bootstrap
              resampled). Tests whether completing genuine TAIL dependence
              -- which no covariance, however fully used, can represent --
              helps beyond what "gauss" already gets.

Known-truth market: bowling_sim.py (pins in tight clusters that cascade
together -- genuine all-or-nothing tail contagion a Gaussian view cannot
encode). Evaluated out-of-sample by ES95 on a held-out half.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from bowling_sim import generate

from allocation import EqualWeight, HierarchicalRiskParity, MinimumVariance, SchurComplementary
from allocation._thurstone.covariance import cov_to_corr
from allocation._thurstone.factor import calibrate_factor, hermite_nodes, winprobs_factor
from allocation._thurstone.transport import race_weights, transport_weights

try:
    from winning.factor.core import qmc_nodes
except ImportError:                                     # pragma: no cover
    qmc_nodes = None


# --------------------------------------------------------------------- nodes

def _nodes(k: int):
    """Gauss-Hermite up to k=4; scrambled-Sobol beyond (matches winning's own
    dispatch in winning/methods/native.py). Q and the Sobol budget are cut
    down from production defaults -- this is an exploratory sweep over many
    (config x factor-count) cells, not a single production calibration."""
    if k <= 4 or qmc_nodes is None:
        return hermite_nodes(max(k, 1), Q=9)
    return qmc_nodes(k, m=10)


# ---------------------------------------------------------- nested-tree fit

def nested_tree_factor(corr: np.ndarray, order: np.ndarray, max_factors):
    """One latent factor per retained internal node of the median-bisection
    tree over `order` (the SAME tree HRP/Schur recurse over -- see
    `allocation._schur.coupling._bisection`). Node gain is the covariance
    mass newly explained at that split (over what ancestors already
    explain); nodes are kept by gain, highest first, up to `max_factors`
    (None = every node -- the full tree). Returns (V, D) for the k-factor
    engine; max_factors=0 gives V=zeros (the independent reference)."""
    n = len(order)
    order = list(int(i) for i in order)
    nodes = []  # (gain, members, c2)

    def recurse(members, floor):
        if len(members) <= 1:
            return
        mid = len(members) // 2
        left, right = members[:mid], members[mid:]
        cross = corr[np.ix_(left, right)]
        c2 = max(0.0, float(cross.mean()) - floor)
        if c2 > 1e-10:
            nodes.append((c2 * len(left) * len(right), list(members), c2))
        recurse(left, floor + c2)
        recurse(right, floor + c2)

    recurse(order, 0.0)
    nodes.sort(key=lambda t: -t[0])
    if max_factors is not None:
        nodes = nodes[:max_factors]
    k = max(1, len(nodes))
    V = np.zeros((n, k))
    for j, (gain, members, c2) in enumerate(nodes):
        V[members, j] = np.sqrt(c2)
    D = np.clip(np.diag(corr) - (V ** 2).sum(axis=1), 1e-3, None)
    return V, D


def repair(w_base, corr_in, order, max_factors, seeds, Lstd_in, boot_idx):
    """theta = calibrate_factor(w_base, C_used); repair toward the Gaussian
    full correlation, and toward the true (resampled) bowling returns.

    The race is loss-side (argmin wins, larger = worse -- see the paper's
    "Centring" remark), so `Lstd_in` must be *losses* (negated, standardized
    in-sample returns), not returns directly. Feeding returns instead of
    losses silently credits the biggest losers in every crash scenario."""
    V, D = nested_tree_factor(corr_in, order, max_factors)
    F, W = _nodes(V.shape[1])
    theta = calibrate_factor(w_base, V, D, F, W, points=251)
    w_gauss = transport_weights(theta, corr_in, seeds)
    w_bowl = race_weights(theta[None, :] + Lstd_in[boot_idx])
    return w_gauss, w_bowl, theta


# --------------------------------------------------------------- baselines

def w_equal(n):
    e = EqualWeight()
    e.fit(np.zeros((2, n)))
    return e.weights_


def w_minvar(Rin):
    m = MinimumVariance()
    m.fit(Rin)
    return m.weights_


def es95(returns):
    q = np.quantile(returns, 0.05)
    return float(returns[returns <= q].mean())          # mean of worst 5% (negative = loss)


# ------------------------------------------------------------------- study

FACTOR_GRID = [0, 1, 2, 3, 5, 8]
GAMMA = 0.5
M_SEEDS = 1 << 14


def run_one(seed, n=20, T=1000, k=3, sd=6.0, rng_seed=123):
    R, *_ = generate(n=n, T=T, seed=seed, k=k, sd=sd)
    Tin = T // 2
    Rin, Rout = R[:Tin], R[Tin:]
    # Robust (median/MAD) centering and scale, not mean/std: bowling returns
    # are extremely leptokurtic (rare huge-displacement cascades drag the
    # MEAN far from the typical value, and inflate the std 50-80x above the
    # quiet-day scale), so mean/std standardizing -- or worse, mixing mean
    # centering with MAD scaling -- crushes quiet days to near-zero noise or
    # injects a spurious common offset; theta (calibrated under a
    # unit-variance reference) then dominates and the race collapses onto
    # one name almost every round. This is the scale-matching caveat the
    # theory already flags ("centering an arbitrary sampler is not enough").
    med = np.median(Rin, axis=0, keepdims=True)
    mad = np.median(np.abs(Rin - med), axis=0) * 1.4826
    Lstd_in = -(Rin - med) / mad                         # loss-side (see `repair`)

    hrp = HierarchicalRiskParity(); hrp.fit(Rin)
    sch = SchurComplementary(gamma=GAMMA, keep_monotonic=False); sch.fit(Rin)

    rng = np.random.default_rng(rng_seed)
    seeds = rng.standard_normal((M_SEEDS, n))
    boot_idx = rng.integers(0, Tin, M_SEEDS)

    rows = {}
    rows[("equal", None)] = w_equal(n)
    rows[("minvar", None)] = w_minvar(Rin)
    for name, est in (("hrp", hrp), (f"schur(g={GAMMA})", sch)):
        cov_in = np.asarray(est._cov_estimator.covariance_, dtype=float)
        corr_in = cov_to_corr(cov_in)
        order = est.order_
        w_base = est.weights_
        rows[(name, "base")] = w_base
        for mf in FACTOR_GRID:
            w_g, w_b, _ = repair(w_base, corr_in, order, mf, seeds, Lstd_in, boot_idx)
            rows[(name, f"gauss[{mf}]")] = w_g
            rows[(name, f"bowl[{mf}]")] = w_b

    return {key: es95(Rout @ w) for key, w in rows.items()}


def main():
    configs = [(s, sd) for s in range(5) for sd in (4.0, 6.0, 8.0, 10.0)]
    agg = {}
    for seed, sd in configs:
        result = run_one(seed, sd=sd)
        for key, val in result.items():
            agg.setdefault(key, []).append(val)

    def mean(key):
        return float(np.mean(agg[key]))

    print(f"{'markets':>8}: {len(configs)}  (seeds x sd in 4,6,8,10; k=3 clusters)\n")
    print(f"{'baseline':<26}{'ES95 (mean, out-of-sample)':>28}")
    print(f"{'equal weight':<26}{mean(('equal', None)):>28.4f}")
    print(f"{'minimum variance':<26}{mean(('minvar', None)):>28.4f}")

    for name in ("hrp", f"schur(g={GAMMA})"):
        print(f"\n--- {name} ---")
        print(f"{'':<10}{'base':>10}" + "".join(f"{'F='+str(mf):>10}" for mf in FACTOR_GRID))
        for target in ("gauss", "bowl"):
            base = mean((name, "base"))
            vals = [mean((name, f"{target}[{mf}]")) for mf in FACTOR_GRID]
            print(f"{target:<10}{base:>10.4f}" + "".join(f"{v:>10.4f}" for v in vals))

        base = mean((name, "base"))
        best_bowl_mf = max(FACTOR_GRID, key=lambda mf: mean((name, f"bowl[{mf}]")))
        best_bowl = mean((name, f"bowl[{best_bowl_mf}]"))
        best_gauss_mf = max(FACTOR_GRID, key=lambda mf: mean((name, f"gauss[{mf}]")))
        best_gauss = mean((name, f"gauss[{best_gauss_mf}]"))
        print(f"\n  best gauss repair: F={best_gauss_mf}  ES95 {best_gauss:+.4f} vs base {base:+.4f} "
              f"({'better' if best_gauss > base else 'worse'})")
        print(f"  best bowl  repair: F={best_bowl_mf}  ES95 {best_bowl:+.4f} vs base {base:+.4f} "
              f"({'better' if best_bowl > base else 'worse'})")
        print(f"  tail-specific gain (bowl - gauss, at each's own best F): {best_bowl - best_gauss:+.4f}")


if __name__ == "__main__":
    main()
