"""Tail-Consistent Thurstone Portfolio: definition, proof, and demonstrator.

PINNED DEFINITION
-----------------
Drive the Thurstone race with a *centered* joint simulation S of asset
performances (any law -- Gaussian, t, copula, ...; the location is the ability
layer's job, NOT the race's -- see the means caveat below). Run the **loss-side**
race -- winner = the WORST performer, ``argmin`` of (centered) performance -- and
take the win-probability **directly** as the weight (no reciprocal "inversion"):

    w_i = P( X_i = min_j X_j ),   X ~ S, centered.

TAIL CONSISTENCY (the property)
-------------------------------
A sub-cluster G that is comonotone in the lower tail (its members crash together)
behaves as a SINGLE competitor on the loss side: its total weight
``w_G = sum_{i in G} w_i`` converges to the win-probability of one representative
-- independent of |G| -- so per name the cluster is de-weighted and the
decorrelated hedge is up-weighted.

Proof (comonotone limit): if X_i = Z_G for all i in G, then min_{i in G} X_i = Z_G,
so {winner in G} = {Z_G is the global min} -- the win event of a single competitor,
with NO dependence on the head count k. Adding identical members cannot raise w_G.
As lower-tail dependence rises, w_G slides from the head-count value k/(k+m) down to
this two-body value (here, A vs a symmetric cluster -> 1/2). It is a functional of
the tail copula of the minimum, so no covariance summary -- full OR downside -- can
reproduce it.

MEANS CAVEAT
------------
The race must run on CENTERED performances. With non-zero means the worst-performer
race just tracks whichever asset has the lowest drift, swamping the tail-dependence
signal. We subtract per-asset means before racing; in the full estimator the ability
calibration carries the location and the tilt races the centered residual.

This script shows: true-simulation w_cluster -> 1/2 as lower-tail dependence
lambda_L -> 1 (the proof), while the full-Sigma and downside-Sigma summaries do not
(they are blind to tail comonotonicity). Run:

    allocation-py312/bin/python experiments/tail_consistency_demo.py
"""

import numpy as np
from scipy.stats import norm

from allocation._thurstone.covariance import nearest_correlation
from allocation._thurstone.transport import symmetric_sqrt

FIG_PATH = "papers/thurstone-portfolios/figures/tail_consistency.png"


def clayton_block(rng, M, n_b, theta, tail):
    """n_b columns, N(0,1) margins, Clayton copula. tail='lower' -> crash together
    (lower-tail dep lambda_L = 2**(-1/theta)); tail='upper' -> rally together."""
    V = rng.gamma(1.0 / theta, 1.0, size=(M, 1))
    E = rng.exponential(1.0, size=(M, n_b))
    U = (1.0 + E / V) ** (-1.0 / theta)
    if tail == "upper":
        U = 1.0 - U
    return norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))


def downside_corr(R):
    """Co-lower-partial-moment correlation (downside semicovariance)."""
    neg = np.minimum(R - R.mean(0), 0.0)
    S = (neg.T @ neg) / R.shape[0]
    d = np.sqrt(np.diag(S))
    return nearest_correlation(S / np.outer(d, d))


def loss_race(X):
    """Loss-side race on CENTERED X: worst performer wins; weight = win-prob."""
    Xc = X - X.mean(0)
    idx = np.argmin(Xc, axis=1)
    c = np.bincount(idx, minlength=X.shape[1]).astype(float)
    return c / c.sum()


def cluster_weights(rng, M, n_b, theta, tail, seeds):
    """w_cluster under the true simulation, the full-Sigma summary, and the
    downside-Sigma summary (Gaussian races driven by each summary matrix)."""
    A = rng.standard_normal((M, 1))
    R = np.concatenate([A, clayton_block(rng, M, n_b, theta, tail)], axis=1)
    C_full = nearest_correlation(np.corrcoef(R, rowvar=False))
    C_low = downside_corr(R)
    w_true = loss_race(R)
    w_full = loss_race(seeds @ symmetric_sqrt(C_full))
    w_low = loss_race(seeds @ symmetric_sqrt(C_low))
    return w_true[1:].sum(), w_full[1:].sum(), w_low[1:].sum()


def main():
    rng = np.random.default_rng(0)
    M, n_b = 150_000, 99
    thetas = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    lam = 2.0 ** (-1.0 / thetas)            # lower-tail dependence coefficient
    seeds = rng.standard_normal((M, 1 + n_b))

    rows = {"true": [], "full": [], "low": []}
    print(f"crash-together cluster (A + {n_b} B's), loss-side race, centered")
    print(f"{'lambda_L':>9} {'w_clust true':>13} {'full-Sigma':>11} {'lower-Sigma':>12}")
    for th, lm in zip(thetas, lam):
        t, f, l = cluster_weights(rng, M, n_b, th, "lower", seeds)
        rows["true"].append(t); rows["full"].append(f); rows["low"].append(l)
        print(f"{lm:>9.3f} {t:>13.4f} {f:>11.4f} {l:>12.4f}")
    print(f"\nproof predicts w_cluster(true) -> 0.5 as lambda_L -> 1 (cluster = one horse)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"\n(matplotlib unavailable: {e}; skipping figure)")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(lam, rows["true"], "o-", lw=2, color="#1b5e20", label="true simulation (sees the tail copula)")
    ax.plot(lam, rows["low"], "s--", lw=1.8, color="#ef6c00", label="downside-$\\Sigma$ summary (Thing 1 only)")
    ax.plot(lam, rows["full"], "^:", lw=1.8, color="#90a4ae", label="full-$\\Sigma$ summary (tail-blind)")
    ax.axhline(0.5, color="#b71c1c", lw=1.2, ls="-.", label="two-body limit (proof): cluster = one horse")
    ax.set_xlabel("lower-tail dependence  $\\lambda_L$  (cluster crashes together)")
    ax.set_ylabel("$w_{\\mathrm{cluster}}$  (total weight on the 99-name cluster)")
    ax.set_title("Tail-Consistent Thurstone: the co-crashing cluster de-duplicates to one horse")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)
    print(f"\nfigure -> {FIG_PATH}")


if __name__ == "__main__":
    main()
