"""Tilt HRP toward only the covariance it deliberately leaves out.

HRP's recursion never reads the covariance between the two halves of a split,
so the information it discards is the cross-cluster part. Tilting it toward
the FULL estimated correlation is therefore doubly wrong: it re-supplies what
HRP already used, with fresh estimation noise, and in a low-rank
approximation it actively discards the within-cluster structure HRP captured.

The targeted version keeps the within-cluster structure and adds only the
cross-cluster part, damped by phi:

    V_tilt = [ group loadings , sqrt(phi) * global loadings ]

so phi = 0 races under the block structure alone, which is HRP's own view of
the world, and phi = 1 adds the global factors it never looked at.
"""
import sys, time
import numpy as np
import winning
from huge_universe import market, sample, true_risk, hrp_big

PHIS = (0.0, 0.25, 0.5, 1.0)


def split_factors(X, labels, k_global=3):
    """Group loadings (what HRP used) and global loadings (what it did not)."""
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1); sd[sd == 0] = 1.0
    Z = Xc / sd
    T, n = Z.shape

    # global: top-k of the T x T gram, so the n x n is never formed
    G = (Z @ Z.T) / (T - 1)
    ev, U = np.linalg.eigh(G)
    idx = np.argsort(ev)[::-1][:k_global]
    lam = np.maximum(ev[idx], 1e-12)
    Vg = (Z.T @ U[:, idx]) / np.sqrt(lam) / np.sqrt(T - 1) * np.sqrt(lam)

    # group: one column per cluster, loading = mean correlation with its own
    # cluster's average, which is the block structure HRP's splits consume
    resid = Z - (Z @ Vg) @ np.linalg.pinv(Vg.T @ Vg) @ Vg.T if False else Z
    groups = np.unique(labels)
    Vb = np.zeros((n, len(groups)))
    for j, g in enumerate(groups):
        m = labels == g
        avg = resid[:, m].mean(1)
        s = avg.std(ddof=1)
        if s > 0:
            Vb[m, j] = (resid[:, m].T @ avg) / ((T - 1) * s) / np.maximum(
                resid[:, m].std(0, ddof=1), 1e-12) * s
    return Vg, Vb


def normalise(Vg, Vb, phi):
    """Stack the kept and tilted loadings and rescale to a unit diagonal."""
    V = np.hstack([Vb, np.sqrt(phi) * Vg]) if phi > 0 else Vb
    q = (V ** 2).sum(1)
    D = np.clip(1.0 - q, 1e-6, None)
    s = np.sqrt(q + D)
    return V / s[:, None], D / s ** 2


def race(a, V, D):
    p = winning.race_probabilities(a, V=V, D=D, points=1025)
    p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
    return p / p.sum()


def run(n=5000, T=104, draws=8, seed=23):
    rng = np.random.default_rng(seed)
    out = {p: [] for p in PHIS}; base = []
    for i in range(draws):
        B, d = market(rng, n)
        X = sample(rng, B, d, T)
        wh, labels = hrp_big(None, X, 50)
        base.append(true_risk(wh, B, d))
        Vg, Vb = split_factors(X, labels, 3)
        # Calibrate under the SAME law the phi=0 race uses, which is the block
        # structure HRP consumed. Calibrating under independence and racing
        # under the blocks makes phi=0 a tilt already, which offsets the whole
        # curve; that was the bug in the previous two attempts.
        V0, D0 = normalise(Vg, Vb, 0.0)
        a = np.asarray(winning.calibrate_abilities(
            np.maximum(wh, 1e-12), V=V0, D=D0, points=1025), float)
        p0 = race(a, V0, D0)
        gap = float(np.abs(p0 - wh / wh.sum()).sum())
        if i == 0:
            print(f"    identity check: phi=0 reproduces HRP to L1 {gap:.2e}", flush=True)
        for phi in PHIS:
            V, D = normalise(Vg, Vb, phi)
            out[phi].append(true_risk(race(a, V, D), B, d))
        print(f"    draw {i+1}/{draws}", flush=True)
    return {p: np.asarray(v) for p, v in out.items()}, np.asarray(base)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 104
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print(f"n = {n}, T = {T}. Tilting HRP toward ONLY the cross-cluster part.\n")
    out, base = run(n, T, draws)
    print(f"\n{'phi':>6s}{'ratio to HRP':>15s}{'beats HRP':>12s}")
    for p in PHIS:
        x = out[p]
        print(f"{p:6.2f}{np.median(x / base):15.4f}{np.mean(x < base):11.0%}")
    print(f"\n  exact HRP {np.median(base):.5f}")
