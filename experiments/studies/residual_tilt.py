"""Tilt a hierarchical portfolio toward only the covariance it left out.

HRP's recursion never reads the covariance between the two halves of a split,
so what it discards is the cross-cluster part. The right tilt therefore keeps
the within-cluster structure and adds only the cross-cluster part, damped.

A two-parameter correlation makes that exact. With average within-cluster
correlation rho_in and cross-cluster rho_out, the market is one global factor
with loading sqrt(rho_out) on every asset plus one group factor per cluster
with loading sqrt(rho_in - rho_out) inside it. Scaling only the global loading
by sqrt(phi) gives

    phi = 0  block structure alone, which is HRP's own view
    phi = 1  the full structure, including what the recursion never saw

and phi = 0 is the identity, so the race must return its benchmark there. That
check is asserted rather than assumed, because three earlier versions of this
experiment failed it and I read past it twice.
"""
import sys
import numpy as np
import winning


def loadings(labels, rho_in, rho_out, phi):
    n = len(labels)
    groups = np.unique(labels)
    V = np.zeros((n, len(groups) + 1))
    V[:, 0] = np.sqrt(max(phi * rho_out, 0.0))
    for j, g in enumerate(groups):
        V[labels == g, j + 1] = np.sqrt(max(rho_in - rho_out, 0.0))
    q = (V ** 2).sum(1)
    D = np.clip(1.0 - q, 1e-6, None)
    s = np.sqrt(q + D)
    return V / s[:, None], D / s ** 2


def race(a, V, D, points=257):
    p = winning.race_probabilities(a, V=V, D=D, points=points)
    p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
    return p / p.sum()


def check_identity(n=200, n_groups=10, rho_in=0.55, rho_out=0.2, seed=0):
    """phi = 0 must return the benchmark. Everything downstream is void if not."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_groups), n // n_groups)
    w = rng.dirichlet(np.full(n, 3.0))
    V0, D0 = loadings(labels, rho_in, rho_out, 0.0)
    a = np.asarray(winning.calibrate_abilities(w, V=V0, D=D0), float)
    return float(np.abs(race(a, V0, D0) - w).sum())


if __name__ == "__main__":
    gap = check_identity()
    print(f"  identity at phi=0, n=200: L1 {gap:.2e}")
    if gap > 1e-6:
        print("  FAILED: the reference and the phi=0 race disagree; stopping.")
        sys.exit(1)
    print("  passed, so the dial below measures the dial\n")

    rng = np.random.default_rng(11)
    n, n_groups, rho_in, rho_out = 200, 10, 0.55, 0.2
    labels = np.repeat(np.arange(n_groups), n // n_groups)
    # the true covariance of that market, used only for scoring
    C = np.full((n, n), rho_out)
    for g in range(n_groups):
        s = slice(g * (n // n_groups), (g + 1) * (n // n_groups))
        C[s, s] = rho_in
    np.fill_diagonal(C, 1.0)
    vol = np.exp(rng.normal(0, 0.4, n))
    Sig = vol[:, None] * C * vol[None, :]

    w = rng.dirichlet(np.full(n, 3.0))         # a stand-in hierarchical book
    V0, D0 = loadings(labels, rho_in, rho_out, 0.0)
    a = np.asarray(winning.calibrate_abilities(w, V=V0, D=D0), float)
    print(f"  {'phi':>6s}{'ratio to the benchmark':>25s}")
    for phi in (0.0, 0.25, 0.5, 0.75, 1.0):
        V, D = loadings(labels, rho_in, rho_out, phi)
        p = race(a, V, D)
        print(f"  {phi:6.2f}{float(p @ Sig @ p) / float(w @ Sig @ w):25.4f}")
