"""The two markets, and the check that each one makes the premise exactly true.

The premise under test is CAPM's: the parent portfolio is optimal for the
parent universe. If that only holds approximately then the experiment measures
the approximation instead of the restriction rules, so both constructions
verify it and `run.py` refuses to score a draw that fails.

`mid` draws a dependence structure from one of six families and pairs it with a
cap-weighted parent, using the same implied-covariance identity as `index`.
It does NOT solve for the parent. Long-only minimum variance is a corner
solution: at four hundred names it held a median of 25 of them and in one draw
6, so 85 to 98 percent of the parent weights were exactly zero. The race floors
those at 1e-12 and the resulting abilities are one-sided bounds rather than
information, and a random sub-universe then contained 0 to 4 names the parent
actually held, which made every rule a split of the same near point mass. That
is a statement about the market, not about restriction. A real index holds
every name it lists.

`index` is the same premise at index scale and never forms a dense matrix. Both
markets put the cap weights first and choose the covariance to make them
optimal:

    Sigma = 11' + eps Q M Q',    u = w / ||w||,    Q = I - u u'

which is positive definite and has w as its exact minimum-variance portfolio.
Since w >= 0 the long-only constraint is inactive, so w is also the long-only
optimum. This is the positive-definite branch of the implied-covariance
identity; the minimal rank-two correction reproduces w as well but is not
positive definite, which is why it is not used here.

Nothing is formed densely in the index case. Sigma[i,j] = 1 + eps (M_ij
- u_i (Mu)_j - (Mu)_i u_j + (u'Mu) u_i u_j), and M is one-factor, so any block
costs O(n) to assemble and the parent check is O(n) too.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np


def premise_residual(w, Sw):
    """How far w is from being the long-only minimum-variance portfolio.

    The KKT conditions for min w'S w subject to 1'w = 1, w >= 0 are

        (S w)_i = c  wherever w_i > 0,      (S w)_i >= c  everywhere,

    with c = w'S w. Stating it that way needs a rule for which weights count
    as held, and a solver leaves a numerical tail just above zero: at a
    threshold of 1e-10 a 120-name draw looks like it holds all 120 and the
    spread reads 11, while the true support is 13 names and the spread there
    is 3e-07. So we use the threshold-free form instead. Complementary
    slackness says w_i ((S w)_i - c) = 0 for every i, which weights each
    departure by how much is actually held, and dual feasibility says no
    (S w)_i falls below c. The residual returned is the larger of the two,
    relative to c.
    """
    c = float(w @ Sw)
    gap = float(np.max(w * np.abs(Sw - c))) / abs(c)
    infeas = max(0.0, float(c - Sw.min())) / abs(c)
    return max(gap, infeas)


# --------------------------------------------------------------------------
# the cap-weight profile, and the identity that makes it optimal
# --------------------------------------------------------------------------

# The cap-weight profile comes from a real index: the 129 month-end
# cross-sections of the S&P 500 in experiments/data. Each draw takes one of
# those dates, sorts it and stretches it onto n names, so the parent inherits
# an observed concentration rather than an invented one.
#
# Concentration is the thing being controlled. Over that decade the index ran
# at an effective name count of 0.10 to 0.31 of its listed count, and every
# name carried a positive weight. A parent more concentrated than the low end
# of that band is not a model of an index, which is the failure the old mid
# market had: it solved for a long-only minimum-variance parent and got a
# corner holding 25 names of 400.
#
# SHAPE_PATH is read when present. A copy of this directory on another machine
# may not have it, so the median profile is kept here as a fallback and the
# market records which one it used.
SHAPE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "data" / \
    "sp500_capweights_2014_2024.parquet"

FALLBACK_Q = np.array([0.000, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200,
                       0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 0.950,
                       0.990, 1.000])
FALLBACK_LOGW = np.array([3.008, 2.949, 2.710, 2.084, 1.801, 1.327, 0.800,
                          0.158, -0.257, -0.568, -0.840, -1.087, -1.304,
                          -1.532, -1.854, -2.094, -2.558, -3.350])

REAL_EFFN_FRACTION = (0.10, 0.31)   # measured over the same panel


@lru_cache(maxsize=1)
def real_shapes():
    """Sorted log cap-weight profiles, one per month-end, relative to 1/n.

    Returns a list of (quantile, log weight) pairs, or None if the panel is
    not reachable from here.
    """
    try:
        import pandas as pd
        df = pd.read_parquet(SHAPE_PATH)
    except Exception:
        return None
    out = []
    for d in df.index:
        w = df.loc[d].dropna().to_numpy(dtype=float)
        if len(w) < 50:
            continue
        w = np.sort(w / w.sum())[::-1]
        out.append(((np.arange(len(w)) + 0.5) / len(w),
                    np.log(w) - np.log(1.0 / len(w))))
    return out or None


def cap_weights(rng, n):
    """A cap-weight vector with the sorted shape of one real index date.

    Every name gets a positive weight, which is the property that matters: an
    index holds what it lists. The order is shuffled, so size carries no
    information about where a name sits in the dependence structure. Making the
    largest names also the most correlated is a modelling claim, and not one
    this study needs.

    Returns the weights and the source, so a draw can record which it used.
    """
    shapes = real_shapes()
    q = (np.arange(n) + 0.5) / n
    if shapes is None:
        logw = np.interp(q, FALLBACK_Q, FALLBACK_LOGW)
        src = "median profile (panel not found)"
    else:
        qe, le = shapes[int(rng.integers(len(shapes)))]
        logw = np.interp(q, qe, le)
        src = "sp500 month-end"
    w = np.exp(logw)
    w = w / w.sum()
    return w[rng.permutation(n)], src


def effective_fraction(w):
    """Effective name count as a fraction of the listed count."""
    return float(1.0 / np.sum(np.asarray(w, float) ** 2) / len(w))


def implied_covariance(w, M, eps):
    """Sigma = 11' + eps Q M Q', which has ``w`` as its exact optimum.

    Q = I - u u' with u = w / ||w|| annihilates w, so Sigma w = 1 exactly and
    w is the minimum-variance portfolio of Sigma. Since w > 0 the long-only
    constraint is inactive and it is the long-only optimum too. Positive
    definite for any positive semi-definite M.
    """
    u = w / np.linalg.norm(w)
    Qm = M - np.outer(u, u @ M) - np.outer(M @ u, u) + float(u @ M @ u) * np.outer(u, u)
    S = 1.0 + eps * Qm
    return (S + S.T) / 2


# --------------------------------------------------------------------------
# mid scale: a random dependence structure, cap-weighted parent
# --------------------------------------------------------------------------

def random_structure(rng, n):
    """A covariance from one of six families, with random parameters.

    Kept byte-identical in behaviour to experiments/studies/robust.py so the
    mid-scale numbers here and there are comparable.
    """
    fam = rng.choice(["blocks", "blocks+factor", "factor", "equicorr",
                      "wishart", "nested"])
    if fam in ("blocks", "blocks+factor"):
        K = int(rng.integers(2, 11))
        cuts = np.sort(rng.choice(np.arange(1, n), size=K - 1, replace=False))
        sizes = np.diff(np.concatenate([[0], cuts, [n]]))
        rho_out = float(rng.uniform(0.0, 0.35))
        C = np.full((n, n), rho_out)
        i = 0
        for s in sizes:
            rin = float(rng.uniform(max(rho_out + 0.05, 0.2), 0.92))
            C[i:i + s, i:i + s] = rin
            i += s
        np.fill_diagonal(C, 1.0)
        if fam == "blocks+factor":
            nf = int(rng.integers(1, 4))
            B = rng.normal(0, rng.uniform(0.2, 0.6), size=(n, nf))
            C = C + B @ B.T
    elif fam == "factor":
        nf = int(rng.integers(1, 5))
        B = rng.normal(rng.uniform(0, 1.0), rng.uniform(0.2, 0.8), size=(n, nf))
        C = B @ B.T + np.diag(np.exp(rng.normal(0, rng.uniform(0.2, 0.8), n)))
    elif fam == "equicorr":
        r = float(rng.uniform(0.05, 0.85))
        C = np.full((n, n), r)
        np.fill_diagonal(C, 1.0)
    elif fam == "wishart":
        df = int(rng.integers(n + 2, 4 * n))
        A = rng.normal(size=(n, df))
        C = A @ A.T / df
    else:                                        # nested hierarchy
        C = np.full((n, n), float(rng.uniform(0.0, 0.2)))
        for lev in (2, 4, 8):
            step = max(n // lev, 1)
            add = float(rng.uniform(0.05, 0.3))
            for b in range(lev):
                s = slice(b * step, min((b + 1) * step, n))
                C[s, s] += add
        np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh((C + C.T) / 2)
    C = V @ np.diag(np.maximum(ev, 1e-6)) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    vol = np.exp(rng.normal(0, float(rng.uniform(0.0, 0.8)), n))
    return vol[:, None] * C * vol[None, :], str(fam)


class MidMarket:
    """Dense. Parent is cap weights, exactly optimal by construction.

    The dependence structure is drawn from one of six families and used as the
    ``M`` of the implied-covariance identity, so the families still separate
    the draws while the parent stays a plausible index. ``solver`` is accepted
    and unused: the parent is no longer solved for, and the oracle does its own
    solving in run.py.
    """

    scale = "mid"

    def __init__(self, rng, n, solver=None, eps=10.0):
        M, self.family = random_structure(rng, n)
        M = M / np.mean(np.diag(M))          # so eps means the same thing across families
        self.n = n
        self.parent, self.weight_source = cap_weights(rng, n)
        self.Sigma = implied_covariance(self.parent, M, eps)
        self.chol = np.linalg.cholesky(
            self.Sigma + 1e-12 * np.eye(n) * np.trace(self.Sigma) / n)

    def block(self, idx):
        S = self.Sigma[np.ix_(idx, idx)]
        return (S + S.T) / 2

    def panel(self, rng, T):
        return rng.normal(size=(T, self.n)) @ self.chol.T

    def sigma_times(self, w):
        return self.Sigma @ w

    def premise_residual(self):
        return premise_residual(self.parent, self.sigma_times(self.parent))


# --------------------------------------------------------------------------
# index scale: cap weights first, covariance chosen to make them optimal
# --------------------------------------------------------------------------

class IndexMarket:
    """Never dense. Parent is cap weights, exactly optimal by construction."""

    scale = "index"

    def __init__(self, rng, n, solver=None, eps=10.0, tail=1.3):
        b = rng.uniform(0.15, 0.75, n)
        s = np.exp(rng.normal(0.0, 0.45, n))
        self.v, self.d = b * s, (1.0 - b ** 2) * s ** 2
        cap = rng.pareto(tail, n) + 1.0
        self.parent = cap / cap.sum()
        self.u = self.parent / np.linalg.norm(self.parent)
        self.Mu = self.v * (self.v @ self.u) + self.d * self.u
        self.uMu = float(self.u @ self.Mu)
        self.eps, self.n, self.family = eps, n, "implied one-factor"

    def block(self, idx):
        """Sigma[idx, idx] in O(n + m^2), without forming Sigma."""
        vs, us, ms = self.v[idx], self.u[idx], self.Mu[idx]
        M = np.outer(vs, vs) + np.diag(self.d[idx])
        C = (M - np.outer(us, ms) - np.outer(ms, us)
             + self.uMu * np.outer(us, us))
        S = 1.0 + self.eps * C
        return (S + S.T) / 2

    def panel(self, rng, T):
        """T observations from the true Sigma, in O(T n)."""
        g = rng.normal(size=T)                                  # the 11' part
        f = rng.normal(size=T)
        Z = f[:, None] * self.v + rng.normal(size=(T, self.n)) * np.sqrt(self.d)
        Z = Z - np.outer(Z @ self.u, self.u)                    # Q M Q'
        return g[:, None] + np.sqrt(self.eps) * Z

    def sigma_times(self, w):
        """Sigma w in O(n), without forming Sigma."""
        Mw = self.v * (self.v @ w) + self.d * w
        uw = float(self.u @ w)
        return (np.full_like(w, float(np.ones_like(w) @ w))
                + self.eps * (Mw - self.u * float(self.u @ Mw)
                              - self.Mu * uw + self.uMu * self.u * uw))

    def premise_residual(self):
        return premise_residual(self.parent, self.sigma_times(self.parent))


MARKETS = {"mid": MidMarket, "index": IndexMarket}
