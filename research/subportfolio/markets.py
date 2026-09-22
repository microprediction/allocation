"""The two markets. In both, the parent is the tangency portfolio.

CAPM's claim is that the market portfolio is mean-variance efficient: for some
vector of expected excess returns m, the cap-weighted book w maximises Sharpe
ratio over the whole universe. Given a covariance Sigma and a long-only w that
is exactly the statement

    Sigma w = c m,   c > 0,

so m is defined by the market rather than drawn: m = Sigma w, the returns the
market has to expect for its own weights to be optimal. Nothing ties Sigma to
w. The covariance is drawn to look like equities and the cap weights to look
like an index, independently, and the premise holds by definition.

Restricting to a sub-universe S is where it stops being trivial. The optimum
there is

    w_S* ~ Sigma_SS^{-1} m_S = w_S + Sigma_SS^{-1} Sigma_{S,S^c} w_{S^c},

proportional restriction plus the departed names' weight projected onto the
survivors they co-moved with. That second term depends on who left and how
they were correlated with who stayed, which is the whole content of the
question. Under a minimum-variance premise (m constant) it vanishes and the
optimum is a function of the survivors alone.

`mid` draws a dependence structure from one of six families. `index` is a
sector market at index scale and is never formed densely: a market factor
with heterogeneous betas, two signed style factors, fifty sectors of unequal size
each with its own loading, and idiosyncratic noise. Its spectrum is one large
eigenvalue and a long tail, which is what equities look like and what a
k-factor estimate cannot capture.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np


def premise_residual(w, Sw, m):
    """How far w is from the long-only maximum-Sharpe portfolio of (Sigma, m).

    KKT: Sigma w / (w'Sigma w) = m / (w'm) wherever w > 0, with >= elsewhere.
    Every parent here holds every name, so only the equality applies. Returned
    relative to the scale of m.
    """
    lhs = Sw / float(w @ Sw)
    rhs = m / float(w @ m)
    return float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs)))


# --------------------------------------------------------------------------
# the cap-weight profile
# --------------------------------------------------------------------------

# The cap-weight profile comes from a real index: the 129 month-end
# cross-sections of the S&P 500 in experiments/data. Each draw takes one of
# those dates, sorts it and stretches it onto n names, so the parent inherits
# an observed concentration rather than an invented one. Over that decade the
# index ran at an effective name count of 0.10 to 0.31 of its listed count,
# with every name carried at a positive weight.
#
# SHAPE_PATH is read when present; a copy of this directory elsewhere may not
# have it, so the median profile is the fallback and the market records which
# one it used.
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



# --------------------------------------------------------------------------
# mid scale: a random dependence structure
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
    """Dense. A random dependence structure with a cap-weighted tangency parent."""

    scale = "mid"

    def __init__(self, rng, n, solver=None):
        M, self.family = random_structure(rng, n)
        self.Sigma = M / np.mean(np.diag(M))
        self.n = n
        self.parent, self.weight_source = cap_weights(rng, n)
        self.m = self.Sigma @ self.parent
        # Sectors are contiguous index ranges, which is where random_structure
        # puts its blocks; cap weights were shuffled so size is independent.
        self.sectors = 10
        self.sector = np.arange(n) * self.sectors // n
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
        return premise_residual(self.parent, self.sigma_times(self.parent), self.m)


# --------------------------------------------------------------------------
# index scale: a sector market, never dense
# --------------------------------------------------------------------------

class IndexMarket:
    """Never dense. Market factor, two style factors, fifty sectors, idiosyncratic.

        Sigma_ij = b_i b_j + e_i e_j + e2_i e2_j + c_i c_j [same sector] + d_i [i = j]

    with every name at unit variance before the volatility scale s_i, so the
    correlation is the same expression without d. Calibrated to
    the S&P 500 panel in experiments/data, 2014-2024:

                                real    market
        pairwise correlation    0.347   0.34
        dispersion              0.120   0.12
        first eigenvalue        36.2%   36%
        second eigenvalue        4.8%    5%
        log volatility sd       0.30    0.30

    Sector sizes are drawn from a Dirichlet so a few sectors are large, which
    is where the eigenvalue tail comes from. Every block, product and panel is
    O(n) or O(n + m^2); Sigma itself is never formed.
    """

    scale = "index"

    def __init__(self, rng, n, solver=None, sectors=50, tail=1.3):
        self.n, self.family, self.sectors = n, "sector market", sectors
        s = np.exp(rng.normal(0.0, 0.30, n))
        b = rng.uniform(0.35, 0.78, n)                    # market beta
        e = rng.uniform(-0.38, 0.38, n)        # signed style loading
        e2 = rng.uniform(-0.32, 0.32, n)                  # a second one, for PC3
        c = rng.uniform(0.25, 0.45, n)                    # sector loading
        # unequal sectors: a Dirichlet with concentration 1 gives a few large
        sizes = rng.dirichlet(np.ones(sectors))
        self.sector = rng.choice(sectors, size=n, p=sizes)
        self.b, self.e, self.e2, self.c = b * s, e * s, e2 * s, c * s
        self.d = np.maximum(1.0 - b ** 2 - e ** 2 - e2 ** 2 - c ** 2, 0.05) * s ** 2
        cap = rng.pareto(tail, n) + 1.0
        self.parent = cap / cap.sum()
        self.m = self.sigma_times(self.parent)

    def _same(self, idx):
        g = self.sector[idx]
        return (g[:, None] == g[None, :]).astype(float)

    def block(self, idx):
        """Sigma[idx, idx] in O(m^2)."""
        bs, es, e2s, cs = self.b[idx], self.e[idx], self.e2[idx], self.c[idx]
        S = (np.outer(bs, bs) + np.outer(es, es) + np.outer(e2s, e2s)
             + np.outer(cs, cs) * self._same(idx) + np.diag(self.d[idx]))
        return (S + S.T) / 2

    def panel(self, rng, T):
        """T observations from the true Sigma, in O(T (n + sectors))."""
        f_m = rng.normal(size=T)
        f_e = rng.normal(size=T)
        f_e2 = rng.normal(size=T)
        f_g = rng.normal(size=(T, self.sectors))
        return (np.outer(f_m, self.b) + np.outer(f_e, self.e) + np.outer(f_e2, self.e2)
                + f_g[:, self.sector] * self.c
                + rng.normal(size=(T, self.n)) * np.sqrt(self.d))

    def sigma_times(self, w):
        """Sigma w in O(n + sectors)."""
        cw = np.bincount(self.sector, weights=self.c * w, minlength=self.sectors)
        return (self.b * float(self.b @ w) + self.e * float(self.e @ w)
                + self.e2 * float(self.e2 @ w)
                + self.c * cw[self.sector] + self.d * w)

    def premise_residual(self):
        return premise_residual(self.parent, self.sigma_times(self.parent), self.m)


MARKETS = {"mid": MidMarket, "index": IndexMarket}
