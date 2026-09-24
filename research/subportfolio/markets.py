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


MAX_WEIGHT = 0.081       # the S&P 500's largest weight over 2014-2024


def _cap_largest(w, cap, iters=50):
    """Hold every weight at or below `cap`, spreading the excess proportionally
    over the names that are under it, until the constraint holds."""
    w = np.asarray(w, float).copy()
    for _ in range(iters):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


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

    Under law="regime" the same covariance is generated by a non-Gaussian law:
    a market crash day, a sector cascade day, and a common volatility regime,
    each built so the total covariance is unchanged. Calibrated to the S&P 500
    panel in experiments/data, 2014-2024, daily:

                                                real    market
        pairwise correlation                    0.347   0.346
        first eigenvalue                        36.2%   37%
        second, third eigenvalue                4.8%, 3.5%   4.8%, 3.7%
        log volatility sd                       0.30    0.30
        random 30-name basket, all down         3.1/yr  3.1/yr
        random 30-name basket, all up           1.9/yr  1.7/yr
        >= 25 of 30 with the same sign          64/yr   69/yr
        lower tail dependence over Gaussian     1.48x   1.39x
        upper tail dependence over Gaussian     1.31x   1.34x
        days a 30-50 name cluster has half its
          names in their 5% residual tail       14      14

    A Gaussian copula at the same correlation gives 1.4/yr for all-down, 1.0x
    for the tail dependence and 1 for the cluster count. Sector sizes are
    Dirichlet, so a few sectors are large, which is where the eigenvalue tail
    comes from. Every block, product and panel is O(n) or O(n + m^2); Sigma
    itself is never formed.
    """

    scale = "index"

    def __init__(self, rng, n, solver=None, sectors=50, tail=1.3, law="gaussian",
                 crash=(0.02, 0.05, 0.4), cascade=(0.006, 0.28), nu=np.inf,
                 vol=(0.25, 1.45), beta_range=(0.38, 0.80)):
        self.n, self.sectors = n, sectors
        self.law = law
        # Two regime layers, off under law="gaussian". Each is (daily
        # probability, share of that factor's variance carried by the regime,
        # and for the crash a dispersion factor). The Gaussian loading of the
        # same factor is shrunk by exactly that share, so the TOTAL covariance
        # is identical under either law: Sigma, m, Black-Litterman and both
        # oracles do not move, only the copula.
        #   crash    a market-wide day: every name takes -a |xi| b_i, and the
        #            idiosyncratic and sector noise is scaled by rho < 1. A
        #            crash is a correlation spike, everyone down modestly with
        #            dispersion collapsing, not a large common shift. The
        #            non-crash idiosyncratic noise is scaled up to keep d exact.
        #   cascade  a sector day: every member of one sector takes -a |eta| c_i
        self.pc, self.fc, self.rho = crash if law == "regime" else (0.0, 0.0, 1.0)
        self.ps, self.fs = cascade if law == "regime" else (0.0, 0.0)
        # A fat-tailed market factor, Student-t with nu degrees of freedom scaled
        # to unit variance, carries the symmetric part of the tail co-movement
        # (Oh and Patton's fat-tailed common factor); the crash layer carries
        # the asymmetry. Covariance is unchanged. Off under gaussian.
        self.nu = nu if law == "regime" else np.inf
        # Volatility clustering as a two-state common scale: with probability
        # pi_h the whole day is turbulent, every component scaled by s_h, else
        # calm at the scale that keeps E[scale^2] = 1. Raises moderate and tail
        # co-movement together, which a fat-tailed factor alone cannot do, and
        # leaves the covariance exactly where it was. A common scale is
        # invisible to any race, since it cannot move an argmin.
        self.pi_h, self.s_h = vol if law == "regime" else (0.0, 1.0)
        self.family = "sector market, " + law
        s = np.exp(rng.normal(0.0, 0.30, n))
        b = rng.uniform(*beta_range, n)                   # market beta
        e = rng.uniform(-0.38, 0.38, n)        # signed style loading
        e2 = rng.uniform(-0.32, 0.32, n)                  # a second one, for PC3
        c = rng.uniform(0.25, 0.45, n)                    # sector loading
        # unequal sectors: a Dirichlet with concentration 1 gives a few large
        sizes = rng.dirichlet(np.ones(sectors))
        self.sector = rng.choice(sectors, size=n, p=sizes)
        self.b, self.e, self.e2, self.c = b * s, e * s, e2 * s, c * s
        self.d = np.maximum(1.0 - b ** 2 - e ** 2 - e2 ** 2 - c ** 2, 0.05) * s ** 2
        # A Pareto tail gives the right median concentration and far too heavy
        # an upper tail: a median largest weight of 4.1% against the real
        # index's 4.6%, but a 90th percentile of 11% against 7.5% and draws
        # reaching 30%. No index has a 30% name. The largest weight is capped
        # at the largest the S&P 500 reached over 2014-2024 and the excess is
        # spread over the rest, which leaves the median untouched.
        cap = rng.pareto(tail, n) + 1.0
        self.parent = _cap_largest(cap / cap.sum(), MAX_WEIGHT)
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

    @staticmethod
    def _shock_scale(p, f):
        """Multiple `a` of the loading such that a centred shock -a|xi| on a
        Bernoulli(p) day carries a share f of the factor's variance."""
        if p <= 0 or f <= 0:
            return 0.0
        kappa = f / (1.0 - f)                 # regime variance / gaussian variance
        return float(np.sqrt(kappa / (p * (1.0 - 2.0 * p / np.pi))))

    def panel(self, rng, T, idx=None):
        """T zero-mean observations from the true law, in O(T (m + sectors)).

        With `idx` only those columns are generated, which is what the tail
        scoring needs at index scale.
        """
        cols = np.arange(self.n) if idx is None else np.asarray(idx)
        b, e, e2, c, d = (self.b[cols], self.e[cols], self.e2[cols],
                          self.c[cols], self.d[cols])
        sec = self.sector[cols]
        g_m = np.sqrt(1.0 - self.fc)          # gaussian share of the market factor
        g_s = np.sqrt(1.0 - self.fs)          # gaussian share of each sector factor
        if np.isfinite(self.nu):
            f_m = rng.standard_t(self.nu, size=T) * np.sqrt((self.nu - 2.0) / self.nu)
        else:
            f_m = rng.normal(size=T)
        f_e = rng.normal(size=T)
        f_e2 = rng.normal(size=T)
        f_g = rng.normal(size=(T, self.sectors))
        # dispersion: rho on crash days, and slightly above one otherwise so
        # that E[scale^2] = 1 and every name keeps its idiosyncratic variance
        hit = rng.random(T) < self.pc
        q = self.pc * self.rho ** 2 + (1.0 - self.pc)
        disp = np.where(hit, self.rho, np.sqrt((1.0 - self.pc * self.rho ** 2) / (1.0 - self.pc))
                        if self.pc > 0 else 1.0)
        disp = disp / np.sqrt(q) if q > 0 else disp
        X = (np.outer(f_m, g_m * b) + np.outer(f_e, e) + np.outer(f_e2, e2)
             + (f_g[:, sec] * (g_s * c) + rng.normal(size=(T, len(cols))) * np.sqrt(d))
             * disp[:, None])
        if self.fc > 0:
            a = self._shock_scale(self.pc, self.fc)
            mag = np.abs(rng.normal(size=T)) * hit
            X += np.outer(-a * mag + a * self.pc * np.sqrt(2.0 / np.pi), b)
        if self.pi_h > 0:
            turb = rng.random(T) < self.pi_h
            s_l = np.sqrt((1.0 - self.pi_h * self.s_h ** 2) / (1.0 - self.pi_h))
            X *= np.where(turb, self.s_h, s_l)[:, None]
        if self.fs > 0:
            a = self._shock_scale(self.ps, self.fs)
            hit = rng.random((T, self.sectors)) < self.ps
            mag = np.abs(rng.normal(size=(T, self.sectors))) * hit
            X += (-a * mag + a * self.ps * np.sqrt(2.0 / np.pi))[:, sec] * c
        return X

    def sigma_times(self, w):
        """Sigma w in O(n + sectors)."""
        cw = np.bincount(self.sector, weights=self.c * w, minlength=self.sectors)
        return (self.b * float(self.b @ w) + self.e * float(self.e @ w)
                + self.e2 * float(self.e2 @ w)
                + self.c * cw[self.sector] + self.d * w)

    def regime(self):
        """The crash layer as (p, a, rho, beta): daily probability, shift multiple,
        dispersion factor, and each name's market loading in unit-volatility
        terms. What the race is handed to run under the true law."""
        a = self._shock_scale(self.pc, self.fc)
        beta = self.b / np.sqrt(self.b ** 2 + self.e ** 2 + self.e2 ** 2 + self.c ** 2 + self.d)
        return (self.pc, a, self.rho, beta)

    def premise_residual(self):
        return premise_residual(self.parent, self.sigma_times(self.parent), self.m)


MARKETS = {"mid": MidMarket, "index": IndexMarket}
