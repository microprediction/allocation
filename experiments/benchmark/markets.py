"""Simulated markets with the truth known by construction.

Every generator returns a callable that produces a return panel of any length
from a fixed population covariance, so realized risk is w'Sigma w exactly and
no backtest noise enters. Three traps this module exists to avoid, all hit at
least once while building it:

  1. A market of independent draws has nothing to forecast. Testing a
     forecasting method there is not a test, so `ar` adds real predictability.
  2. Multivariate t is elliptical, so expected shortfall is proportional to
     standard deviation and ranks portfolios exactly as variance does. A tail
     metric on such a market is a relabelling of the variance metric.
  3. A regime mixture that pushes every correlation uniformly toward one stays
     effectively elliptical. The crash regime needs its OWN structure over a
     different labelling before the tail stops being a function of variance.

`elliptical_spread` measures trap 2 and 3 directly: 1.00 means the market
cannot separate tail risk from variance.
"""
import numpy as np

FAMILIES = ("blocks", "blocks+factor", "factor", "equicorr", "wishart", "nested")


def random_covariance(rng, n, family=None):
    """A population covariance from one of six families, parameters resampled."""
    fam = family or rng.choice(FAMILIES)
    if fam in ("blocks", "blocks+factor"):
        K = int(rng.integers(2, min(11, n)))
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
            B = rng.normal(0, rng.uniform(0.2, 0.6), size=(n, int(rng.integers(1, 4))))
            C = C + B @ B.T
    elif fam == "factor":
        B = rng.normal(rng.uniform(0, 1.0), rng.uniform(0.2, 0.8),
                       size=(n, int(rng.integers(1, 5))))
        C = B @ B.T + np.diag(np.exp(rng.normal(0, rng.uniform(0.2, 0.8), n)))
    elif fam == "equicorr":
        r = float(rng.uniform(0.05, 0.85))
        C = np.full((n, n), r); np.fill_diagonal(C, 1.0)
    elif fam == "wishart":
        df = int(rng.integers(n + 2, 4 * n))
        A = rng.normal(size=(n, df)); C = A @ A.T / df
    else:
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
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    vol = np.exp(rng.normal(0, float(rng.uniform(0.0, 0.8)), n))
    return vol[:, None] * C * vol[None, :], fam


class Market:
    """A population covariance plus a sampler. `sigma` is the truth."""

    def __init__(self, sigma, sampler, label):
        self.sigma = sigma
        self._sampler = sampler
        self.label = label

    def sample(self, rng, rows):
        return self._sampler(rng, rows)


def gaussian(rng, n=40, ar=0.0, family=None):
    Sig, fam = random_covariance(rng, n, family)
    L = np.linalg.cholesky(Sig)

    def sampler(r, rows):
        burn = 50 if ar else 0
        Z = r.normal(size=(rows + burn, n)) @ L.T
        if ar:
            for t in range(1, len(Z)):
                Z[t] += ar * Z[t - 1]
        return Z[burn:]

    # an AR(1) inflates the stationary covariance by 1/(1-ar^2)
    truth = Sig / (1 - ar ** 2) if ar else Sig
    return Market(truth, sampler, f"gaussian/{fam}" + (f"/ar{ar}" if ar else ""))


def student_t(rng, n=40, nu=5.0, family=None):
    """Elliptical on purpose: use it to DEMONSTRATE that a tail metric adds
    nothing, not to measure tail skill."""
    Sig, fam = random_covariance(rng, n, family)
    L = np.linalg.cholesky(Sig * (nu - 2.0) / nu)

    def sampler(r, rows):
        Z = r.normal(size=(rows, n)) @ L.T
        W = r.chisquare(nu, size=(rows, 1)) / nu
        return Z / np.sqrt(W)

    return Market(Sig, sampler, f"student-t/{fam}")


def regime(rng, n=40, p_crash=0.08, vol_mult=3.0, family=None):
    """Calm structure, plus a crash regime with an INDEPENDENT structure over a
    permuted labelling. Genuinely non-elliptical; see `elliptical_spread`."""
    Sc, fam = random_covariance(rng, n, family)
    So, _ = random_covariance(rng, n)
    perm = rng.permutation(n)
    Sx = So[np.ix_(perm, perm)]
    d = np.sqrt(np.diag(Sc)); dc = np.sqrt(np.diag(Sx))
    Sx = (vol_mult * d / dc)[:, None] * Sx * (vol_mult * d / dc)[None, :]
    mu = -2.0 * d
    Lc = np.linalg.cholesky(Sc); Lx = np.linalg.cholesky(Sx)

    def sampler(r, rows):
        crash = r.random(rows) < p_crash
        Z = r.normal(size=(rows, n))
        out = Z @ Lc.T
        if crash.any():
            out[crash] = Z[crash] @ Lx.T + mu
        return out

    # the mixture's own second moment about zero, which is the risk that matters
    truth = (1 - p_crash) * Sc + p_crash * (Sx + np.outer(mu, mu))
    return Market(truth, sampler, f"regime/{fam}")


def elliptical_spread(market, rng, n_port=60, rows=120_000):
    """Spread of shortfall over standard deviation across portfolios of varied
    concentration. 1.00 means the market cannot distinguish a tail metric from
    a variance metric."""
    panel = market.sample(rng, rows)
    n = panel.shape[1]
    W = np.array([rng.dirichlet(np.full(n, 10.0 ** rng.uniform(-1.5, 1.0)))
                  for _ in range(n_port)])
    P = panel @ W.T
    v = P.var(0)
    q = np.quantile(P, 0.05, axis=0)
    es = np.array([-P[P[:, j] <= q[j], j].mean() for j in range(P.shape[1])])
    ratio = es / np.sqrt(v)
    return float(ratio.max() / ratio.min())
