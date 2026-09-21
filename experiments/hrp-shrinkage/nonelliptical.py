"""A market where tail risk is not a function of variance.

Multivariate t is elliptical, so expected shortfall is proportional to
standard deviation and ranks portfolios exactly as variance does. To ask
whether a tail-aware allocator earns anything, the market must break that.

Regime mixture: most of the time assets follow a calm covariance with modest
cross-cluster correlation; with probability p a crash regime fires in which
correlations jump toward one and the mean is negative. Correlation in the
left tail then exceeds correlation overall, which is asymmetric lower-tail
dependence and is invisible to the sample covariance.

First check the market actually does what is claimed, before running anything
on it.
"""
import numpy as np
from robust import random_structure

N = 40


def regime_market(rng, n=N, p_crash=0.06):
    Sig_calm, fam = random_structure(rng, n)
    d = np.sqrt(np.diag(Sig_calm))
    R = Sig_calm / np.outer(d, d)
    # crash regime: correlations pushed toward one, volatility inflated
    R_crash = 0.25 * R + 0.75 * np.ones_like(R)
    np.fill_diagonal(R_crash, 1.0)
    ev, V = np.linalg.eigh(R_crash)
    R_crash = V @ np.diag(np.maximum(ev, 1e-8)) @ V.T
    dd = np.sqrt(np.diag(R_crash)); R_crash = R_crash / np.outer(dd, dd)
    Sig_crash = (2.5 * d)[:, None] * R_crash * (2.5 * d)[None, :]
    mu_crash = -1.5 * d
    return Sig_calm, Sig_crash, mu_crash, p_crash, fam


def sample(rng, spec, rows):
    Sig_calm, Sig_crash, mu_crash, p, _ = spec
    Lc = np.linalg.cholesky(Sig_calm)
    Lx = np.linalg.cholesky(Sig_crash)
    crash = rng.random(rows) < p
    Z = rng.normal(size=(rows, N))
    out = Z @ Lc.T
    if crash.any():
        out[crash] = Z[crash] @ Lx.T + mu_crash
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(31337)
    # Does shortfall stop being a function of variance on this market?
    rows = []
    for _ in range(12):
        spec = regime_market(rng)
        panel = sample(rng, spec, 200_000)
        W = rng.random((60, N)); W = W / W.sum(1, keepdims=True)
        P = panel @ W.T
        v = P.var(0)
        q = np.quantile(P, 0.05, axis=0)
        es = np.array([-P[P[:, j] <= q[j], j].mean() for j in range(P.shape[1])])
        # if elliptical, es / sqrt(v) is constant across portfolios
        ratio = es / np.sqrt(v)
        rows.append(ratio.max() / ratio.min())
    print("Elliptical check: spread of shortfall / standard deviation across")
    print("random long-only portfolios on the SAME market. 1.00 means shortfall")
    print("is a pure function of variance and the market cannot test tail skill.\n")
    print(f"  regime mixture   median spread {np.median(rows):.3f}   max {max(rows):.3f}")

    # the same statistic on the elliptical market used before
    def draw_t(rng, Sig, rows_, nu=5.0):
        L = np.linalg.cholesky(Sig * (nu - 2) / nu)
        Z = rng.normal(size=(rows_, N)) @ L.T
        W_ = rng.chisquare(nu, size=(rows_, 1)) / nu
        return Z / np.sqrt(W_)

    rows2 = []
    for _ in range(12):
        Sig, _ = random_structure(rng, N)
        panel = draw_t(rng, Sig, 200_000)
        W = rng.random((60, N)); W = W / W.sum(1, keepdims=True)
        P = panel @ W.T
        v = P.var(0); q = np.quantile(P, 0.05, axis=0)
        es = np.array([-P[P[:, j] <= q[j], j].mean() for j in range(P.shape[1])])
        r = es / np.sqrt(v); rows2.append(r.max() / r.min())
    print(f"  multivariate t   median spread {np.median(rows2):.3f}   max {max(rows2):.3f}")
