"""The two markets, and the check that each one makes the premise exactly true.

The premise under test is CAPM's: the parent portfolio is optimal for the
parent universe. If that only holds approximately then the experiment measures
the approximation instead of the restriction rules, so both constructions
verify it and `run.py` refuses to score a draw that fails.

`mid` is the ordinary direction. Draw a covariance from one of six families,
solve its long-only minimum-variance portfolio, and call that the parent. Good
to a few hundred names.

`index` is the same premise at index scale, built the other way round. A
long-only minimum-variance parent is the wrong model of an index: at five
thousand names a single factor is so diversifiable that the optimum holds
about eighty of them. So the cap weights come first, from a power law, and the
covariance is then chosen to make them optimal:

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
# mid scale: a random dependence structure, parent = its long-only optimum
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
    """Dense. Parent is the long-only minimum-variance portfolio of Sigma."""

    scale = "mid"

    def __init__(self, rng, n, solver):
        self.Sigma, self.family = random_structure(rng, n)
        self.n = n
        self.parent = solver(self.Sigma)
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
