"""The three fixed markets the studies share, in one place.

`benchmark/markets.py` holds the randomized generator used for the harness.
These are the FIXED markets that specific studies need, where the point is to
hold the structure still and vary one thing. They were previously triplicated
across shrinkage_beats_hrp.py, nco_taper.py and taper_beats_hrp.py with
slightly different signatures.
"""
import numpy as np

N = 40


def blocks(rng, n=N, nb=5, rho_in=0.7, rho_out=0.15, vol=0.4):
    """Five correlated blocks: the structure a hierarchical rule assumes."""
    per = n // nb
    C = np.full((n, n), rho_out)
    for b in range(nb):
        s = slice(b * per, (b + 1) * per)
        C[s, s] = rho_in
    np.fill_diagonal(C, 1.0)
    v = np.exp(rng.normal(0, vol, n))
    return v[:, None] * C * v[None, :]


def one_factor(rng, n=N):
    """One common factor and heterogeneous idiosyncratic risk: no blocks."""
    beta = rng.normal(1.0, 0.4, n)
    idio = np.exp(rng.normal(0, 0.5, n))
    return np.outer(beta, beta) * 0.04 + np.diag(idio ** 2)


def wishart(rng, n=N):
    """No structure at all."""
    A = rng.normal(size=(n, n + 10))
    S = A @ A.T / (n + 10)
    d = np.sqrt(np.diag(S))
    v = np.exp(rng.normal(0, 0.4, n))
    return v[:, None] * (S / np.outer(d, d)) * v[None, :]


ALL = (("five blocks", blocks), ("one factor", one_factor),
       ("Wishart, no structure", wishart))
