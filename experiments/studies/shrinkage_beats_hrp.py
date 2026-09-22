"""Does a plain shrinkage rule beat HRP out of sample?

No oracle anywhere. Ledoit-Wolf picks its own intensity from the data, the
same way a practitioner would, and both sides are long-only so the optimizer
cannot win by shorting. The truth is known by construction, so realized risk
is w' Sigma w exactly rather than a noisy backtest estimate.
"""
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from _markets import blocks, one_factor, wishart
from confound import hrp, min_var

n = 40





_w = cp.Variable(n)
_P = cp.Parameter((n, n), PSD=True)
_prob = cp.Problem(cp.Minimize(cp.quad_form(_w, _P)), [cp.sum(_w) == 1, _w >= 0])


def min_var_long_only(S):
    ev, V = np.linalg.eigh((S + S.T) / 2)
    _P.value = V @ np.diag(np.maximum(ev, 1e-10)) @ V.T
    try:
        _prob.solve(solver=cp.CLARABEL)
        if _w.value is not None:
            w = np.maximum(_w.value, 0); return w / w.sum()
    except Exception:
        pass
    w = np.maximum(min_var(S, 1e-6), 0); return w / max(w.sum(), 1e-12)


def run(gen, T, draws=200, seed=0):
    rng = np.random.default_rng(seed)
    keys = ("HRP", "min-var LO, sample", "min-var LO, Ledoit-Wolf",
            "HRP on Ledoit-Wolf", "inverse variance")
    acc = {k: [] for k in keys}
    for _ in range(draws):
        Sig = gen(rng)
        X = rng.normal(size=(T, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        LW = LedoitWolf(assume_centered=False).fit(X).covariance_
        iv = 1 / np.diag(S); iv /= iv.sum()
        ws = {"HRP": hrp(S), "min-var LO, sample": min_var_long_only(S),
              "min-var LO, Ledoit-Wolf": min_var_long_only(LW),
              "HRP on Ledoit-Wolf": hrp(LW), "inverse variance": iv}
        for k, w in ws.items():
            acc[k].append(float(w @ Sig @ w))
    return {k: np.array(v) for k, v in acc.items()}


if __name__ == "__main__":
    for name, gen in (("five blocks", blocks), ("one factor", one_factor),
                      ("Wishart, no structure", wishart)):
        print(f"\n=== {name} ===")
        print(f"{'T/n':>5s}{'HRP':>9s}{'MV|sample':>11s}{'MV|LW':>9s}"
              f"{'HRP|LW':>9s}{'inv var':>9s}{'LW beats HRP':>14s}")
        for T in (20, 40, 80, 200, 400):
            r = run(gen, T, seed=T + len(name))
            win = float(np.mean(r["min-var LO, Ledoit-Wolf"] < r["HRP"]))
            print(f"{T/n:5.2f}{np.median(r['HRP']):9.4f}"
                  f"{np.median(r['min-var LO, sample']):11.4f}"
                  f"{np.median(r['min-var LO, Ledoit-Wolf']):9.4f}"
                  f"{np.median(r['HRP on Ledoit-Wolf']):9.4f}"
                  f"{np.median(r['inverse variance']):9.4f}{win:13.0%}")
