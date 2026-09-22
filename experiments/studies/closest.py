"""The closest SENSIBLE shrinkage to HRP, judged two ways.

No small family can reproduce HRP exactly, so ask which interpretable one
gets nearest, and whether the nearest one also behaves like HRP. Both
questions matter: a family can sit close in weights and far in realized
risk, or the reverse.
"""
import numpy as np
from scipy.optimize import minimize
from confound import true_sigma, hrp, min_var, tree, quasi_diag
from decompose import lca_depth, Z

n = 40


def targets(S):
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    D = np.diag(np.diag(S))
    rb = (R.sum() - n) / (n * n - n)
    F = np.full_like(R, rb); np.fill_diagonal(F, 1.0); F = d[:, None] * F * d[None, :]
    # single-index target from the leading eigenvector
    ev, V = np.linalg.eigh(S)
    b = V[:, -1] * np.sqrt(max(ev[-1], 0.0))
    resid = np.diag(np.maximum(np.diag(S) - b ** 2, 1e-12))
    I1 = np.outer(b, b) + resid
    return D, F, I1


def clip(S, k):
    ev, V = np.linalg.eigh(S)
    ev = ev.copy()
    if k < n:
        tail = ev[:n - k]
        ev[:n - k] = max(tail.mean(), 1e-12)
    return V @ np.diag(ev) @ V.T


def best_blend(S, w, Tgt, grid=201):
    g = np.linspace(0.0, 1.0, grid)
    vals = [np.abs(min_var(x * S + (1 - x) * Tgt, 1e-10) - w).sum() for x in g]
    i = int(np.argmin(vals))
    return g[i], vals[i], x_to_mat(S, Tgt, g[i])


def x_to_mat(S, Tgt, x):
    return x * S + (1 - x) * Tgt


def leaf_taper(S, L, d, phi):
    T = np.ones((n, n))
    T[(L <= d) & (L > 0)] = 0.0
    T[(L > d)] = phi
    np.fill_diagonal(T, 1.0)
    return S * T


def run(T_obs=120, draws=60, seed=5):
    rng = np.random.default_rng(seed)
    rows = {k: {"l1": [], "var": [], "par": []} for k in
            ("diagonal", "constant correlation", "single index", "eigenvalue clipping",
             "hierarchical filter blend", "two-level taper", "per-level taper")}
    hv = []
    for _ in range(draws):
        Sig, _ = true_sigma(rng, 5, 8, 0.7, 0.15)
        X = rng.normal(size=(T_obs, n)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        Zl, _ = tree(S); L = lca_depth(quasi_diag(Zl, n), n)
        w = hrp(S); hv.append(float(w @ Sig @ w))
        D, F, I1 = targets(S)

        for name, Tgt in (("diagonal", D), ("constant correlation", F), ("single index", I1)):
            x, v, M = best_blend(S, w, Tgt)
            rows[name]["l1"].append(v); rows[name]["par"].append(x)
            wm = min_var(M, 1e-10); rows[name]["var"].append(float(wm @ Sig @ wm))

        best = None
        for k in range(1, n + 1):
            M = clip(S, k); v = np.abs(min_var(M, 1e-10) - w).sum()
            if best is None or v < best[1]:
                best = (k, v, M)
        rows["eigenvalue clipping"]["l1"].append(best[1]); rows["eigenvalue clipping"]["par"].append(best[0])
        wm = min_var(best[2], 1e-10); rows["eigenvalue clipping"]["var"].append(float(wm @ Sig @ wm))

        best = None
        for d in range(1, 7):
            Zd = Z(S, L, d)
            x, v, M = best_blend(S, w, Zd)
            if best is None or v < best[1]:
                best = (d, v, M, x)
        rows["hierarchical filter blend"]["l1"].append(best[1])
        rows["hierarchical filter blend"]["par"].append((best[0], round(best[3], 3)))
        wm = min_var(best[2], 1e-10); rows["hierarchical filter blend"]["var"].append(float(wm @ Sig @ wm))

        best = None
        for d in range(1, 7):
            for phi in np.linspace(0, 1, 41):
                M = leaf_taper(S, L, d, phi)
                v = np.abs(min_var(M, 1e-10) - w).sum()
                if best is None or v < best[1]:
                    best = (d, v, M, phi)
        rows["two-level taper"]["l1"].append(best[1])
        rows["two-level taper"]["par"].append((best[0], round(best[3], 2)))
        wm = min_var(best[2], 1e-10); rows["two-level taper"]["var"].append(float(wm @ Sig @ wm))

        lv = sorted(set(L[L > 0].tolist()))
        def obj(t):
            Tm = np.ones((n, n))
            for i, l in enumerate(lv):
                Tm[L == l] = np.clip(t[i], 0, 1.5)
            np.fill_diagonal(Tm, 1.0)
            return np.abs(min_var(S * Tm, 1e-10) - w).sum()
        r = minimize(obj, np.full(len(lv), 0.3), method="Nelder-Mead",
                     options=dict(maxiter=3000, fatol=1e-9))
        rows["per-level taper"]["l1"].append(r.fun)
        Tm = np.ones((n, n))
        for i, l in enumerate(lv):
            Tm[L == l] = np.clip(r.x[i], 0, 1.5)
        np.fill_diagonal(Tm, 1.0)
        wm = min_var(S * Tm, 1e-10); rows["per-level taper"]["var"].append(float(wm @ Sig @ wm))
        rows["per-level taper"]["par"].append(tuple(round(float(np.clip(v, 0, 1.5)), 2) for v in r.x))
    return rows, np.median(hv)


if __name__ == "__main__":
    rows, hrpvar = run()
    print(f"HRP realized variance (median): {hrpvar:.4f}")
    print(f"\n{'family':28s}{'params':>7s}{'L1 to HRP':>12s}{'its variance':>14s}{'vs HRP':>9s}")
    K = {"diagonal": 1, "constant correlation": 1, "single index": 1,
         "eigenvalue clipping": 1, "hierarchical filter blend": 2,
         "two-level taper": 2, "per-level taper": 6}
    for k, v in sorted(rows.items(), key=lambda kv: np.median(kv[1]["l1"])):
        mv = np.median(v["var"])
        print(f"{k:28s}{K[k]:7d}{np.median(v['l1']):12.4f}{mv:14.4f}{mv/hrpvar:8.2f}x")
    print("\nfitted parameters (median or modal):")
    for k, v in rows.items():
        p = v["par"]
        if isinstance(p[0], tuple):
            from collections import Counter
            print(f"  {k:28s} {Counter(p).most_common(1)[0][0]}")
        else:
            print(f"  {k:28s} {np.median(p):.3f}")
