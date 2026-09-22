"""Five thousand assets, two years of weekly data. T/n = 0.021.

Fifty times more rank-deficient than anything else here, and the regime a
large-universe manager actually faces. The sample covariance has rank 103 out
of 5000, so nothing can be inverted and most of the comparison set does not
exist.

The market is a factor model, Sigma = B B' + diag(d), which lets realized risk
be computed exactly as ||B'w||^2 + sum(d_i w_i^2) without ever forming the
5000 x 5000 matrix. Blocks are carried inside B as group factors, so there is
genuine hierarchical structure for a clustering method to find.
"""
import time
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform


def market(rng, n, n_groups=50, k_global=3):
    """Loadings B (n x k) and idiosyncratic variances d, never the full Sigma."""
    k = k_global + n_groups
    B = np.zeros((n, k))
    B[:, :k_global] = rng.normal(0, 0.35, (n, k_global))
    size = n // n_groups
    for g in range(n_groups):
        s = slice(g * size, (g + 1) * size if g < n_groups - 1 else n)
        B[s, k_global + g] = rng.uniform(0.3, 0.7)
    d = np.exp(rng.normal(0, 0.5, n)) ** 2
    return B, d


def true_risk(w, B, d):
    w = np.asarray(w, dtype=float)
    return float((B.T @ w) @ (B.T @ w) + np.sum(d * w * w))


def sample(rng, B, d, T):
    k = B.shape[1]
    return rng.normal(size=(T, k)) @ B.T + rng.normal(size=(T, len(d))) * np.sqrt(d)


def hrp_big(S_diag, X, n_clusters):
    """HRP needs a full correlation for its linkage. At this size we build the
    tree on a correlation from the sample, which is what a practitioner does."""
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1); sd[sd == 0] = 1.0
    Z = Xc / sd
    R = (Z.T @ Z) / (len(X) - 1)
    np.clip(R, -1, 1, out=R)
    D = np.sqrt(np.maximum(0.5 * (1.0 - R), 0.0))
    np.fill_diagonal(D, 0.0)
    link = linkage(squareform(D, checks=False), method="single")
    order = list(leaves_list(link))
    labels = fcluster(link, t=n_clusters, criterion="maxclust")
    n = len(order)
    w = np.ones(n)
    cl = [order]
    while cl:
        nxt = []
        for c in cl:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            l, r = c[:h], c[h:]
            vl = _naive_var(X, l); vr = _naive_var(X, r)
            a = 1.0 - vl / (vl + vr)
            w[l] *= a; w[r] *= 1.0 - a
            nxt += [l, r]
        cl = nxt
    return w / w.sum(), labels


def _naive_var(X, idx):
    """Variance of the inverse-variance sub-portfolio, from the sample columns
    only, so no n x n block is ever formed."""
    sub = X[:, idx]
    v = sub.var(0, ddof=1); v[v == 0] = 1e-12
    iv = 1.0 / v; iv /= iv.sum()
    p = sub @ iv
    return float(p.var(ddof=1))


def block_min_var(X, labels, ridge=1e-3):
    """Minimum variance inside each cluster, clusters combined by inverse
    variance. The only optimiser that exists at this rank: each block is small
    enough to invert even though the whole matrix is not."""
    n = X.shape[1]
    w = np.zeros(n)
    inner, var = [], []
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        sub = X[:, idx]
        C = np.cov(sub, rowvar=False)
        C = np.atleast_2d(C)
        C = C + ridge * np.trace(C) / max(len(idx), 1) * np.eye(len(idx))
        x = np.linalg.solve(C, np.ones(len(idx)))
        x = np.maximum(x, 0.0)
        x = x / x.sum() if x.sum() > 0 else np.full(len(idx), 1 / len(idx))
        inner.append((idx, x)); var.append(float((sub @ x).var(ddof=1)))
    iv = 1.0 / np.maximum(np.asarray(var), 1e-12); iv /= iv.sum()
    for (idx, x), a in zip(inner, iv):
        w[idx] = a * x
    return w / w.sum()


def run(n=5000, T=104, draws=30, n_clusters=50, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(draws):
        B, d = market(rng, n)
        X = sample(rng, B, d, T)
        t0 = time.time()
        w_hrp, labels = hrp_big(None, X, n_clusters)
        t_hrp = time.time() - t0
        v = X.var(0, ddof=1); v[v == 0] = 1e-12
        iv = (1.0 / v) / np.sum(1.0 / v)
        res = {
            "equal weight": np.full(n, 1.0 / n),
            "inverse variance": iv,
            "HRP": w_hrp,
            "block min-var": block_min_var(X, labels),
        }
        for k, w in res.items():
            out.setdefault(k, []).append(true_risk(w, B, d))
        out.setdefault("_hrp_seconds", []).append(t_hrp)
        if (i + 1) % 10 == 0:
            print(f"    draw {i+1}/{draws}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 104
    print(f"n = {n}, T = {T}, T/n = {T/n:.3f}\n")
    out = run(n=n, T=T, draws=int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    hrp = np.asarray(out["HRP"])
    print(f"\n{'method':20s}{'median true variance':>22s}{'ratio to HRP':>15s}{'beats HRP':>12s}")
    for k in ("equal weight", "inverse variance", "HRP", "block min-var"):
        x = np.asarray(out[k])
        print(f"{k:20s}{np.median(x):22.5f}{np.median(x/hrp):15.3f}"
              f"{np.mean(x < hrp):11.0%}")
    def wilson(p, nn, z=1.96):
        d = 1 + z*z/nn; c = (p + z*z/(2*nn))/d
        h = z*((p*(1-p)/nn + z*z/(4*nn*nn))**0.5)/d
        return max(c-h,0), min(c+h,1)
    print()
    for k in ("equal weight", "inverse variance", "block min-var"):
        x = np.asarray(out[k]); w = float(np.mean(x < hrp))
        lo, hi = wilson(w, len(x))
        print(f"  {k:20s} beats HRP {w:.0%}  95% [{lo:.0%}, {hi:.0%}]  n={len(x)}")
    print(f"\nHRP linkage and recursion: {np.median(out['_hrp_seconds']):.1f}s per fit")
