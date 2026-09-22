"""The 2x2 that the HRP literature does not run.

Rows:    allocator  in {HRP, minimum variance}
Columns: estimator  in {raw sample covariance, clustering-filtered sample covariance}

If HRP's edge is an allocator effect it survives the column change.
If it is an estimator effect it disappears when the optimiser is handed the
same structural information HRP helps itself to.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def true_sigma(rng, n_blocks, per_block, rho_in, rho_out):
    n = n_blocks * per_block
    C = np.full((n, n), rho_out)
    for b in range(n_blocks):
        s = slice(b * per_block, (b + 1) * per_block)
        C[s, s] = rho_in
    np.fill_diagonal(C, 1.0)
    vol = np.exp(rng.normal(0.0, 0.4, n))          # heterogeneous vols
    return (vol[:, None] * C * vol[None, :]), vol


def corr_from_cov(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d), d


def tree(S):
    R, _ = corr_from_cov(S)
    R = np.clip(R, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.5 * (1.0 - R), 0.0))
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method="single"), D


def quasi_diag(Z, n):
    from scipy.cluster.hierarchy import leaves_list
    return list(leaves_list(Z))


def hrp(S):
    """de Prado's recursive bisection with inverse-variance leaves."""
    n = S.shape[0]
    Z, _ = tree(S)
    order = quasi_diag(Z, n)
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            left, right = c[:h], c[h:]
            vl = cluster_var(S, left)
            vr = cluster_var(S, right)
            alpha = 1.0 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def cluster_var(S, idx):
    sub = S[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    iv = iv / iv.sum()
    return float(iv @ sub @ iv)


def min_var(S, ridge=0.0):
    n = S.shape[0]
    A = S + ridge * np.eye(n)
    x = np.linalg.solve(A, np.ones(n))
    return x / x.sum()


def block_filter(S, n_clusters):
    """Tola-style: keep the cluster block structure of the sample correlation,
    average away everything else. Clusters come from the SAMPLE, not the truth."""
    R, d = corr_from_cov(S)
    Z, _ = tree(S)
    lab = fcluster(Z, t=n_clusters, criterion="maxclust")
    Rf = np.zeros_like(R)
    ids = np.unique(lab)
    for a in ids:
        for b in ids:
            ia = np.where(lab == a)[0]
            ib = np.where(lab == b)[0]
            blk = R[np.ix_(ia, ib)]
            m = blk.mean() if a != b else (blk.sum() - len(ia)) / max(len(ia) ** 2 - len(ia), 1)
            Rf[np.ix_(ia, ib)] = m
    np.fill_diagonal(Rf, 1.0)
    # nearest PSD by eigenvalue clipping, then renormalise to a correlation
    ev, V = np.linalg.eigh(Rf)
    Rf = V @ np.diag(np.maximum(ev, 1e-8)) @ V.T
    dd = np.sqrt(np.diag(Rf))
    Rf = Rf / np.outer(dd, dd)
    return d[:, None] * Rf * d[None, :], lab


def run(trials=400, n_blocks=5, per_block=8, T=120, rho_in=0.7, rho_out=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n = n_blocks * per_block
    out = {k: [] for k in ("hrp_raw", "mv_raw", "hrp_flt", "mv_flt")}
    for _ in range(trials):
        Sig, _ = true_sigma(rng, n_blocks, per_block, rho_in, rho_out)
        L = np.linalg.cholesky(Sig)
        X = rng.normal(size=(T, n)) @ L.T
        S = np.cov(X, rowvar=False)
        Sf, _ = block_filter(S, n_blocks)
        for key, w in (
            ("hrp_raw", hrp(S)),
            ("mv_raw", min_var(S)),
            ("hrp_flt", hrp(Sf)),
            ("mv_flt", min_var(Sf)),
        ):
            out[key].append(float(w @ Sig @ w))
    return {k: np.array(v) for k, v in out.items()}, Sig, n


if __name__ == "__main__":
    res, Sig, n = run()
    oracle = float(min_var(Sig) @ Sig @ min_var(Sig))
    print(f"n = {n}, T = 120, 400 markets. True-covariance variance of each portfolio.")
    print(f"{'':10s} {'raw sample':>14s} {'block-filtered':>16s}")
    for a, ra, rf in (("HRP", "hrp_raw", "hrp_flt"), ("min-var", "mv_raw", "mv_flt")):
        print(f"{a:10s} {np.median(res[ra]):14.5f} {np.median(res[rf]):16.5f}")
    print(f"{'oracle':10s} {oracle:14.5f}")
    print()
    print("HRP edge over min-var, same column:")
    for col, ra, rm in (("raw sample", "hrp_raw", "mv_raw"), ("block-filtered", "hrp_flt", "mv_flt")):
        rel = np.median(res[rm] / res[ra]) - 1.0
        print(f"  {col:16s} min-var / HRP - 1 = {rel:+.1%}")
