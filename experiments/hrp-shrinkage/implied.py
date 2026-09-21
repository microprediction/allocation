"""Does every allocator have an equivalent estimator?

Existence is trivial and constructive.  For any w != 0 with 1'w > 0 put
    u = w/||w||,  Q = I - uu',
    X = (1 1')/(1'w) + eps * Q M Q          (M any SPD)
then Xw = 1 and X is positive definite (kernel of X is trivial), so w is
exactly the global minimum-variance portfolio of X.  Long-short included.

So the question worth asking is not whether an implied covariance exists but
how far it sits from the data.  Fix the scale by c = w'Sw and project S onto
{X symmetric : Xw = c 1} in Frobenius norm.  The minimum-norm correction is

    r = c1 - Sw,   D = (r w' + w r')/(w'w) - (r'w)(w w')/(w'w)^2

and ||D||_F / ||S||_F is the distortion: how big a lie about the covariance
the allocator is telling.  Minimum variance scores exactly zero.
"""
import numpy as np
from confound import true_sigma, hrp, min_var, block_filter


def implied(S, w):
    c = float(w @ S @ w)
    r = c * np.ones_like(w) - S @ w
    ww = float(w @ w)
    D = (np.outer(r, w) + np.outer(w, r)) / ww - float(r @ w) * np.outer(w, w) / ww**2
    X = S + D
    ev = np.linalg.eigvalsh(X)
    return (np.linalg.norm(D) / np.linalg.norm(S), ev.min() / ev.max())


def inv_var(S):
    w = 1.0 / np.diag(S)
    return w / w.sum()


def equal(S):
    n = S.shape[0]
    return np.ones(n) / n


def risk_parity(S, iters=600):
    n = S.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = S @ w
        w = w * (1.0 / (n * mrc))
        w = np.maximum(w, 1e-12)
        w /= w.sum()
    return w


def check_existence(rng, n=12):
    """The constructive proof, exercised on a long-short target."""
    w = rng.normal(size=n)
    w = w / w.sum()                       # 1'w = 1 > 0, mixed signs
    u = w / np.linalg.norm(w)
    Q = np.eye(n) - np.outer(u, u)
    M = rng.normal(size=(n, n)); M = M @ M.T + n * np.eye(n)
    one = np.ones(n)
    X = np.outer(one, one) / float(one @ w) + 0.3 * Q @ M @ Q
    back = np.linalg.solve(X, one); back /= back.sum()
    return (np.linalg.eigvalsh(X).min(), np.abs(back - w).max(), (w < 0).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    lam, err, neg = check_existence(rng)
    print("existence, long-short target:")
    print(f"  smallest eigenvalue of the constructed covariance  {lam:.4f}")
    print(f"  its min-variance portfolio reproduces w to         {err:.2e}")
    print(f"  ({neg} of 12 weights were negative)\n")

    ALLOC = {"minimum variance": min_var, "HRP": hrp, "risk parity": risk_parity,
             "inverse variance": inv_var, "equal weight": equal}
    rows = {k: [] for k in ALLOC}
    cond = {k: [] for k in ALLOC}
    for _ in range(300):
        Sig, _ = true_sigma(rng, 5, 8, 0.7, 0.15)
        X = rng.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        for k, f in ALLOC.items():
            d, c = implied(S, f(S))
            rows[k].append(d); cond[k].append(c)
    print("implied-estimator distortion, median over 300 sample covariances")
    print(f"{'allocator':20s} {'||D||/||S||':>12s} {'lambda_min/lambda_max of S+D':>30s}")
    for k in ALLOC:
        print(f"{k:20s} {np.median(rows[k]):12.3f} {np.median(cond[k]):30.4f}")
