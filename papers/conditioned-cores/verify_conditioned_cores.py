"""Certificate for the conditioned-cores note.

Checks, on random covariances satisfying the orbital model:
  1. Proposition 1: conditioning cluster i on the other cores equals
     conditioning it on everything (block and companion vector).
  2. Proposition 2: gamma = 1 with minimum variance at both tiers returns
     the global minimum-variance portfolio.
  3. Proposition 3: the same with the ones vector replaced by a random
     exposure vector u (maximum Sharpe when u is the mean).
And confirms all fail when the orbital model is violated.
"""
import numpy as np

TOL = 1e-10


def random_spd(rng, n, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T / n + np.eye(n))


def orbital_model_cov(rng, sizes, violation=0.0):
    """sizes[j] = number of orbitals in cluster j. Index 0 of each cluster is its core.

    Returns Sigma and the cluster index sets. With violation > 0 the orbital
    residuals are correlated across clusters, breaking the model.
    """
    k = len(sizes)
    n = sum(1 + m for m in sizes)
    scc = random_spd(rng, k)
    idx, pos = [], 0
    for m in sizes:
        idx.append(list(range(pos, pos + 1 + m)))
        pos += 1 + m
    cores = [I[0] for I in idx]
    betas = [rng.standard_normal(m) * 0.8 + 1.0 for m in sizes]
    eps = [random_spd(rng, m, 0.3) for m in sizes]
    sigma = np.zeros((n, n))
    for j, Ij in enumerate(idx):
        Oj = Ij[1:]
        for l, Il in enumerate(idx):
            Ol = Il[1:]
            s = scc[j, l]
            sigma[cores[j], cores[l]] = s
            sigma[np.ix_(Oj, [cores[l]])] = (betas[j] * s)[:, None]
            sigma[np.ix_([cores[j]], Ol)] = (betas[l] * s)[None, :]
            sigma[np.ix_(Oj, Ol)] = np.outer(betas[j], betas[l]) * s
        sigma[np.ix_(Oj, Oj)] += eps[j]
    if violation > 0:
        allO = [i for I in idx for i in I[1:]]
        sigma[np.ix_(allO, allO)] += violation * random_spd(rng, len(allO), 1.0)
    assert np.all(np.linalg.eigvalsh(sigma) > 0)
    return sigma, idx, cores


def full_pair(sigma, I, u=None):
    n = sigma.shape[0]
    u = np.ones(n) if u is None else u
    rest = [i for i in range(n) if i not in I]
    S_ii = sigma[np.ix_(I, I)]
    S_ir = sigma[np.ix_(I, rest)]
    S_rr = sigma[np.ix_(rest, rest)]
    coef = S_ir @ np.linalg.inv(S_rr)
    return S_ii - coef @ S_ir.T, u[I] - coef @ u[rest]


def cheap_pair(sigma, I, other_cores, gamma=1.0, u=None):
    """Conditioned pair against the other cores only, with exposure vector u."""
    n = sigma.shape[0]
    u = np.ones(n) if u is None else u
    S_ii = sigma[np.ix_(I, I)]
    S_ic = sigma[np.ix_(I, other_cores)]
    S_cc = sigma[np.ix_(other_cores, other_cores)]
    coef = gamma * S_ic @ np.linalg.inv(S_cc)
    return S_ii - coef @ S_ic.T, u[I] - coef @ u[other_cores]


def conditioned_cores(sigma, idx, cores, gamma=1.0, u=None):
    """Tactical rule Q^{-1} b per cluster, then the Sigma^{-1}u objective on the span."""
    n = sigma.shape[0]
    u = np.ones(n) if u is None else u
    V = np.zeros((n, len(idx)))
    for i, I in enumerate(idx):
        other = [c for j, c in enumerate(cores) if j != i]
        Q, b = cheap_pair(sigma, I, other, gamma, u)
        v = np.linalg.solve(Q, b)
        V[I, i] = v / v.sum()
    top = V.T @ sigma @ V
    a = np.linalg.solve(top, V.T @ u)
    a /= a.sum()
    return V @ a


def global_solution(sigma, u=None):
    u = np.ones(sigma.shape[0]) if u is None else u
    w = np.linalg.solve(sigma, u)
    return w / w.sum()


def run(rng, sizes, violation):
    sigma, idx, cores = orbital_model_cov(rng, sizes, violation)
    n = sigma.shape[0]
    mu = rng.standard_normal(n) + 3.0
    err1 = 0.0
    for i, I in enumerate(idx):
        other = [c for j, c in enumerate(cores) if j != i]
        for u in (None, mu):
            Qf, bf = full_pair(sigma, I, u)
            Qc, bc = cheap_pair(sigma, I, other, 1.0, u)
            err1 = max(err1, np.abs(Qf - Qc).max(), np.abs(bf - bc).max())
    err2 = np.abs(conditioned_cores(sigma, idx, cores) - global_solution(sigma)).max()
    err3 = np.abs(conditioned_cores(sigma, idx, cores, u=mu) - global_solution(sigma, mu)).max()
    return err1, err2, err3


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst = [0.0, 0.0, 0.0]
    for trial in range(200):
        k = rng.integers(2, 7)
        sizes = list(rng.integers(1, 6, size=k))
        errs = run(rng, sizes, violation=0.0)
        worst = [max(w, e) for w, e in zip(worst, errs)]
    print(f"orbital model holds: |full - cheap complement| = {worst[0]:.2e}, "
          f"|cores - minvar| = {worst[1]:.2e}, |cores - Sigma^-1 u| = {worst[2]:.2e}")
    assert all(w < TOL for w in worst)

    viol = [run(rng, [3, 2, 4], violation=0.5) for _ in range(20)]
    mins = [min(e[j] for e in viol) for j in range(3)]
    print(f"orbital model violated: min errors over 20 trials = "
          f"{mins[0]:.2e}, {mins[1]:.2e}, {mins[2]:.2e}")
    assert mins[0] > 1e-3 and mins[1] > 1e-4 and mins[2] > 1e-4
    print("certificate ok")
