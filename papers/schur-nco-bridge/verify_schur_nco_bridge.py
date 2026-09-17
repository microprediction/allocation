"""Certificate for the Schur bridge note (NCO to unconstrained minimum variance).

Each check_* function returns a dict of measured errors. `main` asserts them.
The repository test suite imports this file and runs the same checks.

  check_sufficiency      Prop 1: complement against other knots = against everything
  check_bridge_right     Prop 2(2,3): gamma = 1 stacks to Sigma^{-1} u, unnormalized
                         and through the two-tier recipe
  check_bridge_left      Prop 2(1): gamma = 0 equals an independent NCO
  check_violation        all of the above fail when the gateway model is violated
  check_diagnostic       regressing on other knots false-accepts; R_j does not
  check_change_of_vars   covariance-only rules: apply to M, map back by x = y / b
  check_proxy            Prop 3: V' Sigma V = K Sigma_PP K + D
  check_symmetric        R_j(c) = (1 - rho_c) lambda_O Cov(f_j, r_{-j})
  check_quotient         complements (and companion vectors) compose, so tree = flat at gamma = 1
"""
import numpy as np

TOL = 1e-10


def random_spd(rng, n, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T / n + np.eye(n))


def clusters(sizes):
    idx, pos = [], 0
    for m in sizes:
        idx.append(list(range(pos, pos + 1 + m)))
        pos += 1 + m
    return idx, [I[0] for I in idx]


def gateway_model_cov(rng, sizes, violation=0.0):
    """Index 0 of each cluster is its knot; sizes[j] remaining members follow.

    violation > 0 correlates member residuals across clusters while keeping
    them orthogonal to every knot.
    """
    k = len(sizes)
    idx, knots = clusters(sizes)
    n = idx[-1][-1] + 1
    scc = random_spd(rng, k)
    betas = [rng.standard_normal(m) * 0.8 + 1.0 for m in sizes]
    eps = [random_spd(rng, m, 0.3) for m in sizes]
    sigma = np.zeros((n, n))
    for j, Ij in enumerate(idx):
        Oj = Ij[1:]
        for l, Il in enumerate(idx):
            Ol = Il[1:]
            s = scc[j, l]
            sigma[knots[j], knots[l]] = s
            sigma[np.ix_(Oj, [knots[l]])] = (betas[j] * s)[:, None]
            sigma[np.ix_([knots[j]], Ol)] = (betas[l] * s)[None, :]
            sigma[np.ix_(Oj, Ol)] = np.outer(betas[j], betas[l]) * s
        sigma[np.ix_(Oj, Oj)] += eps[j]
    if violation > 0:
        allO = [i for I in idx for i in I[1:]]
        sigma[np.ix_(allO, allO)] += violation * random_spd(rng, len(allO))
    assert np.all(np.linalg.eigvalsh(sigma) > 0)
    return sigma, idx, knots


def full_pair(sigma, I, u):
    rest = [i for i in range(sigma.shape[0]) if i not in I]
    coef = sigma[np.ix_(I, rest)] @ np.linalg.inv(sigma[np.ix_(rest, rest)])
    return sigma[np.ix_(I, I)] - coef @ sigma[np.ix_(rest, I)], u[I] - coef @ u[rest]


def cheap_pair(sigma, I, other_knots, gamma, u):
    S_ic = sigma[np.ix_(I, other_knots)]
    coef = gamma * S_ic @ np.linalg.inv(sigma[np.ix_(other_knots, other_knots)])
    return sigma[np.ix_(I, I)] - coef @ S_ic.T, u[I] - coef @ u[other_knots]


def directions(sigma, idx, knots, gamma, u):
    n = sigma.shape[0]
    d = np.zeros(n)
    for i, I in enumerate(idx):
        other = [c for j, c in enumerate(knots) if j != i]
        Q, b = cheap_pair(sigma, I, other, gamma, u)
        d[I] = np.linalg.solve(Q, b)
    return d


def bridge(sigma, idx, knots, gamma, u):
    n = sigma.shape[0]
    d = directions(sigma, idx, knots, gamma, u)
    V = np.zeros((n, len(idx)))
    for i, I in enumerate(idx):
        V[I, i] = d[I] / d[I].sum()
    a = np.linalg.solve(V.T @ sigma @ V, V.T @ u)
    return V @ a, V


def nco(sigma, idx):
    """Independent NCO: min-var inside each block, min-var across cluster portfolios."""
    n = sigma.shape[0]
    V = np.zeros((n, len(idx)))
    for i, I in enumerate(idx):
        x = np.linalg.solve(sigma[np.ix_(I, I)], np.ones(len(I)))
        V[I, i] = x / x.sum()
    a = np.linalg.solve(V.T @ sigma @ V, np.ones(len(idx)))
    return V @ (a / a.sum())


def _random_problem(rng, violation=0.0):
    sizes = list(rng.integers(1, 6, size=rng.integers(2, 7)))
    sigma, idx, knots = gateway_model_cov(rng, sizes, violation)
    mu = rng.standard_normal(sigma.shape[0]) + 3.0
    return sigma, idx, knots, mu


def check_sufficiency(rng, trials=100, violation=0.0):
    err = 0.0
    for _ in range(trials):
        sigma, idx, knots, mu = _random_problem(rng, violation)
        for u in (np.ones(len(mu)), mu):
            for i, I in enumerate(idx):
                other = [c for j, c in enumerate(knots) if j != i]
                Qf, bf = full_pair(sigma, I, u)
                Qc, bc = cheap_pair(sigma, I, other, 1.0, u)
                err = max(err, np.abs(Qf - Qc).max(), np.abs(bf - bc).max())
    return {"pair": err}


def check_bridge_right(rng, trials=100, violation=0.0):
    stacked = recipe = 0.0
    for _ in range(trials):
        sigma, idx, knots, mu = _random_problem(rng, violation)
        for u in (np.ones(len(mu)), mu):
            target = np.linalg.solve(sigma, u)
            stacked = max(stacked, np.abs(directions(sigma, idx, knots, 1.0, u) - target).max())
            w, _ = bridge(sigma, idx, knots, 1.0, u)
            recipe = max(recipe, np.abs(w / w.sum() - target / target.sum()).max())
    return {"stacked": stacked, "recipe": recipe}


def check_bridge_left(rng, trials=100):
    err = 0.0
    for _ in range(trials):
        sigma, idx, knots, _ = _random_problem(rng, violation=0.3)  # no model needed at gamma = 0
        w, _ = bridge(sigma, idx, knots, 0.0, np.ones(sigma.shape[0]))
        err = max(err, np.abs(w / w.sum() - nco(sigma, idx)).max())
    return {"nco": err}


def check_violation(rng, trials=20):
    s = [check_sufficiency(rng, 1, violation=0.5)["pair"] for _ in range(trials)]
    r = [check_bridge_right(rng, 1, violation=0.5) for _ in range(trials)]
    return {"pair": min(s), "stacked": min(x["stacked"] for x in r),
            "recipe": min(x["recipe"] for x in r)}


def residual_R(sigma, I, c):
    O = [m for m in I if m != c]
    rest = [i for i in range(sigma.shape[0]) if i not in I]
    return sigma[np.ix_(O, rest)] - np.outer(sigma[O, c] / sigma[c, c], sigma[c, rest])


def check_diagnostic(rng):
    out = {}
    for name, violation in (("model", 0.0), ("violated", 0.5)):
        sigma, idx, knots = gateway_model_cov(rng, [3, 2, 4], violation)
        coef = R = 0.0
        for j, I in enumerate(idx):
            c, O = I[0], I[1:]
            reg = [c] + [x for l, x in enumerate(knots) if l != j]
            B = sigma[np.ix_(O, reg)] @ np.linalg.inv(sigma[np.ix_(reg, reg)])
            coef = max(coef, np.abs(B[:, 1:]).max())
            R = max(R, np.abs(residual_R(sigma, I, c)).max())
        out[name] = {"other_knot_coef": coef, "R": R}
    return out


def check_change_of_vars(rng, trials=100):
    back = 0.0
    for _ in range(trials):
        m = rng.integers(2, 6)
        Q = random_spd(rng, m)
        b = rng.uniform(0.5, 2.0, size=m) * rng.choice([-1, 1], size=m)
        M = np.diag(1 / b) @ Q @ np.diag(1 / b)
        assert np.allclose(M, np.linalg.inv(np.linalg.inv(Q) * np.outer(b, b)))
        y = np.linalg.solve(M, np.ones(m))
        x = y / b
        t = np.linalg.solve(Q, b)
        back = max(back, np.abs(x / x.sum() - t / t.sum()).max())
    # Q = I, b = (1, 2): minimum variance on M without mapping back gives (1/5, 4/5), not (1/3, 2/3)
    y = np.linalg.solve(np.diag([1.0, 0.25]), np.ones(2))
    return {"mapped_back": back, "unmapped_example": y / y.sum()}


def check_proxy(rng, trials=100):
    err = 0.0
    for _ in range(trials):
        sigma, idx, knots, _ = _random_problem(rng)
        n, k = sigma.shape[0], len(idx)
        V = np.zeros((n, k)); kappa = np.zeros(k); D = np.zeros(k)
        for i, I in enumerate(idx):
            c, O = I[0], I[1:]
            V[I, i] = rng.standard_normal(len(I))  # any portfolio on the cluster
            beta = sigma[O, c] / sigma[c, c]
            E = sigma[np.ix_(O, O)] - np.outer(beta, beta) * sigma[c, c]
            kappa[i] = V[c, i] + beta @ V[O, i]
            D[i] = V[O, i] @ E @ V[O, i]
        rhs = np.diag(kappa) @ sigma[np.ix_(knots, knots)] @ np.diag(kappa) + np.diag(D)
        err = max(err, np.abs(V.T @ sigma @ V - rhs).max())
    return {"identity": err}


def check_symmetric(rng, trials=50):
    err = 0.0
    for _ in range(trials):
        sizes = list(rng.integers(2, 5, size=3))
        idx, _ = clusters([s - 1 for s in sizes])
        n, k = idx[-1][-1] + 1, len(idx)
        F = random_spd(rng, k)
        lam = rng.uniform(0.5, 1.5, size=n)
        L = np.zeros((n, k))
        for j, I in enumerate(idx):
            L[I, j] = lam[I]
        sigma = L @ F @ L.T + np.diag(rng.uniform(0.1, 0.5, size=n))
        for j, I in enumerate(idx):
            rest = [i for i in range(n) if i not in I]
            cov_f_rest = (F[j] @ L[rest].T)
            for c in I:
                O = [m for m in I if m != c]
                rho = lam[c] ** 2 * F[j, j] / sigma[c, c]
                pred = (1 - rho) * np.outer(lam[O], cov_f_rest)
                err = max(err, np.abs(residual_R(sigma, I, c) - pred).max())
    return {"formula": err}


def check_quotient(rng, trials=50):
    """Complementing against D then against A2 equals complementing against A2 u D."""
    err = 0.0
    for _ in range(trials):
        n1, n2, n3 = rng.integers(1, 5, size=3)
        n = n1 + n2 + n3
        sigma = random_spd(rng, n)
        u = rng.standard_normal(n)
        A1 = list(range(n1)); A = list(range(n1 + n2))
        QA, bA = full_pair(sigma, A, u)                 # step 1: A against D
        loc1 = list(range(n1))
        Q2, b2 = full_pair(QA, loc1, bA)                # step 2: A1 against A2, inside the pair
        Q1, b1 = full_pair(sigma, A1, u)                # one step: A1 against everything
        err = max(err, np.abs(Q2 - Q1).max(), np.abs(b2 - b1).max())
    return {"compose": err}


def main():
    rng = np.random.default_rng(0)
    r = check_sufficiency(rng); print("Prop 1 sufficiency        ", r); assert r["pair"] < TOL
    r = check_bridge_right(rng); print("Prop 2 right end          ", r)
    assert r["stacked"] < TOL and r["recipe"] < TOL
    r = check_bridge_left(rng); print("Prop 2 left end = NCO     ", r); assert r["nco"] < TOL
    r = check_violation(rng); print("model violated (min err)  ", r)
    assert r["pair"] > 1e-3 and r["stacked"] > 1e-4 and r["recipe"] > 1e-4
    r = check_diagnostic(rng); print("diagnostic                ", r)
    assert r["model"]["R"] < TOL and r["violated"]["R"] > 1e-2
    assert r["violated"]["other_knot_coef"] < TOL  # the regression test false-accepts
    r = check_change_of_vars(rng); print("change of variables       ", r)
    assert r["mapped_back"] < TOL and np.allclose(r["unmapped_example"], [0.2, 0.8])
    r = check_proxy(rng); print("Prop 3 proxy identity     ", r); assert r["identity"] < TOL
    r = check_symmetric(rng); print("symmetric factor formula  ", r); assert r["formula"] < TOL
    r = check_quotient(rng); print("complements compose       ", r); assert r["compose"] < TOL
    print("certificate ok")


if __name__ == "__main__":
    main()
