"""The candidate restriction rules, in increasing order of what they ask for.

    equal weight      ignores the parent entirely. A floor, not a candidate.
    proportional      renormalise the parent weights. No data. Luce's axiom.
    race              calibrate abilities from the parent weights and race the
                      survivors. No data. Thurstone, which is not IIA.
    race + factor     the same race under a k-factor CORRELATION estimated on
                      the parent universe. Costs k(n+1) numbers.
    estimate + solve  a full sub-covariance and an optimiser. m(m+1)/2 numbers.
    oracle            the long-only optimum of the true sub-covariance.

Volatility is deliberately absent from the middle three, and that is the
point rather than an omission. An optimal parent already holds less of a
volatile name, so the abilities calibrated from the parent weights carry the
volatility. Handing the race a structure scaled by volatility as well applies
it twice: measured, that moves the restricted portfolio by 0.60 in L1 where
correlation alone moves it by 0.06. What restriction actually changes is which
competitors are present, and how the departed names' wins redistribute depends
on how the survivors co-move.

The race must be calibrated on the PARENT field. Calibrating on the already
restricted weights and racing under the same law is the round-trip identity
and returns its input, which makes every row equal to proportional.
"""
import numpy as np
import cvxpy as cp
import winning


def long_only_min_var(C):
    """Long-only minimum variance.

    The problem is rebuilt on every call rather than cached by size. Caching
    a cvxpy Problem and re-solving it through a Parameter carries solver state
    between solves, so the answer depends on what was solved before it, and
    the same draw then lands differently depending on how the run was sharded.
    Measured at 1e-8 relative, which is small and still enough to break the
    invariance check. A fresh build costs 0.12s at m=200, against 26s for the
    calibration in the same draw, so there is nothing to buy here.
    """
    m = C.shape[0]
    if m == 1:
        return np.ones(1)
    ev, V = np.linalg.eigh((C + C.T) / 2)
    P = V @ np.diag(np.maximum(ev, 1e-10)) @ V.T
    w = cp.Variable(m)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(P))),
               [cp.sum(w) == 1, w >= 0]).solve(solver=cp.CLARABEL)
    x = np.maximum(np.asarray(w.value, float), 0.0)
    return x / x.sum()


def long_only_max_sharpe(C, mu):
    """Long-only maximum Sharpe ratio for covariance C and expected returns mu.

    Minimise w'Cw subject to mu'w = 1 and w >= 0, then rescale to the budget.
    The problem is rebuilt on every call for the same reason as
    long_only_min_var: a cached cvxpy Problem carries solver state between
    solves and breaks the sharding invariance.
    """
    m = C.shape[0]
    mu = np.asarray(mu, float)
    if m == 1 or not np.any(mu > 0):
        return np.full(m, 1.0 / m)
    ev, V = np.linalg.eigh((C + C.T) / 2)
    P = V @ np.diag(np.maximum(ev, 1e-10)) @ V.T
    w = cp.Variable(m)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(P))),
               [mu @ w == 1, w >= 0]).solve(solver=cp.CLARABEL)
    if w.value is None:
        return np.full(m, 1.0 / m)
    x = np.maximum(np.asarray(w.value, float), 0.0)
    t = x.sum()
    return x / t if t > 0 else np.full(m, 1.0 / m)


def long_only_min_cvar(scen, mu, alpha=0.95):
    """Long-only portfolio with the least expected shortfall per unit of expected
    return: minimise CVaR_alpha of the loss over scenarios `scen` (S x m, zero
    mean) subject to mu'w = 1 and w >= 0, then rescale to the budget. The
    Rockafellar-Uryasev linear program; the tail counterpart of
    long_only_max_sharpe."""
    S, m = scen.shape
    mu = np.asarray(mu, float)
    if m == 1 or not np.any(mu > 0):
        return np.full(m, 1.0 / m)
    w = cp.Variable(m); t = cp.Variable(); u = cp.Variable(S, nonneg=True)
    prob = cp.Problem(cp.Minimize(t + cp.sum(u) / (S * (1.0 - alpha))),
                      [u >= -(scen @ w) - t, mu @ w == 1, w >= 0])
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return np.full(m, 1.0 / m)
    x = np.maximum(np.asarray(w.value, float), 0.0)
    tot = x.sum()
    return x / tot if tot > 0 else np.full(m, 1.0 / m)


def expected_shortfall(scen, w, alpha=0.95):
    """Mean loss in the worst (1 - alpha) share of scenarios, for portfolio w."""
    loss = -(scen @ np.asarray(w, float))
    k = max(1, int(round((1.0 - alpha) * len(loss))))
    return float(np.sort(loss)[-k:].mean())


def factor_correlation(X, k):
    """A k-factor form of the CORRELATION of X, with unit diagonal.

    The variances are returned too, for callers that want them, but the
    restriction rules do not: see the module docstring.
    """
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Xc / sd
    # Top-k eigenpairs of Z'Z/(T-1) without forming it. At index scale the
    # correlation is 5000x5000 and rank T, so the SVD of the T x n panel gives
    # the same vectors for a thousandth of the work.
    _, sv, Wt = np.linalg.svd(Z, full_matrices=False)
    ev = (sv[:k] ** 2) / (len(Z) - 1)
    V = Wt[:k].T * np.sqrt(np.clip(ev, 0.0, None))
    D = np.clip(1.0 - (V ** 2).sum(1), 1e-6, None)
    sc = np.sqrt((V ** 2).sum(1) + D)
    return sd ** 2, V / sc[:, None], D / sc ** 2


def proportional(parent, idx):
    w = parent[idx]
    return w / w.sum()


def flattened(parent, idx, alpha=0.75):
    """Proportional restriction flattened toward equal weight. The null.

    An independent race is this to within three percent in L1, since with no
    correlation nothing encodes which survivor a departed name resembled and
    all the race can do is de-concentrate. Any claim for the race is measured
    against this row, not against proportional.
    """
    w = proportional(parent, idx) ** alpha
    return w / w.sum()


def race(parent, idx, V=None, D=None):
    """Calibrate on the parent field, then race among the survivors."""
    kw = {k: val for k, val in (("V", V), ("D", D)) if val is not None}
    a = np.asarray(winning.calibrate_abilities(
        np.maximum(parent, 1e-12), target_floor=1e-12, **kw), float)
    sub = {k: val[idx] for k, val in kw.items()}
    p = winning.race_probabilities(a[idx], **sub)
    p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
    return p / p.sum()


def factor_covariance(X, k):
    """A k-factor covariance of X, as (sd, V, D) with V V' + diag(D) the correlation."""
    sd2, V, D = factor_correlation(X, k)
    return np.sqrt(sd2), V, D


def black_litterman(parent, idx, sd, V, D):
    """Reverse-optimize the parent, restrict the implied returns, re-optimize.

    Black-Litterman with no views. The parent is the equilibrium book, so its
    implied excess returns are Pi = Sigma w up to the risk aversion, which
    cancels under the budget. Restricting Pi to the survivors and solving the
    long-only maximum-Sharpe problem under the same covariance estimate is the
    linear answer to the question the race answers non-linearly, and it gets
    the same inputs: the parent weights and a k-factor covariance fitted on
    the parent universe.

    Given the true covariance it is the oracle exactly, since then Pi is the
    market's own m. Its gap from the oracle is therefore the covariance
    estimate and nothing else.
    """
    y = sd * np.asarray(parent, float)
    Pi = sd * (V @ (V.T @ y) + D * y)
    Vi, Di, si = V[idx], D[idx], sd[idx]
    S = np.outer(si, si) * (Vi @ Vi.T + np.diag(Di))
    return long_only_max_sharpe((S + S.T) / 2, Pi[idx])


def crash_law(V, D, regime):
    """The crash-day law as a factor race: the k-factor correlation with
    dispersion scaled by rho, plus one factor for the crash's random magnitude,
    loading a * sd(|xi|) * beta. A deterministic shift and a common scale are
    invisible to a race after calibration; the random magnitude is not."""
    p, a, rho, beta = regime
    Vx = np.column_stack([rho * np.asarray(V, float), np.sqrt(1.0 - 2.0 / np.pi) * a * np.asarray(beta, float)])
    return Vx, rho ** 2 * np.asarray(D, float)


def race_tail(parent, idx, V, D, regime):
    """Calibrate on the parent field under the crash-day law and race the
    survivors under it: the market read as its own answer to crash days."""
    Vx, Dx = crash_law(V, D, regime)
    return race(parent, idx, V=Vx, D=Dx)


def race_avoid_worst(parent, idx, V, D, worst=0.10):
    """Abilities from the parent under the k-factor law, read on the survivors
    through a different chart: the probability of NOT being among the worst
    `worst` share of the field. Weight is tail avoidance rather than winning."""
    from winning.factor.topk import top_k_probabilities
    a = np.asarray(winning.calibrate_abilities(
        np.maximum(parent, 1e-12), target_floor=1e-12, V=V, D=D), float)
    m = len(idx)
    k = max(1, m - int(round(worst * m)))
    q = np.asarray(top_k_probabilities(a[idx], k, V=V[idx], D=D[idx]), float)
    q = np.maximum(q, 0.0)
    return q / q.sum()


def black_litterman_tail(parent, idx, sd, V, D, regime):
    """Black-Litterman handed the crash-day covariance: the second-moment method
    given the same tail law the race is given."""
    Vx, Dx = crash_law(V, D, regime)
    return black_litterman(parent, idx, sd, Vx, Dx)


def sector_structure(X, sector):
    """A nested factor structure from a panel with KNOWN sector labels, in
    correlation units: one global factor (first principal component) with a
    per-name coupling, one private factor per sector with a per-name loading,
    and the idiosyncratic remainder. O(n T)."""
    Z = X - X.mean(0)
    Z = Z / np.where(Z.std(0, ddof=1) > 0, Z.std(0, ddof=1), 1.0)
    T = len(Z)
    _, sv, Wt = np.linalg.svd(Z, full_matrices=False)
    f = Z @ Wt[0]                                   # global factor score
    f = f / f.std(ddof=1)
    coupling = (Z.T @ f) / (T - 1)
    E = Z - np.outer(f, coupling)
    loading = np.zeros(Z.shape[1])
    for g in np.unique(sector):
        mem = np.flatnonzero(sector == g)
        if len(mem) < 2:
            continue
        s_g = E[:, mem].mean(1)
        sd_g = s_g.std(ddof=1)
        if sd_g > 0:
            loading[mem] = (E[:, mem].T @ (s_g / sd_g)) / (T - 1)
    D = np.clip(1.0 - coupling ** 2 - loading ** 2, 0.05, None)
    scale = np.sqrt(coupling ** 2 + loading ** 2 + D)
    return coupling / scale, loading / scale, D / scale ** 2


def race_sectors(parent, idx, sector, coupling, loading, D, iters=60, tol=1e-9):
    """Calibrate on the parent field under the nested law, one private factor
    per sector plus a global factor, and race the survivors under it. The chart
    a k-factor correlation cannot represent: it knows which names share a
    sector, so a departed name's weight can flow to its own sector."""
    from winning.factor.blocks import nested_race_probabilities, abilities_from_block_race
    sector = np.asarray(sector); coupling = np.asarray(coupling, float)
    loading = np.asarray(loading, float); D = np.asarray(D, float)
    target = np.maximum(np.asarray(parent, float), 1e-12)
    target = target / target.sum()

    def fwd(mu, c, l, d, cp):
        return np.asarray(nested_race_probabilities(mu, c, l, d, coupling=cp), float)

    m0 = abilities_from_block_race(target, sector, loading, D)
    mu = np.asarray(m0[0] if isinstance(m0, tuple) else m0, float)
    mu = mu - mu.mean()

    def residual(m_):
        pr = np.maximum(fwd(m_, sector, loading, D, coupling), 1e-300)
        return float(np.abs(pr - target).sum()), pr

    resid, pr = residual(mu)
    damp = 0.3
    for _ in range(iters):
        if resid < tol:
            break
        step = np.log(pr) - np.log(target)
        for _try in range(12):
            cand = mu + damp * step
            cand = cand - cand.mean()
            r_new, pr_new = residual(cand)
            if r_new < resid:
                mu, resid, pr = cand, r_new, pr_new
                damp = min(1.0, damp * 1.3)
                break
            damp *= 0.5
        else:
            break
    if resid > 1e-6:
        raise ValueError(f"race_sectors did not converge: L1 residual {resid:.2e}")
    w = fwd(mu[idx], sector[idx], loading[idx], D[idx], coupling[idx])
    return w / w.sum()


def estimate_and_solve(X, idx):
    m = len(idx)
    return long_only_min_var(np.cov(X[:, idx], rowvar=False) + 1e-8 * np.eye(m))
