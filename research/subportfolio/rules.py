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


def estimate_and_solve(X, idx):
    m = len(idx)
    return long_only_min_var(np.cov(X[:, idx], rowvar=False) + 1e-8 * np.eye(m))
