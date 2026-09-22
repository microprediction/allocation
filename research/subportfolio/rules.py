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


def factor_correlation(X, k):
    """A k-factor form of the CORRELATION of X, with unit diagonal.

    The variances are returned too, for callers that want them, but the
    restriction rules do not: see the module docstring.
    """
    Xc = X - X.mean(0)
    sd = Xc.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Xc / sd
    R = (Z.T @ Z) / (len(Z) - 1)
    ev, U = np.linalg.eigh((R + R.T) / 2)
    i = np.argsort(ev)[::-1][:k]
    V = U[:, i] * np.sqrt(np.clip(ev[i], 0.0, None))
    D = np.clip(1.0 - (V ** 2).sum(1), 1e-6, None)
    sc = np.sqrt((V ** 2).sum(1) + D)
    return sd ** 2, V / sc[:, None], D / sc ** 2


def proportional(parent, idx):
    w = parent[idx]
    return w / w.sum()


def race(parent, idx, V=None, D=None):
    """Calibrate on the parent field, then race among the survivors.

    Returns the weights and the calibration diagnostics. The diagnostics are
    not optional decoration. `calibrate_abilities` returns its last iterate
    after a warning when it fails to converge, and it fails exactly where this
    study puts it: a field in which a few names carry nearly all the mass
    (winning issue #149). A long-only minimum-variance parent is that field,
    holding on the order of fifteen names out of four hundred. A run that does
    not record convergence cannot tell a result from a non-result, so every
    draw carries the residual and the iteration count.
    """
    kw = {k: val for k, val in (("V", V), ("D", D)) if val is not None}
    a, info = winning.calibrate_abilities(
        np.maximum(parent, 1e-12), target_floor=1e-12, return_info=True, **kw)
    a = np.asarray(a, float)
    sub = {k: val[idx] for k, val in kw.items()}
    p = winning.race_probabilities(a[idx], **sub)
    p = np.asarray(p[0] if isinstance(p, tuple) else p, float)
    return p / p.sum(), {"converged": bool(info["converged"]),
                         "residual": float(info["max_log_residual"]),
                         "iterations": int(info["iterations"])}


def estimate_and_solve(X, idx):
    m = len(idx)
    return long_only_min_var(np.cov(X[:, idx], rowvar=False) + 1e-8 * np.eye(m))
