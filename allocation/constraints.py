"""Linear constraints by log-barrier -- box bounds and group caps, kept smooth.

A hard QP projection onto ``{l <= w <= u, sum w = 1, group caps}`` kinks every
time a bound becomes active (the active set changes), which is exactly the
turnover this package avoids. Instead we apply a smooth interior-point operator:
given the base allocator's target ``w*``, return

    argmin_w  1/2 ||w - w*||^2  -  tau * [ sum_i ln(w_i - l_i) + ln(u_i - w_i)
                                           + sum_g ln(c_g - sum_{i in g} w_i) ]
    subject to  sum_i w_i = 1.

The barrier's *domain* is the constraint set, so the result is strictly feasible
for **any** ``tau > 0`` -- feasibility never depends on tuning. ``tau`` only sets
how close to a binding bound we allow; small ``tau`` approaches the exact
projection while staying C^1 in ``w*`` (hence smooth in the covariance, hence low
turnover). Group caps are assumed **disjoint** (each asset in at most one group),
the sector-cap case.
"""

from __future__ import annotations

import numpy as np

__all__ = ["barrier_constrain", "BoxConstrained", "StreamingBoxConstrained"]


def _as_vec(v, n: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or length-{n} array")
    return a.copy()


def _check_groups(groups, group_caps, n: int):
    """Validate disjoint in-range groups with one finite cap each."""
    if not groups and group_caps is None:
        return [], []
    if groups is None or group_caps is None:
        raise ValueError("groups and group_caps must be given together")
    groups = [np.asarray(g) for g in groups]
    caps = [float(c) for c in group_caps]
    if len(groups) != len(caps):
        raise ValueError(f"got {len(groups)} groups but {len(caps)} group_caps")
    seen = np.zeros(n, dtype=bool)
    out = []
    for g in groups:
        if g.size and not np.issubdtype(g.dtype, np.integer):
            raise ValueError("group members must be integer asset indices")
        g = g.astype(int).ravel()
        if np.any((g < 0) | (g >= n)):
            raise ValueError(f"group indices must lie in [0, {n})")
        if np.any(seen[g]) or len(np.unique(g)) != len(g):
            raise ValueError("groups must be disjoint (each asset in at most one group)")
        seen[g] = True
        out.append(g)
    if not np.all(np.isfinite(caps)):
        raise ValueError("group_caps must be finite")
    return out, caps


def _interior_start(lower, upper, groups, caps, budget):
    """A certified strictly-feasible point, given that one exists.

    Treat each group (and each ungrouped asset) as a block whose total can range
    over ``[sum of lowers, min(sum of uppers, cap)]``; put every block at the
    same interior fraction of its range so the totals sum to ``budget``, then
    spread each block's total at a common interior fraction of its members' boxes.
    """
    n = len(lower)
    blocks = [(g, c) for g, c in zip(groups, caps)]
    grouped = np.zeros(n, dtype=bool)
    for g in groups:
        grouped[g] = True
    blocks += [(np.array([i]), np.inf) for i in np.flatnonzero(~grouped)]
    lo = np.array([lower[g].sum() for g, _ in blocks])
    hi = np.array([min(upper[g].sum(), c) for g, c in blocks])
    theta = (budget - lo.sum()) / (hi.sum() - lo.sum())
    w = lower.copy()
    for (g, _), a, b in zip(blocks, lo, hi):
        s = a + theta * (b - a)
        span = upper[g] - lower[g]
        w[g] = lower[g] + (s - a) / span.sum() * span
    return w


def _max_step(w, dw, lower, upper, groups, caps):
    alpha = 1.0
    neg = dw < 0
    if neg.any():
        alpha = min(alpha, float(np.min((lower[neg] - w[neg]) / dw[neg])))
    pos = dw > 0
    if pos.any():
        alpha = min(alpha, float(np.min((upper[pos] - w[pos]) / dw[pos])))
    for g, cap in zip(groups or [], caps or []):
        ds = float(dw[g].sum())
        if ds > 0:
            alpha = min(alpha, (cap - float(w[g].sum())) / ds)
    return alpha


def barrier_constrain(
    target,
    lower=0.0,
    upper=1.0,
    groups=None,
    group_caps=None,
    tau: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """Project ``target`` onto the box / group-cap / simplex set via a log-barrier.

    ``groups`` is a list of index arrays (disjoint); ``group_caps`` the matching
    upper bounds on each group's total weight. Returns a strictly-feasible weight
    vector summing to one.

    Coordinates the constraints pin exactly (``lower == upper``, or a group whose
    cap equals the sum of its lower bounds) are fixed and the barrier runs over
    the rest; a set with only one feasible point returns it. Raises ``ValueError``
    if the constraints are infeasible, or if the feasible set is a face of the
    box/caps with no strict interior that is not simply a point, and
    ``RuntimeError`` if the Newton iteration does not converge.
    """
    target = np.asarray(target, dtype=float)
    n = len(target)
    if not np.all(np.isfinite(target)):
        raise ValueError("target must be finite")
    if not (np.isfinite(tau) and tau > 0):
        raise ValueError("tau must be a positive finite number")
    lower = _as_vec(lower, n, "lower")
    upper = _as_vec(upper, n, "upper")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("lower and upper must be finite")
    if np.any(lower > upper):
        raise ValueError("lower must be <= upper")
    if lower.sum() > 1.0 + 1e-9 or upper.sum() < 1.0 - 1e-9:
        raise ValueError("box [lower, upper] cannot contain a point summing to one")
    groups, caps = _check_groups(groups, group_caps, n)

    eps = 1e-12
    # pin coordinates with no room: a degenerate box, or a group capped at its floor
    fixed = upper - lower <= eps
    for g, cap in zip(groups, caps):
        if cap < lower[g].sum() - 1e-9:
            raise ValueError("group cap is below the sum of its lower bounds")
        if cap <= lower[g].sum() + eps:
            fixed[g] = True
    w = lower.copy()  # fixed coordinates sit at their lower bound
    free = np.flatnonzero(~fixed)
    budget = 1.0 - lower[fixed].sum()
    pos = np.full(n, -1)
    pos[free] = np.arange(len(free))
    sub_groups, sub_caps, capped_below_box = [], [], False
    for g, cap in zip(groups, caps):
        gf = g[~fixed[g]]
        if gf.size:
            sub_groups.append(pos[gf])
            sub_caps.append(cap - lower[g[fixed[g]]].sum())
            capped_below_box |= sub_caps[-1] < upper[gf].sum() - eps
    lo_f, up_f = lower[free], upper[free]
    lo_tot = lo_f.sum()
    hi_tot = up_f.sum() - sum(max(upper[free][g].sum() - c, 0.0) for g, c in zip(sub_groups, sub_caps))
    if budget < lo_tot - 1e-9 or budget > hi_tot + 1e-9:
        raise ValueError("infeasible constraints: no weights within the bounds and group caps sum to one")
    if budget <= lo_tot + eps:
        return w  # every free name at its lower bound is the only feasible point
    if budget >= hi_tot - eps:
        if capped_below_box:
            raise ValueError(
                "feasible set has no strict interior: the budget forces a group cap to "
                "bind exactly; loosen a bound or a cap"
            )
        w[free] = up_f  # every free name at its upper bound is the only feasible point
        return w

    v, ok = _newton(target[free], lo_f, up_f, sub_groups, sub_caps, budget, tau, max_iter, tol)
    w[free] = v
    if not np.all(np.isfinite(w)):
        raise RuntimeError("barrier_constrain produced non-finite weights")
    if not ok:
        raise RuntimeError(f"barrier_constrain did not converge in {max_iter} Newton iterations")
    viol = max(abs(w.sum() - 1.0), float(np.max(lower - w)), float(np.max(w - upper)),
               max((w[g].sum() - c for g, c in zip(groups, caps)), default=0.0))
    if viol > 1e-8:
        raise RuntimeError(f"barrier_constrain result violates the constraints by {viol:.3g}")
    return w


def _newton(target, lower, upper, groups, caps, budget, tau, max_iter, tol):
    """Equality-constrained damped Newton on the barrier objective from a strict interior."""
    n = len(target)
    w = _interior_start(lower, upper, groups, caps, budget)
    ones = np.ones(n)
    for _ in range(max_iter):
        dl = w - lower
        du = upper - w
        grad = (w - target) - tau * (1.0 / dl - 1.0 / du)
        H = np.diag(1.0 + tau * (1.0 / dl**2 + 1.0 / du**2))
        for g, cap in zip(groups, caps):
            s = cap - float(w[g].sum())
            grad[g] += tau / s
            H[np.ix_(g, g)] += tau / s**2
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = H
        M[:n, n] = ones
        M[n, :n] = ones
        rhs = np.empty(n + 1)
        rhs[:n] = -grad
        rhs[n] = -(w.sum() - budget)
        dw = np.linalg.solve(M, rhs)[:n]
        step = min(1.0, 0.99 * _max_step(w, dw, lower, upper, groups, caps))
        w = w + step * dw
        if step * np.max(np.abs(dw)) < tol:
            return w, True
    return w, False


class BoxConstrained:
    """Wrap a batch estimator with box bounds and (disjoint) group caps.

    Parameters
    ----------
    estimator : BaseOnlinePortfolio
        Base allocator; its weights are the target projected each step.
    lower, upper : float or array, default 0.0 / 1.0
        Per-asset bounds (scalar broadcasts). ``upper`` is the per-name cap.
    groups : list of index lists or None
        Disjoint asset groups (e.g. sectors).
    group_caps : list of floats or None
        Upper bound on each group's total weight.
    tau : float, default 1e-4
        Barrier strength (margin from binding bounds); smaller is tighter.
    """

    def __init__(self, estimator, *, lower=0.0, upper=1.0, groups=None, group_caps=None, tau: float = 1e-4):
        self.estimator = estimator
        self.lower = lower
        self.upper = upper
        self.groups = groups
        self.group_caps = group_caps
        self.tau = tau
        self._weights = None

    def _constrain(self) -> np.ndarray:
        return barrier_constrain(
            self.estimator.weights_,
            lower=self.lower,
            upper=self.upper,
            groups=self.groups,
            group_caps=self.group_caps,
            tau=self.tau,
        )

    def fit(self, X, y=None):
        self.estimator.fit(X)
        self._weights = self._constrain()
        return self

    def partial_fit(self, X, y=None):
        if self._weights is None:
            return self.fit(X)
        self.estimator.partial_fit(X)
        self._weights = self._constrain()
        return self

    @property
    def weights_(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("BoxConstrained is not fitted; call fit first.")
        return self._weights

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = X[None, :] if X.ndim == 1 else X
        return X @ self.weights_

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in ("estimator", "lower", "upper", "groups", "group_caps", "tau")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class StreamingBoxConstrained:
    """Wrap a streaming estimator with box bounds and (disjoint) group caps.

    ``lower`` / ``upper`` are scalars or ``{id: bound}`` dicts; ``groups`` maps
    ``{id: group_label}`` and ``group_caps`` maps ``{group_label: cap}``. Bounds
    and caps are assembled for the active ids each step.
    """

    def __init__(self, estimator, *, lower=0.0, upper=1.0, groups=None, group_caps=None, tau: float = 1e-4):
        self.estimator = estimator
        self.lower = lower
        self.upper = upper
        self.groups = groups
        self.group_caps = group_caps
        self.tau = tau
        self._weights: dict = {}

    def _bound(self, spec, ids, default):
        if isinstance(spec, dict):
            return np.array([spec.get(k, default) for k in ids])
        return float(spec)

    def learn_one(self, x: dict) -> "StreamingBoxConstrained":
        self.estimator.learn_one(x)
        target = self.estimator.predict_one()
        if not target:
            return self
        ids = list(target.keys())
        lower = self._bound(self.lower, ids, 0.0)
        upper = self._bound(self.upper, ids, 1.0)
        groups = caps = None
        if self.groups and self.group_caps:
            buckets: dict = {}
            for pos, k in enumerate(ids):
                lab = self.groups.get(k)
                if lab in self.group_caps:
                    buckets.setdefault(lab, []).append(pos)
            if buckets:
                groups = [np.array(v, dtype=int) for v in buckets.values()]
                caps = [self.group_caps[lab] for lab in buckets]
        w = barrier_constrain(
            np.array([target[k] for k in ids]),
            lower=lower, upper=upper, groups=groups, group_caps=caps, tau=self.tau,
        )
        self._weights = dict(zip(ids, w))
        return self

    def predict_one(self, x: dict | None = None) -> dict:
        return dict(self._weights)

    @property
    def weights(self) -> dict:
        return dict(self._weights)
