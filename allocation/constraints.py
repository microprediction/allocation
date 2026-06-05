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


def _interior_start(lower, upper, groups, caps):
    span = upper - lower
    if span.sum() <= 0:
        raise ValueError("upper must exceed lower somewhere")
    t = (1.0 - lower.sum()) / span.sum()
    w = lower + t * span  # box-interior, sums to one
    if not groups:
        return w
    grouped = set().union(*[set(g.tolist()) for g in groups])
    free = np.array([i for i in range(len(w)) if i not in grouped], dtype=int)
    for _ in range(50):
        moved = 0.0
        for g, cap in zip(groups, caps):
            margin = 0.05 * cap
            slack = cap - w[g].sum()
            if slack < margin:
                head = w[g] - lower[g]
                tot = head.sum()
                if tot <= 1e-12:
                    raise ValueError("group cap is below the sum of its lower bounds")
                cut = np.minimum(head, (margin - slack) * head / tot)
                w[g] -= cut
                moved += float(cut.sum())
        if moved <= 1e-15:
            break
        headroom = np.clip(upper - w, 0.0, None)
        pool = free if free.size and headroom[free].sum() > 1e-12 else np.arange(len(w))
        h = headroom[pool]
        if h.sum() <= 1e-12:
            raise ValueError("infeasible box/group caps (no headroom to redistribute)")
        w[pool] += moved * h / h.sum()
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
    """
    target = np.asarray(target, dtype=float)
    n = len(target)
    lower = _as_vec(lower, n, "lower")
    upper = _as_vec(upper, n, "upper")
    if np.any(lower > upper):
        raise ValueError("lower must be <= upper")
    if lower.sum() > 1.0 + 1e-9 or upper.sum() < 1.0 - 1e-9:
        raise ValueError("box [lower, upper] cannot contain a point summing to one")
    groups = [np.asarray(g, dtype=int) for g in groups] if groups else None
    caps = list(map(float, group_caps)) if group_caps is not None else None

    w = _interior_start(lower, upper, groups, caps)
    ones = np.ones(n)
    for _ in range(max_iter):
        dl = w - lower
        du = upper - w
        grad = (w - target) - tau * (1.0 / dl - 1.0 / du)
        H = np.diag(1.0 + tau * (1.0 / dl**2 + 1.0 / du**2))
        for g, cap in zip(groups or [], caps or []):
            s = cap - float(w[g].sum())
            grad[g] += tau / s
            H[np.ix_(g, g)] += tau / s**2
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = H
        M[:n, n] = ones
        M[n, :n] = ones
        rhs = np.empty(n + 1)
        rhs[:n] = -grad
        rhs[n] = -(w.sum() - 1.0)
        dw = np.linalg.solve(M, rhs)[:n]
        step = min(1.0, 0.99 * _max_step(w, dw, lower, upper, groups, caps))
        w = w + step * dw
        if step * np.max(np.abs(dw)) < tol:
            break
    return w


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
