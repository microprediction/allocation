"""Diagnostics for portfolio constructions.

``negative_weight_diagnostic`` answers a single practical question: does a signed
allocator (minimum-variance, factor minimum-variance, ...) actually *need* its
short positions, or are the negatives just estimation noise from inverting a badly
conditioned covariance? Long-only rules never face this; a signed rule should have
to earn its shorts.

It walks the allocator forward and isolates the shorts' contribution exactly, by
re-pricing the *same* weight path with the shorts removed (clipped to zero and
renormalised). Three numbers come out:

* ``sharpe_gain_from_shorts`` -- net Sharpe of the signed book minus the
  long-only projection. If this is not clearly positive, the shorts do not pay
  out of sample.
* ``sign_flip_rate`` -- how often a name that is ever short flips weight sign
  period to period. High means the short book is unstable (noise); low means the
  positions persist (signal).
* ``short_mass`` -- the average gross short fraction ``sum |min(w, 0)|``.

Verdict ``shorts_needed`` is ``True`` only when the shorts add net Sharpe *and*
are stable.
"""

from __future__ import annotations

import numpy as np

from .backtest import portfolio_metrics, walk_forward

__all__ = ["negative_weight_diagnostic"]


def _long_only_projection(W: np.ndarray) -> np.ndarray:
    """Clip each row to the long-only simplex (drop shorts, renormalise)."""
    Wl = np.clip(W, 0.0, None)
    s = Wl.sum(axis=1, keepdims=True)
    n = W.shape[1]
    return np.where(s > 0, Wl / np.where(s > 0, s, 1.0), 1.0 / n)


def negative_weight_diagnostic(
    factory, returns, *, warmup: int = 60, cost_bps: float = 10.0,
    periods_per_year: int = 252, min_gain: float = 0.05, max_flip: float = 0.25,
):
    """Decide whether a signed allocator's short positions are needed.

    ``factory`` builds a fresh signed estimator; ``returns`` is ``(n_obs, n)``.
    Returns a dict of the three measures plus the matched signed / long-only net
    Sharpe and the ``shorts_needed`` verdict.
    """
    returns = np.asarray(returns, dtype=float)
    rets, W = walk_forward(factory, returns, warmup=warmup)
    R = returns[warmup:warmup + len(W)]            # realised next-period returns per path row

    Wl = _long_only_projection(W)
    rets_long = (Wl * R).sum(axis=1)

    signed = portfolio_metrics(rets, W, periods_per_year, cost_bps)
    longonly = portfolio_metrics(rets_long, Wl, periods_per_year, cost_bps)

    short_mass = float(np.mean(np.abs(np.minimum(W, 0.0)).sum(axis=1)))
    ever_short = (W < -1e-9).any(axis=0)
    if ever_short.any() and len(W) > 1:
        signs = np.sign(W[:, ever_short])
        flip_rate = float(np.mean(np.abs(np.diff(signs, axis=0)) > 0))
    else:
        flip_rate = 0.0

    gain = float(signed["net_sharpe"] - longonly["net_sharpe"])
    return {
        "short_mass": short_mass,
        "sign_flip_rate": flip_rate,
        "net_sharpe_signed": float(signed["net_sharpe"]),
        "net_sharpe_longonly": float(longonly["net_sharpe"]),
        "sharpe_gain_from_shorts": gain,
        "var_signed": float(signed["ann_vol"] ** 2),
        "var_longonly": float(longonly["ann_vol"] ** 2),
        "shorts_needed": bool(gain > min_gain and flip_rate < max_flip and short_mass > 1e-6),
    }
