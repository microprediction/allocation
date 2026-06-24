"""Walk-forward backtest and comparison harness (numpy-only).

Runs an estimator forward over a return panel exactly as it would trade: form
weights from the past, earn the next period, then update. Reports the metrics
this package cares about -- out-of-sample Sharpe, **turnover** (the headline,
since smoothness is the whole point), concentration, and Sharpe net of a
proportional trading cost -- and tabulates them across estimators.

    from allocation.backtest import compare, format_table
    rows = compare({"schur": lambda: SchurComplementary(gamma=0.5), ...}, returns)
    print(format_table(rows))
"""

from __future__ import annotations

import numpy as np

__all__ = ["walk_forward", "portfolio_metrics", "compare", "compare_random_subsets",
           "format_table", "make_panel", "load_returns_csv"]


def load_returns_csv(path_or_url: str, skip_cols: int = 0):
    """Load a numeric return panel from a CSV path or URL (stdlib + numpy only).

    No data is vendored into this package; point this at a raw file in a data
    repo (e.g. one of the ``winning*`` repos) at runtime. Assumes a header row
    and ``skip_cols`` leading non-numeric columns (e.g. a date). Non-finite cells
    become 0.0. Returns an ``(n_obs, n_assets)`` array.
    """
    import csv
    import io
    import urllib.request

    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
        handle = io.StringIO(text)
    else:
        handle = open(path_or_url, newline="")
    with handle as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        rows = []
        for row in reader:
            vals = []
            for cell in row[skip_cols:]:
                try:
                    x = float(cell)
                except ValueError:
                    x = 0.0
                vals.append(x if np.isfinite(x) else 0.0)
            if vals:
                rows.append(vals)
    return np.asarray(rows, dtype=float)


def walk_forward(factory, returns, warmup: int = 60):
    """Run a fresh estimator from ``factory()`` forward over ``returns``.

    Fits on the first ``warmup`` rows, then for each later row earns the
    portfolio return with the weights held going in and updates on the realised
    row. Returns ``(port_returns, weight_path)``.
    """
    returns = np.asarray(returns, dtype=float)
    T = returns.shape[0]
    est = factory()
    est.fit(returns[:warmup])
    w = np.asarray(est.weights_, dtype=float)
    rets, path = [], []
    for t in range(warmup, T):
        r = returns[t]
        rets.append(float(w @ r))
        path.append(w)
        est.partial_fit(r)
        w = np.asarray(est.weights_, dtype=float)
    return np.array(rets), np.array(path)


def _expected_shortfall(rets, alpha: float) -> float:
    """Expected shortfall (CVaR) at level ``alpha``: the mean *loss* in the worst
    ``1-alpha`` tail of the return series (e.g. ``alpha=0.95`` -> mean of the worst
    5% of days, reported as a positive loss). This is the headline tail metric --
    the thing a tail-aware allocator should minimise and a variance-only one
    ignores.
    """
    rets = np.asarray(rets, dtype=float)
    if rets.size == 0:
        return 0.0
    q = float(np.quantile(rets, 1.0 - alpha))  # left-tail return cutoff (VaR, signed)
    tail = rets[rets <= q]
    return float(-tail.mean()) if tail.size else float(-q)


def _max_drawdown(rets) -> float:
    """Max peak-to-trough drawdown of the compounded equity curve (a positive
    fraction). The worst-case path metric -- a tail of the *ordering*, not just
    the marginal, so it sees clustered losses that ES on i.i.d. days can miss.
    """
    rets = np.asarray(rets, dtype=float)
    if rets.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def portfolio_metrics(rets, weight_path, periods_per_year: int = 252, cost_bps: float = 10.0):
    """Summary metrics for a realised return series and its weight path.

    Alongside the smoothness/return headline (turnover, Sharpe), reports the
    **tail** block -- expected shortfall at 95/99%, max drawdown, and the Sortino
    ratio (downside-only Sharpe) -- so a tail-aware method (e.g. the Student-t
    Thurstone race) can be scored on the axis it is meant to win, not just on
    variance-flavoured Sharpe.
    """
    rets = np.asarray(rets, dtype=float)
    W = np.asarray(weight_path, dtype=float)
    dW = np.abs(np.diff(W, axis=0)).sum(axis=1) if len(W) > 1 else np.array([0.0])
    turnover = 0.5 * float(np.mean(dW))  # one-way fraction traded per step
    cost = (cost_bps / 1e4) * np.concatenate([[0.0], np.abs(np.diff(W, axis=0)).sum(axis=1)])
    net = rets - cost
    sd = float(np.std(rets))
    sd_net = float(np.std(net))
    downside = float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)))
    ann = np.sqrt(periods_per_year)
    return {
        "sharpe": float(np.mean(rets) / sd * ann) if sd > 0 else 0.0,
        "ann_vol": sd * ann,
        "turnover": turnover,
        "net_sharpe": float(np.mean(net) / sd_net * ann) if sd_net > 0 else 0.0,
        "max_w": float(np.mean(np.max(W, axis=1))),
        "eff_n": float(np.mean(1.0 / np.sum(W**2, axis=1))),
        # tail block
        "cvar_95": _expected_shortfall(rets, 0.95),
        "cvar_99": _expected_shortfall(rets, 0.99),
        "max_dd": _max_drawdown(rets),
        "sortino": float(np.mean(rets) / downside * ann) if downside > 0 else 0.0,
    }


def compare(factories: dict, returns, *, warmup: int = 60, cost_bps: float = 10.0,
            periods_per_year: int = 252):
    """Walk-forward every ``{name: factory}`` and return metric rows, best Sharpe first."""
    rows = []
    for name, factory in factories.items():
        rets, path = walk_forward(factory, returns, warmup=warmup)
        m = portfolio_metrics(rets, path, periods_per_year, cost_bps)
        rows.append({"name": name, **m})
    rows.sort(key=lambda r: r["net_sharpe"], reverse=True)
    return rows


def compare_random_subsets(factories: dict, returns, *, k=50, window=None, n_trials: int = 100,
                           warmup: int = 60, cost_bps: float = 10.0, periods_per_year: int = 252,
                           seed: int = 0):
    """Monte-Carlo comparison over random name-subsets (and optional random windows).

    Each trial draws a random subset of ``k`` columns (``k`` may be a list, sampled
    per trial) and, if ``window`` is set, a random contiguous block of that many
    rows; every ``factory`` is walk-forward tested on that slice. Aggregating over
    trials averages out the universe/period dependence a single backtest suffers,
    giving each method a mean +/- sd net Sharpe, mean turnover, and **win rate**
    (fraction of trials it had the best net Sharpe). Returns rows sorted by mean
    net Sharpe.
    """
    returns = np.asarray(returns, dtype=float)
    T, N = returns.shape
    rng = np.random.default_rng(seed)
    ks = list(k) if isinstance(k, (list, tuple)) else [k]
    windows = list(window) if isinstance(window, (list, tuple)) else ([window] if window else [None])
    stats = {name: {"net": [], "sharpe": [], "turnover": [], "eff_n": [], "wins": 0} for name in factories}
    for _ in range(n_trials):
        kk = min(int(rng.choice(ks)), N)
        cols = rng.choice(N, size=kk, replace=False)
        ww = rng.choice([w for w in windows if w]) if any(windows) else None
        if ww and ww < T:
            s = int(rng.integers(0, T - int(ww) + 1)); panel = returns[s:s + int(ww)][:, cols]
        else:
            panel = returns[:, cols]
        wu = min(warmup, panel.shape[0] // 4)
        best, best_net = None, -np.inf
        for name, fac in factories.items():
            try:
                rets, path = walk_forward(fac, panel, warmup=wu)
                m = portfolio_metrics(rets, path, periods_per_year, cost_bps)
            except Exception:
                continue
            st = stats[name]
            st["net"].append(m["net_sharpe"]); st["sharpe"].append(m["sharpe"])
            st["turnover"].append(m["turnover"]); st["eff_n"].append(m["eff_n"])
            if m["net_sharpe"] > best_net:
                best_net, best = m["net_sharpe"], name
        if best is not None:
            stats[best]["wins"] += 1
    rows = []
    for name, st in stats.items():
        n = len(st["net"])
        if not n:
            continue
        rows.append({"name": name, "trials": n,
                     "net_sharpe": float(np.mean(st["net"])),
                     "net_sd": float(np.std(st["net"])),
                     "sharpe": float(np.mean(st["sharpe"])),
                     "turnover": float(np.mean(st["turnover"])),
                     "eff_n": float(np.mean(st["eff_n"])),
                     "win_rate": st["wins"] / n_trials})
    rows.sort(key=lambda r: r["net_sharpe"], reverse=True)
    return rows


def format_table(rows) -> str:
    """Render comparison rows as a fixed-width leaderboard."""
    cols = [("name", "method", "{:<24}"), ("sharpe", "Sharpe", "{:>7.2f}"),
            ("net_sharpe", "net", "{:>7.2f}"), ("ann_vol", "vol", "{:>7.3f}"),
            ("turnover", "turnover", "{:>9.4f}"), ("max_w", "max_w", "{:>7.3f}"),
            ("eff_n", "eff_N", "{:>6.1f}")]
    head = "".join(("{:<24}" if k == "name" else "{:>" + f"{max(7, len(t)+1)}" + "}").format(t)
                    for k, t, _ in cols)
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append("".join(fmt.format(r[k]) for k, _, fmt in cols))
    return "\n".join(lines)


def make_panel(n_obs: int = 800, n: int = 12, seed: int = 0, regime_at: int | None = None):
    """Synthetic daily-like return panel: two factor blocks, idiosyncratic noise,
    and an optional correlation regime flip halfway, so methods actually differ."""
    rng = np.random.default_rng(seed)
    half = n // 2
    load = np.zeros((n, 2))
    load[:half, 0] = rng.uniform(0.4, 0.9, half)
    load[half:, 1] = rng.uniform(0.4, 0.9, n - half)
    f = rng.standard_normal((n_obs, 2))
    if regime_at is None:
        regime_at = n_obs // 2
    f[regime_at:, 1] *= -1.0  # the second factor's sign flips: a regime change
    idio = rng.standard_normal((n_obs, n)) * rng.uniform(0.3, 0.7, n)
    return (f @ load.T + idio) * 0.01  # ~1% daily scale


if __name__ == "__main__":  # pragma: no cover
    from allocation import (
        EqualWeight, InverseVariance, RiskParity, MinimumVariance,
        MaximumDiversification, HierarchicalRiskParity, SchurComplementary,
        ThurstonePortfolio, TurnoverPenalty, BoxConstrained,
    )

    panel = make_panel()
    factories = {
        "EqualWeight": EqualWeight,
        "InverseVariance": InverseVariance,
        "RiskParity": RiskParity,
        "MinimumVariance": lambda: MinimumVariance(shrinkage=0.1),
        "MaximumDiversification": lambda: MaximumDiversification(shrinkage=0.1),
        "HierarchicalRiskParity": HierarchicalRiskParity,
        "SchurComplementary": lambda: SchurComplementary(gamma=0.5),
        "Thurstone": lambda: ThurstonePortfolio(calib="market"),
        "Schur+turnover(λ=3)": lambda: TurnoverPenalty(SchurComplementary(gamma=0.5), cost=3.0),
        "MinVar+cap(20%)": lambda: BoxConstrained(MinimumVariance(shrinkage=0.1), upper=0.2),
    }
    print(format_table(compare(factories, panel)))
