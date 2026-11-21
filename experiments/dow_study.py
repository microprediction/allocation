"""Dow-constituents walk-forward study.

Downloads ~15y of daily Dow returns and trades each estimator forward with the
numpy-only backtest harness, reporting out-of-sample Sharpe, turnover, and the
tail block (CVaR 95/99, max drawdown, Sortino). Honest comparison: the headline
for the Thurstone tilt is low turnover and sound tail behaviour at parity Sharpe,
not a return miracle.
"""

import numpy as np
import yfinance as yf

from allocation import (
    EqualWeight, InverseVariance, MinimumVariance, RiskParity,
    HierarchicalRiskParity, SchurComplementary, ThurstonePortfolio,
)
from allocation.backtest import walk_forward, portfolio_metrics

TICKERS = ["AAPL","AMGN","AXP","BA","CAT","CSCO","CVX","DIS","GS","HD","HON",
           "IBM","INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG",
           "TRV","UNH","VZ","WMT","CRM","DOW","V"]


def load_returns():
    px = yf.download(TICKERS, start="2010-01-01", end="2024-12-31",
                     auto_adjust=True, progress=False)["Close"].dropna(axis=1, how="any")
    logr = np.log(px / px.shift(1)).dropna().values      # fitted on (covariance)
    simple = (px / px.shift(1) - 1).dropna().values       # earned on (P&L, drift)
    return logr, simple


def main():
    R, S = load_returns()
    T, n = R.shape
    W0 = 252
    print(f"{n} Dow names, {T} trading days\n")

    factories = {
        "EqualWeight":        EqualWeight,
        "InverseVariance":    InverseVariance,
        "RiskParity":         RiskParity,
        "MinimumVariance":    lambda: MinimumVariance(shrinkage=0.1),
        "HRP":                HierarchicalRiskParity,
        "Schur(0.5)":         lambda: SchurComplementary(gamma=0.5),
        "Thurstone":          lambda: ThurstonePortfolio(calib="market", target="equal"),
        "Thurstone(t,nu=4)":  lambda: ThurstonePortfolio(calib="market", target="equal",
                                                         sampler="student_t", nu=4.0),
    }

    hdr = f"{'method':20} {'Sharpe':>7} {'net':>7} {'turn':>7} {'CVaR95':>7} {'maxDD':>6} {'Sortino':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, fac in factories.items():
        rets, path = walk_forward(fac, R, warmup=W0, pnl_returns=S)
        # drift-aware trading turnover; cost is cost_bps on the full traded amount
        m = portfolio_metrics(rets, path, asset_returns=S[W0:W0 + len(path)])
        rows.append((name, m))
        print(f"{name:20} {m['sharpe']:>7.2f} {m['net_sharpe']:>7.2f} {m['turnover']:>7.4f} "
              f"{m['cvar_95']:>7.4f} {m['max_dd']:>6.3f} {m['sortino']:>8.2f}")
    import json, os
    out = os.path.join(os.path.dirname(__file__), "results", "dow_study_results.json")
    with open(out, "w") as f:
        json.dump({name: {k: float(v) for k, v in m.items()} for name, m in rows}, f, indent=1)
    return rows


if __name__ == "__main__":
    main()
