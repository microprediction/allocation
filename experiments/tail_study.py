"""Out-of-sample cross-asset tail study.

Does driving the Thurstone race with a *downside* covariance (the tail-consistent
variant) reduce expected shortfall and drawdown out of sample, versus the
full-covariance race and standard baselines? A cross-asset ETF universe is used
deliberately: it has heterogeneous tail dependence and a genuinely crash-decorrelated
hedge (Treasuries, gold), which is where tail consistency can bite. Honest report:
the numbers fall where they fall.
"""

import numpy as np
import yfinance as yf

from allocation import (
    EqualWeight, MinimumVariance, HierarchicalRiskParity, SchurComplementary,
    ThurstonePortfolio, EwmaCovariance, DownsideSemicovariance,
)
from allocation.backtest import portfolio_metrics

ETFS = ["SPY", "IWM", "EFA", "EEM", "VNQ",       # equities / REIT
        "TLT", "IEF", "LQD", "AGG",              # rates / credit
        "GLD", "DBC"]                            # gold / commodities

STRESS = {
    "GFC 2008":   ("2008-09-01", "2009-03-31"),
    "Aug 2011":   ("2011-07-22", "2011-10-04"),
    "Q4 2018":    ("2018-10-01", "2018-12-31"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
}


def load():
    px = yf.download(ETFS, start="2006-01-01", end="2024-12-31",
                     auto_adjust=True, progress=False)["Close"].dropna(axis=1, how="any")
    r = np.log(px / px.shift(1)).dropna()
    return r.values, r.index


def walk(factory, R, warmup=252):
    est = factory(); est.fit(R[:warmup])
    w = np.asarray(est.weights_, float); rets, path = [], []
    for t in range(warmup, len(R)):
        rets.append(float(w @ R[t])); path.append(w)
        est.partial_fit(R[t]); w = np.asarray(est.weights_, float)
    return np.array(rets), np.array(path)


def max_dd(r):
    eq = np.cumprod(1 + r); return float(np.max(1 - eq / np.maximum.accumulate(eq)))


def main():
    R, idx = load()
    T, n = R.shape
    print(f"{n} ETFs, {T} days ({idx[0].date()}..{idx[-1].date()})\n")

    thurst = dict(calib="diagonal", target="equal", phi=1.0)
    factories = {
        "EqualWeight":     EqualWeight,
        "MinimumVariance": lambda: MinimumVariance(shrinkage=0.1),
        "HRP":             HierarchicalRiskParity,
        "Schur(0.5)":      lambda: SchurComplementary(gamma=0.5),
        "Thurstone full":  lambda: ThurstonePortfolio(covariance_estimator=EwmaCovariance(60), **thurst),
        "Thurstone down":  lambda: ThurstonePortfolio(covariance_estimator=DownsideSemicovariance(60), **thurst),
    }

    warmup = 252
    sdates = idx[warmup:]       # dates aligned with walk() rets
    hdr = f"{'method':17}{'Sharpe':>7}{'turn':>7}{'CVaR95':>8}{'CVaR99':>8}{'maxDD':>7}{'Sortino':>8}"
    print(hdr); print("-" * len(hdr))
    series = {}
    for name, fac in factories.items():
        rets, path = walk(fac, R, warmup)
        series[name] = rets
        m = portfolio_metrics(rets, path)
        print(f"{name:17}{m['sharpe']:>7.2f}{m['turnover']:>7.3f}{m['cvar_95']:>8.4f}"
              f"{m['cvar_99']:>8.4f}{m['max_dd']:>7.3f}{m['sortino']:>8.2f}")

    print("\nStress-window total return (%) and max drawdown (%):")
    print(f"{'window':13}" + "".join(f"{k:>17}" for k in factories))
    for label, (a, b) in STRESS.items():
        mask = (sdates >= a) & (sdates <= b)
        if mask.sum() == 0:
            continue
        cells = []
        for name in factories:
            r = series[name][mask]
            cells.append(f"{(np.prod(1+r)-1)*100:+5.1f}/{max_dd(r)*100:4.1f}")
        print(f"{label:13}" + "".join(f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
