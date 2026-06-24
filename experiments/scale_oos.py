"""Out-of-sample at scale: turnover among the methods that survive n > T.

n = 500 names, T_total = 1000 days, a 252-day window -> the sample covariance is
rank-deficient, so dense minimum variance is undefined. Among the inversion-free
survivors, does the common-seed Thurstone transport deliver the lowest turnover
under streaming (daily) updates? Synthetic factor returns: with no real drift the
Sharpe is noise, so turnover (weight churn from re-estimation) is the metric that
matters and the one the smoothness theorem speaks to.
"""

import numpy as np

from allocation import (ThurstonePortfolio, FactorMinimumVariance, InverseVariance,
                        HierarchicalRiskParity, RiskParity, EqualWeight)
from allocation.backtest import walk_forward, portfolio_metrics


def factor_returns(rng, n, T, k=5):
    B = rng.standard_normal((n, k)) * 0.3
    mu_f = np.array([0.05] + [0.0] * (k - 1))           # mild market premium
    F = mu_f / np.sqrt(252) + rng.standard_normal((T, k))
    return (F @ B.T + 0.5 * rng.standard_normal((T, n))) * 0.01


def main():
    rng = np.random.default_rng(0)
    n, T, k = 500, 1000, 5
    R = factor_returns(rng, n, T, k)
    print(f"n={n} names, T={T} days, window 252 -> n>T (dense MinVar undefined)\n")

    factories = {
        "EqualWeight":     EqualWeight,
        "InverseVariance": InverseVariance,
        "RiskParity":      RiskParity,
        "HRP":             HierarchicalRiskParity,
        "FactorMinVar":    lambda: FactorMinimumVariance(factors=k),
        "Thurstone(f=5)":  lambda: ThurstonePortfolio(calib="diagonal", target="equal",
                                                      factors=k, n_paths=1 << 12),
    }
    hdr = f"{'method':17}{'turnover':>10}{'net Sharpe':>12}{'maxDD':>8}{'eff_N':>8}"
    print(hdr); print("-" * len(hdr))
    for name, fac in factories.items():
        rets, path = walk_forward(fac, R, warmup=252)
        m = portfolio_metrics(rets, path)
        print(f"{name:17}{m['turnover']:>10.4f}{m['net_sharpe']:>12.2f}"
              f"{m['max_dd']:>8.3f}{m['eff_n']:>8.1f}")
    print("\nTurnover is the claim (Thm 2); Sharpe is near-noise on synthetic data.")


if __name__ == "__main__":
    main()
