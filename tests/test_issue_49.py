"""#49: turnover and net Sharpe must charge the trades that undo price drift."""
import numpy as np
import pytest

from allocation import EqualWeight
from allocation.backtest import compare, make_panel, portfolio_metrics


def test_constant_target_with_unequal_returns_trades():
    W = np.tile([0.5, 0.5], (2, 1))
    R = np.array([[0.10, 0.0], [0.0, 0.0]])
    rets = (W * R).sum(axis=1)
    m = portfolio_metrics(rets, W, cost_bps=100, asset_returns=R)
    # pre-trade weights [0.5238, 0.4762]; restoring 50/50 trades L1 0.047619
    assert m["turnover"] == pytest.approx(0.5 * 0.047619047619, rel=1e-9)
    assert m["target_turnover"] == 0.0


def test_net_sharpe_charges_drift_cost_by_hand():
    W = np.tile([0.5, 0.5], (3, 1))
    R = np.array([[0.10, 0.0], [-0.05, 0.0], [0.2, 0.0]])
    rets = (W * R).sum(axis=1)
    m = portfolio_metrics(rets, W, cost_bps=100, asset_returns=R)
    # hand accounting: holdings after period t are w*(1+r)/(1+w@r)
    trades = [0.0]
    for t in (1, 2):
        g = W[t - 1] * (1 + R[t - 1])
        trades.append(np.abs(W[t] - g / g.sum()).sum())
    net = rets - 0.01 * np.array(trades)
    assert m["net_sharpe"] == pytest.approx(np.mean(net) / np.std(net) * np.sqrt(252))
    assert m["net_sharpe"] < m["sharpe"]


def test_buy_and_hold_drift_has_zero_trading():
    R = np.array([[0.10, -0.02], [0.03, 0.05], [0.0, 0.0]])
    w = np.array([0.3, 0.7])
    path = [w]
    for r in R[:-1]:
        w = w * (1 + r) / (1 + w @ r)
        path.append(w)
    W = np.array(path)
    m = portfolio_metrics((W * R).sum(axis=1), W, asset_returns=R)
    assert m["turnover"] == pytest.approx(0.0, abs=1e-15)
    assert m["target_turnover"] > 0


def test_compare_reports_drift_turnover_for_equal_weight():
    rows = compare({"equal": EqualWeight}, make_panel(seed=2), warmup=80)
    assert rows[0]["turnover"] > 1e-4
    assert rows[0]["target_turnover"] < 1e-12
    assert rows[0]["net_sharpe"] < rows[0]["sharpe"]


def test_asset_returns_shape_is_checked():
    with pytest.raises(ValueError, match="asset_returns"):
        portfolio_metrics([0.0, 0.0], np.ones((2, 2)) / 2, asset_returns=np.zeros((3, 2)))
