import numpy as np

from allocation import (
    BoxConstrained,
    EqualWeight,
    HierarchicalRiskParity,
    InverseVariance,
    MaximumDiversification,
    MinimumVariance,
    RiskParity,
    SchurComplementary,
    ThurstonePortfolio,
    TurnoverPenalty,
)
from allocation.backtest import (
    compare,
    format_table,
    make_panel,
    portfolio_metrics,
    walk_forward,
)

FACTORIES = {
    "EqualWeight": EqualWeight,
    "InverseVariance": InverseVariance,
    "RiskParity": RiskParity,
    "MinimumVariance": lambda: MinimumVariance(shrinkage=0.1),
    "MaximumDiversification": lambda: MaximumDiversification(shrinkage=0.1),
    "HierarchicalRiskParity": HierarchicalRiskParity,
    "SchurComplementary": lambda: SchurComplementary(gamma=0.5),
    "Thurstone": lambda: ThurstonePortfolio(calib="market"),
}


def test_compare_runs_all_methods():
    rows = compare(FACTORIES, make_panel(seed=1), warmup=80)
    assert {r["name"] for r in rows} == set(FACTORIES)
    for r in rows:
        assert np.isfinite(r["sharpe"]) and np.isfinite(r["net_sharpe"])
        assert r["turnover"] >= 0 and r["ann_vol"] > 0
        assert r["eff_n"] > 1.0
    # rendering works and is sorted by net Sharpe
    assert "method" in format_table(rows)
    assert [r["net_sharpe"] for r in rows] == sorted((r["net_sharpe"] for r in rows), reverse=True)


def test_equal_weight_has_zero_turnover():
    _, path = walk_forward(EqualWeight, make_panel(seed=2), warmup=80)
    m = portfolio_metrics(_, path)
    assert m["turnover"] < 1e-12


def test_turnover_penalty_lowers_realized_turnover():
    panel = make_panel(seed=3)
    base = portfolio_metrics(*walk_forward(lambda: SchurComplementary(gamma=0.5), panel, 80))
    damped = portfolio_metrics(
        *walk_forward(lambda: TurnoverPenalty(SchurComplementary(gamma=0.5), cost=4.0), panel, 80)
    )
    assert damped["turnover"] < base["turnover"]


def test_box_constraint_caps_concentration_through_backtest():
    panel = make_panel(seed=4)
    _, path = walk_forward(lambda: BoxConstrained(MinimumVariance(shrinkage=0.1), upper=0.2), panel, 80)
    assert np.all(path <= 0.2 + 1e-6)  # cap respected at every rebalance
