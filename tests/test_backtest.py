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
    compare_random_subsets,
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


def test_compare_random_subsets_aggregates():
    panel = make_panel(n_obs=600, n=20, seed=5)
    rows = compare_random_subsets(
        {"equal": EqualWeight, "invvar": InverseVariance, "schur": lambda: SchurComplementary(gamma=0.5)},
        panel, k=[8, 12], window=[300, 400], n_trials=6, warmup=60, seed=1,
    )
    assert {r["name"] for r in rows} == {"equal", "invvar", "schur"}
    total_wins = sum(r["win_rate"] for r in rows)
    assert abs(total_wins - 1.0) < 1e-9  # win rates partition the trials
    for r in rows:
        assert r["trials"] == 6 and r["net_sd"] >= 0
    # reproducible
    rows2 = compare_random_subsets(
        {"equal": EqualWeight, "invvar": InverseVariance, "schur": lambda: SchurComplementary(gamma=0.5)},
        panel, k=[8, 12], window=[300, 400], n_trials=6, warmup=60, seed=1,
    )
    assert [r["name"] for r in rows] == [r["name"] for r in rows2]
    assert rows[0]["net_sharpe"] == rows2[0]["net_sharpe"]


def test_box_constraint_caps_concentration_through_backtest():
    panel = make_panel(seed=4)
    _, path = walk_forward(lambda: BoxConstrained(MinimumVariance(shrinkage=0.1), upper=0.2), panel, 80)
    assert np.all(path <= 0.2 + 1e-6)  # cap respected at every rebalance
