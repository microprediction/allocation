import numpy as np

from allocation import (
    InverseVariance,
    MinimumVariance,
    negative_weight_diagnostic,
)
from allocation.backtest import make_panel


KEYS = {
    "short_mass", "sign_flip_rate", "net_sharpe_signed", "net_sharpe_longonly",
    "sharpe_gain_from_shorts", "var_signed", "var_longonly", "shorts_needed",
}


def test_long_only_method_has_no_shorts():
    # inverse-variance is long-only: no short mass, no flips, shorts not "needed"
    d = negative_weight_diagnostic(InverseVariance, make_panel(seed=1), warmup=80)
    assert set(d) == KEYS
    assert d["short_mass"] < 1e-9
    assert d["sign_flip_rate"] == 0.0
    assert d["shorts_needed"] is False


def test_signed_method_reports_short_book():
    # min-variance can short; the diagnostic should see nonzero short mass and
    # return finite, sane numbers
    d = negative_weight_diagnostic(
        lambda: MinimumVariance(shrinkage=0.05), make_panel(n=20, seed=2), warmup=80
    )
    assert d["short_mass"] > 0.0
    assert 0.0 <= d["sign_flip_rate"] <= 1.0
    assert np.isfinite(d["net_sharpe_signed"]) and np.isfinite(d["net_sharpe_longonly"])
    assert d["var_signed"] > 0 and d["var_longonly"] > 0


def test_gain_is_signed_minus_longonly():
    d = negative_weight_diagnostic(
        lambda: MinimumVariance(shrinkage=0.05), make_panel(n=15, seed=3), warmup=80
    )
    assert abs(d["sharpe_gain_from_shorts"]
               - (d["net_sharpe_signed"] - d["net_sharpe_longonly"])) < 1e-9
