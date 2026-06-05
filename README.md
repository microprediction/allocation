# allocation

**Online (streaming) portfolio construction.** A home for portfolio methods as
scikit-learn / [skfolio](https://skfolio.org)-compatible estimators
(`fit` / `partial_fit` / `predict` / `weights_`), on top of a keyed
dynamic-universe layer so they survive reconstituting universes.

```python
from allocation import ThurstonePortfolio

est = ThurstonePortfolio(calib="market", phi=1.0)
est.fit(returns)            # returns: (n_obs, n_assets)
est.weights_                # long-only weights on the simplex

for batch in stream:        # streaming / rebalancing
    est.partial_fit(batch)  # transports a fixed seed ensemble -> low turnover
    w = est.weights_
```

`score(X)` returns the portfolio's Sharpe ratio (skfolio's scoring convention),
and `to_portfolio(X)` wraps the fitted weights as a skfolio `Portfolio`
(`pip install allocation[skfolio]`) so the output drops into skfolio's metrics,
`Population`, and cross-validation.

## Why a separate package

skfolio is the scikit-learn-native portfolio library, but it is *batch*: a
changing universe is handled by NaN/incomplete-data mechanics, not by carrying
per-asset state across streaming updates. `allocation` is the **online**
complement — estimators with `partial_fit`, a keyed dynamic universe (in the
spirit of [`precise`](https://github.com/microprediction/precise)), and turnover
control by construction — while staying API-compatible so each estimator can also
be contributed upstream to skfolio.

## Methods

| Estimator | Status | Notes |
|-----------|--------|-------|
| `ThurstonePortfolio` | working | Ability tilt: weights are winning probabilities of a correlated race; calibrate to a benchmark under a reference correlation, tilt under the estimate; smooth common-seed transport for `partial_fit`. Built on [`thurstone`](https://github.com/microprediction/thurstone). |
| `SchurComplementary` | planned | Online version of the Schur-complementary construction (the batch version is in skfolio). |
| baselines | planned | Online equal / cap / minimum-variance / risk-parity for benchmarking. |

## Design

```
allocation/
  base.py        # BaseOnlinePortfolio: fit / partial_fit / predict / weights_
  moments.py     # EwmaCovariance (default); any partial_fit/covariance_ estimator plugs in
  universe.py    # keyed dynamic-universe state (planned)
  thurstone.py   # ThurstonePortfolio
  _thurstone/    # calibration + transport engine
```

Covariance is pluggable: the default is a light EWMA, but any online estimator
exposing `partial_fit` and `covariance_` (e.g. a `precise` skater) can be passed
via `covariance=`.

## Status

Early. The Thurstone estimator and the streaming transport work and are tested;
the keyed dynamic-universe layer and the Schur estimator are next. The theory
behind the Thurstone method (feasibility, redundancy consistency, the implied
regularized objective, smoothness/turnover) is written up in the accompanying
paper.

## License

MIT © Peter Cotton
