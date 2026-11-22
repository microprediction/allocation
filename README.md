# allocation

**Online (streaming) portfolio construction.** A home for portfolio methods as
scikit-learn / [skfolio](https://skfolio.org)-compatible estimators
(`fit` / `partial_fit` / `predict` / `weights_`), on top of a keyed
dynamic-universe layer so they survive reconstituting universes.

📖 **[allocation.microprediction.org](https://allocation.microprediction.org)** — documentation for every method
&nbsp;·&nbsp; 🤖 **[SKILL.md](SKILL.md)** — when to reach for `allocation` (for LLMs / code review)

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

Actually Hugo (mostly) and I did move a *batch* version of schur into skfolio and maybe more things here 
will be upstreamed into skfolio or river (Max Halford). Hugo made some improvements that are *not*
replicated in the streaming version of schur here, and the streaming versions of schur and HRP here also
deal with changing universes and weight stability. 

To be clear 
 - `allocation` is the **online** complement in that sense — estimators with `partial_fit`
 - It can also use a keyed dynamic universe (in the
spirit of [`precise`](https://github.com/microprediction/precise) but also river-ml) while staying API-compatible to both.

This is all MIT-licensed and anyone is welcome to take anything here. 

## Methods

Each method has a documentation page with its parameters and an example.

| Method | Batch class | Streaming class | Weights |
|--------|-------------|-----------------|---------|
| [Thurstone portfolio polishing](https://allocation.microprediction.org/methods/thurstone-portfolio-polishing/) | `ThurstonePortfolio` | `StreamingThurstone` | long-only |
| [Schur bridge](https://allocation.microprediction.org/methods/schur-bridge/) | `SchurBridge` | `StreamingSchurBridge` | signed unless `long_only=True` |
| [Schur-complementary](https://allocation.microprediction.org/methods/schur-complementary/) | `SchurComplementary` | `StreamingSchur` | long-only |
| [Hierarchical risk parity](https://allocation.microprediction.org/methods/hierarchical-risk-parity/) | `HierarchicalRiskParity` | `StreamingHRP` | long-only |
| [Risk parity](https://allocation.microprediction.org/methods/risk-parity/) | `RiskParity` | `StreamingRiskParity` | long-only |
| [Minimum variance](https://allocation.microprediction.org/methods/minimum-variance/) | `MinimumVariance`, `FactorMinimumVariance` | `StreamingMinimumVariance` | signed |
| [Maximum diversification](https://allocation.microprediction.org/methods/maximum-diversification/) | `MaximumDiversification`, `MaximumDecorrelation`, `FactorMaximumDiversification` | `StreamingMaximumDiversification`, `StreamingMaximumDecorrelation` | signed |
| [Mean–variance](https://allocation.microprediction.org/methods/mean-variance/) | `MeanVariance` | `StreamingMeanVariance` | signed |
| [Equal weight and inverse variance](https://allocation.microprediction.org/methods/equal-weight-and-inverse-variance/) | `EqualWeight`, `InverseVariance` | `StreamingEqualWeight`, `StreamingInverseVariance` | long-only |

The first three were developed here. The rest are standard methods, implemented so
that `partial_fit` over a drifting covariance moves the weights no more than the
covariance moved.

- **Thurstone portfolio polishing** reads a long-only benchmark as the winning
  probabilities of a race among its assets, calibrates the race, changes the
  dependence between the assets, and races again. It needs no return forecast and
  inverts no covariance. Calibration and racing run through
  [`winning`](https://github.com/microprediction/winning).
- **Schur bridge** runs from a hierarchical heuristic (HRP, HERC, NCO, inverse
  variance) to an optimizer (minimum variance, maximum diversification, tangency),
  and is exact at both ends: `SchurBridge.hrp_to_min_variance(gamma)`.
- **Schur-complementary** is the form of the HRP bridge that matches skfolio to
  machine precision. It is HRP at `gamma=0` and moves toward minimum variance
  without reaching it.

## Guides

- [Streaming and changing universes](https://allocation.microprediction.org/guides/streaming/): the batch interface
  (`fit` / `partial_fit` / `weights_`), the river-style interface
  (`learn_one({id: ret})` / `predict_one()`), and covariance estimators
  (`EwmaCovariance`, `DownsideSemicovariance`, or your own via
  `covariance_estimator=`).
- [Constraints and trading costs](https://allocation.microprediction.org/guides/constraints-and-costs/):
  `BoxConstrained` for position bounds and group caps by log-barrier, and
  `TurnoverPenalty` for a quadratic trading cost. Both wrap any estimator and have
  streaming twins.
- [Comparing methods](https://allocation.microprediction.org/guides/comparing-methods/): `allocation.backtest`, a
  numpy-only walk-forward harness reporting Sharpe, turnover, net Sharpe and tail
  metrics, and `negative_weight_diagnostic` for testing whether a signed method's
  short positions earn anything.

```python
from allocation import SchurComplementary, TurnoverPenalty, MinimumVariance
from allocation.backtest import compare, format_table, make_panel

panel = make_panel()                       # synthetic; or any (n_obs, n_assets) array
print(format_table(compare({
    "schur":         lambda: SchurComplementary(gamma=0.5),
    "schur, damped": lambda: TurnoverPenalty(SchurComplementary(gamma=0.5), cost=3.0),
    "minvar":        lambda: MinimumVariance(shrinkage=0.1),
}, panel)))
```

## Why the weights move smoothly

Read `weights = allocator(cov_estimator(data))` as two factors. For the closed-form
allocators (`MinimumVariance`, `MaximumDiversification`, `InverseVariance`) the
allocator is a smooth function of `Σ`, so a smooth online covariance is enough,
provided `Σ` is kept well-conditioned (`shrinkage`).

Three other things make weights jump, and each has a remedy here: sampling noise
(fixed random seeds, re-coloured as the correlation moves, in Thurstone), re-clustering
(a spectral ordering in place of a dendrogram, in Schur and HRP), and bounds that
start or stop binding (interior solutions and a log-barrier, in risk parity and
`BoxConstrained`).

## Design

```
allocation/
  base.py        # BaseOnlinePortfolio: fit / partial_fit / predict / weights_
  moments.py     # EwmaCovariance (default); any partial_fit/covariance_ estimator plugs in
  baselines.py   # EqualWeight / InverseVariance / RiskParity (ERC)
  convex.py      # MinimumVariance / MaximumDiversification (closed-form, signed)
  thurstone.py   # ThurstonePortfolio
  _thurstone/    # calibration + transport (thin over winning)
  schur.py       # SchurComplementary / HierarchicalRiskParity
  bridge.py      # SchurBridge: the pair-form engine the other allocators are corners of
  _schur/        # Fiedler seriation + Schur coupling engine + the bridge (pairs, dials, knots)
  keyed.py       # river-style streaming twins over a changing universe
  universe.py    # keyed dynamic-universe state for the batch estimators (planned)
```

Covariance is pluggable: the default is a light EWMA, but any online estimator
exposing `partial_fit` and `covariance_` (e.g. a `precise` skater) can be passed
via `covariance_estimator=`.

## Papers

The two novel methods are written up as working papers.

- **Portfolio-Implied Contests, with an Application to Tail Risk**
  ([PDF](https://allocation.microprediction.org/papers/thurstone-portfolios/paper.pdf),
  [SSRN](https://ssrn.com/abstract=7737202),
  [talk slides](https://allocation.microprediction.org/papers/thurstone-portfolios/talk.pdf),
  [demo](https://allocation.microprediction.org/demos/portfolio-contest/)). A long-only portfolio read as the winning
  probabilities of a correlated race: redundant assets share one position, the
  weights respond to tail dependence when the race is driven by a simulation that
  has it, and there is an implied regularized objective and a smoothness bound on
  turnover. Source in [`papers/thurstone-portfolios/`](papers/thurstone-portfolios).
- **Schur-complementary allocation** — robust, inversion-light allocation along a
  spectral (Fiedler) seriation; background and bibliography at
  [schur.microprediction.org](https://schur.microprediction.org).

## Status

Early. The Thurstone, Schur/HRP, and risk-parity estimators all work and are
tested, batch and streaming; the keyed dynamic-universe layer for the *batch*
estimators (so `partial_fit` survives a changing asset count, as the streaming
twins already do) is next. The theory behind Thurstone portfolio polishing
(redundancy consistency, tail consistency, the implied regularized objective,
smoothness and turnover) is in the paper above.

## License

MIT © Peter Cotton
