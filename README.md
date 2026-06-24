# allocation

**Online (streaming) portfolio construction.** A home for portfolio methods as
scikit-learn / [skfolio](https://skfolio.org)-compatible estimators
(`fit` / `partial_fit` / `predict` / `weights_`), on top of a keyed
dynamic-universe layer so they survive reconstituting universes.

📖 **[allocation.microprediction.org](https://allocation.microprediction.org)**
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

skfolio is the scikit-learn-native portfolio library, but it is *batch*: a
changing universe is handled by NaN/incomplete-data mechanics, not by carrying
per-asset state across streaming updates. `allocation` is the **online**
complement — estimators with `partial_fit`, a keyed dynamic universe (in the
spirit of [`precise`](https://github.com/microprediction/precise)), and turnover
control by construction — while staying API-compatible so each estimator can also
be contributed upstream to skfolio. It is all MIT-licensed and meant to be
upstreamed: Max (river) and Hugo (skfolio) are welcome to take anything here and
improve on it.

## Methods

| Estimator | Status | Notes |
|-----------|--------|-------|
| `ThurstonePortfolio` | working | Ability tilt: weights are winning probabilities of a correlated race; calibrate to a benchmark under a reference correlation, tilt under the estimate; smooth common-seed transport for `partial_fit`. Built on [`thurstone`](https://github.com/microprediction/thurstone). |
| `SchurComplementary` | working | Online Schur-complementary construction (`gamma`: HRP at 0 → min-variance as →1) over a smooth **Fiedler seriation** instead of a dendrogram, so `partial_fit` is low-turnover. |
| `HierarchicalRiskParity` | working | Dynamic HRP — the `gamma=0` special case of the Schur construction (recursive-bisection risk parity over the Fiedler order); named for recognisability. |
| `RiskParity` | working | Equal-risk-contribution (ERC); interior convex solution, solved by coordinate descent **warm-started from the previous weights** so updates stay smooth. |
| `EqualWeight`, `InverseVariance` | working | Smooth baselines for benchmarking (`1/n`; `w ∝ 1/σ²`). |
| `MinimumVariance`, `MaximumDiversification` | working | Closed-form `Σ⁻¹1` / `Σ⁻¹σ` with optional `shrinkage` for conditioning. Unconstrained (signed) so they stay smooth — a long-only QP would kink. |

Each has a river-style streaming twin for a *changing* universe — `StreamingThurstone`, `StreamingSchur`, `StreamingHRP`, `StreamingRiskParity`, `StreamingEqualWeight`, `StreamingInverseVariance`, `StreamingMinimumVariance`, `StreamingMaximumDiversification` — with `learn_one({id: ret})` / `predict_one() → {id: weight}`.

**On smoothness.** Each method is written so that `partial_fit` over a drifting covariance moves weights only as much as the covariance moved. It helps to see `weights = allocator(cov_estimator(data))` as a product of two factors:

- For the **closed-form linear allocators** (`MinimumVariance`, `MaximumDiversification`, `InverseVariance`) the allocator factor is already a smooth (rational) function of `Σ`, so weight-smoothness *is* covariance-smoothness — pair them with a smooth online covariance (the default EWMA, a shrinkage estimator, or a `precise` skater) and you're done. Caveats: keep `Σ` well-conditioned (use `shrinkage`, since `Σ⁻¹` swings near a vanishing eigenvalue) and avoid hard long-only QPs (they kink at the zero bound — the smooth long-only min-variance is `SchurComplementary` as `gamma→1`).
- The package's distinctive work is the **other** family, where the allocator factor itself is rough no matter how smooth `Σ` is. Three sources, three primitives: sampling noise → common-seed transport (Thurstone); combinatorial ordering → Fiedler seriation (Schur/HRP); active-set kinks → keep the solution interior (ERC is interior by construction).

### Very large universes (e.g. Russell 3000)

Where the covariance is rank-deficient, every inversion-based allocator (min-variance,
max-diversification, …) is *undefined* (`is_singular()` flags it; `strict=True` refuses).
The robust large-universe methods are the ones that never invert `Σ`: inverse-variance,
HRP/Schur, and the **Thurstone tilt** — benchmark-anchored, no inversion, low turnover.
`ThurstonePortfolio(factors=k)` runs the tilt with a `k`-factor (low-rank) correlation and
an `O(M·n·k)` race instead of the dense `O(M·n²)+O(n³)` one, so it scales to thousands of
names (≈0.4 s/rebalance at n=3000).

To make *minimum-variance itself* well-posed at scale, `FactorMinimumVariance` /
`FactorMaximumDiversification` invert a factor covariance `Σ = BBᵀ + diag(ψ)` via the
**Woodbury identity** (`O(n·k²)`, condition number bounded by an idiosyncratic floor, so the
inverse always exists). Supply a factor covariance (`loadings_` / `idiosyncratic_`, e.g. an
adapted `precise.FactorCovariance`) for the `O(n·k)` path, or let it factor a dense covariance.
And `SchurComplementary(ridge=…)` regularizes the cross-block solve so `gamma>0` stays
well-posed (and self-tempers toward HRP) when blocks are rank-deficient.

### Trading costs

`TurnoverPenalty(estimator, cost=λ)` wraps any estimator with a quadratic trading
cost (market-impact model). Minimising `‖w − w*‖² + λ‖w − w_prev‖²` gives the smooth,
budget-preserving blend `w_t = α w* + (1−α) w_prev` with `α = 1/(1+λ)`, so each step's
turnover is scaled exactly by `α`. This explicit damping composes with the implicit
turnover control (smooth target) and has a streaming twin `StreamingTurnoverPenalty`.

### Linear constraints

`BoxConstrained(estimator, lower=, upper=, groups=, group_caps=)` imposes per-asset
bounds and (disjoint) group caps on any estimator via a **log-barrier**, not a QP.
The barrier's domain *is* the constraint set, so the result is strictly feasible for
any `tau>0` — feasibility never depends on tuning, and the operator stays C¹ (no
active-set kinks), so turnover stays low. Streaming twin: `StreamingBoxConstrained`.

## Comparing methods

`allocation.backtest` is a numpy-only walk-forward harness: it trades each
estimator forward (form weights → earn next period → update) and tabulates
out-of-sample Sharpe, **turnover**, concentration, and Sharpe net of a
proportional cost.

```python
from allocation import SchurComplementary, TurnoverPenalty, MinimumVariance
from allocation.backtest import compare, format_table, make_panel

panel = make_panel()                       # synthetic; or any (n_obs, n_assets) array
print(format_table(compare({
    "schur":   lambda: SchurComplementary(gamma=0.5),
    "schur+λ": lambda: TurnoverPenalty(SchurComplementary(gamma=0.5), cost=3.0),
    "minvar":  lambda: MinimumVariance(shrinkage=0.1),
}, panel)))
```

No data is shipped (to avoid bloat); `compare()` takes any array, and
`load_returns_csv(url)` can pull a raw CSV from a data repo at runtime.

For a robust, universe-independent comparison, `compare_random_subsets(...)` runs
a Monte-Carlo over random name-subsets and windows, reporting each method's mean
± sd net Sharpe, mean turnover, and **win rate** (fraction of trials it had the
best net Sharpe) — averaging out the single-backtest dependence on which names
and period you happened to pick.

## Design

```
allocation/
  base.py        # BaseOnlinePortfolio: fit / partial_fit / predict / weights_
  moments.py     # EwmaCovariance (default); any partial_fit/covariance_ estimator plugs in
  baselines.py   # EqualWeight / InverseVariance / RiskParity (ERC)
  convex.py      # MinimumVariance / MaximumDiversification (closed-form, signed)
  thurstone.py   # ThurstonePortfolio
  _thurstone/    # calibration + transport engine
  schur.py       # SchurComplementary / HierarchicalRiskParity
  _schur/        # Fiedler seriation + Schur coupling engine
  keyed.py       # river-style streaming twins over a changing universe
  universe.py    # keyed dynamic-universe state for the batch estimators (planned)
```

Covariance is pluggable: the default is a light EWMA, but any online estimator
exposing `partial_fit` and `covariance_` (e.g. a `precise` skater) can be passed
via `covariance=`.

## Papers

The two novel methods are written up as working papers.

- **Thurstone Portfolios: Allocation as Winning Probability** — long-only
  allocation as the winning probabilities of a correlated race: free of the
  duplication paradox, tail-consistent when the race is driven by a
  downside-dependent simulation, with an implied regularized objective and a
  smoothness/turnover bound. Draft in
  [`papers/thurstone-portfolios/`](papers/thurstone-portfolios), built on
  [`thurstone`](https://github.com/microprediction/thurstone).
- **Schur-complementary allocation** — robust, inversion-light allocation along a
  smooth Fiedler seriation; background and bibliography at
  [schur.microprediction.org](https://schur.microprediction.org).

## Status

Early. The Thurstone, Schur/HRP, and risk-parity estimators all work and are
tested, batch and streaming; the keyed dynamic-universe layer for the *batch*
estimators (so `partial_fit` survives a changing asset count, as the streaming
twins already do) is next. The theory behind the Thurstone method (feasibility,
redundancy consistency, the implied regularized objective, smoothness/turnover)
is written up in the accompanying paper.

## License

MIT © Peter Cotton
