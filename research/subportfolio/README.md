# How should a sub-portfolio be formed from an optimal index?

## The question

CAPM says the cap-weighted market portfolio is the tangency portfolio for the
whole universe: for the returns the market expects, no book has a higher
Sharpe ratio. It says nothing of the kind about a sector, a screen, an
exclusion list or any other sub-universe, and the usual argument runs the
other way: cap-weighting a subset is generally far from optimal.

The premise is taken literally. The market's expected excess returns are
whatever makes its own weights optimal, m = Sigma w, and every rule is scored
by the Sharpe ratio it achieves against the true sub-covariance and those
returns. On a sub-universe the optimum is

    w_S* ~ Sigma_SS^{-1} m_S = w_S + Sigma_SS^{-1} Sigma_{S,S^c} w_{S^c},

proportional restriction plus the departed names' weight projected onto the
survivors they co-moved with. That second term is the whole question.

So an index holder who wants a sub-portfolio needs a rule. The rule in
universal use is proportional restriction, which is to say renormalise the
weights of the survivors. That is Luce's axiom applied without comment, and it
assumes independence of irrelevant alternatives: the ratio between two holdings
is unaffected by which other names are present.

Thurstone's model does not assume that. Calibrate latent abilities so that a
race among the parent universe reproduces the parent weights, then race only
the survivors. The departed names' weight is redistributed according to who
competed most closely with them, not spread in proportion. The question is
whether that is better, and whether it needs any data to be better.

## The candidates

| rule | what it asks for | what it assumes |
|---|---|---|
| equal weight | nothing | a floor, not a candidate |
| proportional | the parent weights | Luce, IIA |
| flattened | the parent weights | nothing; `w_prop^0.75`, the null the race must clear |
| race | the parent weights | Thurstone, independent field |
| race + factor | those plus a `k`-factor correlation, `k(n+1)` numbers | Thurstone, correlated field |
| black-litterman | the parent weights plus the `k`-factor covariance | CAPM; the estimate is trustworthy |
| estimate + solve | a full sub-covariance, `m(m+1)/2` numbers | ignore the market; minimise variance |
| oracle | the true sub-covariance and the market's returns | unavailable |

Volatility is deliberately absent from the middle three. An optimal parent
already holds less of a volatile name, so the abilities calibrated from the
parent weights carry the volatility. Giving the race a structure scaled by
volatility as well applies it twice: measured, that moves the restricted
portfolio by 0.60 in L1 where correlation alone moves it by 0.06. What
restriction changes is which competitors are present, and how the departed
weight redistributes depends on how the survivors co-move.

## What would falsify the result

The null is `flattened`, not `proportional`. An independent race is
`w_prop^0.75` renormalized to within three percent in L1: with no correlation,
nothing encodes which survivor a departed name resembled, so all the race can
do is de-concentrate. A rule that beats proportional but not `flattened` has
shown nothing.

The `race differs from proportional` check in `smoke.py` is necessary and
nowhere near sufficient, since the whole of that difference is the power
transform. The `race is close to flattened proportional` check is what pins it.

The premise is separately falsifiable and separately checked. If the parent
were not exactly optimal for the parent universe, every comparison here would
be measuring the parent's suboptimality rather than the restriction rules.
`run.py` computes the KKT residual of the parent on every draw and refuses to
score one that fails.

## Layout

```
markets.py    the two markets, and the proof that each makes the premise true
rules.py      the restriction rules
run.py        sharded runner, one JSON per shard
merge.py      combine shards, print the table
smoke.py      twelve property checks, under a minute
check_sharding.py  proves a result does not depend on how work was split
launch.sh     fan across cores and merge
results/      committed outputs
```

`markets.py` carries the part worth reading. The covariance and the cap
weights are drawn independently, the covariance to look like equities and the
weights to look like an index, and the market's expected returns are then
`m = Sigma w`. The index market is a sector market: a market factor with
heterogeneous betas, two signed style factors, fifty sectors of unequal size
and idiosyncratic noise, so its spectrum is one large eigenvalue and a long
tail rather than anything a `k`-factor estimate can capture. Nothing is ever
formed densely; blocks, products and panels are `O(n)` or `O(n + m^2)`.

## Running it

On a new machine:

```bash
git clone https://github.com/microprediction/allocation
cd allocation/research/subportfolio
pip install -r requirements.txt
python smoke.py                 # 15s, must print 0 failures
python check_sharding.py        # a few minutes, must print all identical
```

Nothing outside this directory is imported, so copying the directory alone
works too. Python 3.10 or later.

Check which `winning` you actually got, because the installed version is not
evidence of which one runs:

```bash
python -c "import winning, os; print(winning.__version__, os.path.dirname(winning.__file__))"
```

On the machine this was written, `pip show winning` reported 1.2.0 from
site-packages while the import resolved to a git checkout at 1.5.0. Every
shard therefore records the version and the path it loaded, `merge.py` prints
them, and it warns if the shards were not all built by the same environment.
The numbers committed under `results/` came from winning 1.5.0.

`check_sharding.py` is worth running once on any new machine. It runs the same
six draws as one shard and as three and demands the recorded numbers agree
exactly. It has already caught one defect: a cached cvxpy Problem carried
solver state between draws, which moved the parent portfolio by 1e-8 and, since
the parent feeds every rule, moved `race+factor` by 3.5e-03 while the data-free
rows stayed bit-identical.

Then either study:

```bash
./launch.sh mid                 # 400 names, 60-name sub-universe, k=3
./launch.sh index               # 5000 names, 200-name sub-index, k=2
WORKERS=64 DRAWS=256 ./launch.sh index
```

`launch.sh` pins one BLAS thread per worker, runs `smoke.py` first, fans the
draws across cores and merges. Shards can also be run by hand on separate
hosts and merged later, since `merge.py` only globs for files:

```bash
python run.py --scale index --n 5000 --m 200 --k 2 --draws 256 \
              --shard 7 --shards 64 --tag index-n5000-m200-k2-s4
python merge.py --tag index-n5000-m200-k2-s4
```

`merge.py` names any draw index that no shard produced, so a worker that died
is visible rather than silently reducing the sample.

## Sizing the run

Cost is dominated by one thing: calibrating abilities under a `k`-factor
correlation. `winning`'s factor race is exponential in `k` and linear in `n`,
which is filed as [winning#156](https://github.com/microprediction/winning/issues/156).
Measured on one core, forward race at `n=200`, `points=257`:

| k | seconds |
|---|---|
| 1 | 0.019 |
| 2 | 0.059 |
| 3 | 0.714 |
| 4 | 6.151 |

Calibration converges in about 19 iterations and costs that many forward
passes, so a three-factor calibration on 400 names is 26 seconds against 0.03
for the independent race. On a laptop the `mid` study takes roughly two
minutes per draw, or about 50 minutes for 25 draws, which is why this moved
here.

To budget: **seconds per draw is about `19 * f(k) * n / 200 * (1 + 2 * |Ts|)`**,
where `f(k)` is the table above. Per core. Draws are independent, so wall
clock is that divided by the number of workers.

Two consequences worth knowing before choosing sizes. Raising `k` from 2 to 3
costs an order of magnitude and is the first thing to cut if the run is too
slow. Raising `points` does nothing for `k=1` and doubles the cost for `k>=2`,
so leave it alone.

## Interpreting the table

`merge.py` reports the median ratio to proportional restriction and the share
of draws on which each rule beats it. Both matter. A rule that wins narrowly on
nine draws in ten is a different animal from one that wins large on half, and
only the pair distinguishes them.

The `oracle` row bounds what is available. If a rule sits close to it, the
remaining room is small no matter how good the rule looks against proportional.
