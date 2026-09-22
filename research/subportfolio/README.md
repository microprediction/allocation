# How should a sub-portfolio be formed from an optimal index?

## The question

CAPM says the cap-weighted market portfolio is optimal for the whole universe.
It says nothing of the kind about a sector, a screen, an exclusion list or any
other sub-universe, and the usual argument runs the other way: cap-weighting a
subset is generally far from optimal.

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
| estimate + solve | a full sub-covariance, `m(m+1)/2` numbers | the estimate is trustworthy |
| oracle | the true sub-covariance | unavailable |

Volatility is deliberately absent from the middle three. An optimal parent
already holds less of a volatile name, so the abilities calibrated from the
parent weights carry the volatility. Giving the race a structure scaled by
volatility as well applies it twice: measured, that moves the restricted
portfolio by 0.60 in L1 where correlation alone moves it by 0.06. What
restriction changes is which competitors are present, and how the departed
weight redistributes depends on how the survivors co-move.

## What would falsify the result

Proportional is the wrong null, and using it was the study's second defect.

An independent race is a power transform of proportional restriction and
almost nothing else: fitting `w ~ w_prop^a` to its output gives `a` between
0.75 and 0.81 with an L1 residual of 0.006 to 0.012, against a deviation from
proportional of 0.21 to 0.36. With no correlation anywhere, a departed name's
weight cannot flow toward whichever survivor it most resembled, because
nothing encodes resemblance. All the race can do is de-concentrate.

Measured against proportional, the independent race wins 72 percent of draws
and looks like a result. Measured against `flattened`, which is
`w_prop^0.75` renormalized, it wins 12 of 25 at p = 1.000. Any claim for the
race has to clear the `flattened` row. On the 25 draws run so far only
`race+factor` at T=20 does, by 3.6 percent at p = 0.015, which is one of three
tests and weakens as T grows, so it is not established.

The `race differs from proportional` check in `smoke.py` is therefore
necessary and nowhere near sufficient: it passes at L1 0.31 while the whole
0.31 is the power transform.

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

`markets.py` carries the part worth reading. A long-only minimum-variance
parent is fine at a few hundred names but is the wrong model of an index: at
five thousand names a single factor is so diversifiable that the optimum holds
about eighty of them. So the index market is built the other way round. Cap
weights come first, from a power law, and the covariance is then chosen to
make them optimal:

```
Sigma = 11' + eps Q M Q',   u = w / ||w||,   Q = I - u u'
```

which is positive definite and has `w` as its exact minimum-variance
portfolio, verified to 1e-14. Since `w >= 0` the long-only constraint is
inactive, so `w` is the long-only optimum too. That is the positive-definite
branch of the implied-covariance identity; the minimal rank-two correction
reproduces `w` as well but is not positive definite, which is why it is not
used. Nothing is ever formed densely, so blocks and the premise check are
`O(n)`.

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
