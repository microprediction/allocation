# HRP and shrinkage: settled results

All variance figures are `w' Sigma w` against the true covariance, so there is no
backtest noise. Long-only on both sides wherever an optimizer appears.

## 1. No small shrinkage class reproduces HRP, for counting reasons

A k-parameter family traces a k-dimensional set of weight vectors at fixed S.
The weight vector has n-1 free coordinates. Cliff confirmed at k = n-1:

| n  | last k that misses | its L1 | first k that fits | its L1 |
|----|--------------------|--------|-------------------|--------|
| 16 | 15                 | 1.2e-02| 16                | 3.1e-10|
| 24 | 22                 | 9.6e-03| 23                | 2.0e-07|
| 40 | 38                 | 1.1e-02| 39                | 1.9e-04|

Control check: the diagonal family recovers a diagonal-shrinkage portfolio to
0.00 and still misses HRP by 0.222, so the miss is informative about HRP and
not an artefact of small families fitting nothing.

## 2. The closest sensible shrinkage is near-diagonal

Fitted per market, n=40, T=120. HRP realized variance 0.1660.

| family                    | params | L1 to HRP | its variance | vs HRP |
|---------------------------|--------|-----------|--------------|--------|
| per-level taper           | 6      | 0.2017    | 0.1558       | 0.94x  |
| two-level taper           | 2      | 0.2023    | 0.1597       | 0.96x  |
| diagonal target           | 1      | 0.2303    | 0.1615       | 0.97x  |
| hierarchical filter blend | 2      | 0.2303    | 0.1615       | 0.97x  |
| eigenvalue clipping       | 1      | 0.8447    | 0.2427       | 1.46x  |
| single index              | 1      | 0.8857    | 0.1325       | 0.80x  |
| constant correlation      | 1      | 1.1925    | 0.0716       | 0.43x  |

Two readings: the hierarchical filter blend chose depth 6 and intensity 0, i.e.
the plain diagonal, so HRP's weights are approximated better by ignoring all
correlation than by using its own block structure. And the ranking is close to
monotone in the wrong direction: the better the estimator, the further from HRP.

## 3. HRP sits at the dominated corner of a family containing it

Five blocks, T/n=10, HRP 0.1661. Cell = median variance (share beating HRP).
Parameters fixed a priori, never fitted to HRP.

| depth | phi=0        | phi=0.25     | phi=0.5      | phi=0.75     | phi=1        |
|-------|--------------|--------------|--------------|--------------|--------------|
| 1     | 0.1676 (43%) | 0.1184 (100%)| 0.1033 (100%)| 0.0970 (100%)| 0.0939 (100%)|
| 3     | 0.1676 (43%) | 0.1418 (100%)| 0.1231 (100%)| 0.1147 (100%)| 0.1101 (100%)|
| 5     | 0.1676 (43%) | 0.1646 (65%) | 0.1611 (87%) | 0.1577 (93%) | 0.1562 (95%) |

Monotone in phi along every row. Tapering by phi equals damping the complement
at gamma = phi^2, so this is the interiority claim in estimator form.

## 4. NCO is on the frontier, HRP is not

Same clusters for both. Five blocks, T/n=10: HRP 0.1661, NCO 0.0892.
Best taper cell beats HRP 100% of draws by 47%, and beats NCO 70% by 0.06%.

## 5. Robustness: 500 randomized structures

Family, cluster count and sizes, correlation levels, extra factors, vol
dispersion, n, T/n and tail weight all resampled. Shrinkage is a single fixed
phi = 0.5, never tuned.

| comparison                                | wins | median ratio |
|-------------------------------------------|------|--------------|
| fixed taper beats HRP                     | 78%  | 0.740        |
| NCO beats HRP                             | 77%  | 0.738        |
| long-only min-var on raw sample beats HRP | 76%  | 0.726        |
| fixed taper beats NCO                     | 43%  | 1.005        |

By family: blocks+factor 89, blocks 88, nested 83, equicorr 81, factor 71,
wishart 60. By ratio: 0.5 -> 46%, 1 -> 69%, 2 -> 87%, 5 -> 93%, 10 -> 96%.

The third row matters most: the no-short-sale constraint alone, which is
Jagannathan-Ma shrinkage, beats HRP three times in four.

## 6. The rank-deficient corner, HRP's actual claim

Random structures, n in {40,100}. Cell = share beating HRP / median ratio.

| T/n  | LW min-var  | phi=0.5 taper | block-diagonal | NCO         | inverse variance |
|------|-------------|---------------|----------------|-------------|------------------|
| 0.10 | 50% / 1.002 | 46% / 1.033   | 49% / 1.006    | 36% / 1.171 | 59% / 0.958      |
| 0.20 | 47% / 1.027 | 45% / 1.052   | 45% / 1.049    | 44% / 1.094 | 52% / 0.996      |
| 0.30 | 54% / 0.974 | 54% / 0.930   | 55% / 0.954    | 55% / 0.920 | 44% / 1.015      |
| 0.50 | 65% / 0.879 | 63% / 0.811   | 63% / 0.861    | 64% / 0.804 | 40% / 1.018      |
| 0.75 | 72% / 0.814 | 69% / 0.710   | 70% / 0.737    | 69% / 0.700 | 35% / 1.030      |
| 1.00 | 77% / 0.754 | 75% / 0.680   | 75% / 0.687    | 75% / 0.666 | 35% / 1.040      |

HRP does edge the inverting methods below T/n = 0.2, by 3 to 5 percent, and
beats NCO there by 17 percent because NCO still has to invert cluster blocks.
But inverse variance beats HRP in that same corner. So the window closes from
both sides and HRP has no interval where it is the right answer.

## 7. Thurstone tilt of HRP: negative on variance

Thurstone calibrated to reproduce HRP's own weights, then tilted toward the
estimated correlation by phi. Paired under common random numbers (fixed seed
ensemble transported across phi), n=40, 16384 paths, 30 random structures.

Monte Carlo pedestal: the phi=0 race misses exact HRP by 0.0279 in L1, but its
realized VARIANCE is within 0.2% of exact HRP (1.0022 at T/n=0.1), so the
pedestal is negligible in the metric being measured.

variance(tilt at phi) / variance(tilt at 0), median (share below 1):

| T/n  | phi=0.25     | phi=0.5      | phi=0.75     | phi=1.0      |
|------|--------------|--------------|--------------|--------------|
| 0.10 | 1.0034 (43%) | 1.0323 (33%) | 1.0879 (30%) | 1.1917 (27%) |
| 0.30 | 0.9947 (53%) | 0.9959 (53%) | 0.9915 (53%) | 1.0050 (47%) |

At T/n=0.1 tilting monotonically HURTS, 19% worse variance at full tilt. At
T/n=0.3 it is a coin flip with effects under 1%. So the tilt does not rescue
HRP in the rank-deficient corner.

This is a vindication of the dial's MEANING rather than a failure of it: phi
says how much to trust the estimated correlation, and at 4 observations for 40
assets that correlation is nearly pure noise, so the correct setting is near
zero and the sweep says so.

CAVEATS, all material:
- Variance objective only. The Thurstone construction is designed as a TAIL
  statistic (student_t race, downside semicovariance); judging it on variance
  may be the scoring mismatch studied earlier in this session.
- Diagonal calibration only. Market calibration costs ~12s/fit and was not swept.
- Gaussian race only, n=40 only, T/n in {0.1, 0.3} only.
- Tilting EQUAL WEIGHT is clearly bad here: 1.3x to 2.3x HRP's variance,
  winning 13-30% of draws. The benchmark tilted FROM matters more than the tilt.

An earlier run at 4096 paths and n=100 was discarded: the phi=0 fidelity gap
was 0.118 in L1 against effects of 2-4%, so it was noise-dominated. The gap
scales as 1/sqrt(paths) with constant 4.3 (n=40) and 7.8 (n=100), confirming
Monte Carlo error rather than a calibration fault.

## Scripts (session scratchpad)

confound.py, implied.py, bridge_distortion.py, psd_gamma.py, what_shrinkage.py,
dimension.py, closest.py, decompose.py, shrinkage_beats_hrp.py,
taper_beats_hrp.py, nco_taper.py, robust.py, rank_deficient.py,
thurstone_rank.py, thurstone_dial.py


## 8. Thurstone tilt DOES repair HRP, on a regime-switching market

Market: calm dependence structure, plus a crash regime firing 8% of the time
with an INDEPENDENT structure over a permuted labelling, 3x volatility and
negative mean. Validated non-elliptical: shortfall / standard deviation varies
17% across portfolios here against 1% under multivariate t, so the tail is not
a function of variance.

Paired under common random numbers, tilt(phi) against tilt(0) on the same seed
ensemble. n=40, 16384 paths, 60 draws, standard error 6% on every win rate.

variance ratio, win rate:

| T/n  | phi=0.5          | phi=1.0          |
|------|------------------|------------------|
| 0.30 | 0.9873  67% +/-6 | 0.9689  63% +/-6 |
| 1.00 | 0.9690  73% +/-6 | 0.9208  67% +/-6 |

expected shortfall ratio, win rate:

| T/n  | phi=0.5          | phi=1.0          |
|------|------------------|------------------|
| 0.30 | 0.9930  63% +/-6 | 0.9742  62% +/-6 |
| 1.00 | 0.9840  70% +/-6 | 0.9574  70% +/-6 |

All win rates sit 2 to 4 standard errors above a coin, so the effect is real.
Full tilt beats half tilt on variance at both ratios, so more tilt is better
once there is data.

Combined with section 7, the dial behaves exactly like a trust parameter:
- T/n=0.10: tilting HURTS monotonically, 19% worse at full tilt. Optimum near 0.
- T/n=0.30: tilting helps 1-3%, 62-67% of draws.
- T/n=1.00: tilting helps 3-8%, 67-73% of draws.

IMPORTANT distinction: on a stationary Gaussian market the same paired test at
T/n=0.3 was NEUTRAL (0.9915-1.0050, 47-53%). The repair appears on the
regime-switching market and not the stationary one, which is interpretable: the
race is a non-linear functional of the correlation, so it can express structure
a single covariance cannot.

NEGATIVE within this: the Student-t race is consistently WORSE than the
Gaussian race, on both objectives, at every ratio. The tail-aware sampler does
not earn its keep even on a market with asymmetric tail dependence.

## 8b. Replicated across three seeds; the effect is monotone in sample size

Run 1 (60 draws, seed 13579), run 2 (40 draws, seed 86420), run 3 (50 draws,
seed 24680). Paired tilt(phi)/tilt(0), crash market, Gaussian race.

Full tilt, variance ratio and win rate, best estimate pooling available runs:

| T/n  | variance ratio | win rate | status                                  |
|------|----------------|----------|-----------------------------------------|
| 0.10 | 1.19           | 27%      | tilting HURTS (from section 7, Gaussian mkt) |
| 0.30 | 0.97 - 1.01    | 48-63%   | borderline; run 2 killed the phi=1 cell |
| 1.00 | 0.98           | 60-67%   | modest, replicates                      |
| 2.00 | 0.90 - 0.92    | 82-84%   | REPLICATED EXACTLY (z=0.00, +0.25)      |
| 5.00 | 0.87           | 88%      | strongest cell, one run                 |

Expected shortfall follows the same shape: neutral at parity (50-55% in run 2,
against 70% in run 1, so the parity tail benefit did NOT replicate), but 78-84%
at T/n of 2 and 5, with ratios 0.93 to 0.98.

Honest note on run 2: all four cells shared with run 1 moved DOWN (67->62,
63->48, 73->62, 67->60). All pass a two-sigma check, but four of four in the
same direction is regression to the mean, so run 1's magnitudes were optimistic.
The high-data cells did not show this: run 3 reproduced run 2's T/n=2 numbers
to within a quarter of a standard error.

CONCLUSION: the tilt is worth nothing when the correlation is noise, and worth
13% of variance when it is not. The dial is a trust parameter and its optimum
walks from 0 to 1 as coverage grows. That is the same shape as the Schur
coupling and the same shape as the taper.

## 6b. HRP's best case: a market that really is a nested hierarchy

The obvious objection to sections 5 and 6 is that they aggregate structural
families, and HRP assumes a hierarchy. Clean dyadic nesting, n=64, correlation
rising at all four levels, 120 draws. Cell = share beating HRP / median ratio.

| T/n  | inv var     | Ledoit-Wolf | taper 0.5   | NCO         | taper, TRUE tree |
|------|-------------|-------------|-------------|-------------|------------------|
| 0.10 | 85% / 0.870 | 15% / 1.323 | 36% / 1.197 | 22% / 1.342 | 63% / 0.918      |
| 0.25 | 68% / 0.968 | 36% / 1.072 | 97% / 0.739 | 93% / 0.733 | 100% / 0.650     |
| 0.50 | 52% / 0.999 | 92% / 0.823 | 100% / 0.588| 100% / 0.575| 100% / 0.562     |
| 1.00 | 22% / 1.016 | 100% / 0.657| 100% / 0.550| 100% / 0.548| 100% / 0.542     |
| 2.00 | 11% / 1.021 | 100% / 0.560| 100% / 0.527| 100% / 0.522| 100% / 0.521     |
| 10.0 |  2% / 1.026 | 100% / 0.497| 100% / 0.503| 100% / 0.502| 100% / 0.503     |

The objection sharpens the finding rather than rescuing the method. Past
T/n=0.25 it loses essentially every draw by a FACTOR OF TWO: at T/n=10 it is
0.1350 against 0.0671. Below that the inverting methods do lose and inverse
variance wins instead, 85% of draws.

The decisive column is the one that does not move. Over a hundredfold increase
in sample size HRP improves 24% (0.1778 -> 0.1350) while its rivals improve
56% (0.1538 -> 0.0671). A method that barely learns from data is not
recovering a hierarchy from it, whatever hierarchy is present.

Script: `nested_best_case.py`.

## 6c. The seriation works; the split rule cannot use it

HRP has two ideas: quasi-diagonalization (reorder so similar assets are
adjacent) and recursive inverse-variance bisection down that order. Hold each
fixed, vary the other. Clean nested market so the TRUE order is known.

Ratio of realized variance to the same construction on a RANDOM ordering or
grouping; below 1 is an improvement:

| T/n  | seriation inside the recursion | same grouping inside a taper |
|------|--------------------------------|------------------------------|
| 0.25 | 1.018                          | 0.896                        |
| 0.50 | 1.013                          | 0.922                        |
| 1.00 | 1.000                          | 0.947                        |
| 2.00 | 0.991                          | 0.949                        |
| 5.00 | 0.991                          | 0.952                        |
| 10.0 | 0.993                          | 0.958                        |

Inside the recursion the ordering is worth NOTHING: HRP on its own seriation
is within 2% of HRP on a random permutation at every sample size, and slightly
worse at light sampling. Handing it the TRUE order does not help either.

Handed to a taper the identical clustering is worth 4-10%, and the true
grouping is better still. The ordering carries real information; the recursion
cannot spend it.

This is the sharpest form of the thesis. The seriation is not the problem. The
inverse-variance split is what makes the information inert, because the
variance of an inverse-variance sub-portfolio barely depends on how its
members were ordered.

Script: `seriation_value.py`.

## 6d. THE BOUNDING RESULT: 5000 assets, two years of weekly data

Everything else here runs at n=40 to 100, which is where a known-truth
simulation is cheap. The regime a large-universe manager faces is the
opposite. Factor market with 50 group factors so realized risk is exact
without forming the covariance. 30 draws each, T=104 throughout.

| assets | T/n   | inv var beats HRP | 95% Wilson  |
|--------|-------|-------------------|-------------|
| 500    | 0.208 | 83%               | [66%, 93%]  |
| 2000   | 0.052 | 47%               | [30%, 64%]  |
| 5000   | 0.021 | 10%               | [3%, 26%]   |

At n=5000 HRP is the BEST method tested: 0.00487 against 0.00504 inverse
variance, 0.00549 clustered optimizer, 0.00563 equal weight, winning every
draw against the last two.

**This reverses the headline of sections 5 and 6 and bounds every other result
in this file.** The deciding variable is the NUMBER OF ASSETS, not T/n: at
T/n=0.1 with n=40 HRP loses; at T/n=0.021 with n=5000 it wins. Crossover near
n=2000.

Conjectured mechanism, NOT isolated: inverse variance is exposed to the worst
of n independent variance estimates, which degrades as n grows at fixed T,
while the top levels of the bisection consume aggregates over thousands of
names that are well determined from 104 observations.

Script: `huge_universe.py`.

## 9. Scope limit on section 5

Ledoit-Wolf min-var, which beat HRP in 76% of stationary structures, is 1.21x
HRP's variance at T/n=0.1 on the crash market and 1.10x at 0.3. Shrinking
toward a single target cannot represent two dependence structures. So "a simple
shrinkage rule beats HRP" holds on stationary markets and REVERSES under regime
switching. This is a scope limit, not a footnote.

## Failed measurements, kept deliberately

1. Thurstone at 4096 paths, n=100: phi=0 fidelity gap 0.118 L1 against effects
   of 2-4%. Noise-dominated. Gap scales as 1/sqrt(paths), constant 4.3 (n=40),
   7.8 (n=100), so Monte Carlo error not a calibration fault.
   SELF-INFLICTED: the path budget is nearly free. 4,096 to 262,144 paths costs
   30% more time, not 64 times more, because the cost is the ability
   calibration and not the race. Running at 262,144 would have put the gap near
   0.015 and the paired design would not have been necessary. Check what a knob
   actually costs before economising on it.
2. Tail test on multivariate t: elliptical, so ES ratio came out as exactly
   sqrt(variance ratio) and the tail column was a relabelling of the variance
   column. Could not test tail skill in principle.
3. First crash market pushed all correlations uniformly toward 1: spread only
   1.009 against 1.004 for multivariate t, so still effectively elliptical.
   Fixed by giving the crash regime its own independent structure.


## 10. CORRECTION to 7 and 8: the race is a diversification operator

Earlier sections concluded the tilt only helps on a regime-switching market.
That was the wrong reading. The benchmark tilted in those tests was HRP, which
already discounts correlated names, so there was little correlation left to
act on. What governs the gain is how much correlation the BENCHMARK ignores.

One race, one correlation, three starting points. One-factor market with
heterogeneous variances, n=120, long-only optimum solved with cvxpy:

| benchmark                  | before  | after race | change |
|----------------------------|---------|------------|--------|
| equal weight               | 0.32059 | 0.16882    | -47.3% |
| inverse variance           | 0.12494 | 0.06148    | -50.8% |
| long-only minimum variance | 0.02705 | 0.02709    |  +0.1% |

It halves the variance of benchmarks that ignore correlation and is nearly a
FIXED POINT on the optimum. Direction check: raced weights correlate -0.996
with factor loading against -0.989 for long-only min-var, and the race captures
84% of the variance reduction available between equal weight and the optimum.

This explains the n=5000 negative result: HRP is the best construction at that
size, so the race was being asked to improve on the ceiling.

Also corrected: the race is now evaluated by QUADRATURE, not simulated. At
phi=0 it reproduces the benchmark to 1e-10 in 0.1s where the simulation needed
65536 paths and 2.6 GB to reach 0.2. Simulation resolution tracks paths PER
ASSET; at n=5000 with 4096 paths there is less than one path per asset and the
returned book differs from its own input by 82%. Two dial sweeps reported
earlier in this file were measuring that noise.

Scripts: the benchmark comparison is inline above; `residual_tilt.py` holds the
verifiable construction with the phi=0 identity asserted.

## 11. Four attempts that failed the same way

Tilting toward the covariance HRP discards is a good idea and I got it wrong
three times before measuring it. Every failure was the same: the abilities were
calibrated under one law and the phi=0 race run under another, so zero on the
dial was already a tilt and the curve was measured from the wrong origin.

1. Calibrated under independence, raced under the full correlation.
2. Approximated the target by its top-3 eigenvectors, which are the GLOBAL
   factors, so the tilt deleted the block structure instead of adding to it.
3. Kept the blocks in the race but still calibrated under independence; phi=0
   came out at 1.0018 rather than 1.0000 and I read past it.
4. A literal `if False else Z` left in the factor split, so group loadings were
   fitted to data still containing the global factors.

THE CHECK: phi=0 must reproduce the benchmark to ~1e-10. Assert it and exit if
it fails, before reading any other number.
