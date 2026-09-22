# Hierarchical Risk Parity Has No Regime of Advantage

Finds that hierarchical risk parity earns its keep at large universe size and
not below it. At five thousand assets with two years of weekly data it is the
best method we can run, beating inverse variance in 90 percent of draws and
every feasible optimizer in all of them. At five hundred assets with the same
history it loses to inverse variance in 83 percent. The crossover sits near two
thousand, so the deciding variable is the number of assets rather than the
ratio of observations to them.

Below that scale it has no interval of advantage: beaten by inverse variance
where nothing can be inverted, by any regularised optimizer where something
can, and by long-only minimum variance on the raw sample in 76 percent of five
hundred random structures.

Supporting results: no shrinkage class with fewer than `n-1` parameters can
reproduce the rule, confirmed at three universe sizes; and nested clustered
optimization on the same clusters tracks the shrinkage frontier while
hierarchical risk parity does not, which puts the cost in the split rule
rather than in the hierarchy.

Reproduce with `experiments/studies` and `experiments/benchmark`; the numbers
are recorded in `experiments/studies/RESULTS.md` sections 1 to 6 and 9.

Build: `./build.sh hierarchical-risk-parity-regime`
