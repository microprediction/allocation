# Winning Probabilities as Credit

- **Status**: draft
- **Authors**: Peter Cotton
- **Slug**: `thurstone-credit`

## Abstract

The probability that a competitor wins a noisy race is a rule for attributing
credit among correlated contributors: a probability share, symmetric, and
redundancy-aware (contributors that move together share a single contributor's
credit rather than double-counting), differentiable in the contributors'
abilities, updates online, and costs O(Mn) in a single Monte-Carlo pass with no
coalition enumeration. It is not a Shapley value and does not try to be:
Shapley measures coalitional cooperation, the winning probability measures
selection relevance, how often a contributor is the single best in the field
at hand. We show the headline redundancy property, an even split of credit
among near-duplicates, is a property of the calibrated equal-ability race
rather than of raw scores, and demonstrate the rule on forecast combination
and feature attribution. This develops the credit reading of the construction
in the companion paper on Thurstone portfolios.

## Files

- `paper.tex` — manuscript (imports `../shared/preamble.tex`, cites `../refs.bib`)

## Build

```sh
cd .. && ./build.sh thurstone-credit   # -> paper.pdf (requires tectonic)
```
