# Portfolio-Implied Contests, with an Application to Tail Risk

- **Author**: Peter Cotton
- **Posted**: [SSRN 7737202](https://ssrn.com/abstract=7737202); submitted to arXiv
- **PDF**: [paper.pdf](https://allocation.microprediction.org/papers/thurstone-portfolios/paper.pdf)

## Abstract

We use a recently developed contest inversion algorithm as a device for portfolio
construction and augmentation, and in particular we demonstrate how an investor can
inject views about tail dependence into an existing portfolio by this means. Each
asset is a runner in a Thurstone race whose noisy performances may be correlated
through factor or hierarchical structure, and its weight is the probability that it
finishes first. A fast deterministic inverse recovers abilities from benchmark
weights, and a simulation re-runs the race under new dependence. Neither inverts a
covariance or solves an optimization. Given a factor model the cost is linear in the
number of assets, and thousands of names fit in seconds. Near-duplicates split a
position rather than double it. We prove benchmark reproduction, an implied convex
objective and a turnover bound.

## Around the paper

- Method documentation: [Thurstone portfolio polishing](https://allocation.microprediction.org/methods/thurstone-portfolio-polishing/)
- Interactive demo: [A portfolio is a race](https://allocation.microprediction.org/demos/portfolio-contest/)
- Slides: [talk](https://allocation.microprediction.org/papers/thurstone-portfolios/talk.pdf),
  [introductory presentation](https://allocation.microprediction.org/papers/thurstone-portfolios/intro-presentation.pdf)
- Literature: [interactive map](https://allocation.microprediction.org/maps/portfolio-implied-contests.html),
  [LITERATURE-MAP.md](LITERATURE-MAP.md)
- The inverse: Cotton (2021), [SIAM J. Financial Math. 12(1)](https://doi.org/10.1137/19M1276261);
  Cotton (2026), [arXiv:2609.01133](https://arxiv.org/abs/2609.01133)
- Code: `allocation.ThurstonePortfolio`, with calibration and racing in
  [`winning`](https://github.com/microprediction/winning)

## Files

- `paper.tex`, `sections/` — manuscript (cites `../refs.bib`)
- `figures/` — figures used in the paper
- `talk/` — Beamer source of the talk
- `arxiv/` — arXiv submission files
- `REVISIONS.md` — revision notes
- `LITERATURE-MAP.md` — related work, as text
