# Literature map — Tail-sensitive Black–Litterman

Where the construction sits: *reverse-optimize a benchmark to latent abilities, then
re-weight on the simplex under a view on **lower-tail co-movement**, via a fast exact
nonlinear inverse, never inverting a covariance and never running an optimizer.*

The map is organized along the three axes the method composes. Each axis is well
populated; the **intersection is not** (see "White space"). Honest reading: every
*ingredient* exists — what is unoccupied is the specific composition.

---

## Axis A — Reverse optimization / implied views

The BL reverse step recovers a **mean vector**; dependence is the fixed metric, never the
recovered latent.

- **Sharpe (1974), *JFQA* 9(3):463–472** — the original reverse step: impute expected
  returns that make held weights MV-optimal. Returns only; Σ given.
- **Black & Litterman (1992), *FAJ* 48(5):28–43** — `Π = λ Σ w_mkt`, then Bayesian view
  update `P μ = q + ε`; `τΣ` confidence scaling. Canonical. Views and inversion on returns.
- **Idzorek (2005)** — practitioner guide; view-confidence → `Ω` calibration (the dial our
  `φ` plays). *(working-paper venue — verify before citing)*
- **Qian & Gorman (2001), *FAJ* 57(2):44–51** — BL with views on **volatilities and
  correlations**, yielding a *conditional covariance*. The canonical "view on dependence"
  BL — but second-moment only (no tail), and it re-inverts a covariance for MV.

## Axis B — Inverse optimization & discrete-choice (share→latent) inversion

- **Ahuja & Orlin (2001), *Oper. Res.* 49(5):771–783** — foundational inverse-optimization
  theory (recover the objective making an observed solution optimal; itself an LP by duality).
- **Bertsimas, Gupta & Paschalidis (2012), *Oper. Res.* 60(6):1389–1403** — **the closest
  reverse-optimization neighbor**: recasts BL as inverse optimization, extends to **coherent
  (CVaR) risk**. But it inverts via a *conic optimization on returns/risk*, not a fast exact
  nonlinear inverse to abilities, and the view object is risk/returns, not tail co-movement.
- **Inverse-portfolio risk-preference learning (Yu/Chan; arXiv:2010.01687, 2510.06986)** —
  recover `λ` from holdings; *documents ill-conditioning* of the inverse-of-an-optimizer —
  an argument for having an **exact** inverse.
- **McFadden (1974); Berry (1994), *RAND* 25(2)** — logit/nested-logit share→utility inverse
  is **closed form** (`δ_j = log s_j − log s_0`). The one exact-and-fast regime.
- **Berry, Levinsohn & Pakes — BLP (1995), *Econometrica* 63(4)** — random-coefficients case
  has **no closed form**; inverted by a **contraction-mapping fixed point**. The workhorse
  when tastes are heterogeneous/correlated.
- **Berry, Gandhi & Haile (2013), *Econometrica* 81(5)** — "connected substitutes":
  conditions for global invertibility of a share system to the latent index (our existence/
  uniqueness backing).
- **Probit / correlated races**: *no closed-form share→utility inverse* — simulation or BLP
  iteration only. This is exactly the gap a fast exact correlated-race inverse fills.

## Axis C — Tail / non-normal / dependence-aware views and tilts

### Meucci — the most general "views on tails" machinery
- **Meucci (2006), *Risk* 19(9) — Copula-Opinion Pooling** — views on **realizations**,
  skew-t marginals glued by a copula; posterior is the blended distribution → downstream
  optimizer picks weights.
- **Meucci (2008), *Risk* 21(10) (arXiv:1012.2848) — Entropy Pooling** — views as
  moment/inequality constraints (means, **correlations, CVaR, tail probabilities, rankings**)
  on **scenario probabilities**; posterior = min-KL projection. Reweights a simplex of
  *scenario* probabilities via a **convex (dual-Newton) solve**; a separate optimizer still
  turns the posterior into weights. The strongest prior art for "tilt a normalized object
  toward a tail/dependence view" — but scenario-probabilities, not portfolio shares/abilities,
  and an optimization, not a closed-form inverse.

### BL with non-normality / higher moments / CVaR
- **Giacometti, Bertocchi, Rachev & Fabozzi (2007), *Quant. Fin.* 7(4)** — stable/t priors,
  VaR/CVaR risk in BL.
- **Closed-form BL-CVaR under elliptical returns (2017), *Oper. Res. Lett.* S0167637717306582.**
- **Harvey, Liechty, Liechty & Müller (2010), *Quant. Fin.* 10(5)** — full Bayesian portfolio
  choice under a **skew-normal** likelihood (skew/kurtosis + parameter uncertainty); MCMC +
  utility maximization.
- **Hidden-truncation BL (arXiv:2310.12333)** — skew posterior via hidden-truncation family.
- **Martellini & Ziemann (2010), *RFS* 23(4)** — improved coskewness/cokurtosis estimators;
  the "villain" of estimation-heavy higher-moment allocation (tensor dimensionality).

### Copula-BL — the explicit tail-aware-BL lineage to differentiate against
- **Sahamkhadam, Stephan & Östermark (2022), *EJOR* 297(3):1055–1070** — BL posterior via
  **R-vine copulas** (asymmetric tail dependence) → mean-CVaR optimize. **But the copula
  supplies the return scenarios; the reverse step is still returns-only `Π = λΣw`.**
- **BL + copula views in mean-CVaR (2023), *Econ. Change & Restructuring* 56(1).**

### Tail-dependence-aware allocation (mostly *not* BL — where our de-dup primitive lives)
- **Lohre, Rother & Schäfer (2020), SSRN 3513399** — **HRP using lower-tail dependence** as
  linkage; re-weights on the simplex by recursive bisection, **no covariance inversion** —
  closest in *mechanism* to "reweight the simplex without inverting Σ", but no view, no
  reverse-opt, no inverse map (a clustering heuristic).
- **De Luca & Zuccolotto (2011); "Maximal tail-dependence clustering," *Risks* 6(4):115 (2018)**
  — cluster co-crashers by lower-tail dependence (the de-dup primitive, as a hard pre-step).
- **Tail-risk parity** — equalize **expected-shortfall** contributions; weights driven by
  tail-fatness and tail correlations.
- **Mallela & Leonelli (2026), arXiv:2606.16840** — Hüsler–Reiss extremal graphs: lower-tail
  network near-complete, upper-tail thin ("crash together, rally apart"). *Independent
  empirical support for the premise* — cite as motivation, not competition.

## Axis D — Fast exact inverse of a *correlated* Thurstonian race

- **Thurstone (1927), *Psych. Rev.* 34** — law of comparative judgment (Case V = probit).
- **Clark (1961), *Oper. Res.* 9(2):145–162** — canonical moment-matching approximation of
  `E[max]` of correlated Gaussians (still the standard tool); both forward race and inverse
  are *approximate/iterative* in the standard literature.
- **Maydeu-Olivares & Böckenholt (2005), *Psych. Methods*** — latent-scale recovery by ML over
  order-statistic probabilities (numerical, not closed form).
- **Bottom line**: there is **no published fast *exact* inverse of a *correlated*
  order-statistic race** from win-shares to abilities. Exact-but-independent (logit log-shares)
  or correlated-but-approximate (Clark / BLP / simulation). The fast exact correlated-race
  inverse (the FAT) is novel in its own right.

---

## Closest neighbors and white space

**The triple — (reverse optimization) × (lower-tail co-movement view) × (fast exact
nonlinear share→ability inverse for a correlated race) — is unoccupied.** Each pairwise edge
exists; the triple does not:

1. Reverse optimization recovers **means**, never dependence (Sharpe; BL; even Bertsimas
   et al. 2012 inverts a risk objective, with Σ fixed).
2. Tail/dependence **views** exist but are bolted onto the *forward* step: copula-BL and
   entropy pooling state tail views, then the reverse map underneath is still returns-only
   `Π = λΣw`. No "tail-sensitive reverse step."
3. Discrete-choice **share→latent inversion** (Berry/BLP, connected-substitutes) is never
   connected to portfolio reverse optimization; and it has no fast exact form for a
   *correlated* race.

**Honest novelty statement (for the paper):** the contribution is not a new IIA fix (RFC/BLP
did that) nor a new tail-clustering rule (HRP-LTD, tail-dependence clustering did that). It is
the **composition**: treat benchmark weights as win-shares of a correlated Thurstonian race,
invert them *fast and exactly* to latent abilities, and let the *dependence of that race* carry
a tail-specific (downside/LPM-semicovariance) view — a tail-sensitive analogue of `Π = λΣw`
that never inverts a covariance and never runs an optimizer. Differentiate explicitly against:
Bertsimas et al. 2012 (inverse-opt BL, coherent risk, not dependence-inverted), Meucci entropy
pooling (tail views, but a distributional projection + convex solve, not a reverse-optimized
ability vector), Qian–Gorman 2001 (views on correlation, but second-moment + MV inversion),
Sahamkhadam 2022 (copula tail scenarios, but returns-only reverse step), Lohre 2020 (simplex
reweight without Σ-inversion, but a clustering heuristic with no view/inverse).

---
*Compiled from two literature sweeps (BL tail/non-normal extensions; reverse-optimization &
implied dependence views). Citations to be added to `papers/refs.bib`; venues flagged
"verify" should be confirmed before final submission.*
