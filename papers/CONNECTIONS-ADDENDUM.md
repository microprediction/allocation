# Connections addendum — missing academic anchors

Cross-paper literature that the construction sits on but the bib did not yet cite. All entries
below are now in `papers/refs.bib` (Crossref-verified DOIs where a DOI exists; open-access ML
venues and older proceedings carry venue/arXiv only, matching the `berthet2020` convention).

Organizing idea: the whole package is one object — the win-share field
`w = ∇G_S(θ)` with `G_S(θ) = E[max_i(θ_i + η_i)]`, dual to `w = argmax_{p∈Δ}{⟨θ,p⟩ − Ω_S(p)}`,
`Ω_S = G_S^*`. Each cluster below is a literature that already owns one face of that object.
Ranked by how much it changes a current claim.

**Wiring status (2026-08).** 17 of the 25 are now `\parencite`d into the manuscripts and the
three edited papers build clean (no undefined citations): clusters 1, 2, 4, most of 3 and 5.
Still reservoir-only, with reasons: `mcmahan2011` (redundant with `hazan2016` in the online
paper); `fiedler1973`/`atkins1998`/`friedman2008` (the Schur manuscript is `cotton2024schur`,
a separate repo — no `.tex` here to wire them into); and the lower-priority trio
`cuturi2013`/`peyre2019`/`pickands1981`/`artzner1999` (decorative — left for when the relevant
prose is written). Unused shared-bib entries emit nothing at build time; this is the reservoir
pattern the bib already uses.

---

## 1. The redundancy/IIA argument had no cited backbone

- **`yellott1977`** — *J. Math. Psychology* 15(2):109–144. IIA holds **iff** the race noise is
  Gumbel. This is the exact theorem behind the pivot: any Luce/softmax/cap-weight rule inherits
  red-bus/blue-bus, and escaping it *requires* leaving the Gumbel world (Gaussian, tail-dependent).
  Slots into `thurstone-portfolios/sections/00-motivation.tex` §capm and `20-theory.tex`
  (Prop. redundancy) — turns "Lucian rules fail" from assertion into a cited characterization.
- **`tideman1987`** — *Social Choice and Welfare* 4(3):185–206. "Independence of clones" is the
  established name, in voting theory, for the property the paper calls clone-consistency. Gives
  it priority and an axiomatic pedigree beyond the choice-modelling anecdote. Same sections.

## 2. The "implied objective" identity is a live ML object (a dependence-aware Ω)

`argmax_{p∈Δ}{⟨θ,p⟩ − Ω(p)}` is the Fenchel–Young / regularized-prediction map; the paper is
proposing a **dependence-aware** Ω. Slots into `thurstone-portfolios/sections/20-theory.tex`
(Thm. objective, §temperature) and the `thurstone-credit` MoE/attention framing.

- **`blondel2020`** — Fenchel–Young losses (JMLR); the general framework `Ω_S` is a special case
  of. `berthet2020` (already cited) is the same school.
- **`martins2016`** — sparsemax; the Euclidean-Ω sibling of softmax, precedent for "swap the
  regularizer, get a different simplex map."
- **`nesterov2005`** — smooth minimization of non-smooth functions; cleaner machinery for the
  smoothness/Lipschitz theorem (`20-theory.tex` §smoothness) than Price's theorem alone — the
  smoothed-max *is* a Nesterov/Moreau smoothing, and smoothness is the paper's discriminator.
- **`papandreou2011`**, **`jang2017`**, **`maddison2017`** — perturb-and-MAP and
  Gumbel-softmax/Concrete. The operator is the **Gaussian-noise** analogue of Gumbel-softmax;
  these make the MoE-router and attention experiments legible to an ML reader.

## 3. Biggest free lunch: the operator is Follow-the-Perturbed-Leader

"Perturb abilities, take the argmax, low turnover" is textbook FTPL. The
`online-portfolio-regimes` paper is currently empirical with no theory anchor — this supplies
regret theory and recasts the Lipschitz/turnover bound as FTPL stability.

- **`hannan1957`**, **`kalaivempala2005`** — FTPL origin and the efficient-algorithms form.
- **`cover1991`** — universal portfolios; the canonical online-portfolio regret benchmark.
- **`mcmahan2011`**, **`hazan2016`** — FTRL ≡ mirror-descent equivalence, and the OCO reference.
  Entropic `Ω` = Hedge/multiplicative weights; the Gaussian `Ω` is a sibling regularizer.
  Slots into `online-portfolio-regimes/paper.tex` (intro + a new "why these rules" paragraph).

## 4. The credit paper cites ML attribution but not the finance-native version

"Credit = gradient of a potential" already exists in finance as **Euler / gradient risk-capital
allocation** — arguably a closer peer to the win-prob rule than SHAP, and conspicuously absent
next to `lundberg2017`. Slots into `thurstone-credit/paper.tex` (related work + the Prop. bridge
discussion).

- **`denault2001`** — coherent allocation of risk capital (the Euler idea).
- **`kalkbrener2005`** — axiomatic capital allocation.
- **`tasche2008`** — the Euler principle for sub-portfolios; the direct gradient-allocation cite.

## 5. The entropy Ω has an economic reading; the Schur ordering has a name

- **`matejka2015`**, **`fosgerau2020`** — rational inattention. The `G_S ↔ Ω_S` duality *is* the
  RUM–inattention equivalence; `Ω_S` becomes an attention/information cost. Slots into
  `thurstone-portfolios/sections/20-theory.tex` §temperature (the "generalized entropy" reading).
- **`fiedler1973`**, **`atkins1998`** — algebraic connectivity and spectral seriation; the
  established names for the Fiedler ordering in `allocation/_schur/seriation.py`.
- **`friedman2008`** — graphical lasso; Schur block-diagonality ⇔ a Gaussian conditional-
  independence graph. Both slot into the Schur write-up / `online-portfolio-regimes` §wellposed.

## Lower priority (real, more decorative)

- **`cuturi2013`**, **`peyre2019`** — entropic OT / Sinkhorn; common-seed transport (recolor a
  fixed seed by `C^{1/2}`) is a coupling, so the `_thurstone/transport.py` module has an OT reading.
- **`pickands1981`** — the Pickands dependence function, to formalize "tail-aware" via
  max-stable dependence rather than only Clayton/LPM examples.
- **`artzner1999`** — coherent risk measures; if you want subadditivity/diversification to
  *explain why* clone-splitting is the correct response rather than just exhibiting it.

---
*Compiled 2026-08 from a repo-wide method map vs. `refs.bib`. DOIs Crossref-verified; a few
open-access ML venues (JMLR/ICLR/ICML/AISTATS/NeurIPS) and older proceedings (Hannan 1957,
Pickands 1981) carry no Crossref DOI and are cited by venue + arXiv.*
