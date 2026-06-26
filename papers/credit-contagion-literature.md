# Literature map — credit, default contagion, systemic risk, and the limits of pairwise dependence

Compiled from four background literature sweeps (2026-06) to position the **bowling /
tetrachoric demo** (`docs/demos/bowling/`) and the credit-modeling side of the
tail-sensitive work. Demo claim: a Gaussian copula calibrated to reproduce **every pairwise
default probability** (tetrachoric) *and* every marginal still catastrophically under-prices
**systemic** (many-name simultaneous) default — the joint tail lives *above* all second-order
information. DOIs below are Crossref-verified unless flagged; bib keys refer to
`papers/refs.bib`.

---

## 1. The accurate crisis narrative (correcting the "formula that killed Wall Street" myth)

The popular "a single Gaussian-copula formula killed Wall Street" ([Salmon 2009, *Wired*];
Jones, *FT* 2009) is a **myth**. The accurate picture:

- **Sell-side** desks did *not* naively trust the copula. They used the one-factor Gaussian
  copula as a **quoting convention** — the credit analogue of Black–Scholes implied vol — and
  traded the **implied-correlation surface**. When tranche-by-tranche *compound correlation*
  proved non-unique/ill-posed, JPMorgan's **base correlation** (McGinty, Beinstein, Ahluwalia
  & Watts 2004; echoed by Lehman's O'Kane–Livesey) became the quoting standard, and desks
  traded the **correlation smile/skew** — extending the model (Andersen–Sidenius 2005),
  pricing it fast to *invert* tranche quotes for implied correlation [[hullwhite2004]], and
  openly debating its tail flaws years before 2008 (Whitehouse, WSJ 2005; the May-2005 GM/Ford
  "correlation crisis" that blew up the long-equity/short-mezzanine trade).
- **Buy-side** over-reliance sat on **rating-agency "non-models"** — Moody's Binomial Expansion
  Technique + diversity score, S&P's CDO Evaluator, Fitch's VECTOR — which fed **historical
  default-rate matrices + *assumed*, rating/sector correlations** (not market-implied
  dependence) into ratings (Fender & Kiff 2004, [[fenderkiff2004]]). Those AAA grades were
  "extremely fragile" to small errors in exactly those assumed-correlation inputs
  ([[coval2009]]); the alchemy is documented in [[benmelech2009]] and the FCIC report.
- **The actuarial (P) vs market-implied (Q) divide**: risk-neutral default probabilities from
  CDS/bond spreads run far above historical/actuarial frequencies (Hull–Predescu–White 2005,
  *J. Credit Risk* 1(2):53–60 — *DOI uncertain, cite by venue*). The catastrophe concentrated
  where market-implied dependence was *ignored* in favor of actuarial ratings.
- **Scholarly rebuttal (cite this):** [[mackenzie2014]] (114 practitioner interviews): quants
  disparaged the copula as "not really a model" yet kept it as a communication / risk-control /
  **P&L-booking** device — a model with no causal force "in itself"; Salmon's personalization on
  Li was "quite misplaced." Companion: MacKenzie & Spears, "A device for being able to book
  P&L" (DOI 10.1177/0306312713517158). Quant rebuttal: [[donnelly2010]] — the math was sound;
  the failure was application, parameterization, and ignoring tail dependence.

## 2. The correlated-default models being critiqued

- **[[li2000]]** — the Gaussian copula of default *times*; asset-correlation ≡ normal copula.
  Pairwise-only; no contagion. **(a)**
- **Vasicek (2002, *Risk*)** — single systematic Gaussian factor + idiosyncratic noise; the
  large-portfolio (ASRF) loss limit underlying Basel II IRB. One factor ⇒ one uniform pairwise
  correlation; conditional independence given the factor. **(a)**  *(Risk magazine, no DOI.)*
- **Merton (1974)** / **CreditMetrics (Gupton–Finger–Bhatia 1997)** — the structural /
  latent-Gaussian-threshold backbone the copula inherits. **(a)**
- **[[freymcneil2002]]** (and Frey–McNeil 2003, *J. Risk*) — **the key formal antecedent**: all
  standard latent-variable credit models map to **Bernoulli mixture models**, and the
  *mixing-distribution tail* (a higher-order object) governs default clustering and large-
  portfolio tail loss. Fitting marginals + correlation does **not** pin down the loss tail.

## 3. Default contagion / cascade models (the "bowling" mechanism)

- **[[davislo2001]]** — *infectious defaults*: a defaulter can topple others (domino-capable).
  The canonical mechanical-contagion generator, closest in spirit to a pins/cascade. **(c)**
- **[[jarrowyu2001]]** — *looping default*: one firm's default raises survivors' intensities
  (counterparty contagion) beyond common-factor correlation. **(c)**
- **Giesecke–Weber (2004)**, **Egloff–Leippold–Vanini (2007)** (*J. Banking & Finance*) —
  interacting-particle / micro-structural contagion on a partner network; even moderate local
  contagion fattens the loss tail of a well-diversified book. **(c)**

## 4. Systemic-risk networks & cascade dynamics

- **[[eisenbergnoe2001]]** — clearing-payment fixed point; the canonical mechanical-clearing
  engine a rigid-body cascade resembles. **[[gaikapadia2010]]** — analytic random-graph default
  cascades, **"robust-yet-fragile"** (rare but, past a percolation threshold, system-wide).
  **[[acemoglu2015]]** — phase transition: dense links absorb small shocks, propagate large
  ones. **[[battiston2012]]** (DebtRank) — empirical feedback-centrality systemic-impact score
  (FED data).
- **[[watts2002]]** — binary-threshold *global cascades*: rare but system-wide, **bimodal**
  cascade-size law. **The single most direct theoretical template for the bowling mechanic.**
  **[[baktangwiesenfeld1987]]** — self-organized-criticality sandpile: one grain can trigger a
  power-law avalanche; the physical "one hit clears the rack" metaphor.
- **Fire-sale / overlapping-portfolio contagion**: Greenwood–Landier–Thesmar (2015),
  Caccioli et al. (2014, branching-process threshold), Cont–Schaanning (indirect contagion).
- **Measurement targets** (how a regulator scores the joint tail): **[[adrianbrunnermeier2016]]**
  (CoVaR) and **[[brownleesengle2017]]** (SRISK) — conditional joint-tail statistics.

## 5. Higher-order vs pairwise — why matched correlations miss the joint tail

- **[[sibuya1960]]** — the foundational result: a bivariate normal with any ρ<1 is
  **asymptotically independent** (tail-dependence λ=0); only ρ=1 gives λ=1.
- **[[ledfordtawn1996]]** (and Ledford–Tawn 1997) — the coefficient of tail dependence; for the
  Gaussian copula η=(1+ρ)/2<1, so joint exceedance decays **strictly faster** than any
  tail-dependent copula. Coles–Heffernan–Tawn (1999): the (χ,χ̄) diagnostic — Gaussian sits at
  χ=0, χ̄=ρ (residual-only, not genuine, tail dependence).
- **[[huajoe2011]]** — extends η to the **d-dimensional tail order**: the cleanest *d-wise*
  statement that Gaussian joint-exceedance decays faster. **The sharpest rigorous backbone for
  the demo's claim.**
- **Embrechts–McNeil–Straumann (2002)** / **McNeil–Frey–Embrechts (QRM, 2015)** — "correlation
  is not dependence"; same correlation, wildly different joint tails; Gaussian has zero tail
  dependence. *(Book/chapter, ISBN not DOI.)*
- **Constructive alternatives**: t-copula (Demarta–McNeil 2005), Clayton/Archimedean
  (lower-tail), nested/hierarchical Archimedean, **vine copulas** (Aas et al. 2009) — the
  conditioning *tree* supplies the higher-order glue that unconditional pairwise lacks. Meucci
  panic copula (scenario reweighting).
- **Multivariate extremal dependence**: Hüsler–Reiss (1989); **Schlather–Tawn (2003)** — pairwise
  extremal coefficients, even when self-consistent, do **not** determine the higher-order ones;
  **Engelke–Hitz (2020)** — graphical models for extremes (sparse higher-order CI structure).
- **Max-entropy / k-marginal reconstruction**: **[[schneidman2006]]** — weak *pairwise*
  correlations imply strongly correlated network states (pairwise *sufficient* at small N);
  counterpoints Roudi et al. (2009), Ganmor et al. (2011) — pairwise *fails* at scale, a sparse
  set of third-order terms restores it. Finance analogue: Bury (2013, *Physica A*). The crash
  regime sits right at that crossover, where the marginal-problem under-determination
  (Jaynes 1957; Vorob'ev) bites.

---

## 6. Where the bowling / tetrachoric work sits — white space & honest novelty

**Established (do NOT claim as novel):** the Gaussian copula's zero tail dependence
(Sibuya; Embrechts–McNeil–Straumann); that matching marginals + correlation does not
determine the loss tail ([[freymcneil2002]] via the Bernoulli-mixture mapping); that pairwise
binary moments leave all order-≥3 interactions free (multivariate-Bernoulli literature); that
static Gaussian-copula CDO calibration empirically fails on senior tranches (the
correlation-skew / base-correlation literature); that **tetrachoric = the pairwise-only latent
Gaussian threshold model**, with documented joint-tail limits.

**The genuine white space:** there is **no single published theorem** of the form
> *for a Gaussian copula and a tail-dependent copula matched on ALL pairwise default
> correlations and marginals, P(≥k of d default) is strictly smaller for the Gaussian — hence
> the d-wise joint (senior-tranche) tail is under-priced.*

[[donnelly2010]] says it qualitatively; [[sibuya1960]]/[[ledfordtawn1996]]/[[huajoe2011]]
prove the decay-rate half with no default framing; the CDO comparisons show it numerically as
"correlation skew." **The rigorous bridge — a pairwise-matched, d-wise joint-tail pricing-gap
inequality — appears genuinely unwritten**, and it dovetails with the redundancy=singularity
framing (Gaussian's λ=0 is exactly the "blind to co-extremity" failure).

**The demo's contribution is synthesis + exhibit, not a new theorem:**
1. a clean, mechanical, *exactly-pairwise-calibrated* reproduction — calibrate the full
   tetrachoric matrix and every marginal, then watch the k-name joint blow up — rather than the
   usual asymptotic/limit statement;
2. positioning a rigid-body **cascade generator** (Davis–Lo / Watts / sandpile lineage) as the
   explicit *filler* of the order-≥3 free parameters that the Gaussian copula discards — a
   controllable known-truth **test fixture** bridging the *mechanism* literature (§3–4) and the
   *measurement* literature (CoVaR/SRISK, §4).

**If citing exactly two for the claim:** [[donnelly2010]] (on-point narrative) +
[[huajoe2011]] (d-wise rigor). Cite [[freymcneil2002]] head-on as the reviewer's "this gap is
known" reference, and frame the contribution as the exhibit + the contagion-as-higher-order-
filler synthesis. The crisis framing should follow §1 (implied correlation / rating-agency
non-models / [[mackenzie2014]]), **not** the Salmon myth.

---
*Verification: bib keys above are Crossref-verified (2026-06). Flagged uncertain — cite by
venue, not DOI: Hull–Predescu–White (J. Credit Risk 1(2)); Vasicek (Risk, no DOI);
Embrechts–McNeil–Straumann and QRM (books, ISBN); McGinty et al., O'Kane–Livesey, agency
technical docs (dealer/agency notes, no DOI). Raw agent reports retain the full detail.*
