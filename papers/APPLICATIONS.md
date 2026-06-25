# Applications of the dependence-aware share-tilt operator

The core object is a map on the simplex: take a share vector `w ∈ Δ` (shares summing to 1),
calibrate latent abilities so an *independence* reference race reproduces it, then re-run the
race under a *correlation- or tail-dependent* simulation and read off the tilted shares. It
makes co-moving / redundant alternatives **share** weight (clone consistency / red-bus–blue-bus
/ IIA fix), is **tail-aware** (responds to extreme co-movement a covariance can't encode), and is
**smooth** (small input change → small share change → low churn under repeated/online updates).

**Meta-finding (consistent across five field searches).** Each field has the *ingredients* in
separate silos — a rich correlation/IIA-fix literature and a separate smoothness/stability
literature — but **nobody unifies dependence-aware clone-consistency with smoothness in one
share→share operator.** Where clone-consistency is already solved (Shapley in attribution, DPP in
ranking, similarity-diversity indices in ecology), the wedge is the **smoothness + tail-awareness**
combination, not clone-consistency alone. The share→share tilt operator itself appears novel
(closest prior art, GEV / Hotz–Miller inversion, stops at estimation).

## What makes a *direct* hit (and why MoE is not one)

The operator is a smooth, dependence-aware, entropy-regularized tilt of a soft distribution:
`w = argmax_{p∈Δ} ⟨θ,p⟩ − Ω_S(p)`. It is a **direct hit** iff:
1. the deliverable is a **soft distribution on the full simplex**, consumed as a distribution;
2. the semantics are **choice / selection probability** (or entropy-diversification is genuinely the
   objective);
3. there is redundancy/dependence among the alternatives to tilt for; and
4. smoothness / low churn is valued (repeated updates).

It is **not** a fit when there is a hard **cardinality** constraint (top-k) or the value is purely
**coalitional-combination** (a subset's cooperative output). Those are constrained-portfolio /
submodular / ℓ0 problems with their own machinery; the race (single-winner *selection*, no
coalitions — see `experiments/coalition_credit.py`) can at best supply a redundancy-aware *prior*
to such a solver, not the solution.

**MoE top-k routing fails on both counts** — it is cardinality-constrained (exactly k experts) and
its value is the *combined* output of the chosen experts (a constrained portfolio). The router demo
(`experiments/moe_router.py`) shows the race correctly de-duplicates redundant experts — a useful
sub-problem — but it does not solve the constrained combination, so MoE is at most a *partial* fit.
**Recommender top-k slates** and **sparse feature selection** sit on the same boundary (cardinality
+ slate-combination value). LARS/LASSO live on the *sparse* side of this line; the race is the
smooth, full-simplex analogue of their correlated-predictor sharing.

## Opportunities by field (fit, the gap, best-in-field)

| Field | Best application | Fit | The gap the operator fills |
|---|---|---|---|
| **ML systems** | **MoE router / gating weights** | **STRONG** | softmax routing is correlation-blind (IIA ≡ zero covariance); redundant/collapsed experts + router *churn* are named costly pains; load-balancing ≠ redundancy — nobody tilts for expert correlation. DeepSeek-V3's aux-loss-free out-of-gradient bias is a ready scaffold. |
| ML systems | Ensemble / stacking weights | STRONG | clone-consistency is *proven* here (elastic-net grouping bound); the "forecast-combination puzzle" motivates smoothness; existing fixes are static L2 — none online + correlation-aware. **Matches our M4 result.** |
| **Marketing** | **MMM budget reallocation (Meridian/Robyn)** | **STRONG** | redundancy handled only at *estimation* (Ridge); reallocation churn handled by a crude ±30 % box constraint + manual refresh — no entropy/KL smoothness term. Clean drop-in for the box. |
| Marketing | RTB pacing / cross-campaign budget | STRONG | smoothness is the field's *native* obsession (pacing control theory); audience-overlap double-counting is a named pain; entropy-mirror allocation already lives here in spirit. |
| **Epidemiology** | **Variant-frequency nowcasting** | **STRONG** | the deployed model (multinomial-logistic / GARW) *is* a Luce race with an unaddressed IIA/clone flaw; low weekly churn is an explicit CDC design goal; antigenic-similarity kernels exist next door. |
| **Politics** | **Electoral-college correlated state errors** | **STRONG** | tail co-movement *is* the EC win-probability; 538's tail correlations are a documented bug (242 negatively-correlated state pairs); Taleb gives a published smoothness anchor. |
| Politics | Multi-candidate primary consolidation | STRONG | spoiler / clone problem in pure form; shares update over months (smoothness genuinely needed); transfer matrices are descriptive, not a fitted share-redistribution model. |
| **OR / IR / sports** | **Ranking diversification + stability** | **STRONG** | DPP/MMR own diversity, CASPER/RLS own stability — *no one unifies them*; high-frequency, churn-sensitive, falsifiable metric (Rank List Sensitivity). |
| OR / IR / sports | Sports outright (multi-way win) markets | STRONG (best **demo**) | BT/PL *literally is* the noisy race; copulas are used only for same-game parlays, never to de-duplicate co-moving contenders in a multi-way field. |
| Politics | Approximate-clone-independence (social choice) | MEDIUM (high novelty) | Delemazure 2026 proves *no ranked rule* is independent of approximate clones (≥4 cands); a continuous share method could satisfy a *smooth* clone-independence ranked rules provably can't. |
| Ecology | Tail-conditioned effective-N of populations | MEDIUM (most novel) | no named index down-weights species *because they crash together*; portfolio-effect φ is covariance-only, tail-dependence work never becomes an effective-N. |
| Marketing | Multi-touch attribution | MEDIUM | Shapley already owns clone-fairness; but Shapley estimates are *volatile* and smoothness is unsolved — the wedge is smoothness only. |

Honest weak fits: neural attention (no persistent online weight vector → smoothness lever absent);
replicator dynamics / similarity-diversity indices (already smooth / already clone-consistent from an
*exogenous* similarity matrix — the race's novelty is deriving dependence from *co-movement*);
ad budget allocation as single-winner (continuous-split mechanism mismatch; portfolio theory already
owns correlation+turnover).

## Top cross-field picks (re-ranked by the direct-hit criterion)
Most direct = soft simplex, native choice/win-probability semantics, no cardinality constraint:
1. **Sports / prediction-market outright win-probability fields** — the *most* direct hit: BT/PL
   literally *is* the race, the object is `P(win)`, no cardinality, tilt for correlated contenders.
2. **Discrete-choice shares** — market share, vote/poll share, pathogen-variant frequency — genuine
   choice-probability distributions you tilt for substitution; each has a documented incumbent
   failure (cannibalization; correlated polling/tail errors; ad-hoc clade collapsing).
3. **Continuous allocation via the implied objective** — portfolio weights, MMM / RTB budget split
   (soft, churn-sensitive); direct because the entropy-regularized-diversification objective is what
   is wanted, even though the value is combinational.
4. **Ensemble / stacking weights** — soft combination weights; clone-consistency provable, validated
   by our M4 result. (Mild caveat: the *value* is combinational, so it leans on the implied objective
   rather than selection semantics.)

Boundary / partial fits (cardinality or coalition — the race is a prior, not the solution):
**MoE top-k routing**, **recommender top-k slates**, **sparse feature selection**.

## Strategic implication
The breadth (ML, marketing, epidemiology, politics, IR/sports — not just finance) supports the
conceptual reviewer's two calls: (i) reposition around the unifying object — a *dependence-aware
share-tilt operator* / "choice-geometric allocation and attribution" — and (ii) elevate the general
(credit/attribution) paper, since portfolios is one projection of a much broader object. The
differentiated pitch everywhere is the same triple: **clone-consistent + tail-aware + smooth, in one
operator**, with smoothness the discriminator that the share→share form uniquely provides.
