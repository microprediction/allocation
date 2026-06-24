# Thurstone Portfolios — revision checklist (referee pass)

External mathematical read. Verdict: strong spine, sound core idea (weights are gradients of
an expected-max potential; changing the perturbation law changes the regularizer; dependence-aware
perturbations de-duplicate without covariance inversion). Fixes below are about **exactness, scope,
and assumptions** — narrowing/formalizing claims so they can't be attacked. Not that the idea is wrong.

Two biggest vulnerabilities flagged:
- **conceptual:** the calibration ↔ redundancy interaction (item 3).
- **technical:** overclaiming from λ_L to general tail-monotonicity (item 5).

Severity: **[B]** blocker before wide circulation · **[S]** should-fix · **[N]** nice-to-have.

---

## Tier 1 — do before circulating

- [ ] **[B] 1. State the population map up front.** Before the algorithm, give the clean object:
  `G_S(θ)=E_{η∼S}[max_i(θ_i+η_i)]`, `W_S(θ)=∇G_S(θ)∈Δ`; calibration `θ_0=W_{S_0}^{-1}(w_0)` (mod
  constant); tilt `T_{S_0→S_1}(w_0)=W_{S_1}(θ_0)`. Then index reproduction is `T_{S_0→S_0}=id`,
  the implied objective is Fenchel duality, Markowitz is a change of regularizer, smoothness is
  smoothness of `W_S`, redundancy is a property of `W_S(θ)` at fixed θ. *Agree — one display
  organizes the whole paper. Pure exposition, high payoff.*

- [ ] **[B] 2. Add an existence/uniqueness (calibration) theorem.** For Gaussian `S=N(0,C)`, `C≻0`:
  `W_C : R^n/span{1} → int Δ` is a smooth bijection with smooth inverse; equivalently every strictly
  positive benchmark has a unique ability vector (mod constants). Separate this existence result from
  the *numerical* inverse (the Fast Ability Transform handles the diagonal case; one-factor/general
  laws also need the existence claim). **Assume `w_0 ∈ int Δ`**; for `w_i^0=0` the inverse sends
  `θ_i→−∞`, so state that zero-weight assets are dropped or floored. *Agree — currently load-bearing
  but silent.*

- [x] **[B] 3a. Hessian/Jacobian slip in §6.3 — FIXED.** §6.3 read "its Hessian the Jacobian of the
  winning probabilities"; the Markowitz section correctly has the **inverse** Jacobian. Corrected to
  `∇²Ω_S = J_S^{-1}` on the tangent space (consistent with `H_0=[∇²G]^{-1}=[∇_θ w]^{-1}`). The
  gradient identity `∇G=W` (verified numerically to 7e-5 in `experiments/shapley_bridge.py`)
  confirms `∇²G=J`, `∇²Ω=J^{-1}`. The correct local Bregman penalty is
  `D_{Ω_S}(w_0+δ,w_0)=½ δ^T J_S(θ_0)^{-1} δ + O(‖δ‖³)`.

- [x] **[B] 3b. Narrow redundancy consistency to fixed abilities — DONE.** Added `rem:fixedability`
  after Prop 4 with the verified counterexample (`(.8,.2)→(4/9,4/9,1/9)`, `|θ_A−θ_B|≈1.01`, comonotone
  cluster `≈0.76≠0.8`). Prop 4 is correct *for the race
  at fixed abilities* (equal abilities + comonotone ⇒ tie-split halves the single-horse prob). But the
  full cap-anchored algorithm recalibrates `θ_new=W_{S_0,new}^{-1}(w_new^0)` to the renormalized
  benchmark when a duplicate enters, and **calibration need not commute with duplication**. Referee's
  Gaussian counterexample: benchmark `(A,B)=(.8,.2)`; duplicate A at equal cap → `(4/9,4/9,1/9)`;
  independent-reference calibration gives `θ_A−θ_B≈1.014`; making the two A's comonotone yields
  combined weight `Φ(1.014/√2)≈0.763 ≠ 0.8`. So the race de-duplicates, but exact pre/post-universe
  invariance fails under recalibration through a Lucian benchmark. **State:** redundancy consistency is
  a property of the race at fixed abilities; the calibrated tilt inherits it exactly only if the
  calibration is itself duplication-consistent (or duplicated assets inherit the same ability rather
  than being recalibrated). *This is the single most important clarification — verify our Prop 4
  wording and surrounding rhetoric; add the caveat and likely the counterexample.*

- [ ] **[B] 5. Reformulate tail consistency via the group minimum, not λ_L.** Prop 6 proves only the
  comonotone limit, not monotonicity in λ_L. The object controlling combined weight is
  `M_G=min_{i∈G}X_i` vs `Y=min_{j∉G}X_j`, `w_G=P(M_G<Y)` — the race responds to the *whole
  distribution of M_G*, not the pairwise asymptotic λ_L (which is not a total order on copulas; equal
  λ_L can have different mid-tail behavior that matters at finite n). **Stronger theorem:** if as r↑,
  `M_G^{(r)}` increases in first-order stochastic order with Y fixed and independent of G, then
  `w_G^{(r)}=P(M_G^{(r)}<Y)` is non-increasing; the comonotone case is the endpoint. Keep "reads the
  tail copula" as intuition but state mathematically it reads the distribution of the group minimum.
  *Agree — biggest technical vulnerability. Our Clayton sweep is a family-specific instance; frame it
  that way.*

## Tier 2 — should fix

- [ ] **[S] 6. Tangent-space care throughout.** Since `G_C(θ+c1)=G_C(θ)+c`, all gradients/Hessians/
  inverses live on `TΔ={v:1^T v=0}`; every `[∇_θ w]^{-1}` should read "inverse on the simplex tangent
  space / Moore–Penrose modulo the constant direction" (in full R^n the Jacobian is singular). *The
  §6.3 fix already added this qualifier; sweep the rest (Markowitz §, Tweedie §).*

- [ ] **[S] 7. Separate the three regularity regimes for "any simulation."** Convexity of `G_S` holds
  for any centered S, but `w=∇G_S` and uniqueness need regularity. State three regimes:
  (i) full-rank continuous law → `G_S` smooth, `w=∇G_S` unique; (ii) singular/tie-prone law → `G_S`
  convex but possibly nonsmooth, tie-splitting selects a subgradient `w∈∂G_S(θ)`, redundancy holds but
  uniqueness/smoothness may fail; (iii) finite-MC → empirical `G_{S,M}` piecewise-linear, weights are
  step functions of parameters. *Agree — reconciles the smooth-Gaussian and singular-duplicate claims.*

- [ ] **[S] 8. Soften "minimum variance is undefined" at rank deficiency.** The unconstrained inverse
  formula is undefined when Σ singular, but the long-only QP `min_{p∈Δ} p^T Σ p` is well-defined for
  PSD Σ (possibly non-unique/unstable, not undefined). Table 5 / §scale should say "dense **inverse
  formula** undefined" or "unregularized inverse implementation undefined," not "minimum variance
  undefined." *Agree — easy target for critics; fix wording in 30-empirics Table `tab:scale` + §robust.*

- [ ] **[S] 9. Sharpen the CAPM restriction proposition with the block-matrix wedge.** Replace the
  proof sketch with: from `μ−r1=γΣ_U m`, restricted tangency `q_S ∝ Σ_SS^{-1}(μ_S−r1) =
  γ(m_S + Σ_SS^{-1}Σ_SX m_X)`; cap weighting uses `m_S`; they coincide iff `Σ_SS^{-1}Σ_SX m_X ∝ m_S`.
  That is the precise "excluded-covariance" wedge. *Agree — makes Prop 1 hard to object to.*

- [ ] **[S] 12. Define the performance/loss variable in the forecast-combination section.** Running the
  race on signed errors makes the "winner" the most negatively-biased forecaster. Competitors'
  performances should be **losses** (`|e_{k,t}|`, `e²_{k,t}`, sMAPE contribution), centered/scaled,
  so lower performance = better forecast. State this explicitly in §combination. *Agree — and check
  `experiments/m4_combination.py`/`combine.py` actually feed errors consistently with the chosen
  convention (we feed `truth−forecast` as "returns"; document the min-vs-max convention).*

## Tier 3 — strengthen / elevate

- [ ] **[S] 4. Advertise Prop 5 (diversification monotonicity) narrowly.** Correct under its stated
  structure (equal-ability Gaussian subgroup, independent of complement, equicorrelation ρ inside;
  Slepian). Do **not** imply general monotonicity for arbitrary clusters, correlation changes,
  non-Gaussian laws, or changes touching the group–complement dependence. State exactly: `ρ↑ ⇒
  min_{i∈T}X_i` stochastically larger ⇒ `P(T wins)↓`.

- [ ] **[N] 7-bis. Cleaner smoothness proof (Theorem 2).** Optionally replace Price's-theorem/boundary
  integrals with total variation: `‖w(C_1)−w(C_0)‖_1 ≤ 2 TV(N(0,C_1),N(0,C_0))`, then KL/Pinsker on
  `λ_min(C)≥λ` gives `‖Δw‖_1 ≤ K(n,λ)‖C_1−C_0‖_F`, `K∼dim·(1/λ)`. **Define the matrix norm in eq. (9)**
  (the table uses an ℓ1-style norm; the theorem just writes `‖C_1−C_0‖`). *Cleaner and avoids
  distributional derivatives of cone indicators.*

- [ ] **[B-ish] 7-ter. Finite-MC turnover is probabilistic, not Lipschitz.** Don't say the finite-M
  map "inherits" smoothness. It's piecewise-constant and jumps at boundary crossings. Correct claim
  (fixed seeds, boundary-density condition): `E‖ŵ(C+ΔC)−ŵ(C)‖_1 = O(‖ΔC‖)+O(‖ΔC‖/√M)`. It inherits
  **low turnover in expectation/high probability**, not literal smoothness. *We already corrected Thm 2
  toward the O(1/√M) term in the empirics; make sure the Theorem 2 statement itself carries this.*

- [ ] **[S] 10. Markowitz analogy — one warning sentence.** Make explicit: the Thurstone geometry is a
  **diversification** geometry induced by the perturbation law, not automatically a **risk** geometry
  in return space. Unless S is tied to an investor loss functional, the objective rationalizes the
  choice rule; it is not a welfare theorem.

- [ ] **[S] 11. Elevate the selection/Tweedie identity to a lemma.** `E[X|I=i]=θ+C∇_θ log w_i(θ)`, hence
  `∇_θ w_i = w_i C^{-1}(E[X|I=i]−θ)`. Gives a concrete object behind `[∇_θ w]^{-1}` (selection
  sensitivities). Label as Lemma/Proposition; caveat `C≻0` (limits/pseudoinverse in singular duplicate
  cases). *Agree — supports the choice-space-inverse story; also corroborated by the gradient identity.*

---

## Suggested order of attack
1. Items **1, 3a✓, 6** — exposition + the confirmed Hessian/tangent-space fixes (cheap, internal).
2. Item **3b** — the redundancy/calibration caveat + counterexample (the key conceptual fix).
3. Item **5** — restate tail consistency via the group-minimum stochastic-order theorem.
4. Items **2, 9, 11** — calibration existence theorem, CAPM block-matrix wedge, Tweedie lemma.
5. Items **7, 7-bis, 7-ter, 8, 12, 4, 10** — regularity regimes, smoothness cleanup, wording softenings.
