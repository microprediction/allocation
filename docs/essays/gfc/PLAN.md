# Reshape plan — "We Asked the Chinese Quant from The Big Short What Caused the Credit Crisis"

## Conceit
A mock interview. **Jiang** — "the Chinese quant from *The Big Short*" — answers; the
**narrator/interviewer** plays *translator*, building the demonstrations so the reader can
follow Jiang's sophisticated, faintly irritated replies. His contempt for the popular
explanations is barely concealed. (Source clip: *That's My Quant*, youtu.be/FoYC_8cutb0.)

### Running gags / motifs (use lightly, don't overdo)
- Everyone calls him **Yang**; his name is **Jiang**. He corrects it. ("Use correct spelling.")
- "I got **second** in that competition, not first." / "And I *do* speak English."
- **Ryan Gosling knows more about CDOs than Nassim Taleb by virtue of reading the script.**
- "That's my quant." (Vennett's line — the narrator can borrow it.)
- The narrator keeps having to *build a demo* to render Jiang's curt answers legible.

### Voice
- **Jiang**: terse, exact, mildly contemptuous, precise; corrects the record; hates wasting
  breath on nonsense; warms only when the actual mechanics come up.
- **Narrator**: earnest translator; admires Jiang; supplies the demos and the context.
- Folds (`details.think`) become **Jiang off-the-record** asides / continuations that would
  otherwise interrupt — relabel summaries ("Off the record, Jiang…", "Jiang, unprompted…").

## Structure (fluid path through the *believed* reasons, then the truth)

0. **Header / setup.** New title + subtitle. Keep the three-measures box (P/Q/R). Short framing:
   we tracked Jiang down; the Yang/Jiang/second-place gag; what he agreed to.
1. **Exhibit A — lead with the bowling-pin demo (it's cool).** Narrator brings the *sophisticated*
   theory first: hidden tail dependence / contagion. Bowling iframe. Jiang: elegant — I even built
   the demo and a whole portfolio method off it — *and the wrong culprit.* (Part 2 material, led with.)
2. **The famous one — Taleb / bell curve / "the formula that killed Wall Street."** Jiang dismisses:
   Gaussian *copula*, not Normal *marginals*; marginals pinned by CDS; the one free knob was
   correlation. (Part 1 + Taleb fold.) Gosling-script joke lands here.
3. **The behaviourist angle — Thaler / the movie.** Jiang dismisses: hindsight bias dressed up;
   Howie Hubler was *short* and still lost $9bn. (Thaler fold material.)
4. **"So, Jiang — what actually happened?"** PIVOT. Finally the mechanics:
   - **A company that does nothing** (ASCII). Bodega / car-wash comparison — the ONLY thing a
     structured-finance vehicle does is raise debt and invest excess cash. Contempt at calling it "complex."
   - **The arbitrage** (rating-arb widget): market-implied vs rating-agency default prob; the free lunch.
   - **The three measures** P/Q/R; R = the rating measure, the one you were never taught.
   - **The fast model** (quadrature widget): quants were *not* blinded; conditioning/convolution/quadrature.
   - **Ugly, not beautiful** (hazard-jump widget): Sklar on time, jumps, same name/two baskets, one hedge/two risks.
   - **Base correlation** (bccv widget): the kludge — the model screaming it's wrong, written down as a quote.
   - **The quants knew** (Julius Finance). **The truth / incentives.** **Diversity score** (dscv).
     **Implied correlation** (iccv) + **Indispensable Markets Hypothesis** + M6.
5. **The investigation that wasn't — the new payoff section.** Jiang's contempt at full volume:
   Bernanke ("complex, opaque, unwieldy"), Kevin Rudd ("house on fire"), the dignitaries, the FCIC's
   "The CDO Machine" chapter. **Contrast with real accident investigations:** NTSB *bans lawyers by
   law* and convenes the engineers; the Rogers Commission put Feynman in front of a glass of ice water.
   **The FCIC's ten:** list them; not one structured-finance practitioner; the closest was a retired
   *healthcare* equity analyst. "You see where this is going."
6. **Twenty years later** (Part 7): symmetry, de Finetti, the conformal-prediction second life,
   COVID/epidemiology — keep as Jiang's "indulge me" tangent (fold or aside).
7. **Back to the story** (Part 8) + close.

## Assets to preserve verbatim (do NOT touch the JS)
- `<head>` + `<style>` unchanged. All `<script>` blocks after `</main>` unchanged.
- Reuse every widget HTML block with its existing IDs (rating-arb `ra*`, fast model `chart/rho/p/nq`,
  hazard `hz`, base corr `bc*`, diversity `ds*`/`drho`, implied corr `ic*`, symmetrization `sym*`/`s*`),
  the company ASCII, the measures box, the orbit SVG, the `eqn`, the bowling iframe. Only the prose moves/changes.

## Research facts (cite/lean on)
- The Big Short: Jared Vennett (Gosling) → "that's my quant… his name is Yang… won a national math
  competition in China, doesn't even speak English." Jiang (Stanley Wong) corrects: name's Jiang,
  speaks English, got second.
- NTSB "go team" + party system; **attorneys prohibited by law**. Rogers Commission: Feynman + Armstrong
  + Ride + Yeager; Feynman's O-ring/ice-water demo; excluded current NASA staff for independence.
- FCIC ten commissioners (above); chapter 8 = "The CDO Machine"; none built/modelled CDOs.

## Mechanics of writing
Replace the `<main>…</main>` block only. Keep widgets+IDs; scripts bind by ID afterward, so widget
order is free. Preserve everything outside `<main>`.
