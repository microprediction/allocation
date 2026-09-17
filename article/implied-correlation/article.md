# The number that priced the World Cup — and mispriced the mortgage market

*A short tour of **implied correlation**, with two interactive toys you can spin yourself.*

![Implied correlation is the tilt of the cloud that makes the corner match the market price](images/hero-tetrachoric.png)

---

There is a number in finance you can never look up. You can't measure it, you can't observe
it — you can only **back it out of prices**. It is called *implied correlation*, and once you
see it once, you see it everywhere: in an equity index option, in a mortgage CDO, and — this
week — in the odds that two teams both reach the World Cup final.

Here is the whole idea in one picture.

## Two events, one corner

Take any two yes/no events. "France reaches the final." "Argentina reaches the final." Each one,
on its own, has a probability — a *marginal*. Draw them as two lines on a plane, and they carve it
into four rectangles. The chance that **both** happen is the mass of a two-dimensional bell curve
sitting in the corner where both are true.

Now here is the move. Correlation **tilts the bell**. Spin it one way and the corner fills up; spin
it the other and it empties — *while the two marginals never move*. So if the market quotes you a
price for "both reach the final," there is exactly one tilt that reproduces it. That tilt is the
implied correlation. You didn't measure it. You solved for it.

![Spin the cloud until the "both reach the final" corner matches the market — that tilt is the implied correlation](images/spinner-worldcup.png)

👉 **Spin it yourself:** [Tetrachoric correlation, World Cup edition](https://allocation.microprediction.org/demos/world-cup/)
— drag the cloud until the corner matches Kalshi's live "meet in the final" price.

## What it says about the tournament

Do it for every pair of teams and rank them, and a real structure falls out. Some teams'
runs to the final **co-move** — when one is having a deep-run kind of tournament, so is the other.
Others are **substitutes** — one team's charge to the final tends to come at the other's expense.
The board as a whole sits close to independence, which is itself the honest headline; the *spread*
around it is the signal.

![Every cross-half pair, ranked by the implied correlation of reaching the final](images/world-cup-ranking.png)

None of this needs a model of the teams. Two market prices — each side's chance of reaching the
final, and the chance the two meet there — pin the correlation directly.

## The same number, a darker room

Swap the teams for two mortgage bonds and "reaches the final" for "defaults." Now the corner where
both fall is the **joint default** — and the value of a senior CDO tranche is very nearly *only* a
bet on that corner. A desk never measured the correlation that governs it. It read it off tranche
prices, exactly the way you just spun it off a football match. That backed-out number is the
implied correlation, and it is the one input that was wrong in 2008.

![The same spinner, recast for two defaulting names — with default barriers at five horizons, 2 to 10 years, so you can watch the joint-default corner deepen with maturity](images/spinner-cdo.png)

![*The Big Short* explaining a synthetic CDO, in full](images/big-short-cdo.png)

Not the "complexity" the post-mortems blamed — a CDO is the simplest company on earth, an empty box
that buys bonds and pays them out in order of seniority. Not the normal distribution, and not one
famous formula. The hard, decisive thing was always putting a number on *joint* defaults, and that
number came in stale, from the wrong measure of probability.

That story is told properly — by someone who led the CDO modelling desk — in the companion essay,
which now carries the same spin-the-cloud toy recast for two defaulting names, with 5-year and
10-year barriers:

👉 **Read it:** [Opiate for the Mathless: the R-measure and the Global Financial Crisis](https://allocation.microprediction.org/essays/gfc/)

---

*Both toys are live and interactive on [allocation.microprediction.org](https://allocation.microprediction.org).
World Cup prices are a live snapshot from Kalshi. This is a teaching tour of implied correlation, not
investment advice.*

---

### Posting notes (not part of the article)

- **Images, in order:** `hero-tetrachoric.png` (opener) · `spinner-worldcup.png` (live World Cup widget) ·
  `world-cup-ranking.png` (the ranking) · `spinner-cdo.png` (live CDO widget) · `big-short-cdo.png` (movie still).
  Also available: `big-short-quant.png`, `big-short-manage.png`, `margin-call.png` (movie stills, use as you like).
- **Note on movie stills:** *The Big Short* / *Margin Call* frames are for personal drafting; check usage rights before publishing on Medium/LinkedIn.
- **Links to keep live (the interactive versions are the payoff):**
  - Demo: https://allocation.microprediction.org/demos/world-cup/
  - Essay: https://allocation.microprediction.org/essays/gfc/
- **Suggested title A (LinkedIn):** "The number that priced the World Cup — and mispriced the mortgage market"
- **Suggested title B (Medium):** "Implied correlation, explained by spinning a cloud"
- **Suggested tags:** quantitative finance, risk, prediction markets, financial crisis, statistics
