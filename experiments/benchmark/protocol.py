"""One evaluation protocol, so results across methods are comparable.

Rules, each of which exists because breaking it produced a wrong answer at
some point:

  Paired. Every method sees the identical sample on every draw, so comparisons
  are paired and the market-to-market variation cancels. Win rates are computed
  per draw, never from marginal medians, which can disagree with the pairing.

  Known truth. Risk is w'Sigma w against the population covariance, and
  shortfall is measured on one large out-of-sample panel per draw. Nothing is
  estimated in the scoring.

  Uncertainty reported. Every win rate carries a standard error. A 64% rate on
  25 draws is 1.5 standard errors from a coin and means nothing on its own.

  Failures visible. A method that cannot fit is recorded as missing, not
  replaced by a fallback, and its coverage is reported beside its score.
"""
import numpy as np


class _Failed(Exception):
    """A method that legitimately could not produce a portfolio."""


def evaluate(methods, market_factory, ratios, draws, n=40, oos=80_000,
             seed=0, progress=None):
    """Returns per-(ratio, method) arrays of variance, shortfall and coverage.

    Distinguishes three outcomes rather than two: a portfolio, a legitimate
    no-fit, and a silent fallback. The third is the one this harness exists to
    catch, and detecting it by exception alone does not.
    """
    from collections import Counter, defaultdict
    rng = np.random.default_rng(seed)
    keys = [m.name for m in methods]
    fallbacks = {r: Counter() for r in ratios}
    errors = {r: defaultdict(Counter) for r in ratios}
    var = {r: {k: [] for k in keys} for r in ratios}
    es = {r: {k: [] for k in keys} for r in ratios}
    labels = []
    for i in range(draws):
        market = market_factory(rng, n)
        labels.append(market.label)
        panel = market.sample(np.random.default_rng(rng.integers(1 << 62)), oos)
        Sig = market.sigma
        for r in ratios:
            T = max(int(round(r * n)), 4)
            X = market.sample(np.random.default_rng(rng.integers(1 << 62)), T)
            for m in methods:
                try:
                    w = np.asarray(m.fn(X), dtype=float)
                    if not np.isfinite(w).all():
                        raise _Failed("non-finite weights")
                    # A method that cannot fit and returns a plausible fallback
                    # rather than raising is the case this harness exists to
                    # catch. SchurBridge returns exactly 1/n on a singular
                    # covariance, which reads as a result. Equal weight is the
                    # one method allowed to be equal weight.
                    if (m.name != "equal weight"
                            and np.abs(w - 1.0 / len(w)).max() < 1e-12):
                        fallbacks[r][m.name] += 1
                        raise _Failed("returned exactly equal weight")
                    p = panel @ w
                    q = np.quantile(p, 0.05)
                    var[r][m.name].append(float(w @ Sig @ w))
                    es[r][m.name].append(float(-p[p <= q].mean()))
                except _Failed:
                    var[r][m.name].append(np.nan)
                    es[r][m.name].append(np.nan)
                except Exception as exc:               # a real bug, not a no-fit
                    errors[r][m.name][type(exc).__name__] += 1
                    var[r][m.name].append(np.nan)
                    es[r][m.name].append(np.nan)
        if progress and (i + 1) % progress == 0:
            print(f"  ...{i + 1}/{draws}", flush=True)
    for r in ratios:
        for name, cnt in sorted(fallbacks[r].items()):
            print(f"  NOTE  {name} returned exactly equal weight in {cnt} of "
                  f"{draws} draws at T/n={r}; scored as no-fit, not as a result")
        for name, kinds in sorted(errors[r].items()):
            for kind, cnt in kinds.items():
                print(f"  NOTE  {name} raised {kind} in {cnt} of {draws} draws "
                      f"at T/n={r}; that is a bug or a missing import, not a no-fit")
    to_arr = lambda d: {r: {k: np.asarray(v) for k, v in dd.items()} for r, dd in d.items()}
    return to_arr(var), to_arr(es), labels


def _wilson(p, n, z=1.96):
    """A 95% interval that stays informative at 0 and 1.

    The Wald form ``sqrt(p(1-p)/n)`` collapses to exactly zero when a method
    wins every draw or none, so the least informative cells printed perfect
    certainty, which is the opposite of this module's stated rule.
    """
    if n == 0:
        return 0.0, 1.0
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(centre - half, 0.0), min(centre + half, 1.0)


def summarise(scores, methods, ratios, reference, metric_name):
    """Median ratio to the reference, paired win rate with its standard error,
    and coverage. Printed rather than returned, since this is a report."""
    ref = reference
    print(f"\n{metric_name}: median ratio to '{ref}', paired win rate "
          f"[95% Wilson], coverage\n")
    head = f"{'method':24s}{'long-only':>11s}" + "".join(f"{('T/n=' + str(r)):>28s}"
                                                        for r in ratios)
    print(head)
    for m in methods:
        if m.name == ref:
            continue
        cells = []
        for r in ratios:
            x = scores[r][m.name]; y = scores[r][ref]
            ok = np.isfinite(x) & np.isfinite(y)
            cov = ok.mean()
            if ok.sum() < 2:
                cells.append("        no fit            ")
                continue
            med = np.median(x[ok] / y[ok])
            # ties count as half a win; scoring them as losses makes two
            # identical methods read as total defeat
            win = float(np.mean(x[ok] < y[ok]) + 0.5 * np.mean(x[ok] == y[ok]))
            lo, hi = _wilson(win, int(ok.sum()))
            cells.append(f"{med:7.3f} {win:4.0%}[{lo:.0%},{hi:.0%}] {cov:4.0%}")
        print(f"{m.name:24s}{('yes' if m.long_only else 'no'):>11s}"
              + "".join(f"{c:>28s}" for c in cells))
    print(f"\n  reference '{ref}' median level: "
          + "  ".join(f"T/n={r}: {np.nanmedian(scores[r][ref]):.4f}" for r in ratios))
