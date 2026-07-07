"""The implied tetrachoric correlation of the World Cup, read off Kalshi's joint markets.

A single team's markets (its game, its make-final, its winner price) never pin a correlation --
they are nested conditionals, self-consistent under independence. And in a Thurstone / Bradley-Terry
model the interesting quantity, the correlation between teams' latent *abilities*, is unidentifiable:
a common ability factor cancels out of every pairwise match, so it drops out of the game odds AND the
bracket odds alike (calibrate the bracket at any equicorrelation and the win probabilities are
identical -- only the ratings rescale by sqrt(1-rho)). The correlation between latent abilities is,
in that sense, unknowable from the marginals.

What IS identifiable is the correlation between OUTCOMES, and for that you need a genuine *joint*
price. Kalshi lists one: KXWCMATCHUP, "will A play B in the final?". Because the final is exactly one
team from each half of the draw, for two opposite-half teams

    P(A and B both reach the final)  ==  P(A plays B in the final),

so that market price IS the joint. Put it beside the two make-final marginals and invert the bivariate
normal and you get the implied tetrachoric correlation between "A reaches the final" and "B reaches the
final" -- with no free parameters. Independent halves predict exactly zero (joint = product); the
market prices a structure around it.

We de-vig both markets (make-final to 1 per half; the matchup grid to 1 overall), compute the implied
tetrachoric for every cross-half pair, and RANK them. The mean sits near zero -- the board is broadly
consistent with independence -- but the spread is the story: favourites' runs to the final co-move
(positive), favourite-vs-longshot pairs substitute (negative). That co-move / substitute axis is the
Thurstone redundancy structure, priced.

Ratings are calibrated to the game markets with a Thurstone bracket (P(i beats j)=Phi((th_i-th_j)/sqrt2))
purely to draw the board and to show the independent-rollout baseline. Live from Kalshi; cached to
world_cup_copula_data.json (pass --offline for the cache).
"""
import os, sys, json, urllib.request, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import norm, multivariate_normal as MVN
from scipy.optimize import least_squares, brentq

SQRT2 = np.sqrt(2.0)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "world_cup_copula_data.json")
SEEDJS = os.path.join(HERE, "world_cup_copula_seed.json")
BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"winner": "KXMENWORLDCUP", "advance": "KXWCADVANCE",
          "round": "KXWCROUND", "matchup": "KXWCMATCHUP"}


# ------------------------------------------------------------------ Kalshi data
def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def _mid(m):
    b, a = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    if b and a and float(a) > 0:
        return 0.5 * (float(b) + float(a))
    lp = m.get("last_price_dollars")
    return float(lp) if lp else None


def fetch_live():
    out = {"winner": {}, "advance": {}, "final": {}, "semi": {},
           "games": {}, "codes": {}, "final_matchup": {}}
    for m in _get(f"/markets?series_ticker={SERIES['winner']}&status=open&limit=100")["markets"]:
        p = _mid(m)
        if p is not None:
            out["winner"][m["yes_sub_title"]] = p
    for m in _get(f"/markets?series_ticker={SERIES['round']}&status=open&limit=200")["markets"]:
        rnd, p = m["ticker"].split("-")[1], _mid(m)
        if p is None:
            continue
        if rnd.endswith("SEMI"):
            out["semi"][m["yes_sub_title"]] = p
        elif rnd.endswith("FINAL"):
            out["final"][m["yes_sub_title"]] = p
    for m in _get(f"/markets?series_ticker={SERIES['advance']}&status=open&limit=100")["markets"]:
        p = _mid(m)
        if p is None:
            continue
        matchid, code = m["ticker"].split("-")[1], m["ticker"].split("-")[2]
        team = m["yes_sub_title"].replace(" advances", "").strip()
        out["advance"][team] = p
        out["codes"][code] = team
        out["games"].setdefault(matchid, {})[code] = team
    for m in _get(f"/markets?series_ticker={SERIES['matchup']}&status=open&limit=200")["markets"]:
        if m["ticker"].split("-")[1] != "26FIN":
            continue
        p = _mid(m)
        if p is None or " vs " not in (m.get("yes_sub_title") or ""):
            continue
        a, b = [x.strip() for x in m["yes_sub_title"].split(" vs ")]
        out["final_matchup"][f"{a}|{b}"] = p
    return out


def load(offline=False):
    if not offline:
        try:
            d = fetch_live()
            if d["winner"] and d["final"] and d["advance"] and d["final_matchup"]:
                json.dump(d, open(CACHE, "w"), indent=1)
                print(f"[data] live snapshot from Kalshi ({len(d['winner'])} teams, "
                      f"{len(d['final_matchup'])} final-matchup pairs) -> cached")
                return d
            print("[data] live fetch incomplete; falling back to cache")
        except Exception as e:  # noqa: BLE001
            print(f"[data] live fetch failed ({e}); using cache")
    if not os.path.exists(CACHE):
        sys.exit("no cached data and live fetch failed")
    print("[data] using cached snapshot")
    return json.load(open(CACHE))


# ------------------------------------------------------------------ bracket tree
def leaf(t):
    return ("t", t)


def match(l, r, lab):
    return ("m", l, r, lab)


def leaves(nd):
    return [nd[1]] if nd[0] == "t" else leaves(nd[1]) + leaves(nd[2])


def build_bracket(data):
    games = {mid: list(sides.values()) for mid, sides in data["games"].items()}
    r16, single = [], []
    for _, sides in games.items():
        adv = max(data["advance"].get(s, 0) for s in sides)
        semi = max(data["semi"].get(s, 0) for s in sides)
        node = match(leaf(sides[0]), leaf(sides[1]), "R16" if semi < adv - 0.05 else "QF")
        (r16 if node[3] == "R16" else single).append(node)
    slots = []
    if len(r16) == 2:
        slots.append(match(r16[0], r16[1], "QF"))
    elif len(r16) == 1:
        slots.append(r16[0])
    slots += single
    fin = data["final"]
    hm = lambda s: sum(fin.get(t, 0) for t in leaves(s))
    best, bdev = None, 1e9
    for (h1, h2) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        dev = abs((hm(slots[h1[0]]) + hm(slots[h1[1]])) - (hm(slots[h2[0]]) + hm(slots[h2[1]])))
        if dev < bdev:
            best, bdev = (h1, h2), dev
    (h1, h2) = best
    sf1 = match(slots[h1[0]], slots[h1[1]], "SF")
    sf2 = match(slots[h2[0]], slots[h2[1]], "SF")
    return match(sf1, sf2, "F"), sf1, sf2


# ------------------------------------------------------------------ implied tetrachoric
def tetrachoric(pa, pb, pab):
    """Correlation of the latent bivariate normal reproducing marginals (pa,pb) and joint pab."""
    if not (0 < pa < 1 and 0 < pb < 1):
        return np.nan
    lo, hi = max(0.0, pa + pb - 1), min(pa, pb)
    pab = min(max(pab, lo + 1e-6), hi - 1e-6)
    ha, hb = norm.ppf(pa), norm.ppf(pb)
    f = lambda r: MVN(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([ha, hb]) - pab
    try:
        return brentq(f, -0.999, 0.999, xtol=1e-4)
    except ValueError:
        return -0.999 if pab <= lo + 1e-5 else 0.999


def implied_ranking(data, half1, half2):
    """Implied tetrachoric correlation of 'both reach the final' for every cross-half pair,
    de-vigging make-final (to 1 per half) and the matchup grid (to 1 overall)."""
    fin = data["final"]
    s1 = sum(fin[t] for t in half1 if t in fin)
    s2 = sum(fin[t] for t in half2 if t in fin)
    f = {t: fin[t] / s1 for t in half1 if t in fin}
    f.update({t: fin[t] / s2 for t in half2 if t in fin})
    sM = sum(data["final_matchup"].values())
    rows = []
    for key, v in data["final_matchup"].items():
        a, b = key.split("|")
        if a in f and b in f:
            rows.append((a, b, f[a], f[b], v / sM, tetrachoric(f[a], f[b], v / sM)))
    rows.sort(key=lambda r: -r[5])
    return rows, f


# ------------------------------------------------------------------ Thurstone ratings (for the board)
def label_nodes(final):
    teams = sorted(set(leaves(final)))
    return teams, {t: i for i, t in enumerate(teams)}


def solve_indep(nd, theta, ix):
    n = len(ix)
    if nd[0] == "t":
        v = np.zeros(n); v[ix[nd[1]]] = 1.0
        return v
    pL, pR = solve_indep(nd[1], theta, ix), solve_indep(nd[2], theta, ix)
    P = norm.cdf((theta[:, None] - theta[None, :]) / SQRT2)
    return pL * (P @ pR) + pR * (P @ pL)


def fit_ratings(final, sf1, sf2, ix, data):
    teams = list(ix)
    sched = [tuple(s.values()) for s in data["games"].values()]

    def resid(th):
        theta = th - th.mean()
        win = solve_indep(final, theta, ix)
        rf = solve_indep(sf1, theta, ix) + solve_indep(sf2, theta, ix)
        r = [norm.cdf((theta[ix[a]] - theta[ix[b]]) / SQRT2) - data["advance"][a] for a, b in sched]
        for t, i in ix.items():
            if t in data["final"]:
                r.append(rf[i] - data["final"][t])
            if t in data["winner"]:
                r.append(win[i] - data["winner"][t])
        return np.array(r)

    x0 = np.array([norm.ppf(np.clip(data["winner"].get(t, .02), 1e-3, .99)) for t in teams])
    sol = least_squares(resid, x0, method="lm", max_nfev=6000)
    return sol.x - sol.x.mean()


# ------------------------------------------------------------------ report / seed / figure
def report(rows):
    print("\n===== implied tetrachoric correlation: 'both reach the final' (ranked) =====")
    print("  (from Kalshi's final-matchup market vs de-vigged make-final marginals;")
    print("   independent halves would give exactly 0 -- the spread is what the market prices)\n")
    print(f"  {'pair':26s}{'Pf(A)':>7s}{'Pf(B)':>7s}{'joint':>7s}{'tetra rho':>11s}")
    for a, b, fa, fb, j, r in rows:
        tag = "  co-move" if r > 0.05 else ("  substitute" if r < -0.05 else "")
        print(f"  {a+' vs '+b:26s}{fa:7.3f}{fb:7.3f}{j:7.3f}{r:+11.3f}{tag}")
    vals = [r[5] for r in rows]
    print(f"\n  mean {np.nanmean(vals):+.3f}   range [{np.nanmin(vals):+.3f}, {np.nanmax(vals):+.3f}]")


def value_bets(rows):
    """If opposite-half finalists are independent (which any single-factor ability model implies),
    the fair matchup price is the product of the make-final marginals. Departures are value."""
    bets = [(a, b, fa * fb, j, (fa * fb - j) * 100) for a, b, fa, fb, j, r in rows]
    bets.sort(key=lambda x: -abs(x[4]))
    print("\n===== value screen: final-matchup market vs independence (de-vigged) =====")
    print("  fair = Pf(A)*Pf(B) (independent halves); edge = fair - market, in cents\n")
    print(f"  {'matchup':26s}{'fair':>7s}{'market':>8s}{'edge¢':>7s}   action")
    for a, b, fair, mkt, edge in bets[:8]:
        act = "BUY  (market underprices)" if edge > 0 else "SELL (market overprices)"
        print(f"  {a+' vs '+b:26s}{fair:7.3f}{mkt:8.3f}{edge:+7.1f}   {act}")
    print("  (edges are vig-neutral; the live matchup book also carries ~10-13% overround)")


def export_seed(data, final, sf1, sf2, theta, ix, rows, fdev):
    def ser(nd):
        return {"t": nd[1]} if nd[0] == "t" else {"r": nd[3], "l": ser(nd[1]), "R": ser(nd[2])}
    seed = {"tree": ser(final), "half1": leaves(sf1), "half2": leaves(sf2),
            "theta": {t: round(float(theta[ix[t]]), 4) for t in ix},
            "winner": data["winner"], "final": data["final"], "advance": data["advance"],
            "final_dev": {t: round(v, 4) for t, v in fdev.items()},
            "final_matchup": data["final_matchup"], "codes": data["codes"],
            "pairs": [{"a": a, "b": b, "fa": round(fa, 4), "fb": round(fb, 4),
                       "joint": round(j, 4), "rho": round(float(r), 4),
                       "edge": round((fa * fb - j) * 100, 2)}
                      for a, b, fa, fb, j, r in rows]}
    json.dump(seed, open(SEEDJS, "w"), indent=1)
    print(f"\nweb seed -> {SEEDJS}")


def figure(path, data, final, theta, ix, rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.8], wspace=0.34)

    # (1) bracket
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("Live Kalshi bracket  (win% = winner market)", fontsize=10)
    lv = leaves(final); ypos = {t: len(lv) - k for k, t in enumerate(lv)}
    def draw(nd):
        if nd[0] == "t":
            t = nd[1]; ax.text(0, ypos[t], f"{t} {data['winner'].get(t,0)*100:.0f}%", fontsize=7.5,
                               va="center", bbox=dict(boxstyle="round,pad=0.15", fc="#eef3fb", ec="#9bb"))
            return 0, ypos[t]
        (xl, yl), (xr, yr) = draw(nd[1]), draw(nd[2])
        x = max(xl, xr) + 1; ym = (yl + yr) / 2
        ax.plot([xl, x, x, xr], [yl, yl, yr, yr], color="#889", lw=0.8)
        ax.text(x + .02, ym, nd[3] if nd[3] != "F" else "Final", fontsize=6.5, color="#556", va="center")
        return x, ym
    draw(final); ax.set_xlim(-0.5, 5.4)

    # (2) ranked implied tetrachoric bars
    ax2 = fig.add_subplot(gs[0, 1])
    top = rows[:9] + rows[-9:]
    labels = [f"{a[:3]}–{b[:3]}" for a, b, *_ in top]
    vals = [r[5] for r in top]
    cols = ["#2f8f5b" if v > 0 else "#b2495b" for v in vals]
    y = np.arange(len(top))[::-1]
    ax2.barh(y, vals, color=cols, height=0.7)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=7)
    ax2.axvline(0, color="k", lw=0.6)
    ax2.set_xlabel("implied tetrachoric ρ  (both reach the final)")
    ax2.set_title("Co-move (green) vs substitute (red)\nreach-final correlation, ranked", fontsize=9.5)
    ax2.grid(axis="x", alpha=0.3)

    # (3) joint vs independence
    ax3 = fig.add_subplot(gs[0, 2])
    ind = [fa * fb for _, _, fa, fb, _, _ in rows]; jt = [j for *_, j, _ in rows]
    ax3.scatter(ind, jt, s=20, c=[("#2f8f5b" if r[5] > 0 else "#b2495b") for r in rows], alpha=0.8)
    m = max(max(ind), max(jt)) * 1.05
    ax3.plot([0, m], [0, m], "k:", lw=0.7)
    ax3.set_xlabel("independent halves  Pf(A)·Pf(B)"); ax3.set_ylabel("market joint  P(A&B final)")
    ax3.set_title("Above the line = co-move", fontsize=9.5)
    ax3.grid(alpha=0.3); ax3.set_aspect("equal")

    fig.suptitle("World Cup on Kalshi: implied tetrachoric correlation of reaching the final", fontsize=12.5)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"figure -> {path}")


def main():
    data = load(offline="--offline" in sys.argv)
    final, sf1, sf2 = build_bracket(data)
    teams, ix = label_nodes(final)
    h1, h2 = leaves(sf1), leaves(sf2)
    rows, fdev = implied_ranking(data, h1, h2)
    theta = fit_ratings(final, sf1, sf2, ix, data)
    print(f"\ninferred halves:\n  half 1: {', '.join(sorted(h1))}\n  half 2: {', '.join(sorted(h2))}")
    report(rows)
    value_bets(rows)
    export_seed(data, final, sf1, sf2, theta, ix, rows, fdev)
    figure(os.path.join(HERE, "world_cup_copula.png"), data, final, theta, ix, rows)


if __name__ == "__main__":
    main()
