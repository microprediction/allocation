"""Thurstone as a stable forecast combiner — M4 evidence.

Forecast combination is the natural home for the ability tilt: feed the K base
models' error series as the race's "returns" and read off simplex combination
weights. The claim is not that this beats the least-squares-optimal combination
everywhere, but that it is the *stable* member of the family --- it wins exactly
where the combination puzzle lives, when forecasters are similar-quality and the
data available to estimate weights is scarce (so the error covariance is badly
estimated and the optimal combination overfits it).

We test on the canonical M4 benchmark. For each category we fit the standard
statistical base-forecaster pool (statsforecast), then estimate combination
weights on a small set of "fit" series and evaluate on held-out series, sweeping
the number of fit series --- the small-sample lever. Four combiners:

  equal          : 1/K (the combination-puzzle champion)
  bates_granger  : Sigma_e^{-1} 1, normalized (the error-variance-optimal combo)
  nnls           : Breiman non-negative stacking, projected to the simplex
  thurstone      : ability tilt on the error series (stable simplex combiner)

Honest report: the numbers fall where they fall.

Extra deps (not package deps): statsforecast, datasetsforecast, scipy, matplotlib.
Forecasts are cached per category so the figure can be re-tuned without refitting.
"""
import os, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from numpy.linalg import pinv
from scipy.optimize import nnls
from statsforecast import StatsForecast
from statsforecast.models import (AutoETS, AutoCES, AutoTheta, DynamicOptimizedTheta,
                                   SeasonalNaive, RandomWalkWithDrift, HistoricAverage,
                                   WindowAverage)
from datasetsforecast.m4 import M4, M4Info
from allocation import ThurstonePortfolio

CACHE = os.environ.get("M4_CACHE", "/tmp/m4cache")
os.makedirs(CACHE, exist_ok=True)
N_SERIES = {"Yearly": 500, "Quarterly": 500, "Monthly": 400}
N_EVAL = 200                      # held-out series for evaluation
N_TRIALS = 25                     # random fit-set draws per (category, n_fit)
FIT_GRID = [5, 10, 20, 50, 200]
# the comparable statistical-forecaster family (the combination-puzzle regime)
COMPARABLE = ["AutoETS", "CES", "AutoTheta", "DynamicOptimizedTheta", "RWD", "WindowAverage"]
KEYS = ["equal", "bates_granger", "nnls", "thurstone"]


def smape(y, f):
    return float(np.mean(200 * np.abs(y - f) / (np.abs(y) + np.abs(f) + 1e-9)))


def load_category(group, rng):
    """Fit the base pool on a sample of M4 series; cache the per-series forecasts.

    Returns F (id -> (H,K) forecasts), truth (id->(H,)), scale (level), mase
    (in-sample seasonal-naive MAE), and the method names.
    """
    info = M4Info[group]
    H, seas = info.horizon, info.seasonality
    npz = f"{CACHE}/{group}.npz"
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        return (d["F"].item(), d["truth"].item(), d["scale"].item(),
                d["mase"].item(), list(d["mcols"]), H)

    Y, _, _ = M4.load(directory=CACHE, group=group)
    ids = rng.choice(Y["unique_id"].unique(), size=N_SERIES[group], replace=False)
    train = Y[Y["unique_id"].isin(ids)].sort_values(["unique_id", "ds"]).copy()
    train["ds"] = train.groupby("unique_id").cumcount() + 1            # integer step index

    test_wide = pd.read_csv(glob.glob(f"{CACHE}/**/{group}-test.csv", recursive=True)[0]).set_index("V1")
    truth = {u: test_wide.loc[u].to_numpy(float) for u in ids}

    models = [AutoETS(season_length=seas), AutoCES(season_length=seas),
              AutoTheta(season_length=seas), DynamicOptimizedTheta(season_length=seas),
              SeasonalNaive(season_length=seas), RandomWalkWithDrift(),
              HistoricAverage(), WindowAverage(window_size=max(2, seas))]
    print(f"[{group}] fitting {len(models)} base methods on {len(ids)} series (H={H}) ...")
    fc = StatsForecast(models=models, freq=1, n_jobs=1).forecast(df=train, h=H)
    mcols = [c for c in fc.columns if c not in ("unique_id", "ds")]

    F, scale, mase = {}, {}, {}
    for u in ids:
        g = fc[fc.unique_id == u].sort_values("ds")
        F[u] = g[mcols].to_numpy(float)
        yt = train[train.unique_id == u]["y"].to_numpy()
        scale[u] = np.mean(np.abs(yt))
        d = np.abs(yt[seas:] - yt[:-seas]) if len(yt) > seas else np.abs(np.diff(yt))
        mase[u] = max(np.mean(d), 1e-9)
    np.savez(npz, F=F, truth=truth, scale=scale, mase=mase, mcols=np.array(mcols, object))
    return F, truth, scale, mase, mcols, H


def simplex(w):
    w = np.clip(w, 0, None); s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1 / len(w))


def fit_weights(fit_ids, cols, F, truth, scale):
    kk = len(cols)
    Estack = np.vstack([(truth[u][:, None] - F[u][:, cols]) / scale[u] for u in fit_ids])
    out = {"equal": np.full(kk, 1 / kk)}
    wbg = pinv(np.cov(Estack.T)) @ np.ones(kk); out["bates_granger"] = wbg / wbg.sum()
    Pn = np.vstack([F[u][:, cols] / scale[u] for u in fit_ids])
    yn = np.concatenate([truth[u] / scale[u] for u in fit_ids])
    coef, _ = nnls(Pn, yn); out["nnls"] = simplex(coef)
    th = ThurstonePortfolio(calib="diagonal", target="diagonal", phi=1.0, n_paths=1 << 14)
    th.fit(Estack); out["thurstone"] = np.asarray(th.weights_)
    return out


def evaluate(w, eval_ids, cols, F, truth, mase):
    s = np.mean([smape(truth[u], F[u][:, cols] @ w) for u in eval_ids])
    m = np.mean([np.mean(np.abs(truth[u] - F[u][:, cols] @ w)) / mase[u] for u in eval_ids])
    return float(s), float(m)


def run():
    rng = np.random.default_rng(0)
    results = {}            # group -> dict(n_fit -> dict(key -> (smape_mean, smape_sd, mase_mean)))
    for group in N_SERIES:
        F, truth, scale, mase, mcols, H = load_category(group, rng)
        ids = np.array(list(F))
        single = {m: np.mean([smape(truth[u], F[u][:, j]) for u in ids])
                  for j, m in enumerate(mcols)}
        cols = [j for j, m in enumerate(mcols) if m in COMPARABLE]
        print(f"\n### {group}  (comparable pool: {[mcols[c] for c in cols]})")
        print("  single-method sMAPE: " +
              ", ".join(f"{m} {single[m]:.1f}" for m in mcols))
        perm = rng.permutation(ids); eval_ids, pool = perm[:N_EVAL], perm[N_EVAL:]

        results[group] = {}
        print(f"  {'fit':>5} | " + "  ".join(f"{k:>14}" for k in KEYS) + "   (sMAPE | best)")
        for n_fit in FIT_GRID:
            n_trials = 1 if n_fit >= len(pool) else N_TRIALS
            acc = {k: [] for k in KEYS}; macc = {k: [] for k in KEYS}
            for _ in range(n_trials):
                fit_ids = rng.choice(pool, min(n_fit, len(pool)), replace=False)
                w = fit_weights(fit_ids, cols, F, truth, scale)
                for k in KEYS:
                    s, m = evaluate(w[k], eval_ids, cols, F, truth, mase)
                    acc[k].append(s); macc[k].append(m)
            results[group][n_fit] = {k: (np.mean(acc[k]), np.std(acc[k]), np.mean(macc[k]))
                                     for k in KEYS}
            cells = "  ".join(f"{np.mean(acc[k]):>6.2f}±{np.std(acc[k]):>4.2f}" for k in KEYS)
            best = min(KEYS, key=lambda k: np.mean(acc[k]))
            print(f"  {n_fit:>5} | {cells}   {best}")
    return results


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(results)
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.4), sharey=False)
    colors = {"equal": "#888", "bates_granger": "#d62728", "nnls": "#1f77b4", "thurstone": "#2ca02c"}
    for ax, g in zip(axes, groups):
        xs = sorted(results[g])
        for k in KEYS:
            ys = [results[g][n][k][0] for n in xs]
            sd = [results[g][n][k][1] for n in xs]
            ax.plot(xs, ys, "-o", ms=3, color=colors[k], label=k)
            ax.fill_between(xs, np.array(ys) - np.array(sd), np.array(ys) + np.array(sd),
                            color=colors[k], alpha=0.12)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(xs)
        ax.set_title(f"M4 {g}"); ax.set_xlabel("# series to fit weights")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out sMAPE")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Forecast combination: stable simplex (Thurstone) vs optimal (Bates–Granger / NNLS) "
                 "vs equal — comparable-quality pool", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    res = run()
    figure(res, os.path.join(os.path.dirname(__file__), "m4_combination.png"))
