"""Rodriguez Dominguez, Shahzad and Hong (2025), implemented and benchmarked.

"Multi-Hypothesis Prediction for Portfolio Optimization", arXiv:2501.03919,
Expert Systems with Applications 292:128633.

The construction, faithfully:

  1. One forecaster per asset, each predicting only its own next return from
     its own lagged returns. Their predictors are small MLPs; here they are
     ridge autoregressions, which keeps the structure and removes a training
     loop that would dominate the runtime without changing what is measured.
  2. Trained under their competition rule, equation 5: on each sample the
     lowest-loss predictor is updated with weight 1 - eps and the rest share
     eps/(M-1). Implemented as per-sample weights in the ridge fit.
  3. Weights are the ridge regression of the EQUAL-WEIGHTED portfolio's
     realised return series onto the stacked forecasts, equation 8, then
     clipped at zero and renormalised, as the paper specifies.

Note eps = (M-1)/M is where 1-eps equals eps/(M-1) and the competition
vanishes; the paper's grid tops out at 0.5 and never reaches it for M > 2.
"""
import numpy as np


def _ar_features(X, lags):
    """Rows of lagged returns per asset: (T-lags, n, lags)."""
    T, n = X.shape
    return np.stack([X[lags - 1 - k: T - 1 - k] for k in range(lags)], axis=2)


def mhp_weights(X, eps=0.35, lam_s=3.0, lam_f=1.0, lags=3):
    """Portfolio weights from the multi-hypothesis construction."""
    T, n = X.shape
    if T <= lags + 2:
        return np.full(n, 1.0 / n)
    F = _ar_features(X, lags)          # (m, n, lags)
    Y = X[lags:]                       # (m, n) targets
    m = Y.shape[0]

    # --- stage 1: per-asset forecasters, fitted once to get the competition
    # weights, then refitted with them (one pass, as the rule is a reweighting)
    def fit(sample_w):
        out = np.empty((m, n))
        for j in range(n):
            A = F[:, j, :]
            sw = sample_w[:, j][:, None]
            G = A.T @ (A * sw) + lam_f * np.eye(lags)
            b = A.T @ (Y[:, j] * sample_w[:, j])
            out[:, j] = A @ np.linalg.solve(G, b)
        return out

    pred0 = fit(np.ones((m, n)))
    # Scale-free competition. Comparing raw squared error across assets makes
    # the lowest-volatility name win almost every sample, which is a scale
    # artifact rather than a better forecaster: with lognormal(0, 0.6) vols the
    # correlation between an asset's volatility and its win count is about
    # -0.6, and at T=20, n=40 only 14 of 40 assets ever win, leaving the rest
    # with near-zero sample weight and a collapsed forecast column. The paper's
    # competition is among hypotheses for the same target, so normalise.
    loss = (pred0 - Y) ** 2 / np.maximum(Y.var(axis=0), 1e-12)
    winner = loss.argmin(axis=1)
    sw = np.full((m, n), eps / max(n - 1, 1))
    sw[np.arange(m), winner] = 1.0 - eps
    pred = fit(sw * n)                 # rescale so the mean weight is order one

    # --- stage 2: regress the equal-weighted realised return on the forecasts
    rbar = Y.mean(axis=1)
    G = pred.T @ pred + lam_s * np.eye(n)
    w = np.linalg.solve(G, pred.T @ rbar)
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


if __name__ == "__main__":
    import sys
    from sklearn.covariance import LedoitWolf
    from confound import hrp
    from robust import random_structure, mv_lo, clusters, nco, taper

    RATIOS = (0.5, 1.0, 2.0, 5.0)
    # Their method is built on forecasts having skill. A market of independent
    # draws has nothing to forecast, so it is tested twice: once on such a
    # market and once with a genuine AR(1) component that the per-asset
    # forecasters can actually find.
    PHI_AR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    EPS = (0.0, 0.35, 0.5)
    rng = np.random.default_rng(4242)
    keys = ["hrp", "equal weight", "inverse variance", "min-var LW", "taper 0.5", "nco"] + \
           [f"mhp eps={e}" for e in EPS]
    out = {r: {k: [] for k in keys} for r in RATIOS}
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    for i in range(draws):
        n = 40
        Sig, fam = random_structure(rng, n)
        L = np.linalg.cholesky(Sig)
        for r in RATIOS:
            T = max(int(round(r * n)), 8)
            Z = rng.normal(size=(T + 50, n)) @ L.T
            if PHI_AR:
                for t in range(1, len(Z)):
                    Z[t] += PHI_AR * Z[t - 1]
            X = Z[50:]
            S = np.cov(X, rowvar=False)
            cl = clusters(S, 5)
            v = lambda w: float(np.asarray(w, float) @ Sig @ np.asarray(w, float))
            d = out[r]
            d["hrp"].append(v(hrp(S)))
            d["equal weight"].append(v(np.full(n, 1.0 / n)))
            iv = 1 / np.diag(S); iv /= iv.sum(); d["inverse variance"].append(v(iv))
            d["min-var LW"].append(v(mv_lo(LedoitWolf(assume_centered=False).fit(X).covariance_)))
            d["taper 0.5"].append(v(mv_lo(taper(S, cl, 0.5, n))))
            d["nco"].append(v(nco(S, cl, n)))
            for e in EPS:
                d[f"mhp eps={e}"].append(v(mhp_weights(X, eps=e)))
        if (i + 1) % 30 == 0:
            print(f"  ...{i+1}/{draws}", flush=True)

    print(f"\nMedian true variance over {draws} random structures, n = 40, "
          f"AR(1) coefficient {PHI_AR}\n")
    print(f"{'T/n':>6s}" + "".join(f"{k:>19s}" for k in keys))
    for r in RATIOS:
        print(f"{r:6.2f}" + "".join(f"{np.median(out[r][k]):19.4f}" for k in keys))
    print(f"\nShare of draws where each beats EQUAL WEIGHT\n")
    print(f"{'T/n':>6s}" + "".join(f"{k:>19s}" for k in keys if k != "equal weight"))
    for r in RATIOS:
        ew = np.array(out[r]["equal weight"])
        print(f"{r:6.2f}" + "".join(f"{np.mean(np.array(out[r][k]) < ew):19.0%}"
                                   for k in keys if k != "equal weight"))
