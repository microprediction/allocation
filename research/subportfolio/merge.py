"""Combine shards and print the table. Reads every JSON under --out.

    python merge.py --tag index-n5000-m200-k2-s4

Reports each rule's median Sharpe ratio, its median ratio to proportional
restriction, since proportional is what index providers actually do, and the
share of draws on which it beats it. Higher is better throughout. The share is the robustness claim and the ratio is the size of
the effect; a rule that wins narrowly on most draws and loses badly on a few
is a different animal from one that wins big on half, so both are printed.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(out, tag):
    files = sorted(Path(out).glob(f"{tag}-shard*of*.json"))
    if not files:
        raise SystemExit(f"no shards matching {tag} under {out}/")
    cfg, rows, seen, env = None, [], set(), set()
    for f in files:
        blob = json.loads(f.read_text())
        cfg = cfg or blob["config"]
        env.add((blob.get("winning", "?"), blob.get("winning_path", "?"),
                 blob.get("numpy", "?"), blob.get("python", "?")))
        for r in blob["rows"]:
            if r["draw"] in seen:
                continue
            seen.add(r["draw"])
            rows.append(r)
    expected = set(range(cfg["draws"]))
    missing = sorted(expected - seen)
    return cfg, rows, files, missing, sorted(env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    cfg, rows, files, missing, env = load(a.out, a.tag)

    skipped = [r for r in rows if r.get("skipped")]
    rows = [r for r in rows if not r.get("skipped")]
    Ts = cfg["Ts"]
    est_Ts = cfg.get("Ts_est") or Ts
    order = (["equal weight", "proportional", "flattened", "race"]
             + [f"race+factor T={t}" for t in Ts]
             + [f"race avoid-worst-10% T={t}" for t in Ts]
             + [f"race+sectors T={t}" for t in Ts]
             + ["race+sectors (true structure)"]
             + [f"race+tail T={t}" for t in Ts]
             + [f"black-litterman T={t}" for t in est_Ts]
             + [f"BL tail T={t}" for t in est_Ts]
             + [f"estimate+solve T={t}" for t in est_Ts]
             + ["oracle", "tail oracle"])
    # A rule is reported over the draws that have it, paired with proportional
    # on those same draws, so a run restarted with new rules keeps its old draws.
    order = [k for k in order if any(k in r for r in rows)]
    have = {k: [r for r in rows if k in r] for k in order}
    res = {k: np.array([r[k] for r in have[k]]) for k in order}
    bases = {k: np.array([r["proportional"] for r in have[k]]) for k in order}
    base = bases["proportional"]

    print(f"\n{cfg['scale']} scale: parent {cfg['n']} names, sub-universe "
          f"{cfg['m']}, {len(rows)} draws, {cfg['k']}-factor correlation.")
    print(f"{len(files)} shard files."
          + (f"  MISSING DRAWS: {missing}" if missing else "")
          + (f"  {len(skipped)} draws failed the premise check."
             if skipped else ""))
    worst = max(r["premise_residual"] for r in rows)
    print(f"premise: the parent is the exact optimum of the true covariance "
          f"to {worst:.1e} relative, worst draw.")
    print(f"markets: {', '.join(sorted({r['family'] for r in rows}))}; "
          f"parent holds {np.nanmedian([r['parent_effective_n'] for r in rows]):.0f} "
          f"effective names.")
    for win, path, npv, pyv in env:
        print(f"built by: winning {win} ({path}), numpy {npv}, python {pyv}")
    if len(env) > 1:
        print("WARNING: shards were not all built by the same environment.")
    print()

    # Sharpe differences in bps of excess return at an index volatility of 16%,
    # so "small" has a number: the bar is 25 to 50 bps.
    VOL = 0.16
    failed = {}
    for r in rows:
        for note in r.get("failed", []):
            failed[note.split(":")[0]] = failed.get(note.split(":")[0], 0) + 1
    for k, c in sorted(failed.items()):
        print(f"{k}: did not converge on {c} of {len(rows)} draws (left out of its row)")
    print(f"{'method':26s}{'sharpe':>9s}{'x prop':>9s}{'bps vs prop':>13s}{'beats it':>10s}")
    for k in order:
        x, b = res[k], bases[k]
        ok = ~np.isnan(x)
        bps = "" if k == "proportional" else f"{1e4 * VOL * np.nanmedian(x - b):+12.0f}"
        note = "" if k == "proportional" else f"{np.mean(x[ok] > b[ok]):9.0%}"
        n_k = "" if len(x) == len(rows) else f"  ({len(x)} draws)"
        print(f"{k:26s}{np.nanmedian(x):9.4f}{np.nanmedian(x / b):9.3f}{bps:>13s}{note}{n_k}")

    # The tail analogue of Sharpe: expected return per unit of expected
    # shortfall at 95% under the true law (the STARR ratio). Higher is better,
    # and it is what the tail oracle maximises. Raw es95 is kept in the rows.
    st_keys = [k for k in order if any(("starr " + k) in r for r in rows)]
    if st_keys:
        print()
        print(f"{'method':26s}{'starr':>9s}{'x prop':>9s}{'beats it':>10s}")
        for k in st_keys:
            hk = [r for r in rows if ("starr " + k) in r]
            x = np.array([r["starr " + k] for r in hk]); b = np.array([r["starr proportional"] for r in hk])
            ok = ~np.isnan(x)
            note = "" if k == "proportional" else f"{np.mean(x[ok] > b[ok]):9.0%}"
            print(f"{k:26s}{np.nanmedian(x):9.4f}{np.nanmedian(x / b):9.3f}{note}")


if __name__ == "__main__":
    main()
