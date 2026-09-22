"""Combine shards and print the table. Reads every JSON under --out.

    python merge.py --tag index-n5000-m200-k2-s4

Reports the median ratio to proportional restriction, since proportional is
what index providers actually do, along with the share of draws on which each
rule beats it. The share is the robustness claim and the ratio is the size of
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
    cfg, rows, seen = None, [], set()
    for f in files:
        blob = json.loads(f.read_text())
        cfg = cfg or blob["config"]
        for r in blob["rows"]:
            if r["draw"] in seen:
                continue
            seen.add(r["draw"])
            rows.append(r)
    expected = set(range(cfg["draws"]))
    missing = sorted(expected - seen)
    return cfg, rows, files, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    cfg, rows, files, missing = load(a.out, a.tag)

    skipped = [r for r in rows if r.get("skipped")]
    rows = [r for r in rows if not r.get("skipped")]
    Ts = cfg["Ts"]
    order = (["equal weight", "proportional", "race"]
             + [x for t in Ts for x in (f"race+factor T={t}",
                                        f"estimate+solve T={t}")]
             + ["oracle"])
    res = {k: np.array([r[k] for r in rows]) for k in order}
    base = res["proportional"]

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
          f"parent holds {np.median([r['parent_effective_n'] for r in rows]):.0f} "
          f"effective names.\n")

    print(f"{'method':26s}{'variance':>11s}{'vs proportional':>17s}{'beats it':>10s}")
    for k in order:
        x = res[k]
        note = "" if k == "proportional" else f"{np.mean(x < base):9.0%}"
        print(f"{k:26s}{np.median(x):11.5f}{np.median(x / base):17.3f}{note}")


if __name__ == "__main__":
    main()
