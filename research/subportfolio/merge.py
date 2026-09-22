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
          f"effective names.")
    for win, path, npv, pyv in env:
        print(f"built by: winning {win} ({path}), numpy {npv}, python {pyv}")
    if len(env) > 1:
        print("WARNING: shards were not all built by the same environment.")
    print()

    print(f"{'method':26s}{'variance':>11s}{'vs proportional':>17s}{'beats it':>10s}")
    for k in order:
        x = res[k]
        note = "" if k == "proportional" else f"{np.mean(x < base):9.0%}"
        print(f"{k:26s}{np.median(x):11.5f}{np.median(x / base):17.3f}{note}")

    # Calibration either converged or the row above is the last iterate of a
    # failed solve, which winning returns after a warning (winning #149). A
    # concentrated parent is where that happens, so the concentration is
    # printed beside it.
    calib = [k for k in rows[0] if k.startswith("calib ")]
    if calib:
        print()
        held = np.array([r["parent_names_held"] for r in rows])
        top = np.array([r["parent_top_weight"] for r in rows])
        print(f"parent concentration: holds {np.median(held):.0f} names "
              f"(min {held.min()}), top weight {np.median(top):.3f} "
              f"(max {top.max():.3f})")
        for k in calib:
            ok = np.mean([r[k]["converged"] for r in rows])
            it = np.median([r[k]["iterations"] for r in rows])
            worst = max(r[k]["residual"] for r in rows)
            flag = "" if ok == 1.0 else "   <-- SOME ROWS ARE NOT RESULTS"
            print(f"  {k[6:]:24s} converged {ok:6.0%}, median {it:3.0f} "
                  f"iterations, worst residual {worst:.1e}{flag}")
        if any(np.mean([r[k]["converged"] for r in rows]) < 1.0 for k in calib):
            print("\nA failed calibration returns its last iterate, so the "
                  "affected rows are not measurements of the method.")


if __name__ == "__main__":
    main()
