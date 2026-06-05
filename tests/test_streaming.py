import numpy as np

from allocation import StreamingThurstone

MIN_OBS = 10


def test_streaming_changing_universe():
    rng = np.random.default_rng(0)
    est = StreamingThurstone(phi=1.0, n_paths=1 << 12, min_obs=MIN_OBS)

    for t in range(160):
        active = ["A", "B", "C", "E"]
        if t >= 40:                       # D enters the observation stream at t=40
            active = active + ["D"]
        if t >= 110:                      # A leaves at t=110
            active = [a for a in active if a != "A"]
        f = rng.standard_normal()
        x = {a: 0.6 * f + 0.5 * rng.standard_normal() for a in active}
        est.learn_one(x)
        w = est.predict_one()
        # weights are always a valid simplex over a subset of the active names
        assert set(w).issubset(set(active))
        if w:
            assert abs(sum(w.values()) - 1.0) < 1e-6
            assert all(v >= -1e-9 for v in w.values())

    # entrant warmed in and is held; departed name is gone
    assert "D" in est.weights
    assert "A" not in est.weights


def test_streaming_is_smooth_step_to_step():
    # Within a stable universe the correlation drifts slowly, so consecutive
    # weights should move only a little -- the common-seed transport at work.
    rng = np.random.default_rng(1)
    est = StreamingThurstone(phi=1.0, n_paths=1 << 12, min_obs=MIN_OBS)
    assets = ["A", "B", "C", "D"]
    prev = None
    max_step = 0.0
    for t in range(200):
        f = rng.standard_normal()
        x = {a: 0.5 * f + 0.5 * rng.standard_normal() for a in assets}
        est.learn_one(x)
        w = est.predict_one()
        if t > 40 and prev:
            max_step = max(max_step, sum(abs(w[k] - prev[k]) for k in w))
        prev = w
    assert max_step < 0.15  # small per-step turnover
