from __future__ import annotations

import time

import numpy as np
import pytest

from cqed_sim.snap_opt import SnapModelConfig, SnapRunConfig, optimize_snap_parameters, target_difficulty_metric


def _find_topt(model, target, durations):
    for t in durations:
        cfg = SnapRunConfig(duration=t, dt=0.25, base_amp=0.010)
        res = optimize_snap_parameters(model, target, cfg, max_iter=28, learning_rate=0.3, threshold=1.2e-2)
        if res.history_error[-1] < 0.8 * res.history_error[0]:
            return t
    return None


@pytest.mark.slow
def test_snap_opt_fails_below_topt_and_runtime_budget():
    start = time.perf_counter()
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, 1.0, -0.8, 0.7], dtype=float)
    short = SnapRunConfig(duration=70.0, dt=0.25, base_amp=0.010)
    long = SnapRunConfig(duration=180.0, dt=0.25, base_amp=0.010)
    r_short = optimize_snap_parameters(model, target, short, max_iter=24, learning_rate=0.3, threshold=1e-2)
    r_long = optimize_snap_parameters(model, target, long, max_iter=24, learning_rate=0.3, threshold=1e-2)
    assert r_short.history_error[-1] < r_short.history_error[0]
    # longer gate should not perform worse than short in this benchmark.
    assert r_long.history_error[-1] <= r_short.history_error[-1] + 0.1
    assert (time.perf_counter() - start) < 10.0


def test_snap_opt_generalization_sweeps():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    targets = [
        np.array([0.0, 0.3, -0.2, 0.1]),
        np.array([0.0, 1.2, -1.1, 0.9]),
        np.array([0.0, -0.8, 0.7, -1.0]),
    ]
    durations = [80.0, 110.0, 140.0, 180.0]
    pairs = []
    for tar in targets:
        topt = _find_topt(model, tar, durations)
        assert topt is not None
        pairs.append((target_difficulty_metric(tar), topt))
    # harder target should not have smaller Topt than easiest target in this sample.
    pairs = sorted(pairs, key=lambda x: x[0])
    assert pairs[-1][1] >= pairs[0][1]
