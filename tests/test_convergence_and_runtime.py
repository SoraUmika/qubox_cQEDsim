from __future__ import annotations

import time

import numpy as np

from cqed_sim.snap_prl133.model import SnapModelConfig
from cqed_sim.snap_prl133.optimize import optimize_snap_prl133
from cqed_sim.snap_prl133.pulses import SnapToneParameters
from cqed_sim.snap_opt.experiments import SnapRunConfig


def test_convergence_and_runtime():
    start = time.perf_counter()
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, -np.pi / 4.0, np.pi / 2.0], dtype=float)
    cfg = SnapRunConfig(duration=140.0, dt=0.25, base_amp=0.010)
    res = optimize_snap_prl133(
        model,
        target,
        cfg,
        initial_params=SnapToneParameters.vanilla(target),
        max_iter=36,
        learning_rate=0.3,
        threshold=1e-5,
        local_refine_maxiter=0,
    )
    assert res.history_epsilon_coh[-1] <= res.history_epsilon_coh[0]
    tail = np.asarray(res.history_epsilon_coh[-5:], dtype=float)
    if tail.size > 1:
        assert np.all(np.diff(tail) <= 1e-8)
    assert (time.perf_counter() - start) < 10.0
