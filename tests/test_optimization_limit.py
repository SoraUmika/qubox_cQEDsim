from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.snap_prl133.model import SnapModelConfig
from cqed_sim.snap_prl133.optimize import optimize_snap_prl133
from cqed_sim.snap_prl133.pulses import SnapToneParameters
from cqed_sim.snap_opt.experiments import SnapRunConfig


@pytest.mark.slow
def test_optimization_limit():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, -np.pi / 4.0, np.pi / 2.0], dtype=float)
    eps_target = 1e-5
    iter_cap = 80

    short_cfg = SnapRunConfig(duration=70.0, dt=0.25, base_amp=0.010)
    long_cfg = SnapRunConfig(duration=180.0, dt=0.25, base_amp=0.010)

    short = optimize_snap_prl133(
        model,
        target,
        short_cfg,
        initial_params=SnapToneParameters.vanilla(target),
        max_iter=iter_cap,
        learning_rate=0.3,
        threshold=eps_target,
        local_refine_maxiter=8,
    )
    long = optimize_snap_prl133(
        model,
        target,
        long_cfg,
        initial_params=SnapToneParameters.vanilla(target),
        max_iter=iter_cap,
        learning_rate=0.3,
        threshold=eps_target,
        local_refine_maxiter=8,
    )

    assert long.final_metric.epsilon_coh < eps_target
    assert short.final_metric.epsilon_coh >= eps_target
