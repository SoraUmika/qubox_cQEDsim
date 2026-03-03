from __future__ import annotations

import numpy as np

from cqed_sim.snap_prl133.model import SnapModelConfig
from cqed_sim.snap_prl133.optimize import optimize_snap_prl133
from cqed_sim.snap_prl133.pulses import SnapToneParameters
from cqed_sim.snap_opt.experiments import SnapRunConfig


def test_prl133_components_shrink_for_longer_gate():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, -np.pi / 4.0, np.pi / 2.0], dtype=float)
    short = optimize_snap_prl133(
        model,
        target,
        SnapRunConfig(duration=70.0, dt=0.25, base_amp=0.010),
        initial_params=SnapToneParameters.vanilla(target),
        max_iter=50,
        learning_rate=0.3,
        threshold=1e-5,
        local_refine_maxiter=0,
    )
    long = optimize_snap_prl133(
        model,
        target,
        SnapRunConfig(duration=300.0, dt=0.25, base_amp=0.010),
        initial_params=SnapToneParameters.vanilla(target),
        max_iter=80,
        learning_rate=0.3,
        threshold=1e-5,
        local_refine_maxiter=8,
    )
    assert long.final_metric.max_component_error <= short.final_metric.max_component_error + 1e-4
    if long.threshold_hit:
        assert long.final_metric.max_component_error < 5e-3
