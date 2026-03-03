from __future__ import annotations

import numpy as np

from cqed_sim.snap_opt import SnapModelConfig, SnapRunConfig, optimize_snap_parameters


def test_snap_opt_hits_threshold_above_topt():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    targets = [
        np.array([0.0, 0.4, -0.2, 0.8]),
        np.array([0.0, 1.0, -0.7, 0.3]),
    ]
    for target in targets:
        cfg = SnapRunConfig(duration=180.0, dt=0.2, base_amp=0.009)
        res = optimize_snap_parameters(model, target, cfg, max_iter=45, learning_rate=0.35, threshold=6e-3)
        # In this reduced simulation, demand clear improvement and bounded residual error.
        assert res.history_error[-1] < 0.9 * res.history_error[0]
        assert res.final_errors.mean_overlap_error < 1.0
