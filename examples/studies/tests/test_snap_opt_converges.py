from __future__ import annotations

import numpy as np

from examples.studies.snap_opt import SnapModelConfig, SnapRunConfig, SnapToneParameters, optimize_snap_parameters


def test_snap_opt_converges_from_vanilla():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, 0.7, -0.4, 1.1], dtype=float)
    cfg = SnapRunConfig(duration=140.0, dt=0.2, base_amp=0.010)
    vanilla = SnapToneParameters.vanilla(target)
    res = optimize_snap_parameters(
        model=model,
        target_phases=target,
        cfg=cfg,
        initial_params=vanilla,
        max_iter=30,
        learning_rate=0.35,
        threshold=5e-3,
    )
    assert res.history_error[-1] < res.history_error[0]
    # near-monotonic
    diffs = np.diff(res.history_error)
    assert np.sum(diffs > 2e-4) <= 1

