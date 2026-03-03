from __future__ import annotations

import numpy as np

from cqed_sim.snap_opt import (
    SnapModelConfig,
    SnapRunConfig,
    optimize_snap_parameters,
)


def test_no_extra_frequencies_introduced():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, 0.5, 0.8, -0.6], dtype=float)
    cfg = SnapRunConfig(duration=150.0, dt=0.2, base_amp=0.010)
    res = optimize_snap_parameters(model, target, cfg, max_iter=25, learning_rate=0.3, threshold=1e-2)
    # Protocol keeps the same tone set; optimizer only changes A_n, delta_n, phi_n.
    assert res.params.amplitudes.shape == target.shape
    assert res.params.detunings.shape == target.shape
    assert res.params.phases.shape == target.shape
    assert np.all(np.isfinite(res.params.amplitudes))
    assert np.all(np.isfinite(res.params.detunings))
    assert np.all(np.isfinite(res.params.phases))
