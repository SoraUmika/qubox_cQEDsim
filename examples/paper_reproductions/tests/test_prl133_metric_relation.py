from __future__ import annotations

import numpy as np

from examples.paper_reproductions.snap_prl133.model import SnapModelConfig
from examples.paper_reproductions.snap_prl133.optimize import optimize_snap_prl133
from examples.paper_reproductions.snap_prl133.pulses import SnapToneParameters
from examples.studies.snap_opt.experiments import SnapRunConfig


def test_prl133_error_vector_norm_is_secondary_and_related():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, -np.pi / 4.0, np.pi / 2.0], dtype=float)
    res = optimize_snap_prl133(
        model,
        target,
        SnapRunConfig(duration=160.0, dt=0.25, base_amp=0.010),
        initial_params=SnapToneParameters.vanilla(target),
        max_iter=40,
        learning_rate=0.3,
        threshold=1e-5,
        local_refine_maxiter=0,
    )
    eps = np.asarray(res.history_epsilon_coh, dtype=float)
    norm = np.asarray(res.history_error_vector_norm, dtype=float)
    assert eps.size == norm.size
    if eps.size > 2:
        corr = np.corrcoef(eps, norm)[0, 1]
        assert np.isfinite(corr)
        assert corr > 0.5
