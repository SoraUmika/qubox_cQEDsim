from __future__ import annotations

import numpy as np
import pytest

from examples.paper_reproductions.snap_prl133.errors import CoherentMetricResult, _validate_metric_bounds, compute_mean_squared_overlap
from examples.paper_reproductions.snap_prl133.model import SnapModelConfig
from examples.paper_reproductions.snap_prl133.pulses import SnapToneParameters
from examples.studies.snap_opt.experiments import SnapRunConfig


def test_metric_bounds_and_per_manifold_bounds():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, -np.pi / 4.0, np.pi / 2.0], dtype=float)
    cfg = SnapRunConfig(duration=140.0, dt=0.25, base_amp=0.010)
    metric = compute_mean_squared_overlap(model, target, cfg, SnapToneParameters.vanilla(target))
    assert 0.0 <= metric.fidelity <= 1.0
    assert 0.0 <= metric.epsilon_coh <= 1.0
    assert np.all(metric.per_manifold_overlap >= 0.0)
    assert np.all(metric.per_manifold_overlap <= 1.0)


def test_metric_bounds_fail_fast():
    bad = CoherentMetricResult(
        fidelity=1.2,
        epsilon_coh=-0.2,
        per_manifold_overlap=np.array([1.1, -0.1]),
        dtheta=np.zeros(2),
        eps_l=np.zeros(2),
        eps_t=np.zeros(2),
        error_vector_norm=0.0,
        max_component_error=0.0,
        excited_amplitudes=np.zeros(2, dtype=np.complex128),
        ground_amplitudes=np.zeros(2, dtype=np.complex128),
    )
    with pytest.raises(ValueError):
        _validate_metric_bounds(bad, context="test")

