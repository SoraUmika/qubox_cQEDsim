from __future__ import annotations

import numpy as np

from examples.workflows.kerr_free_evolution import verify_kerr_sign


def test_notebook_kerr_diagnostic_matches_documented_runtime_convention():
    result = verify_kerr_sign(comparison_time_s=1.0e-6, alpha=2.0, n_cav=30, n_tr=3)
    assert result.documented_kerr_hz < 0.0
    assert result.matches_documented_sign
    assert np.imag(result.cavity_mean_documented) > 0.0
    assert np.imag(result.cavity_mean_flipped) < 0.0
    assert result.documented_phase_rad != result.flipped_phase_rad
