from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from examples.workflows.kerr_free_evolution import verify_kerr_sign


def test_kerr_sign_verification_matches_documented_runtime_sign():
    result = verify_kerr_sign(comparison_time_s=1.0e-6, alpha=2.0, n_cav=30, n_tr=3)

    assert np.isclose(result.documented_kerr_hz, -107.9e3, rtol=0.0, atol=1.0)
    assert np.isclose(result.flipped_kerr_hz, 107.9e3, rtol=0.0, atol=1.0)
    assert result.matches_documented_sign
    assert np.sign(np.imag(result.cavity_mean_documented)) == -np.sign(np.imag(result.cavity_mean_flipped))


def test_kerr_tutorial_points_users_to_sign_comparison_exercise():
    notebook = Path("tutorials/14_kerr_free_evolution.ipynb")
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    content = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "self-Kerr" in content
    assert "Reverse the sign of the Kerr coefficient and compare the direction of the phase-space bending." in content
