from __future__ import annotations

import json
from pathlib import Path


def _notebook_text(path: str) -> str:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_analysis_tutorial_uses_public_ramsey_fit_key_and_explicit_frequency_offset_label():
    content = _notebook_text("tutorials/23_analysis_fitting_and_result_extraction.ipynb")

    assert 'ramsey.fitted_parameters["delta_omega"]' in content
    assert 'ramsey.fitted_parameters["detuning"]' not in content
    assert "Drive frequency relative to bare omega_q [MHz]" in content


def test_end_to_end_calibration_tutorial_uses_delta_omega_summary_key():
    content = _notebook_text("tutorials/25_small_calibration_workflow_end_to_end.ipynb")

    assert 'ramsey.fitted_parameters["delta_omega"]' in content
    assert 'ramsey.fitted_parameters["detuning"]' not in content
    assert '"ramsey_delta_omega_hz"' in content
    assert "Drive frequency relative to bare omega_q [MHz]" in content
