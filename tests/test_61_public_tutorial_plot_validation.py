from __future__ import annotations

import json

import numpy as np
import pytest

from tools import generate_tutorial_plots as tutorial_plots


def test_cross_kerr_public_plot_tracks_signed_conditional_phase() -> None:
    data = tutorial_plots.compute_cross_kerr_phase_data()

    times_ns = np.asarray(data["times_ns"], dtype=float)
    conditional_phase = np.asarray(data["conditional_phase_rad"], dtype=float)
    theory_phase = np.asarray(data["theory_phase_rad"], dtype=float)

    assert times_ns[0] == 0.0
    assert conditional_phase[-1] < -6.0
    assert abs(float(data["fitted_slope_hz"]) + 1.5e6) < 5.0e2
    assert float(data["max_abs_phase_error_rad"]) < 1.0e-5
    assert np.max(np.abs(conditional_phase - theory_phase)) < 1.0e-5


def test_floquet_public_plot_highlights_the_hybridized_branch_pair() -> None:
    data = tutorial_plots.compute_floquet_quasienergy_scan_data()

    scan_detunings = np.asarray(data["scan_detunings_mhz"], dtype=float)
    highlighted_gap = np.asarray(data["highlighted_gap_mhz"], dtype=float)
    center_pair_overlaps = np.asarray(data["center_pair_max_overlaps"], dtype=float)

    min_index = int(np.argmin(highlighted_gap))

    assert 0 < min_index < len(highlighted_gap) - 1
    assert abs(float(data["resonance_detuning_mhz"])) < 1.0e-6
    assert 0.0 < float(data["min_gap_mhz"]) < 0.05
    assert np.all(center_pair_overlaps < 0.6)
    assert highlighted_gap[min_index] < highlighted_gap[0]
    assert highlighted_gap[min_index] < highlighted_gap[-1]
    assert scan_detunings[min_index] == pytest.approx(0.0, abs=1.0e-6)


def test_public_plot_generators_emit_png_and_validation_summaries(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "tutorials"
    validation_dir = out_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(tutorial_plots, "OUT_DIR", out_dir)
    monkeypatch.setattr(tutorial_plots, "VALIDATION_DIR", validation_dir)

    tutorial_plots.plot_cross_kerr_phase()
    tutorial_plots.plot_floquet_quasienergy_scan()

    cross_png = out_dir / "cross_kerr_phase.png"
    floquet_png = out_dir / "floquet_quasienergy_scan.png"
    cross_summary = validation_dir / "cross_kerr_phase.json"
    floquet_summary = validation_dir / "floquet_quasienergy_scan.json"

    assert cross_png.is_file()
    assert floquet_png.is_file()
    assert cross_summary.is_file()
    assert floquet_summary.is_file()

    cross_payload = json.loads(cross_summary.read_text(encoding="utf-8"))
    floquet_payload = json.loads(floquet_summary.read_text(encoding="utf-8"))

    assert abs(float(cross_payload["fitted_slope_hz"]) + 1.5e6) < 5.0e2
    assert 0.0 < float(floquet_payload["min_gap_mhz"]) < 0.05
