"""Smoke tests for cqed_sim.plotting — headless rendering verification.

Each test verifies that the plotting function runs without error and returns a
matplotlib Figure. All tests use the Agg backend to avoid display requirements.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cqed_sim.plotting import (
    plot_energy_levels,
    plot_bloch_track,
    plot_sqr_calibration_result,
    save_figure,
)


# ---- fixtures ----

@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


def _make_energy_spectrum():
    from cqed_sim.core import (
        DispersiveTransmonCavityModel,
        FrameSpec,
        compute_energy_spectrum,
    )
    model = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 5.0e9,
        omega_q=2 * np.pi * 6.0e9,
        alpha=2 * np.pi * (-200.0e6),
        chi=2 * np.pi * (-2.84e6),
        kerr=2 * np.pi * (-2.0e3),
        n_cav=4,
        n_tr=2,
    )
    return compute_energy_spectrum(model, frame=FrameSpec(), levels=6)


def _make_bloch_track(n_gates: int = 3):
    """Build a minimal bloch-track dict that plot_bloch_track accepts."""
    indices = list(range(n_gates))
    snapshots = [
        {"index": i, "gate_type": "Rotation", "top_label": f"R{i}"}
        for i in indices
    ]
    return {
        "indices": indices,
        "x": [0.0] * n_gates,
        "y": [0.0] * n_gates,
        "z": [1.0] * n_gates,
        "snapshots": snapshots,
    }


# ---- plot_energy_levels ----

class TestPlotEnergyLevels:
    def test_returns_figure(self):
        spectrum = _make_energy_spectrum()
        fig = plot_energy_levels(spectrum)
        assert isinstance(fig, plt.Figure)

    def test_max_levels_limits_plot(self):
        spectrum = _make_energy_spectrum()
        fig = plot_energy_levels(spectrum, max_levels=3)
        assert isinstance(fig, plt.Figure)

    def test_custom_energy_scale_and_label(self):
        spectrum = _make_energy_spectrum()
        fig = plot_energy_levels(
            spectrum,
            energy_scale=1.0 / (2.0 * np.pi * 1.0e6),
            energy_unit_label="MHz",
        )
        assert isinstance(fig, plt.Figure)

    def test_annotate_false(self):
        spectrum = _make_energy_spectrum()
        fig = plot_energy_levels(spectrum, annotate=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self):
        spectrum = _make_energy_spectrum()
        fig = plot_energy_levels(spectrum, title="Test Title")
        assert isinstance(fig, plt.Figure)

    def test_with_provided_ax(self):
        spectrum = _make_energy_spectrum()
        _, ax = plt.subplots()
        fig = plot_energy_levels(spectrum, ax=ax)
        assert isinstance(fig, plt.Figure)


# ---- plot_bloch_track ----

class TestPlotBlochTrack:
    def test_returns_figure(self):
        track = _make_bloch_track(n_gates=4)
        fig = plot_bloch_track(track, title="Smoke test", label_stride=1)
        assert isinstance(fig, plt.Figure)

    def test_longer_sequence(self):
        track = _make_bloch_track(n_gates=8)
        fig = plot_bloch_track(track, title="Long track", label_stride=2)
        assert isinstance(fig, plt.Figure)


# ---- save_figure (smoke) ----

class TestSaveFigure:
    def test_save_figure_writes_file(self, tmp_path):
        spectrum = _make_energy_spectrum()
        fig = plot_energy_levels(spectrum)
        result = save_figure(fig, tmp_path, "output.png")
        # save_figure returns a Path or None; if it returns a Path the file must exist
        if result is not None:
            assert result.exists()
            assert result.stat().st_size > 0


# ---- plot_sqr_calibration_result ----


class TestPlotSQRCalibrationResult:
    def _make_result(self, max_n: int = 4):
        from cqed_sim.calibration.sqr import SQRCalibrationResult
        return SQRCalibrationResult(
            sqr_name="SQR_test",
            max_n=max_n,
            d_lambda=[0.01 * (i + 1) for i in range(max_n + 1)],
            d_alpha=[0.005 * i for i in range(max_n + 1)],
            d_omega_rad_s=[2 * np.pi * 100.0 * i for i in range(max_n + 1)],
            theta_target=[np.pi / 2] * (max_n + 1),
            phi_target=[0.0] * (max_n + 1),
            initial_loss=[0.1] * (max_n + 1),
            optimized_loss=[1e-4] * (max_n + 1),
        )

    def test_returns_figure(self):
        result = self._make_result()
        fig = plot_sqr_calibration_result(result)
        assert isinstance(fig, plt.Figure)

    def test_single_level(self):
        result = self._make_result(max_n=0)
        fig = plot_sqr_calibration_result(result)
        assert isinstance(fig, plt.Figure)
