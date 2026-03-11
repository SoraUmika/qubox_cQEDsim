from __future__ import annotations

import numpy as np

from .common import CalibrationResult, _argmax_peak


def _lorentzian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    gamma = max(float(width), 1.0e-15)
    return 1.0 / (1.0 + ((np.asarray(x, dtype=float) - float(center)) / gamma) ** 2)


def run_spectroscopy(
    model,
    drive_frequencies: np.ndarray,
    *,
    linewidth: float | None = None,
    excited_state_fraction: float = 0.0,
) -> CalibrationResult:
    """Estimate ``omega_01`` and ``omega_12`` from effective spectroscopy sweeps."""

    drive_frequencies = np.asarray(drive_frequencies, dtype=float)
    omega_01 = float(model.omega_q)
    omega_12 = float(model.omega_q + model.alpha)
    width = abs(float(model.alpha)) / 10.0 if linewidth is None else abs(float(linewidth))

    ground_trace = _lorentzian(drive_frequencies, omega_01, width)
    excited_trace = _lorentzian(drive_frequencies, omega_12, width)
    signal = (1.0 - float(excited_state_fraction)) * ground_trace + float(excited_state_fraction) * excited_trace

    omega_01_fit, omega_01_err = _argmax_peak(drive_frequencies, ground_trace)
    omega_12_fit, omega_12_err = _argmax_peak(drive_frequencies, excited_trace)
    return CalibrationResult(
        fitted_parameters={"omega_01": float(omega_01_fit), "omega_12": float(omega_12_fit)},
        uncertainties={"omega_01": float(omega_01_err), "omega_12": float(omega_12_err)},
        raw_data={
            "drive_frequencies": drive_frequencies,
            "response": signal,
            "ground_response": ground_trace,
            "excited_response": excited_trace,
        },
        metadata={"linewidth": float(width)},
    )


__all__ = ["run_spectroscopy"]
