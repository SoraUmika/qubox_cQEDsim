from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .common import CalibrationResult


def _rabi_curve(amplitudes: np.ndarray, omega_scale: float, duration: float) -> np.ndarray:
    return np.sin(0.5 * omega_scale * np.asarray(amplitudes, dtype=float) * float(duration)) ** 2


def run_rabi(
    model,
    amplitudes: np.ndarray,
    *,
    duration: float = 40.0e-9,
    omega_scale: float = 1.0,
) -> CalibrationResult:
    """Fit the Rabi-rate versus amplitude scaling."""

    amplitudes = np.asarray(amplitudes, dtype=float)
    populations = _rabi_curve(amplitudes, float(omega_scale), duration)
    popt, pcov = curve_fit(
        lambda amp, scale: _rabi_curve(amp, scale, duration),
        amplitudes,
        populations,
        p0=(float(omega_scale),),
    )
    return CalibrationResult(
        fitted_parameters={"omega_scale": float(popt[0]), "duration": float(duration)},
        uncertainties={"omega_scale": float(np.sqrt(np.diag(pcov))[0]) if pcov.size else 0.0},
        raw_data={"amplitudes": amplitudes, "excited_population": populations},
        metadata={"omega_q": float(model.omega_q)},
    )


__all__ = ["run_rabi"]
