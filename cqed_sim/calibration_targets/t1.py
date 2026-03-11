from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .common import CalibrationResult


def _decay_curve(times: np.ndarray, t1: float) -> np.ndarray:
    return np.exp(-np.asarray(times, dtype=float) / float(t1))


def run_t1(
    model,
    delays: np.ndarray,
    *,
    t1: float = 30.0e-6,
) -> CalibrationResult:
    """Fit an energy-relaxation curve."""

    delays = np.asarray(delays, dtype=float)
    populations = _decay_curve(delays, float(t1))
    popt, pcov = curve_fit(_decay_curve, delays, populations, p0=(float(t1),))
    sigma = np.sqrt(np.diag(pcov)) if pcov.size else np.zeros(1, dtype=float)
    return CalibrationResult(
        fitted_parameters={"t1": float(popt[0])},
        uncertainties={"t1": float(sigma[0])},
        raw_data={"delays": delays, "excited_population": populations},
        metadata={"omega_q": float(model.omega_q)},
    )


__all__ = ["run_t1"]
