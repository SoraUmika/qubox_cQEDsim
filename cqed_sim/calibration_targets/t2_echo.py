from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .common import CalibrationResult


def _echo_curve(delays: np.ndarray, t2_echo: float) -> np.ndarray:
    return 0.5 * (1.0 + np.exp(-np.asarray(delays, dtype=float) / float(t2_echo)))


def run_t2_echo(
    model,
    delays: np.ndarray,
    *,
    t2_echo: float = 40.0e-6,
) -> CalibrationResult:
    """Fit a Hahn-echo coherence envelope."""

    delays = np.asarray(delays, dtype=float)
    populations = _echo_curve(delays, float(t2_echo))
    popt, pcov = curve_fit(_echo_curve, delays, populations, p0=(float(t2_echo),))
    sigma = np.sqrt(np.diag(pcov)) if pcov.size else np.zeros(1, dtype=float)
    return CalibrationResult(
        fitted_parameters={"t2_echo": float(popt[0])},
        uncertainties={"t2_echo": float(sigma[0])},
        raw_data={"delays": delays, "excited_population": populations},
        metadata={"omega_q": float(model.omega_q)},
    )


__all__ = ["run_t2_echo"]
