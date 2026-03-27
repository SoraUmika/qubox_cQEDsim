from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .common import CalibrationResult


def _ramsey_curve(delays: np.ndarray, detuning: float, t2_star: float) -> np.ndarray:
    delays = np.asarray(delays, dtype=float)
    return 0.5 * (1.0 + np.exp(-delays / float(t2_star)) * np.cos(float(detuning) * delays))


def run_ramsey(
    model,
    delays: np.ndarray,
    *,
    detuning: float,
    t2_star: float = 20.0e-6,
) -> CalibrationResult:
    """Fit detuning and ``T2*`` from a Ramsey fringe envelope."""

    delays = np.asarray(delays, dtype=float)
    populations = _ramsey_curve(delays, float(detuning), float(t2_star))
    popt, pcov = curve_fit(
        _ramsey_curve,
        delays,
        populations,
        p0=(float(detuning), float(t2_star)),
    )
    sigma = np.sqrt(np.diag(pcov)) if pcov.size else np.zeros(2, dtype=float)
    return CalibrationResult(
        fitted_parameters={"delta_omega": float(popt[0]), "t2_star": float(popt[1])},
        uncertainties={"delta_omega": float(sigma[0]), "t2_star": float(sigma[1])},
        raw_data={"delays": delays, "excited_population": populations},
        metadata={"omega_q": float(model.omega_q)},
    )


__all__ = ["run_ramsey"]
