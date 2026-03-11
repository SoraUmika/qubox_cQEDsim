from __future__ import annotations

import numpy as np

from .common import CalibrationResult


def run_drag_tuning(
    model,
    drag_values: np.ndarray,
    *,
    optimal_drag: float | None = None,
    baseline_leakage: float = 1.0e-3,
    curvature: float = 0.25,
) -> CalibrationResult:
    """Fit the DRAG value minimizing leakage."""

    drag_values = np.asarray(drag_values, dtype=float)
    if optimal_drag is None:
        optimal_drag = float(0.0 if model.alpha == 0.0 else -1.0 / model.alpha)
    leakage = float(baseline_leakage) + float(curvature) * (drag_values - float(optimal_drag)) ** 2
    coeffs, covariance = np.polyfit(drag_values, leakage, 2, cov=True)
    fitted_optimal = float(-0.5 * coeffs[1] / coeffs[0]) if abs(coeffs[0]) > 1.0e-15 else float(optimal_drag)
    sigma = float(np.sqrt(np.diag(covariance))[1]) if covariance.size else 0.0
    return CalibrationResult(
        fitted_parameters={"drag_optimal": fitted_optimal},
        uncertainties={"drag_optimal": sigma},
        raw_data={"drag_values": drag_values, "leakage": leakage},
        metadata={"baseline_leakage": float(baseline_leakage), "curvature": float(curvature)},
    )


__all__ = ["run_drag_tuning"]
