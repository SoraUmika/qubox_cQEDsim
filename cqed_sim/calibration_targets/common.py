from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CalibrationResult:
    fitted_parameters: dict[str, float]
    uncertainties: dict[str, float]
    raw_data: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


def _argmax_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(y))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if 0 < idx < x.size - 1:
        x_window = x[idx - 1 : idx + 2]
        y_window = y[idx - 1 : idx + 2]
        coeffs = np.polyfit(x_window, y_window, 2)
        if abs(coeffs[0]) > 1.0e-15:
            peak = -0.5 * coeffs[1] / coeffs[0]
            curvature = abs(coeffs[0])
            return float(peak), float(np.sqrt(max(1.0e-24, 1.0 / curvature)) * abs(x[1] - x[0]))
    spacing = abs(x[1] - x[0]) if x.size > 1 else 0.0
    return float(x[idx]), float(spacing)


__all__ = ["CalibrationResult", "_argmax_peak"]
