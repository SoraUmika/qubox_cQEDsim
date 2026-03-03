from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultitoneTone:
    manifold: int
    omega_rad_s: float
    amp_rad_s: float
    phase_rad: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": int(self.manifold),
            "omega_rad_s": float(self.omega_rad_s),
            "amp_rad_s": float(self.amp_rad_s),
            "phase_rad": float(self.phase_rad),
        }


def square_envelope(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def gaussian_envelope(t_rel: np.ndarray, sigma: float, center: float | None = None) -> np.ndarray:
    center = 0.5 if center is None else center
    return np.exp(-0.5 * ((t_rel - center) / sigma) ** 2).astype(np.complex128)


def cosine_rise_envelope(t_rel: np.ndarray, rise_fraction: float = 0.1) -> np.ndarray:
    y = np.ones_like(t_rel, dtype=float)
    rf = max(1e-6, min(0.49, rise_fraction))
    left = t_rel < rf
    right = t_rel > (1.0 - rf)
    y[left] = 0.5 * (1 - np.cos(np.pi * t_rel[left] / rf))
    y[right] = 0.5 * (1 - np.cos(np.pi * (1.0 - t_rel[right]) / rf))
    return y.astype(np.complex128)


def gaussian_area_fraction(sigma_fraction: float, n_pts: int = 4097) -> float:
    grid = np.linspace(0.0, 1.0, n_pts)
    env = np.asarray(gaussian_envelope(grid, sigma=sigma_fraction), dtype=np.complex128)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(np.real(env), grid))


def normalized_gaussian(t_rel: np.ndarray, sigma_fraction: float) -> np.ndarray:
    base = np.asarray(gaussian_envelope(t_rel, sigma=sigma_fraction), dtype=np.complex128)
    area = gaussian_area_fraction(sigma_fraction)
    return base if abs(area) < 1.0e-12 else base / area


def multitone_gaussian_envelope(
    t_rel: np.ndarray,
    duration_s: float,
    sigma_fraction: float,
    tone_specs: list[MultitoneTone],
) -> np.ndarray:
    env = normalized_gaussian(t_rel, sigma_fraction=sigma_fraction)
    t = t_rel * duration_s
    coeff = np.zeros_like(t, dtype=np.complex128)
    for spec in tone_specs:
        coeff += spec.amp_rad_s * np.exp(1j * spec.phase_rad) * np.exp(-1j * spec.omega_rad_s * t)
    return env * coeff
