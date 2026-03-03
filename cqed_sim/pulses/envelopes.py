from __future__ import annotations

import numpy as np


def square_envelope(_: np.ndarray) -> np.ndarray:
    return np.ones_like(_, dtype=np.complex128)


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

