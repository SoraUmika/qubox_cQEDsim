from __future__ import annotations

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.envelopes import MultitoneTone
from cqed_sim.snap_opt.model import manifold_transition_frequency


def displacement_square_amplitude(alpha: complex, duration_s: float) -> complex:
    return 1j * alpha / duration_s


def rotation_gaussian_amplitude(theta: float, duration_s: float) -> float:
    return float(theta) / (2.0 * duration_s)


def pad_parameter_array(values: list[float] | tuple[float, ...], n_cav_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size > n_cav_dim:
        arr = arr[:n_cav_dim]
    if arr.size < n_cav_dim:
        arr = np.pad(arr, (0, n_cav_dim - arr.size))
    return arr


def pad_sqr_angles(
    theta_values: list[float] | tuple[float, ...],
    phi_values: list[float] | tuple[float, ...],
    n_cav_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    return pad_parameter_array(theta_values, n_cav_dim), pad_parameter_array(phi_values, n_cav_dim)


def build_sqr_tone_specs(
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    theta_values: list[float] | tuple[float, ...],
    phi_values: list[float] | tuple[float, ...],
    duration_s: float,
    tone_cutoff: float = 1.0e-10,
) -> list[MultitoneTone]:
    theta, phi = pad_sqr_angles(theta_values, phi_values, model.n_cav)
    tone_specs: list[MultitoneTone] = []
    for n, (theta_n, phi_n) in enumerate(zip(theta, phi)):
        if abs(theta_n) <= tone_cutoff:
            continue
        tone_specs.append(
            MultitoneTone(
                manifold=int(n),
                omega_rad_s=float(manifold_transition_frequency(model, n, frame=frame)),
                amp_rad_s=rotation_gaussian_amplitude(float(theta_n), duration_s),
                phase_rad=float(phi_n),
            )
        )
    return tone_specs
