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


def sqr_lambda0_rad_s(duration_s: float) -> float:
    duration = float(duration_s)
    if abs(duration) <= 1.0e-15:
        return 0.0
    return float(np.pi / (2.0 * duration))


def sqr_rotation_coefficient(theta: float, d_lambda_norm: float) -> float:
    # Lab-aligned additive convention in normalized coefficient space:
    #   s_n = theta_n/pi + d_lambda_norm_n
    # where d_lambda_norm = d_lambda_abs / lambda0.
    return float(theta) / np.pi + float(d_lambda_norm)


def sqr_tone_amplitude_rad_s(theta: float, duration_s: float, d_lambda_norm: float = 0.0) -> float:
    # Equivalent physical amplitude (rad/s):
    #   amp_n = lambda0 * s_n = theta/(2T) + lambda0*d_lambda_norm.
    lam0 = sqr_lambda0_rad_s(duration_s)
    return float(lam0 * sqr_rotation_coefficient(theta, d_lambda_norm))


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
    d_lambda_values: list[float] | tuple[float, ...] | None = None,
    include_all_levels: bool = False,
    tone_cutoff: float = 1.0e-10,
) -> list[MultitoneTone]:
    theta, phi = pad_sqr_angles(theta_values, phi_values, model.n_cav)
    d_lambda = (
        np.zeros(model.n_cav, dtype=float)
        if d_lambda_values is None
        else pad_parameter_array(d_lambda_values, model.n_cav)
    )
    tone_specs: list[MultitoneTone] = []
    for n, (theta_n, phi_n, d_lambda_n) in enumerate(zip(theta, phi, d_lambda)):
        amp_n = sqr_tone_amplitude_rad_s(float(theta_n), duration_s, d_lambda_norm=float(d_lambda_n))
        if (
            not include_all_levels
            and abs(amp_n) <= tone_cutoff
            and abs(theta_n) <= tone_cutoff
            and abs(d_lambda_n) <= tone_cutoff
        ):
            continue
        # Canonical waveform convention uses exp(+i*omega*t), so SQR tone frequency
        # parameter is the negative of manifold_transition_frequency(...), which was
        # previously paired with exp(-i*omega*t).
        omega_waveform = -float(manifold_transition_frequency(model, n, frame=frame))
        tone_specs.append(
            MultitoneTone(
                manifold=int(n),
                omega_rad_s=omega_waveform,
                amp_rad_s=float(amp_n),
                phase_rad=float(phi_n),
            )
        )
    return tone_specs
