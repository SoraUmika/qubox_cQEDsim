from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.drive_targets import SidebandDriveSpec
from cqed_sim.core.frequencies import drive_frequency_from_internal_carrier
from cqed_sim.io.gates import DisplacementGate, RotationGate, SQRGate
from cqed_sim.pulses.calibration import (
    build_sqr_tone_specs,
    displacement_square_amplitude,
    rotation_gaussian_amplitude,
)
from cqed_sim.pulses.envelopes import MultitoneTone, multitone_gaussian_envelope, normalized_gaussian, square_envelope
from cqed_sim.pulses.pulse import Pulse

if TYPE_CHECKING:
    from cqed_sim.calibration.sqr import SQRCalibrationResult


def build_displacement_pulse(
    gate: DisplacementGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    duration_s = float(config["duration_displacement_s"])
    eps = displacement_square_amplitude(gate.alpha, duration_s)
    pulse = Pulse(
        "storage",
        0.0,
        duration_s,
        square_envelope,
        amp=float(abs(eps)),
        phase=float(np.angle(eps)) if abs(eps) > 0 else 0.0,
        label=gate.name,
    )
    return [pulse], {"storage": "cavity"}, {
        "mapping": "Square cavity drive with analytic rotating-frame calibration alpha = -i * integral epsilon(t) dt.",
        "duration_s": duration_s,
        "drive_amp": pulse.amp,
        "drive_phase": pulse.phase,
        "target_alpha": gate.alpha,
    }


def build_rotation_pulse(
    gate: RotationGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    duration_s = float(config["duration_rotation_s"])
    sigma_fraction = float(config["rotation_sigma_fraction"])

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return normalized_gaussian(t_rel, sigma_fraction=sigma_fraction)

    pulse = Pulse(
        "qubit",
        0.0,
        duration_s,
        envelope,
        amp=rotation_gaussian_amplitude(gate.theta, duration_s),
        phase=gate.phi,
        label=gate.name,
    )
    return [pulse], {"qubit": "qubit"}, {
        "mapping": "Gaussian qubit drive with analytic RWA calibration theta = 2 * integral Omega(t) dt.",
        "duration_s": duration_s,
        "drive_amp": pulse.amp,
        "drive_phase": pulse.phase,
        "sigma_fraction": sigma_fraction,
    }


def build_sideband_pulse(
    target: SidebandDriveSpec,
    *,
    duration_s: float,
    amplitude_rad_s: float,
    channel: str = "sideband",
    carrier: float = 0.0,
    phase: float = 0.0,
    sigma_fraction: float | None = None,
    label: str | None = None,
) -> tuple[list[Pulse], dict[str, SidebandDriveSpec], dict[str, Any]]:
    if sigma_fraction is None:
        envelope = square_envelope
        envelope_name = "square"
    else:
        sigma = float(sigma_fraction)

        def envelope(t_rel: np.ndarray) -> np.ndarray:
            return normalized_gaussian(t_rel, sigma_fraction=sigma)

        envelope_name = "normalized_gaussian"

    pulse = Pulse(
        str(channel),
        0.0,
        float(duration_s),
        envelope,
        carrier=float(carrier),
        phase=float(phase),
        amp=float(amplitude_rad_s),
        label=label,
    )
    return [pulse], {str(channel): target}, {
        "mapping": "Effective multilevel sideband drive using the structured SidebandDriveSpec target.",
        "duration_s": float(duration_s),
        "amplitude_rad_s": float(amplitude_rad_s),
        "carrier": float(carrier),
        "phase": float(phase),
        "envelope": envelope_name,
        "sideband_target": {
            "mode": str(target.mode),
            "lower_level": int(target.lower_level),
            "upper_level": int(target.upper_level),
            "sideband": str(target.sideband),
        },
    }


def build_sqr_multitone_pulse(
    gate: SQRGate,
    model,
    config: Mapping[str, Any],
    *,
    frame: FrameSpec | None = None,
    calibration: SQRCalibrationResult | None = None,
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    duration_s = float(config["duration_sqr_s"])
    sigma_fraction = float(config["sqr_sigma_fraction"])
    if frame is None:
        if bool(config.get("use_rotating_frame", False)):
            frame = FrameSpec(omega_c_frame=float(model.omega_c), omega_q_frame=float(model.omega_q))
        else:
            frame = FrameSpec()
    d_lambda_values = None if calibration is None else list(calibration.d_lambda)
    raw_tone_specs = build_sqr_tone_specs(
        model=model,
        frame=frame,
        theta_values=list(gate.theta),
        phi_values=list(gate.phi),
        duration_s=duration_s,
        d_lambda_values=d_lambda_values,
        fock_fqs_hz=config.get("fock_fqs_hz"),
        tone_cutoff=float(config["sqr_theta_cutoff"]),
    )
    tone_specs: list[MultitoneTone] = []
    for spec in raw_tone_specs:
        if calibration is None:
            tone_specs.append(spec)
            continue
        _d_lambda, d_alpha, d_omega_rad_s = calibration.correction_for_n(spec.manifold)
        omega_rad_s = float(spec.omega_rad_s + d_omega_rad_s)
        tone_specs.append(
            MultitoneTone(
                manifold=spec.manifold,
                omega_rad_s=omega_rad_s,
                amp_rad_s=float(spec.amp_rad_s),
                phase_rad=float(spec.phase_rad + d_alpha),
                drive_frequency_rad_s=float(
                    drive_frequency_from_internal_carrier(omega_rad_s, frame.omega_q_frame)
                ),
            )
        )

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return multitone_gaussian_envelope(
            t_rel,
            duration_s=duration_s,
            sigma_fraction=sigma_fraction,
            tone_specs=tone_specs,
        )

    pulse = Pulse("qubit", 0.0, duration_s, envelope, amp=1.0, phase=0.0, label=gate.name)
    mapping = (
        "Simplified multitone Gaussian SQR with canonical waveform convention w(t)~exp(+i*omega*t); "
        "tone amplitudes follow additive correction amp_n = theta/(2T) + lambda0*d_lambda_norm "
        "(equivalent to coefficient theta/pi + d_lambda_norm)."
    )
    if calibration is not None:
        mapping += " Applied cached/numerically optimized per-manifold corrections to amplitude, phase, and tone frequency."
    return [pulse], {"qubit": "qubit"}, {
        "mapping": mapping,
        "duration_s": duration_s,
        "drive_amp": None,
        "drive_phase": None,
        "sigma_fraction": sigma_fraction,
        "active_tones": [spec.as_dict() for spec in tone_specs],
        "calibration_applied": calibration is not None,
        "calibration_summary": None if calibration is None else calibration.improvement_summary(),
    }


__all__ = [
    "build_displacement_pulse",
    "build_rotation_pulse",
    "build_sideband_pulse",
    "build_sqr_multitone_pulse",
]
