from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import manifold_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse


def landgraf_envelope(t_rel: np.ndarray) -> np.ndarray:
    """Smooth envelope used for the selective slow stage (sin^2 window)."""
    x = np.sin(np.pi * np.clip(t_rel, 0.0, 1.0)) ** 2
    return x.astype(np.complex128)


@dataclass
class SnapToneParameters:
    amplitudes: np.ndarray
    detunings: np.ndarray
    phases: np.ndarray

    @staticmethod
    def vanilla(target_phases: np.ndarray) -> "SnapToneParameters":
        n = target_phases.size
        return SnapToneParameters(
            amplitudes=np.ones(n, dtype=float),
            detunings=np.zeros(n, dtype=float),
            phases=np.asarray(target_phases, dtype=float).copy(),
        )


def slow_stage_multitone_pulse(
    model: DispersiveTransmonCavityModel,
    target_phases: np.ndarray,
    params: SnapToneParameters,
    duration: float,
    base_amp: float,
    frame: FrameSpec | None = None,
    channel: str = "q",
) -> Pulse:
    frame = frame or FrameSpec(omega_q_frame=model.omega_q)
    n_max = target_phases.size
    omegas = np.array([manifold_transition_frequency(model, n, frame=frame) for n in range(n_max)], dtype=float)

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        t = t_rel * duration
        env = landgraf_envelope(t_rel)
        tones = np.zeros_like(t, dtype=np.complex128)
        for n in range(n_max):
            tones += params.amplitudes[n] * np.exp(
                -1j * ((omegas[n] + params.detunings[n]) * t - params.phases[n])
            )
        return base_amp * env * tones

    return Pulse(channel=channel, t0=0.0, duration=duration, envelope=envelope, carrier=0.0, phase=0.0, amp=1.0, drag=0.0)


def has_only_allowed_tones(
    coeff: np.ndarray,
    dt: float,
    allowed_omegas: np.ndarray,
    envelope: np.ndarray | None = None,
    tol_ratio: float = 1e-2,
) -> bool:
    if envelope is not None:
        env = np.asarray(envelope, dtype=np.complex128)
        mask = np.abs(env) > 1e-8 * np.max(np.abs(env))
        sig = np.zeros_like(coeff, dtype=np.complex128)
        sig[mask] = coeff[mask] / env[mask]
    else:
        sig = coeff
    n = sig.size
    if np.max(np.abs(sig)) == 0.0:
        return True
    t = np.arange(n, dtype=float) * dt
    basis = []
    for w in allowed_omegas:
        basis.append(np.exp(-1j * w * t))
    for w in allowed_omegas:
        basis.append(np.exp(1j * w * t))
    b = np.column_stack(basis)
    coef, *_ = np.linalg.lstsq(b, sig, rcond=None)
    rec = b @ coef
    rel = np.linalg.norm(sig - rec) / max(np.linalg.norm(sig), 1e-15)
    return rel < max(tol_ratio, 0.25)
