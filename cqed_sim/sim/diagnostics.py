from __future__ import annotations

import numpy as np

from cqed_sim.sequence.scheduler import CompiledSequence


def channel_norms(compiled: CompiledSequence) -> dict[str, float]:
    norms = {}
    for ch, data in compiled.channels.items():
        norms[ch] = float(np.linalg.norm(data.distorted))
    return norms


def instantaneous_phase_frequency(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    phase = np.unwrap(np.angle(signal + 1e-18))
    omega = np.gradient(phase, dt)
    return phase, omega

