from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


EnvelopeFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Pulse:
    """A single pulsed drive on one channel.

    The sampled waveform at time ``t`` is::

        amp * envelope(t_rel) * exp(i * (carrier * t + phase))

    where ``t_rel = (t - t0) / duration`` is the normalised time within [0, 1).

    Carrier sign convention
    -----------------------
    ``carrier`` stores the **angular frequency of the drive tone**
    (negative for a tone that rotates in the counter-clockwise direction in the
    rotating frame).  When targeting a transition at angular frequency
    ``omega_transition`` (relative to the rotating frame defined by
    :class:`~cqed_sim.core.frame.FrameSpec`), the carrier should be set to::

        carrier = -omega_transition(frame)

    That is, the carrier is the **negative** of the transition frequency in the
    rotating frame.  This sign convention ensures the drive Hamiltonian term
    (proportional to ``exp(i * carrier * t)``) rotates at the same rate as the
    transmon or cavity transition, producing a resonant interaction in the
    doubly-rotating frame.

    For on-resonance qubit drives in the rotating frame defined by
    ``omega_q_frame = omega_q``, the residual detuning is zero and ``carrier = 0``
    is the correct choice.  For off-resonance drives set ``carrier`` to the
    detuning (negative sign included).

    The library is unit-coherent: it does not enforce specific physical units
    for frequencies or times. Any internally consistent unit system is valid
    (for example, rad/s with times in seconds, or rad/ns with times in
    nanoseconds). The recommended convention used in the main examples and
    calibration function naming is rad/s and seconds. All time and frequency
    fields within a single ``Pulse`` instance must use the same unit system.
    """

    channel: str
    """Drive channel name (e.g. ``'qubit'``, ``'storage'``, ``'sideband'``)."""
    t0: float
    """Pulse start time, in the user's chosen unit system (e.g. seconds)."""
    duration: float
    """Pulse duration, in the same time unit as ``t0``."""
    envelope: EnvelopeFunc | np.ndarray
    """Normalised envelope function ``f(t_rel)`` with ``t_rel`` in [0, 1),
    or a pre-sampled array of complex amplitudes."""
    carrier: float = 0.0
    """Angular frequency of the drive carrier tone, in the user's chosen
    angular-frequency unit (e.g. rad/s).

    Convention: ``carrier = -omega_transition(frame)`` (negative of the
    transition frequency in the rotating frame).  Set to 0 for an on-resonance
    drive in the frame where the transition frequency is already subtracted.
    """
    phase: float = 0.0
    """Initial phase offset of the carrier (rad)."""
    amp: float = 1.0
    """Peak amplitude of the pulse, in angular-frequency units
    (e.g. rad/s for Hamiltonian-strength drives)."""
    drag: float = 0.0
    """DRAG coefficient.  A non-zero value adds the in-phase derivative of the
    envelope as a quadrature component to suppress leakage to higher transmon
    levels."""
    sample_rate: float | None = None
    """Samples per time unit (e.g. samples per second).  Required when
    ``envelope`` is a pre-sampled array."""
    label: str | None = None
    """Optional human-readable label for display and debugging."""

    @property
    def t1(self) -> float:
        return self.t0 + self.duration

    def _sample_analytic(self, t: np.ndarray) -> np.ndarray:
        t_rel = (t - self.t0) / self.duration
        in_support = (t_rel >= 0.0) & (t_rel < 1.0)
        out = np.zeros_like(t, dtype=np.complex128)
        if np.any(in_support):
            env = np.asarray(self.envelope(t_rel[in_support]), dtype=np.complex128)
            if self.drag != 0.0:
                # DRAG-like quadrature from envelope derivative.
                d_env = np.gradient(env.real, t[in_support], edge_order=1)
                env = env + 1j * self.drag * d_env
            phase = np.exp(1j * (self.carrier * t[in_support] + self.phase))
            out[in_support] = self.amp * env * phase
        return out

    def _sample_discrete(self, t: np.ndarray) -> np.ndarray:
        arr = np.asarray(self.envelope, dtype=np.complex128)
        if self.sample_rate is None:
            raise ValueError("sample_rate is required when envelope is sampled.")
        in_support = (t >= self.t0) & (t < self.t1)
        out = np.zeros_like(t, dtype=np.complex128)
        idx = np.floor((t[in_support] - self.t0) * self.sample_rate).astype(int)
        idx = np.clip(idx, 0, arr.size - 1)
        phase = np.exp(1j * (self.carrier * t[in_support] + self.phase))
        out[in_support] = self.amp * arr[idx] * phase
        return out

    def sample(self, t: np.ndarray) -> np.ndarray:
        if callable(self.envelope):
            return self._sample_analytic(t)
        return self._sample_discrete(t)

