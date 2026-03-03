from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


EnvelopeFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Pulse:
    channel: str
    t0: float
    duration: float
    envelope: EnvelopeFunc | np.ndarray
    carrier: float = 0.0
    phase: float = 0.0
    amp: float = 1.0
    drag: float = 0.0
    sample_rate: float | None = None
    label: str | None = None

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

