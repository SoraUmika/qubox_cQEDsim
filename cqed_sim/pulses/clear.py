from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from cqed_sim.pulses.pulse import Pulse


AWGTransfer = Callable[[np.ndarray], np.ndarray] | Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SampledReadoutPulse:
    samples: np.ndarray
    dt: float
    drive_frequency: float
    phase: float = 0.0
    label: str = "readout"

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.complex128).reshape(-1)
        if samples.size < 1:
            raise ValueError("samples must contain at least one value.")
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be positive.")
        self.samples = samples
        self.dt = float(self.dt)
        self.drive_frequency = float(self.drive_frequency)
        self.phase = float(self.phase)

    @property
    def duration(self) -> float:
        return float(self.samples.size * self.dt)

    @property
    def tlist(self) -> np.ndarray:
        return np.arange(self.samples.size + 1, dtype=float) * self.dt

    def to_pulse(self, *, channel: str = "readout", t0: float = 0.0, carrier: float = 0.0) -> Pulse:
        return Pulse(
            channel=str(channel),
            t0=float(t0),
            duration=self.duration,
            envelope=np.asarray(self.samples, dtype=np.complex128),
            carrier=float(carrier),
            phase=float(self.phase),
            amp=1.0,
            sample_rate=1.0 / self.dt,
            label=self.label,
        )


def apply_awg_transfer(
    samples: Sequence[complex],
    *,
    dt: float,
    transfer: AWGTransfer | None = None,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex128)
    if transfer is None:
        return values
    freqs = 2.0 * np.pi * np.fft.fftfreq(values.size, d=float(dt))
    spectrum = np.fft.fft(values)
    try:
        response = transfer(freqs)  # type: ignore[misc]
    except TypeError:
        response = transfer(freqs, spectrum)  # type: ignore[misc]
    return np.fft.ifft(spectrum * np.asarray(response, dtype=np.complex128))


def _n_samples(duration: float, dt: float) -> int:
    if float(duration) <= 0.0:
        raise ValueError("duration must be positive.")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive.")
    return max(1, int(np.round(float(duration) / float(dt))))


def square_readout_seed(
    *,
    amplitude: complex | float,
    duration: float,
    dt: float,
    drive_frequency: float,
    phase: float = 0.0,
    awg_transfer: AWGTransfer | None = None,
) -> SampledReadoutPulse:
    samples = np.full(_n_samples(duration, dt), complex(amplitude) * np.exp(1j * float(phase)), dtype=np.complex128)
    samples = apply_awg_transfer(samples, dt=float(dt), transfer=awg_transfer)
    return SampledReadoutPulse(samples=samples, dt=float(dt), drive_frequency=float(drive_frequency), label="square_readout")


def ramped_readout_seed(
    *,
    amplitude: complex | float,
    duration: float,
    dt: float,
    drive_frequency: float,
    rise_time: float,
    phase: float = 0.0,
    awg_transfer: AWGTransfer | None = None,
) -> SampledReadoutPulse:
    n = _n_samples(duration, dt)
    t = np.arange(n, dtype=float) * float(dt)
    envelope = np.ones(n, dtype=float)
    rise = max(float(rise_time), 0.0)
    if rise > 0.0:
        rise_mask = t < rise
        fall_mask = t > float(duration) - rise
        envelope[rise_mask] = 0.5 * (1.0 - np.cos(np.pi * t[rise_mask] / rise))
        envelope[fall_mask] = 0.5 * (1.0 - np.cos(np.pi * (float(duration) - t[fall_mask]) / rise))
    samples = complex(amplitude) * envelope * np.exp(1j * float(phase))
    samples = apply_awg_transfer(samples, dt=float(dt), transfer=awg_transfer)
    return SampledReadoutPulse(samples=samples, dt=float(dt), drive_frequency=float(drive_frequency), label="ramped_readout")


def gaussian_readout_seed(
    *,
    amplitude: complex | float,
    duration: float,
    dt: float,
    drive_frequency: float,
    sigma_fraction: float = 0.18,
    phase: float = 0.0,
    awg_transfer: AWGTransfer | None = None,
) -> SampledReadoutPulse:
    n = _n_samples(duration, dt)
    x = (np.arange(n, dtype=float) + 0.5) / float(n)
    sigma = max(float(sigma_fraction), 1.0e-6)
    envelope = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    envelope /= max(float(np.max(envelope)), 1.0e-30)
    samples = complex(amplitude) * envelope * np.exp(1j * float(phase))
    samples = apply_awg_transfer(samples, dt=float(dt), transfer=awg_transfer)
    return SampledReadoutPulse(samples=samples, dt=float(dt), drive_frequency=float(drive_frequency), label="gaussian_readout")


def clear_readout_seed(
    *,
    amplitude: complex | float,
    duration: float,
    dt: float,
    drive_frequency: float,
    kick_fraction: float = 0.15,
    depletion_fraction: float = 0.20,
    kick_amplitude: complex | float | None = None,
    depletion_amplitude: complex | float | None = None,
    phase: float = 0.0,
    awg_transfer: AWGTransfer | None = None,
) -> SampledReadoutPulse:
    """Build a CLEAR-like kick-up, plateau, kick-down/depletion seed."""

    n = _n_samples(duration, dt)
    kick_n = max(1, int(np.round(float(kick_fraction) * n)))
    dep_n = max(1, int(np.round(float(depletion_fraction) * n)))
    if kick_n + dep_n >= n:
        scale = (n - 1) / max(kick_n + dep_n, 1)
        kick_n = max(1, int(np.floor(kick_n * scale)))
        dep_n = max(1, int(np.floor(dep_n * scale)))
    plateau_n = max(0, n - kick_n - dep_n)
    plateau = complex(amplitude)
    kick = complex(1.8 * plateau if kick_amplitude is None else kick_amplitude)
    depletion = complex(-0.75 * plateau if depletion_amplitude is None else depletion_amplitude)
    samples = np.concatenate(
        [
            np.full(kick_n, kick, dtype=np.complex128),
            np.full(plateau_n, plateau, dtype=np.complex128),
            np.full(dep_n, depletion, dtype=np.complex128),
        ]
    )
    if samples.size < n:
        samples = np.concatenate([samples, np.full(n - samples.size, depletion, dtype=np.complex128)])
    samples = samples[:n] * np.exp(1j * float(phase))
    samples = apply_awg_transfer(samples, dt=float(dt), transfer=awg_transfer)
    return SampledReadoutPulse(samples=samples, dt=float(dt), drive_frequency=float(drive_frequency), label="clear_readout")


__all__ = [
    "AWGTransfer",
    "SampledReadoutPulse",
    "apply_awg_transfer",
    "clear_readout_seed",
    "gaussian_readout_seed",
    "ramped_readout_seed",
    "square_readout_seed",
]
