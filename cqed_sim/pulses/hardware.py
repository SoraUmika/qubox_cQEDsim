from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardwareConfig:
    lo_freq: float = 0.0
    if_freq: float = 0.0
    gain_i: float = 1.0
    gain_q: float = 1.0
    quadrature_skew: float = 0.0
    dc_i: float = 0.0
    dc_q: float = 0.0
    image_leakage: float = 0.0
    channel_gain: float = 1.0
    zoh_samples: int = 1
    lowpass_bw: float | None = None
    detuning: float = 0.0
    timing_quantum: float | None = None
    amplitude_bits: int | None = None


def apply_timing_quantization(t0: float, quantum: float | None) -> float:
    if quantum is None:
        return t0
    # Round to nearest sample with half-up rule.
    return np.floor(t0 / quantum + 0.5) * quantum


def apply_zoh(x: np.ndarray, zoh_samples: int) -> np.ndarray:
    if zoh_samples <= 1:
        return x.copy()
    y = x.copy()
    for i in range(0, len(x), zoh_samples):
        y[i : i + zoh_samples] = x[i]
    return y


def apply_first_order_lowpass(x: np.ndarray, dt: float, bw: float | None) -> np.ndarray:
    if bw is None or bw <= 0.0:
        return x.copy()
    tau = 1.0 / (2.0 * np.pi * bw)
    alpha = dt / (tau + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def apply_amplitude_quantization(x: np.ndarray, bits: int | None) -> np.ndarray:
    if bits is None or bits <= 0:
        return x.copy()
    levels = 2**bits
    max_amp = max(float(np.max(np.abs(x.real))), float(np.max(np.abs(x.imag))), 1e-15)
    step = 2.0 * max_amp / (levels - 1)
    qr = np.round(x.real / step) * step
    qi = np.round(x.imag / step) * step
    return (qr + 1j * qi).astype(np.complex128)


def apply_iq_distortion(baseband: np.ndarray, t: np.ndarray, hw: HardwareConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return distorted complex envelope and physical RF waveform."""
    i = baseband.real
    q = baseband.imag
    i1 = hw.gain_i * i + hw.dc_i
    q1 = hw.gain_q * q + hw.dc_q
    q2 = q1 * np.cos(hw.quadrature_skew) + i1 * np.sin(hw.quadrature_skew)
    distorted = hw.channel_gain * (i1 + 1j * q2)
    if hw.image_leakage != 0.0:
        image = hw.image_leakage * np.conj(distorted) * np.exp(-2j * hw.if_freq * t)
        distorted = distorted + image
    if hw.detuning != 0.0:
        distorted = distorted * np.exp(1j * hw.detuning * t)
    omega_lo = hw.lo_freq
    rf = distorted.real * np.cos(omega_lo * t) - distorted.imag * np.sin(omega_lo * t)
    return distorted, rf


def image_ratio_db(gain_i: float, gain_q: float) -> float:
    """Approximate image suppression from gain mismatch only."""
    eps = (gain_i - gain_q) / (gain_i + gain_q)
    ratio = np.clip(np.abs(eps), 1e-15, 1.0)
    return 20.0 * np.log10(ratio)
