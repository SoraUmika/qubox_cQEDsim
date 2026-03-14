from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

ns = 1.0e-9
us = 1.0e-6
ms = 1.0e-3


def hz_to_angular(frequency_hz: float) -> float:
    return float(2.0 * np.pi * float(frequency_hz))


def angular_to_hz(angular_frequency: float) -> float:
    return float(float(angular_frequency) / (2.0 * np.pi))


def khz(value: float) -> float:
    return hz_to_angular(float(value) * 1.0e3)


def MHz(value: float) -> float:
    return hz_to_angular(float(value) * 1.0e6)


def GHz(value: float) -> float:
    return hz_to_angular(float(value) * 1.0e9)


def angular_to_mhz(angular_frequency: float) -> float:
    return angular_to_hz(angular_frequency) / 1.0e6


def angular_to_ghz(angular_frequency: float) -> float:
    return angular_to_hz(angular_frequency) / 1.0e9


def final_expectation(result, key: str) -> float:
    return float(np.real(np.asarray(result.expectations[key], dtype=np.complex128)[-1]))


@dataclass(frozen=True)
class FitResult:
    name: str
    parameters: dict[str, float]
    covariance: np.ndarray | None = None
    model_y: np.ndarray | None = None


def lorentzian(
    x: np.ndarray,
    center: float,
    width: float,
    amplitude: float,
    offset: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    gamma = max(abs(float(width)), 1.0e-15)
    return float(offset) + float(amplitude) / (1.0 + ((x - float(center)) / gamma) ** 2)


def fit_lorentzian_peak(
    x: np.ndarray,
    y: np.ndarray,
    *,
    p0: tuple[float, float, float, float] | None = None,
) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if p0 is None:
        peak_index = int(np.argmax(y))
        p0 = (
            float(x[peak_index]),
            float(max(1.0e-15, 0.05 * max(np.ptp(x), 1.0e-12))),
            float(max(y) - min(y)),
            float(min(y)),
        )
    popt, pcov = curve_fit(lorentzian, x, y, p0=p0, maxfev=20000)
    model_y = lorentzian(x, *popt)
    return FitResult(
        name="lorentzian",
        parameters={
            "center": float(popt[0]),
            "width": float(abs(popt[1])),
            "amplitude": float(popt[2]),
            "offset": float(popt[3]),
        },
        covariance=np.asarray(pcov, dtype=float),
        model_y=np.asarray(model_y, dtype=float),
    )


def rabi_vs_amplitude(
    amplitudes: np.ndarray,
    omega_scale: float,
    duration: float,
    offset: float,
) -> np.ndarray:
    amplitudes = np.asarray(amplitudes, dtype=float)
    return float(offset) + np.sin(0.5 * float(omega_scale) * amplitudes * float(duration)) ** 2


def fit_rabi_vs_amplitude(
    amplitudes: np.ndarray,
    populations: np.ndarray,
    *,
    duration: float,
    p0: tuple[float, float] = (1.0, 0.0),
) -> FitResult:
    amplitudes = np.asarray(amplitudes, dtype=float)
    populations = np.asarray(populations, dtype=float)
    popt, pcov = curve_fit(
        lambda amp, omega_scale, offset: rabi_vs_amplitude(amp, omega_scale, duration, offset),
        amplitudes,
        populations,
        p0=p0,
        maxfev=20000,
    )
    model_y = rabi_vs_amplitude(amplitudes, popt[0], duration, popt[1])
    pi_amplitude = float(np.pi / (float(popt[0]) * float(duration)))
    return FitResult(
        name="rabi_vs_amplitude",
        parameters={
            "omega_scale": float(popt[0]),
            "offset": float(popt[1]),
            "duration": float(duration),
            "pi_amplitude": pi_amplitude,
            "pi_over_two_amplitude": 0.5 * pi_amplitude,
        },
        covariance=np.asarray(pcov, dtype=float),
        model_y=np.asarray(model_y, dtype=float),
    )


def rabi_vs_duration(
    durations: np.ndarray,
    omega_rabi: float,
    amplitude: float,
    offset: float,
    phase: float,
) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    return float(offset) + float(amplitude) * np.sin(0.5 * float(omega_rabi) * durations + float(phase)) ** 2


def fit_rabi_vs_duration(
    durations: np.ndarray,
    populations: np.ndarray,
    *,
    p0: tuple[float, float, float, float] = (2.0 * np.pi * 10.0e6, 1.0, 0.0, 0.0),
) -> FitResult:
    durations = np.asarray(durations, dtype=float)
    populations = np.asarray(populations, dtype=float)
    popt, pcov = curve_fit(rabi_vs_duration, durations, populations, p0=p0, maxfev=20000)
    model_y = rabi_vs_duration(durations, *popt)
    pi_time = float(np.pi / float(popt[0]))
    return FitResult(
        name="rabi_vs_duration",
        parameters={
            "omega_rabi": float(popt[0]),
            "amplitude": float(popt[1]),
            "offset": float(popt[2]),
            "phase": float(popt[3]),
            "pi_time_s": pi_time,
            "pi_over_two_time_s": 0.5 * pi_time,
        },
        covariance=np.asarray(pcov, dtype=float),
        model_y=np.asarray(model_y, dtype=float),
    )


def exponential_decay(
    times: np.ndarray,
    t_const: float,
    amplitude: float,
    offset: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return float(offset) + float(amplitude) * np.exp(-times / float(t_const))


def fit_exponential_decay(
    times: np.ndarray,
    values: np.ndarray,
    *,
    p0: tuple[float, float, float] | None = None,
    parameter_name: str = "t_const",
) -> FitResult:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if p0 is None:
        p0 = (float(max(np.median(times), 1.0e-9)), float(values[0] - values[-1]), float(values[-1]))
    popt, pcov = curve_fit(exponential_decay, times, values, p0=p0, maxfev=20000)
    model_y = exponential_decay(times, *popt)
    return FitResult(
        name="exponential_decay",
        parameters={
            parameter_name: float(popt[0]),
            "amplitude": float(popt[1]),
            "offset": float(popt[2]),
        },
        covariance=np.asarray(pcov, dtype=float),
        model_y=np.asarray(model_y, dtype=float),
    )


def ramsey_signal(
    delays: np.ndarray,
    detuning: float,
    t2_star: float,
    amplitude: float,
    offset: float,
    phase: float,
) -> np.ndarray:
    delays = np.asarray(delays, dtype=float)
    return float(offset) + float(amplitude) * np.exp(-delays / float(t2_star)) * np.cos(float(detuning) * delays + float(phase))


def fit_ramsey_signal(
    delays: np.ndarray,
    populations: np.ndarray,
    *,
    p0: tuple[float, float, float, float, float] = (2.0 * np.pi * 0.5e6, 20.0e-6, 0.5, 0.5, 0.0),
) -> FitResult:
    delays = np.asarray(delays, dtype=float)
    populations = np.asarray(populations, dtype=float)
    popt, pcov = curve_fit(ramsey_signal, delays, populations, p0=p0, maxfev=30000)
    model_y = ramsey_signal(delays, *popt)
    return FitResult(
        name="ramsey_signal",
        parameters={
            "detuning": float(popt[0]),
            "t2_star": float(popt[1]),
            "amplitude": float(popt[2]),
            "offset": float(popt[3]),
            "phase": float(popt[4]),
        },
        covariance=np.asarray(pcov, dtype=float),
        model_y=np.asarray(model_y, dtype=float),
    )


def echo_signal(
    delays: np.ndarray,
    t2_echo: float,
    amplitude: float,
    offset: float,
) -> np.ndarray:
    delays = np.asarray(delays, dtype=float)
    return float(offset) + float(amplitude) * np.exp(-delays / float(t2_echo))


def fit_echo_signal(
    delays: np.ndarray,
    populations: np.ndarray,
    *,
    p0: tuple[float, float, float] = (30.0e-6, 0.5, 0.5),
) -> FitResult:
    delays = np.asarray(delays, dtype=float)
    populations = np.asarray(populations, dtype=float)
    popt, pcov = curve_fit(echo_signal, delays, populations, p0=p0, maxfev=20000)
    model_y = echo_signal(delays, *popt)
    return FitResult(
        name="echo_signal",
        parameters={
            "t2_echo": float(popt[0]),
            "amplitude": float(popt[1]),
            "offset": float(popt[2]),
        },
        covariance=np.asarray(pcov, dtype=float),
        model_y=np.asarray(model_y, dtype=float),
    )


def coherent_alpha_from_state_moment(moment_a: complex) -> tuple[float, float]:
    alpha = complex(moment_a)
    return float(np.real(alpha)), float(np.imag(alpha))


__all__ = [
    "FitResult",
    "GHz",
    "MHz",
    "angular_to_ghz",
    "angular_to_hz",
    "angular_to_mhz",
    "coherent_alpha_from_state_moment",
    "echo_signal",
    "exponential_decay",
    "final_expectation",
    "fit_echo_signal",
    "fit_exponential_decay",
    "fit_lorentzian_peak",
    "fit_rabi_vs_amplitude",
    "fit_rabi_vs_duration",
    "fit_ramsey_signal",
    "hz_to_angular",
    "khz",
    "lorentzian",
    "ms",
    "ns",
    "rabi_vs_amplitude",
    "rabi_vs_duration",
    "ramsey_signal",
    "us",
]
