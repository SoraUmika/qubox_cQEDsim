from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import qutip as qt


TransferFunction = Callable[[np.ndarray], np.ndarray] | Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class OutputSignal:
    times: np.ndarray
    field: np.ndarray
    I: np.ndarray
    Q: np.ndarray

    @property
    def complex_iq(self) -> np.ndarray:
        return self.I + 1j * self.Q


def output_operator(
    model: object,
    *,
    kappa_r: float = 0.0,
    kappa_f: float = 0.0,
    filter_present: bool | None = None,
) -> qt.Qobj:
    """Select the input-output operator.

    Without a filter, the output field is `sqrt(kappa_r) a`.  With an explicit
    filter, it is `sqrt(kappa_f) f`.
    """

    ops = model.operators()  # type: ignore[attr-defined]
    has_filter = ("f" in ops) if filter_present is None else bool(filter_present)
    if has_filter:
        if "f" not in ops:
            raise ValueError("filter_present=True but the model has no filter operator 'f'.")
        return np.sqrt(max(float(kappa_f), 0.0)) * ops["f"]
    if "a" not in ops:
        raise ValueError("The model has no readout operator 'a'.")
    return np.sqrt(max(float(kappa_r), 0.0)) * ops["a"]


def output_from_states(
    states: Sequence[qt.Qobj],
    out_op: qt.Qobj,
) -> np.ndarray:
    values = []
    for state in states:
        rho = state if state.isoper else state.proj()
        values.append(complex((out_op * rho).tr()))
    return np.asarray(values, dtype=np.complex128)


def output_from_expectations(
    lowering_mean: Sequence[complex],
    *,
    kappa: float,
) -> np.ndarray:
    return np.sqrt(max(float(kappa), 0.0)) * np.asarray(lowering_mean, dtype=np.complex128)


def apply_transfer_function(
    samples: Sequence[complex],
    *,
    dt: float,
    transfer: TransferFunction | None = None,
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


def build_output_signal(
    times: Sequence[float],
    field: Sequence[complex],
    *,
    transfer: TransferFunction | None = None,
) -> OutputSignal:
    t = np.asarray(times, dtype=float)
    values = np.asarray(field, dtype=np.complex128)
    if t.size != values.size:
        raise ValueError("times and field must have the same length.")
    if t.size > 1 and transfer is not None:
        values = apply_transfer_function(values, dt=float(t[1] - t[0]), transfer=transfer)
    return OutputSignal(
        times=t,
        field=values,
        I=np.asarray(np.real(values), dtype=float),
        Q=np.asarray(np.imag(values), dtype=float),
    )


def integrate_iq(
    signal: OutputSignal | Sequence[complex],
    *,
    times: Sequence[float] | None = None,
    kernel: Sequence[complex] | None = None,
) -> complex:
    if isinstance(signal, OutputSignal):
        values = signal.field
        t = signal.times
    else:
        values = np.asarray(signal, dtype=np.complex128)
        if times is None:
            t = np.arange(values.size, dtype=float)
        else:
            t = np.asarray(times, dtype=float)
    if values.size != t.size:
        raise ValueError("signal and times must have the same length.")
    weights = np.ones(values.size, dtype=np.complex128) if kernel is None else np.asarray(kernel, dtype=np.complex128)
    if weights.size != values.size:
        raise ValueError("kernel must match the signal length.")
    return complex(np.trapezoid(values * np.conjugate(weights), x=t))


def linear_pointer_response(
    drive_envelope: Sequence[complex],
    *,
    dt: float,
    kappa: float,
    detuning: float = 0.0,
    initial_alpha: complex = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the standard linear pointer equation.

    Convention: `dot(alpha) = -(kappa/2 + i*Delta) alpha - i epsilon(t)`.
    """

    envelope = np.asarray(drive_envelope, dtype=np.complex128).reshape(-1)
    tlist = np.arange(envelope.size + 1, dtype=float) * float(dt)
    alpha = np.empty(envelope.size + 1, dtype=np.complex128)
    alpha[0] = complex(initial_alpha)
    lam = 0.5 * float(kappa) + 1j * float(detuning)
    for idx, epsilon in enumerate(envelope):
        if abs(lam) <= 1.0e-15:
            alpha[idx + 1] = alpha[idx] - 1j * complex(epsilon) * float(dt)
        else:
            steady = -1j * complex(epsilon) / lam
            decay = np.exp(-lam * float(dt))
            alpha[idx + 1] = steady + (alpha[idx] - steady) * decay
    return tlist, alpha


__all__ = [
    "OutputSignal",
    "TransferFunction",
    "apply_transfer_function",
    "build_output_signal",
    "integrate_iq",
    "linear_pointer_response",
    "output_from_expectations",
    "output_from_states",
    "output_operator",
]
