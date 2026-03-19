from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from cqed_sim.pulses import Pulse, square_envelope

if TYPE_CHECKING:
    from .problems import ControlTerm


@dataclass(frozen=True)
class PiecewiseConstantTimeGrid:
    step_durations_s: tuple[float, ...]
    t0_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.step_durations_s:
            raise ValueError("PiecewiseConstantTimeGrid requires at least one time slice.")
        durations = tuple(float(value) for value in self.step_durations_s)
        if any(value <= 0.0 for value in durations):
            raise ValueError("All control slice durations must be positive.")
        object.__setattr__(self, "step_durations_s", durations)

    @classmethod
    def uniform(cls, *, steps: int, dt_s: float, t0_s: float = 0.0) -> "PiecewiseConstantTimeGrid":
        if int(steps) <= 0:
            raise ValueError("steps must be positive.")
        return cls(step_durations_s=tuple(float(dt_s) for _ in range(int(steps))), t0_s=float(t0_s))

    @property
    def steps(self) -> int:
        return len(self.step_durations_s)

    @property
    def duration_s(self) -> float:
        return float(np.sum(np.asarray(self.step_durations_s, dtype=float)))

    def boundaries_s(self) -> np.ndarray:
        durations = np.asarray(self.step_durations_s, dtype=float)
        return float(self.t0_s) + np.concatenate(([0.0], np.cumsum(durations)))

    def midpoints_s(self) -> np.ndarray:
        boundaries = self.boundaries_s()
        return 0.5 * (boundaries[:-1] + boundaries[1:])


@dataclass(frozen=True)
class ControlParameterization(ABC):
    time_grid: PiecewiseConstantTimeGrid
    control_terms: tuple["ControlTerm", ...]

    def __post_init__(self) -> None:
        if not self.control_terms:
            raise ValueError(f"{type(self).__name__} requires at least one control term.")
        dim = int(np.asarray(self.control_terms[0].operator).shape[0])
        for term in self.control_terms[1:]:
            if np.asarray(term.operator).shape != (dim, dim):
                raise ValueError("All control terms must have the same matrix shape.")

    @property
    def n_controls(self) -> int:
        return len(self.control_terms)

    @property
    @abstractmethod
    def n_slices(self) -> int:
        raise NotImplementedError

    @property
    def n_time_slices(self) -> int:
        return self.time_grid.steps

    @property
    def hilbert_dim(self) -> int:
        return int(np.asarray(self.control_terms[0].operator).shape[0])

    @property
    def parameter_shape(self) -> tuple[int, int]:
        return (self.n_controls, self.n_slices)

    @property
    def waveform_shape(self) -> tuple[int, int]:
        return (self.n_controls, self.n_time_slices)

    def zero_array(self) -> np.ndarray:
        return np.zeros(self.parameter_shape, dtype=float)

    def zero_schedule(self) -> "ControlSchedule":
        return ControlSchedule(self, self.zero_array())

    def bounds(self) -> tuple[tuple[float, float], ...]:
        bounds: list[tuple[float, float]] = []
        for term in self.control_terms:
            lower, upper = term.amplitude_bounds
            bounds.extend([(float(lower), float(upper))] * self.n_slices)
        return tuple(bounds)

    def flatten(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.shape != self.parameter_shape:
            raise ValueError(f"Control values must have shape {self.parameter_shape}, received {data.shape}.")
        return data.reshape(-1)

    def unflatten(self, vector: np.ndarray) -> np.ndarray:
        data = np.asarray(vector, dtype=float).reshape(-1)
        expected = self.n_controls * self.n_slices
        if data.size != expected:
            raise ValueError(f"Expected flattened control vector of length {expected}, received {data.size}.")
        return data.reshape(self.parameter_shape)

    def clip(self, values: np.ndarray) -> np.ndarray:
        clipped = np.asarray(values, dtype=float).copy()
        if clipped.shape != self.parameter_shape:
            raise ValueError(f"Control values must have shape {self.parameter_shape}, received {clipped.shape}.")
        for term_index, term in enumerate(self.control_terms):
            lower, upper = term.amplitude_bounds
            clipped[term_index, :] = np.clip(clipped[term_index, :], float(lower), float(upper))
        return clipped

    @abstractmethod
    def command_values(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pullback(self, gradient_command: np.ndarray, values: np.ndarray, *, command_values: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def parameterization_metrics(self, values: np.ndarray, command_values: np.ndarray | None = None) -> dict[str, Any]:
        _ = values
        _ = command_values
        return {
            "parameterization": type(self).__name__,
            "parameter_slices": int(self.n_slices),
            "time_slices": int(self.n_time_slices),
            "effective_update_period_s": float(self.time_grid.duration_s / max(self.n_slices, 1)),
        }

    def _channel_coefficients(self, waveform_values: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        data = np.asarray(waveform_values, dtype=float)
        if data.shape != self.waveform_shape:
            raise ValueError(f"Waveform values must have shape {self.waveform_shape}, received {data.shape}.")
        boundaries = self.time_grid.boundaries_s()
        channels: dict[str, Any] = {}
        for term in self.control_terms:
            if term.export_channel is None:
                continue
            channel_state = channels.setdefault(
                str(term.export_channel),
                {
                    "target": term.drive_target,
                    "coefficients": np.zeros(self.n_time_slices, dtype=np.complex128),
                },
            )
            if channel_state["target"] != term.drive_target:
                raise ValueError(
                    f"Export channel '{term.export_channel}' is associated with multiple incompatible drive targets."
                )
        for term_index, term in enumerate(self.control_terms):
            if term.export_channel is None:
                continue
            contribution = np.asarray(data[term_index, :], dtype=np.complex128)
            if term.quadrature.upper() == "Q":
                contribution = -1j * contribution
            channels[str(term.export_channel)]["coefficients"] += contribution
        return channels, boundaries

    def to_pulses(self, values: np.ndarray, *, waveform_values: np.ndarray | None = None) -> tuple[list[Pulse], dict[str, Any], dict[str, Any]]:
        waveform = self.command_values(values) if waveform_values is None else np.asarray(waveform_values, dtype=float)
        channels, boundaries = self._channel_coefficients(waveform)
        durations = np.asarray(self.time_grid.step_durations_s, dtype=float)
        pulses: list[Pulse] = []
        drive_ops: dict[str, Any] = {}
        channel_summary: dict[str, Any] = {}
        for channel_name, payload in channels.items():
            drive_ops[channel_name] = payload["target"]
            coefficients = np.asarray(payload["coefficients"], dtype=np.complex128)
            channel_summary[channel_name] = {
                "max_abs_amp": float(np.max(np.abs(coefficients))) if coefficients.size else 0.0,
                "nonzero_slices": int(np.count_nonzero(np.abs(coefficients) > 1.0e-14)),
            }
            for step_index, coefficient in enumerate(coefficients):
                if abs(coefficient) <= 1.0e-14:
                    continue
                pulses.append(
                    Pulse(
                        channel=channel_name,
                        t0=float(boundaries[step_index]),
                        duration=float(durations[step_index]),
                        envelope=square_envelope,
                        carrier=0.0,
                        phase=float(np.angle(coefficient)),
                        amp=float(abs(coefficient)),
                        label=f"optimal_control_{channel_name}_{step_index}",
                    )
                )
        metadata = {
            "mapping": (
                "Rotating-frame controls exported as square-envelope pulses on the propagation grid. "
                "For repository drive targets, I/Q quadratures are combined into the complex channel coefficient "
                "c(t) = I(t) - i Q(t), which matches cqed_sim.sim.runner Hamiltonian assembly."
            ),
            "parameterization": type(self).__name__,
            "parameter_slices": int(self.n_slices),
            "time_grid_s": [float(value) for value in self.time_grid.step_durations_s],
            "channels": channel_summary,
        }
        return pulses, drive_ops, metadata


@dataclass(frozen=True)
class PiecewiseConstantParameterization(ControlParameterization):
    @property
    def n_slices(self) -> int:
        return self.time_grid.steps

    def command_values(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.shape != self.parameter_shape:
            raise ValueError(f"Control values must have shape {self.parameter_shape}, received {data.shape}.")
        return np.array(data, copy=True)

    def pullback(self, gradient_command: np.ndarray, values: np.ndarray, *, command_values: np.ndarray | None = None) -> np.ndarray:
        _ = values
        _ = command_values
        gradient = np.asarray(gradient_command, dtype=float)
        if gradient.shape != self.waveform_shape:
            raise ValueError(f"Command-waveform gradients must have shape {self.waveform_shape}, received {gradient.shape}.")
        return np.array(gradient, copy=True)


@dataclass(frozen=True)
class HeldSampleParameterization(ControlParameterization):
    sample_period_s: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.sample_period_s) <= 0.0:
            raise ValueError("HeldSampleParameterization.sample_period_s must be positive.")

    @property
    def n_slices(self) -> int:
        duration = float(self.time_grid.duration_s)
        count = int(np.ceil(duration / float(self.sample_period_s) - 1.0e-15))
        return max(count, 1)

    def sample_indices(self) -> np.ndarray:
        starts = self.time_grid.boundaries_s()[:-1] - float(self.time_grid.t0_s)
        indices = np.floor(np.maximum(starts, 0.0) / float(self.sample_period_s) + 1.0e-15).astype(int)
        return np.clip(indices, 0, self.n_slices - 1)

    def command_values(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.shape != self.parameter_shape:
            raise ValueError(f"Control values must have shape {self.parameter_shape}, received {data.shape}.")
        return np.asarray(data[:, self.sample_indices()], dtype=float)

    def pullback(self, gradient_command: np.ndarray, values: np.ndarray, *, command_values: np.ndarray | None = None) -> np.ndarray:
        _ = values
        _ = command_values
        gradient = np.asarray(gradient_command, dtype=float)
        if gradient.shape != self.waveform_shape:
            raise ValueError(f"Command-waveform gradients must have shape {self.waveform_shape}, received {gradient.shape}.")
        reduced = np.zeros(self.parameter_shape, dtype=float)
        for time_index, sample_index in enumerate(self.sample_indices()):
            reduced[:, sample_index] += gradient[:, time_index]
        return reduced

    def parameterization_metrics(self, values: np.ndarray, command_values: np.ndarray | None = None) -> dict[str, Any]:
        metrics = super().parameterization_metrics(values, command_values)
        metrics.update(
            {
                "parameterization": type(self).__name__,
                "sample_period_s": float(self.sample_period_s),
                "hold_ratio": float(self.n_time_slices / max(self.n_slices, 1)),
            }
        )
        return metrics


@dataclass(frozen=True)
class FourierParameterization(ControlParameterization):
    """Truncated real Fourier series parameterization for band-limited controls.

    Represents each control channel as a sum of cosine and sine modes:

        u(t_n) = sum_{k=0}^{K-1} A_k * cos(2*pi*k*t_n/T) + B_k * sin(2*pi*k*t_n/T)

    where T is the total duration and t_n is the midpoint of time slice n.

    Parameters have shape ``(n_controls, 2*n_modes)`` — the first ``n_modes`` columns
    are cosine amplitudes A_k, the last ``n_modes`` columns are sine amplitudes B_k.
    The DC sine (k=0, B_0) is structurally zero and has no effect on the waveform.

    This enforces a hard bandwidth limit: the highest frequency present in the command
    waveform is ``(n_modes - 1) / T`` Hz.  Gradients flow through the linear basis
    evaluation exactly (no approximation).

    Args:
        n_modes: Number of frequency modes, including DC.  Must satisfy
            ``1 <= n_modes <= n_time_slices // 2 + 1`` (Nyquist limit).
    """

    n_modes: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.n_modes) < 1:
            raise ValueError("FourierParameterization.n_modes must be at least 1.")
        max_modes = self.n_time_slices // 2 + 1
        if int(self.n_modes) > max_modes:
            raise ValueError(
                f"FourierParameterization.n_modes={self.n_modes} exceeds the Nyquist limit "
                f"{max_modes} for {self.n_time_slices} time slices."
            )

    @property
    def n_slices(self) -> int:
        return 2 * int(self.n_modes)

    def _basis_matrix(self) -> np.ndarray:
        """Return the ``(2*n_modes, n_time_slices)`` real Fourier basis matrix.

        Row ``k``: cos(2*pi*k*t/T) at each slice midpoint.
        Row ``K+k``: sin(2*pi*k*t/T) at each slice midpoint.
        """
        K = int(self.n_modes)
        T = float(self.time_grid.duration_s)
        t0 = float(self.time_grid.t0_s)
        midpoints = np.asarray(self.time_grid.midpoints_s(), dtype=float) - t0
        basis = np.zeros((2 * K, self.n_time_slices), dtype=float)
        for k in range(K):
            phase = 2.0 * np.pi * k * midpoints / T
            basis[k, :] = np.cos(phase)
            basis[K + k, :] = np.sin(phase)
        return basis

    def command_values(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.shape != self.parameter_shape:
            raise ValueError(f"Control values must have shape {self.parameter_shape}, received {data.shape}.")
        return np.asarray(data @ self._basis_matrix(), dtype=float)

    def pullback(self, gradient_command: np.ndarray, values: np.ndarray, *, command_values: np.ndarray | None = None) -> np.ndarray:
        _ = values
        _ = command_values
        gradient = np.asarray(gradient_command, dtype=float)
        if gradient.shape != self.waveform_shape:
            raise ValueError(f"Command-waveform gradients must have shape {self.waveform_shape}, received {gradient.shape}.")
        return np.asarray(gradient @ self._basis_matrix().T, dtype=float)

    def bounds(self) -> tuple[tuple[float, float], ...]:
        bounds: list[tuple[float, float]] = []
        for term in self.control_terms:
            lower, upper = term.amplitude_bounds
            if np.isfinite(float(lower)) and np.isfinite(float(upper)):
                max_abs = max(abs(float(lower)), abs(float(upper)))
            else:
                max_abs = float("inf")
            bounds.extend([(float(-max_abs), float(max_abs))] * self.n_slices)
        return tuple(bounds)

    def parameterization_metrics(self, values: np.ndarray, command_values: np.ndarray | None = None) -> dict[str, Any]:
        metrics = super().parameterization_metrics(values, command_values)
        T = float(self.time_grid.duration_s)
        K = int(self.n_modes)
        metrics.update(
            {
                "parameterization": "FourierParameterization",
                "n_modes": K,
                "max_frequency_hz": float((K - 1) / T) if K > 1 else 0.0,
                "effective_bandwidth_hz": float(K / T),
            }
        )
        return metrics


@dataclass(frozen=True)
class LinearInterpolatedParameterization(ControlParameterization):
    """Coarse-grid parameterization with linear interpolation onto the propagation grid.

    Stores ``n_control_points`` values at uniformly-spaced nodes spanning the total
    duration [0, T] and linearly interpolates them onto the simulation time-grid midpoints.

    This is smoother than :class:`HeldSampleParameterization` (zero-order hold) while
    being simpler to implement than cubic splines.  Gradients flow through the sparse
    linear interpolation matrix exactly.

    Args:
        n_control_points: Number of coarse control-point nodes (≥ 2).
    """

    n_control_points: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.n_control_points) < 2:
            raise ValueError("LinearInterpolatedParameterization.n_control_points must be at least 2.")

    @property
    def n_slices(self) -> int:
        return int(self.n_control_points)

    def _interpolation_matrix(self) -> np.ndarray:
        """Return the ``(n_time_slices, n_control_points)`` linear interpolation matrix."""
        K = int(self.n_control_points)
        T = float(self.time_grid.duration_s)
        t0 = float(self.time_grid.t0_s)
        midpoints = np.asarray(self.time_grid.midpoints_s(), dtype=float) - t0
        knot_times = np.linspace(0.0, T, K)
        M = np.zeros((self.n_time_slices, K), dtype=float)
        for i, t in enumerate(midpoints):
            t_clipped = float(np.clip(t, 0.0, T))
            k = int(np.searchsorted(knot_times, t_clipped, side="right")) - 1
            k = int(np.clip(k, 0, K - 2))
            dt_knot = float(knot_times[k + 1] - knot_times[k])
            alpha = float((t_clipped - knot_times[k]) / max(dt_knot, 1.0e-18))
            M[i, k] = 1.0 - alpha
            M[i, k + 1] = alpha
        return M

    def command_values(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.shape != self.parameter_shape:
            raise ValueError(f"Control values must have shape {self.parameter_shape}, received {data.shape}.")
        M = self._interpolation_matrix()
        return np.asarray(data @ M.T, dtype=float)

    def pullback(self, gradient_command: np.ndarray, values: np.ndarray, *, command_values: np.ndarray | None = None) -> np.ndarray:
        _ = values
        _ = command_values
        gradient = np.asarray(gradient_command, dtype=float)
        if gradient.shape != self.waveform_shape:
            raise ValueError(f"Command-waveform gradients must have shape {self.waveform_shape}, received {gradient.shape}.")
        M = self._interpolation_matrix()
        return np.asarray(gradient @ M, dtype=float)

    def parameterization_metrics(self, values: np.ndarray, command_values: np.ndarray | None = None) -> dict[str, Any]:
        metrics = super().parameterization_metrics(values, command_values)
        metrics.update(
            {
                "parameterization": "LinearInterpolatedParameterization",
                "n_control_points": int(self.n_control_points),
                "upsampling_factor": float(self.n_time_slices / max(self.n_control_points, 1)),
            }
        )
        return metrics


@dataclass
class ControlSchedule:
    parameterization: ControlParameterization
    values: np.ndarray

    def __post_init__(self) -> None:
        data = np.asarray(self.values, dtype=float)
        if data.shape != self.parameterization.parameter_shape:
            raise ValueError(
                "ControlSchedule.values must match the parameterization shape "
                f"{self.parameterization.parameter_shape}, received {data.shape}."
            )
        self.values = data

    @classmethod
    def from_flattened(cls, parameterization: ControlParameterization, vector: np.ndarray) -> "ControlSchedule":
        return cls(parameterization=parameterization, values=parameterization.unflatten(vector))

    def copy(self) -> "ControlSchedule":
        return ControlSchedule(self.parameterization, np.array(self.values, copy=True))

    def clipped(self) -> "ControlSchedule":
        return ControlSchedule(self.parameterization, self.parameterization.clip(self.values))

    def flattened(self) -> np.ndarray:
        return self.parameterization.flatten(self.values)

    def command_values(self) -> np.ndarray:
        return self.parameterization.command_values(self.values)

    def to_pulses(self, *, waveform_values: np.ndarray | None = None) -> tuple[list[Pulse], dict[str, Any], dict[str, Any]]:
        return self.parameterization.to_pulses(self.values, waveform_values=waveform_values)

    def max_abs_amplitude(self) -> float:
        return float(np.max(np.abs(self.values))) if self.values.size else 0.0

    def rms_amplitude(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.values)))) if self.values.size else 0.0


__all__ = [
    "PiecewiseConstantTimeGrid",
    "ControlParameterization",
    "PiecewiseConstantParameterization",
    "HeldSampleParameterization",
    "FourierParameterization",
    "LinearInterpolatedParameterization",
    "ControlSchedule",
]