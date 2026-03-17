from __future__ import annotations

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
class PiecewiseConstantParameterization:
    time_grid: PiecewiseConstantTimeGrid
    control_terms: tuple["ControlTerm", ...]

    def __post_init__(self) -> None:
        if not self.control_terms:
            raise ValueError("PiecewiseConstantParameterization requires at least one control term.")
        dim = int(np.asarray(self.control_terms[0].operator).shape[0])
        for term in self.control_terms[1:]:
            if np.asarray(term.operator).shape != (dim, dim):
                raise ValueError("All control terms must have the same matrix shape.")

    @property
    def n_controls(self) -> int:
        return len(self.control_terms)

    @property
    def n_slices(self) -> int:
        return self.time_grid.steps

    @property
    def hilbert_dim(self) -> int:
        return int(np.asarray(self.control_terms[0].operator).shape[0])

    def zero_array(self) -> np.ndarray:
        return np.zeros((self.n_controls, self.n_slices), dtype=float)

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
        if data.shape != (self.n_controls, self.n_slices):
            raise ValueError(
                f"Control values must have shape {(self.n_controls, self.n_slices)}, received {data.shape}."
            )
        return data.reshape(-1)

    def unflatten(self, vector: np.ndarray) -> np.ndarray:
        data = np.asarray(vector, dtype=float).reshape(-1)
        expected = self.n_controls * self.n_slices
        if data.size != expected:
            raise ValueError(f"Expected flattened control vector of length {expected}, received {data.size}.")
        return data.reshape(self.n_controls, self.n_slices)

    def clip(self, values: np.ndarray) -> np.ndarray:
        clipped = np.asarray(values, dtype=float).copy()
        if clipped.shape != (self.n_controls, self.n_slices):
            raise ValueError(
                f"Control values must have shape {(self.n_controls, self.n_slices)}, received {clipped.shape}."
            )
        for term_index, term in enumerate(self.control_terms):
            lower, upper = term.amplitude_bounds
            clipped[term_index, :] = np.clip(clipped[term_index, :], float(lower), float(upper))
        return clipped

    def _channel_coefficients(self, values: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        data = np.asarray(values, dtype=float)
        boundaries = self.time_grid.boundaries_s()
        channels: dict[str, Any] = {}
        for term in self.control_terms:
            if term.export_channel is None:
                continue
            channel_state = channels.setdefault(
                str(term.export_channel),
                {
                    "target": term.drive_target,
                    "coefficients": np.zeros(self.n_slices, dtype=np.complex128),
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

    def to_pulses(self, values: np.ndarray) -> tuple[list[Pulse], dict[str, Any], dict[str, Any]]:
        channels, boundaries = self._channel_coefficients(values)
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
                "Piecewise-constant rotating-frame controls exported as square-envelope pulses. "
                "For repository drive targets, I/Q quadratures are combined into the complex channel coefficient "
                "c(t) = I(t) - i Q(t), which matches cqed_sim.sim.runner Hamiltonian assembly."
            ),
            "time_grid_s": [float(value) for value in self.time_grid.step_durations_s],
            "channels": channel_summary,
        }
        return pulses, drive_ops, metadata


@dataclass
class ControlSchedule:
    parameterization: PiecewiseConstantParameterization
    values: np.ndarray

    def __post_init__(self) -> None:
        data = np.asarray(self.values, dtype=float)
        if data.shape != (self.parameterization.n_controls, self.parameterization.n_slices):
            raise ValueError(
                "ControlSchedule.values must match the parameterization shape "
                f"{(self.parameterization.n_controls, self.parameterization.n_slices)}, received {data.shape}."
            )
        self.values = data

    @classmethod
    def from_flattened(cls, parameterization: PiecewiseConstantParameterization, vector: np.ndarray) -> "ControlSchedule":
        return cls(parameterization=parameterization, values=parameterization.unflatten(vector))

    def copy(self) -> "ControlSchedule":
        return ControlSchedule(self.parameterization, np.array(self.values, copy=True))

    def clipped(self) -> "ControlSchedule":
        return ControlSchedule(self.parameterization, self.parameterization.clip(self.values))

    def flattened(self) -> np.ndarray:
        return self.parameterization.flatten(self.values)

    def to_pulses(self) -> tuple[list[Pulse], dict[str, Any], dict[str, Any]]:
        return self.parameterization.to_pulses(self.values)

    def max_abs_amplitude(self) -> float:
        return float(np.max(np.abs(self.values))) if self.values.size else 0.0

    def rms_amplitude(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.values)))) if self.values.size else 0.0


__all__ = [
    "PiecewiseConstantTimeGrid",
    "PiecewiseConstantParameterization",
    "ControlSchedule",
]