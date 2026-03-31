from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from .grape import GrapeIterationRecord, _evaluate_schedule, _prepare_leakage_penalties, _prepare_objective
from .hardware import apply_control_pipeline, selected_control_indices
from .parameterizations import ControlSchedule, PiecewiseConstantTimeGrid, waveform_values_to_pulses
from .problems import ControlProblem, ModelControlChannelSpec, build_control_problem_from_model
from .result import ControlResult
from .utils import finite_bound_scale


def _axes_list(axes: Any) -> list[Any]:
    if isinstance(axes, np.ndarray):
        return [axis for axis in axes.reshape(-1)]
    return [axes]


def _make_agg_subplots(nrows: int, *, figsize: tuple[float, float], sharex: bool = False) -> tuple[Figure, list[Any]]:
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    axes = figure.subplots(nrows, 1, sharex=sharex)
    return figure, _axes_list(axes)


@dataclass(frozen=True)
class PulseParameterSpec:
    name: str
    lower_bound: float
    upper_bound: float
    default: float
    description: str = ""
    units: str | None = None

    def __post_init__(self) -> None:
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        default = float(self.default)
        if lower > upper:
            raise ValueError("PulseParameterSpec requires lower_bound <= upper_bound.")
        if default < lower - 1.0e-15 or default > upper + 1.0e-15:
            raise ValueError(
                f"PulseParameterSpec default {default} for '{self.name}' must lie within [{lower}, {upper}]."
            )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "default", default)

    def clip(self, value: float) -> float:
        return float(np.clip(float(value), float(self.lower_bound), float(self.upper_bound)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "default": float(self.default),
            "description": str(self.description),
            "units": None if self.units is None else str(self.units),
        }


class StructuredPulseFamily(ABC):
    @property
    @abstractmethod
    def family_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameter_specs(self) -> tuple[PulseParameterSpec, ...]:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_complex_envelope(self, time_rel_s: np.ndarray, duration_s: float, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def defaults(self) -> np.ndarray:
        return np.asarray([spec.default for spec in self.parameter_specs], dtype=float)

    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(spec.lower_bound), float(spec.upper_bound)) for spec in self.parameter_specs)

    def clip(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float).reshape(-1)
        if data.size != len(self.parameter_specs):
            raise ValueError(
                f"{type(self).__name__} expects {len(self.parameter_specs)} parameters, received {data.size}."
            )
        return np.asarray([spec.clip(value) for spec, value in zip(self.parameter_specs, data, strict=True)], dtype=float)

    def evaluate(self, time_rel_s: np.ndarray, duration_s: float, values: Sequence[float] | np.ndarray) -> np.ndarray:
        clipped = self.clip(values)
        waveform = np.asarray(self._evaluate_complex_envelope(time_rel_s, duration_s, clipped), dtype=np.complex128)
        expected = np.asarray(time_rel_s, dtype=float).shape
        if waveform.shape != expected:
            raise ValueError(
                f"{type(self).__name__} must return waveform shape {expected}, received {waveform.shape}."
            )
        return waveform

    def waveform_and_jacobian(
        self,
        time_rel_s: np.ndarray,
        duration_s: float,
        values: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        clipped = self.clip(values)
        waveform = self.evaluate(time_rel_s, duration_s, clipped)
        jacobian = np.zeros((len(self.parameter_specs), waveform.size), dtype=np.complex128)
        for index, spec in enumerate(self.parameter_specs):
            scale = max(abs(float(clipped[index])), abs(float(spec.upper_bound)), 1.0)
            epsilon = 1.0e-6 * scale
            plus = np.array(clipped, copy=True)
            minus = np.array(clipped, copy=True)
            plus[index] = spec.clip(float(clipped[index]) + epsilon)
            minus[index] = spec.clip(float(clipped[index]) - epsilon)
            delta = float(plus[index] - minus[index])
            if abs(delta) <= 1.0e-18:
                continue
            jacobian[index, :] = (
                self.evaluate(time_rel_s, duration_s, plus) - self.evaluate(time_rel_s, duration_s, minus)
            ) / delta
        return waveform, jacobian


@dataclass(frozen=True)
class GaussianDragPulseFamily(StructuredPulseFamily):
    amplitude_bounds: tuple[float, float] = (0.0, 8.0e7)
    sigma_fraction_bounds: tuple[float, float] = (0.08, 0.35)
    center_fraction_bounds: tuple[float, float] = (0.35, 0.65)
    phase_bounds: tuple[float, float] = (-np.pi, np.pi)
    drag_bounds: tuple[float, float] = (-0.6, 0.6)
    default_amplitude: float = 2.5e7
    default_sigma_fraction: float = 0.18
    default_center_fraction: float = 0.5
    default_phase: float = -0.5 * np.pi
    default_drag: float = 0.0

    @property
    def family_name(self) -> str:
        return "GaussianDragPulseFamily"

    @property
    def parameter_specs(self) -> tuple[PulseParameterSpec, ...]:
        return (
            PulseParameterSpec(
                name="amplitude",
                lower_bound=float(self.amplitude_bounds[0]),
                upper_bound=float(self.amplitude_bounds[1]),
                default=float(self.default_amplitude),
                description="Peak complex-envelope amplitude.",
                units="rad/s",
            ),
            PulseParameterSpec(
                name="sigma_fraction",
                lower_bound=float(self.sigma_fraction_bounds[0]),
                upper_bound=float(self.sigma_fraction_bounds[1]),
                default=float(self.default_sigma_fraction),
                description="Gaussian sigma as a fraction of the total duration.",
            ),
            PulseParameterSpec(
                name="center_fraction",
                lower_bound=float(self.center_fraction_bounds[0]),
                upper_bound=float(self.center_fraction_bounds[1]),
                default=float(self.default_center_fraction),
                description="Envelope center as a fraction of the total duration.",
            ),
            PulseParameterSpec(
                name="phase_rad",
                lower_bound=float(self.phase_bounds[0]),
                upper_bound=float(self.phase_bounds[1]),
                default=float(self.default_phase),
                description="Global complex-envelope phase.",
                units="rad",
            ),
            PulseParameterSpec(
                name="drag_alpha",
                lower_bound=float(self.drag_bounds[0]),
                upper_bound=float(self.drag_bounds[1]),
                default=float(self.default_drag),
                description="Dimensionless DRAG derivative weight.",
            ),
        )

    def _evaluate_complex_envelope(self, time_rel_s: np.ndarray, duration_s: float, values: np.ndarray) -> np.ndarray:
        amplitude, sigma_fraction, center_fraction, phase_rad, drag_alpha = np.asarray(values, dtype=float)
        if float(duration_s) <= 0.0:
            raise ValueError("GaussianDragPulseFamily requires a positive duration.")
        tau = np.asarray(time_rel_s, dtype=float) / float(duration_s)
        u = tau - float(center_fraction)
        sigma = max(float(sigma_fraction), 1.0e-12)
        gaussian = np.exp(-0.5 * np.square(u / sigma))
        derivative_tau = -(u / (sigma * sigma)) * gaussian
        envelope = gaussian + 1j * float(drag_alpha) * derivative_tau
        return np.asarray(float(amplitude) * envelope * np.exp(1j * float(phase_rad)), dtype=np.complex128)

    def waveform_and_jacobian(
        self,
        time_rel_s: np.ndarray,
        duration_s: float,
        values: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        amplitude, sigma_fraction, center_fraction, phase_rad, drag_alpha = np.asarray(self.clip(values), dtype=float)
        tau = np.asarray(time_rel_s, dtype=float) / float(duration_s)
        u = tau - float(center_fraction)
        sigma = max(float(sigma_fraction), 1.0e-12)
        x = u / sigma
        gaussian = np.exp(-0.5 * np.square(x))
        derivative_tau = -(u / (sigma * sigma)) * gaussian
        phase_factor = np.exp(1j * float(phase_rad))
        envelope = gaussian + 1j * float(drag_alpha) * derivative_tau
        waveform = np.asarray(float(amplitude) * envelope * phase_factor, dtype=np.complex128)

        d_gaussian_d_sigma = gaussian * np.square(x) / sigma
        d_derivative_d_sigma = gaussian * x * (2.0 - np.square(x)) / (sigma * sigma)
        d_envelope_d_sigma = d_gaussian_d_sigma + 1j * float(drag_alpha) * d_derivative_d_sigma

        d_gaussian_d_center = -derivative_tau
        d_derivative_d_center = gaussian * (1.0 - np.square(x)) / (sigma * sigma)
        d_envelope_d_center = d_gaussian_d_center + 1j * float(drag_alpha) * d_derivative_d_center

        jacobian = np.vstack(
            [
                envelope * phase_factor,
                float(amplitude) * d_envelope_d_sigma * phase_factor,
                float(amplitude) * d_envelope_d_center * phase_factor,
                1j * waveform,
                float(amplitude) * 1j * derivative_tau * phase_factor,
            ]
        )
        return waveform, np.asarray(jacobian, dtype=np.complex128)


@dataclass(frozen=True)
class FourierSeriesPulseFamily(StructuredPulseFamily):
    n_modes: int = 3
    coefficient_bound: float = 4.0e7

    def __post_init__(self) -> None:
        if int(self.n_modes) < 1:
            raise ValueError("FourierSeriesPulseFamily.n_modes must be at least 1.")
        if float(self.coefficient_bound) <= 0.0:
            raise ValueError("FourierSeriesPulseFamily.coefficient_bound must be positive.")

    @property
    def family_name(self) -> str:
        return "FourierSeriesPulseFamily"

    @property
    def parameter_specs(self) -> tuple[PulseParameterSpec, ...]:
        specs: list[PulseParameterSpec] = []
        bound = float(self.coefficient_bound)
        for prefix in ("i_cos", "i_sin", "q_cos", "q_sin"):
            start = 0 if prefix.endswith("cos") else 1
            for mode in range(start, int(self.n_modes)):
                basis = "cosine" if prefix.endswith("cos") else "sine"
                quadrature = "I" if prefix.startswith("i_") else "Q"
                specs.append(
                    PulseParameterSpec(
                        name=f"{prefix}_{mode}",
                        lower_bound=-bound,
                        upper_bound=bound,
                        default=0.0,
                        description=(
                            f"{quadrature}-quadrature {basis} coefficient for Fourier mode {mode}."
                        ),
                        units="rad/s",
                    )
                )
        return tuple(specs)

    def _basis(self, time_rel_s: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(time_rel_s, dtype=float) / float(duration_s)
        cosine = np.vstack([np.cos(2.0 * np.pi * mode * tau) for mode in range(int(self.n_modes))])
        if int(self.n_modes) <= 1:
            sine = np.zeros((0, tau.size), dtype=float)
        else:
            sine = np.vstack([np.sin(2.0 * np.pi * mode * tau) for mode in range(1, int(self.n_modes))])
        return cosine, sine

    def _evaluate_complex_envelope(self, time_rel_s: np.ndarray, duration_s: float, values: np.ndarray) -> np.ndarray:
        cosine, sine = self._basis(time_rel_s, duration_s)
        data = np.asarray(values, dtype=float).reshape(-1)
        n_cos = int(self.n_modes)
        n_sin = max(int(self.n_modes) - 1, 0)
        split_0 = n_cos
        split_1 = split_0 + n_sin
        split_2 = split_1 + n_cos
        i_cos = data[:split_0]
        i_sin = data[split_0:split_1]
        q_cos = data[split_1:split_2]
        q_sin = data[split_2:]
        i_waveform = i_cos @ cosine
        if n_sin:
            i_waveform = i_waveform + i_sin @ sine
        q_waveform = q_cos @ cosine
        if n_sin:
            q_waveform = q_waveform + q_sin @ sine
        return np.asarray(i_waveform + 1j * q_waveform, dtype=np.complex128)

    def waveform_and_jacobian(
        self,
        time_rel_s: np.ndarray,
        duration_s: float,
        values: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        clipped = self.clip(values)
        cosine, sine = self._basis(time_rel_s, duration_s)
        waveform = self.evaluate(time_rel_s, duration_s, clipped)
        jacobian_rows: list[np.ndarray] = []
        for mode in range(int(self.n_modes)):
            jacobian_rows.append(np.asarray(cosine[mode], dtype=np.complex128))
        for mode in range(max(int(self.n_modes) - 1, 0)):
            jacobian_rows.append(np.asarray(sine[mode], dtype=np.complex128))
        for mode in range(int(self.n_modes)):
            jacobian_rows.append(np.asarray(1j * cosine[mode], dtype=np.complex128))
        for mode in range(max(int(self.n_modes) - 1, 0)):
            jacobian_rows.append(np.asarray(1j * sine[mode], dtype=np.complex128))
        return waveform, np.asarray(jacobian_rows, dtype=np.complex128)


@dataclass(frozen=True)
class StructuredControlChannel:
    name: str
    pulse_family: StructuredPulseFamily
    export_channel: str | None = None
    control_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.export_channel is None and not self.control_names:
            raise ValueError("StructuredControlChannel requires export_channel or control_names.")
        object.__setattr__(self, "control_names", tuple(str(value) for value in self.control_names))


@dataclass(frozen=True)
class _ResolvedStructuredChannel:
    channel: StructuredControlChannel
    control_indices: tuple[int, ...]
    mode: str
    parameter_slice: slice


@dataclass(frozen=True)
class StructuredPulseParameterization:
    time_grid: PiecewiseConstantTimeGrid
    control_terms: tuple[Any, ...]
    channels: tuple[StructuredControlChannel, ...]

    def __post_init__(self) -> None:
        control_terms = tuple(self.control_terms)
        channels = tuple(self.channels)
        if not control_terms:
            raise ValueError("StructuredPulseParameterization requires at least one control term.")
        if not channels:
            raise ValueError("StructuredPulseParameterization requires at least one structured channel.")
        dim = int(np.asarray(control_terms[0].operator).shape[0])
        for term in control_terms[1:]:
            if np.asarray(term.operator).shape != (dim, dim):
                raise ValueError("All control terms must share the same matrix shape.")
        channel_names = [channel.name for channel in channels]
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("StructuredPulseParameterization channel names must be unique.")

        resolved_channels: list[_ResolvedStructuredChannel] = []
        parameter_specs: list[PulseParameterSpec] = []
        offset = 0
        for channel in channels:
            indices = selected_control_indices(
                control_terms,
                control_names=tuple(channel.control_names),
                export_channels=()
                if channel.export_channel is None
                else (str(channel.export_channel),),
            )
            if not indices:
                target = channel.control_names if channel.control_names else (channel.export_channel,)
                raise ValueError(f"Structured control channel '{channel.name}' did not resolve any control terms: {target}.")

            quadratures = tuple(str(control_terms[index].quadrature).upper() for index in indices)
            if len(indices) == 2 and set(quadratures) == {"I", "Q"}:
                i_index = next(index for index in indices if str(control_terms[index].quadrature).upper() == "I")
                q_index = next(index for index in indices if str(control_terms[index].quadrature).upper() == "Q")
                resolved_indices = (int(i_index), int(q_index))
                mode = "iq_pair"
            elif len(indices) == 1:
                quad = quadratures[0]
                if quad == "I":
                    mode = "i_only"
                elif quad == "Q":
                    mode = "q_only"
                elif quad == "SCALAR":
                    mode = "scalar"
                else:
                    raise ValueError(f"Unsupported quadrature '{quad}' for structured channel '{channel.name}'.")
                resolved_indices = (int(indices[0]),)
            else:
                raise ValueError(
                    "StructuredControlChannel must resolve either one control term or one I/Q pair sharing an export channel. "
                    f"Channel '{channel.name}' resolved {len(indices)} terms."
                )

            local_specs = tuple(channel.pulse_family.parameter_specs)
            resolved_channels.append(
                _ResolvedStructuredChannel(
                    channel=channel,
                    control_indices=resolved_indices,
                    mode=mode,
                    parameter_slice=slice(offset, offset + len(local_specs)),
                )
            )
            for spec in local_specs:
                parameter_specs.append(
                    PulseParameterSpec(
                        name=f"{channel.name}.{spec.name}",
                        lower_bound=float(spec.lower_bound),
                        upper_bound=float(spec.upper_bound),
                        default=float(spec.default),
                        description=str(spec.description),
                        units=None if spec.units is None else str(spec.units),
                    )
                )
            offset += len(local_specs)

        object.__setattr__(self, "control_terms", control_terms)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "_resolved_channels", tuple(resolved_channels))
        object.__setattr__(self, "_parameter_specs", tuple(parameter_specs))

    @property
    def n_controls(self) -> int:
        return len(self.control_terms)

    @property
    def n_slices(self) -> int:
        return len(self._parameter_specs)

    @property
    def n_time_slices(self) -> int:
        return int(self.time_grid.steps)

    @property
    def parameter_shape(self) -> tuple[int]:
        return (len(self._parameter_specs),)

    @property
    def waveform_shape(self) -> tuple[int, int]:
        return (self.n_controls, self.n_time_slices)

    @property
    def parameter_specs(self) -> tuple[PulseParameterSpec, ...]:
        return tuple(self._parameter_specs)

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._parameter_specs)

    def parameter_records(self, values: Sequence[float] | np.ndarray) -> tuple[dict[str, Any], ...]:
        clipped = self.clip(values)
        return tuple(
            {
                **spec.as_dict(),
                "value": float(value),
            }
            for spec, value in zip(self._parameter_specs, clipped, strict=True)
        )

    def zero_array(self) -> np.ndarray:
        return np.asarray([spec.default for spec in self._parameter_specs], dtype=float)

    def zero_schedule(self) -> ControlSchedule:
        return ControlSchedule(self, self.zero_array())

    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(spec.lower_bound), float(spec.upper_bound)) for spec in self._parameter_specs)

    def flatten(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.shape != self.parameter_shape:
            raise ValueError(f"Structured parameters must have shape {self.parameter_shape}, received {data.shape}.")
        return np.array(data.reshape(-1), copy=True)

    def unflatten(self, vector: np.ndarray) -> np.ndarray:
        data = np.asarray(vector, dtype=float).reshape(-1)
        expected = int(self.n_slices)
        if data.size != expected:
            raise ValueError(f"Expected flattened structured parameter vector of length {expected}, received {data.size}.")
        return np.asarray(data, dtype=float)

    def clip(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.shape != self.parameter_shape:
            raise ValueError(f"Structured parameters must have shape {self.parameter_shape}, received {data.shape}.")
        clipped = np.array(data, copy=True)
        for index, spec in enumerate(self._parameter_specs):
            clipped[index] = spec.clip(float(clipped[index]))
        return clipped

    def _relative_midpoints_s(self) -> np.ndarray:
        return np.asarray(self.time_grid.midpoints_s(), dtype=float) - float(self.time_grid.t0_s)

    def _channel_waveform(self, resolved_channel: _ResolvedStructuredChannel, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_values = np.asarray(values[resolved_channel.parameter_slice], dtype=float)
        duration = float(self.time_grid.duration_s)
        return resolved_channel.channel.pulse_family.waveform_and_jacobian(
            self._relative_midpoints_s(),
            duration,
            local_values,
        )

    def command_values(self, values: np.ndarray) -> np.ndarray:
        clipped = self.clip(values)
        waveform = np.zeros(self.waveform_shape, dtype=float)
        for resolved_channel in self._resolved_channels:
            channel_waveform, _jacobian = self._channel_waveform(resolved_channel, clipped)
            if resolved_channel.mode == "iq_pair":
                i_index, q_index = resolved_channel.control_indices
                waveform[i_index, :] += np.asarray(channel_waveform.real, dtype=float)
                waveform[q_index, :] += np.asarray(channel_waveform.imag, dtype=float)
            elif resolved_channel.mode in {"i_only", "scalar"}:
                (index,) = resolved_channel.control_indices
                waveform[index, :] += np.asarray(channel_waveform.real, dtype=float)
            elif resolved_channel.mode == "q_only":
                (index,) = resolved_channel.control_indices
                waveform[index, :] += np.asarray(channel_waveform.imag, dtype=float)
            else:
                raise ValueError(f"Unsupported structured control mode '{resolved_channel.mode}'.")
        return waveform

    def pullback(
        self,
        gradient_command: np.ndarray,
        values: np.ndarray,
        *,
        command_values: np.ndarray | None = None,
    ) -> np.ndarray:
        _ = command_values
        gradient = np.asarray(gradient_command, dtype=float)
        if gradient.shape != self.waveform_shape:
            raise ValueError(
                f"Structured command-waveform gradients must have shape {self.waveform_shape}, received {gradient.shape}."
            )
        clipped = self.clip(values)
        reduced = np.zeros(self.parameter_shape, dtype=float)
        for resolved_channel in self._resolved_channels:
            _waveform, jacobian = self._channel_waveform(resolved_channel, clipped)
            if resolved_channel.mode == "iq_pair":
                i_index, q_index = resolved_channel.control_indices
                gradient_real = np.asarray(gradient[i_index, :], dtype=float)
                gradient_imag = np.asarray(gradient[q_index, :], dtype=float)
            elif resolved_channel.mode in {"i_only", "scalar"}:
                (index,) = resolved_channel.control_indices
                gradient_real = np.asarray(gradient[index, :], dtype=float)
                gradient_imag = np.zeros_like(gradient_real, dtype=float)
            elif resolved_channel.mode == "q_only":
                (index,) = resolved_channel.control_indices
                gradient_real = np.zeros(gradient.shape[1], dtype=float)
                gradient_imag = np.asarray(gradient[index, :], dtype=float)
            else:
                raise ValueError(f"Unsupported structured control mode '{resolved_channel.mode}'.")

            local_gradient = np.sum(
                gradient_real[None, :] * np.asarray(jacobian.real, dtype=float)
                + gradient_imag[None, :] * np.asarray(jacobian.imag, dtype=float),
                axis=1,
            )
            reduced[resolved_channel.parameter_slice] += np.asarray(local_gradient, dtype=float)
        return reduced

    def parameterization_metrics(self, values: np.ndarray, command_values: np.ndarray | None = None) -> dict[str, Any]:
        _ = command_values
        clipped = self.clip(values)
        return {
            "parameterization": "StructuredPulseParameterization",
            "parameter_count": int(self.n_slices),
            "time_slices": int(self.n_time_slices),
            "structured_channel_count": int(len(self.channels)),
            "parameter_names": list(self.parameter_names()),
            "structured_channels": [
                {
                    "name": str(resolved.channel.name),
                    "family": resolved.channel.pulse_family.family_name,
                    "mode": str(resolved.mode),
                    "control_terms": [str(self.control_terms[index].name) for index in resolved.control_indices],
                    "parameters": {
                        spec.name.split(".", 1)[1]: float(clipped[index])
                        for index, spec in enumerate(self._parameter_specs)
                        if resolved.parameter_slice.start <= index < resolved.parameter_slice.stop
                    },
                }
                for resolved in self._resolved_channels
            ],
        }

    def to_pulses(self, values: np.ndarray, *, waveform_values: np.ndarray | None = None) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
        waveform = self.command_values(values) if waveform_values is None else np.asarray(waveform_values, dtype=float)
        return waveform_values_to_pulses(
            control_terms=self.control_terms,
            time_grid=self.time_grid,
            waveform_values=waveform,
            parameterization_name="StructuredPulseParameterization",
            extra_metadata={
                "parameter_count": int(self.n_slices),
                "structured_channels": self.parameterization_metrics(values).get("structured_channels", []),
            },
        )


@dataclass(frozen=True)
class StructuredControlConfig:
    optimizer_method: str = "L-BFGS-B"
    maxiter: int = 200
    ftol: float = 1.0e-9
    gtol: float = 1.0e-6
    initial_guess: Any = "defaults"
    random_scale: float = 0.15
    seed: int | None = None
    history_every: int = 1
    apply_hardware_in_forward_model: bool = True
    report_command_reference: bool = True
    use_gradients: bool = True
    engine: str = "numpy"
    jax_device: str | None = None
    scipy_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.initial_guess, str) and self.initial_guess.lower() not in {"defaults", "random"}:
            raise ValueError("StructuredControlConfig.initial_guess must be 'defaults', 'random', or an explicit parameter array.")
        if int(self.maxiter) <= 0:
            raise ValueError("StructuredControlConfig.maxiter must be positive.")
        if float(self.ftol) < 0.0 or float(self.gtol) < 0.0:
            raise ValueError("StructuredControlConfig tolerances must be non-negative.")
        if int(self.history_every) <= 0:
            raise ValueError("StructuredControlConfig.history_every must be positive.")
        if str(self.engine).lower() not in {"numpy", "jax"}:
            raise ValueError("StructuredControlConfig.engine must be 'numpy' or 'jax'.")


class StructuredControlSolver:
    def __init__(self, config: StructuredControlConfig | None = None) -> None:
        self.config = StructuredControlConfig() if config is None else config

    def _initial_schedule(self, problem: ControlProblem, initial_schedule: ControlSchedule | np.ndarray | None) -> ControlSchedule:
        if initial_schedule is None:
            if isinstance(self.config.initial_guess, ControlSchedule):
                return self.config.initial_guess.clipped()
            if isinstance(self.config.initial_guess, str):
                if self.config.initial_guess.lower() == "defaults":
                    return problem.parameterization.zero_schedule()
                bounds = tuple(problem.parameterization.bounds())
                defaults = np.asarray(problem.parameterization.zero_array(), dtype=float).reshape(-1)
                rng = np.random.default_rng(self.config.seed)
                random_values = np.array(defaults, copy=True)
                for index, (lower, upper) in enumerate(bounds):
                    if np.isfinite(lower) and np.isfinite(upper):
                        half_range = 0.5 * float(upper - lower)
                        width = float(self.config.random_scale) * half_range
                        random_values[index] = np.clip(
                            defaults[index] + rng.uniform(-width, width),
                            float(lower),
                            float(upper),
                        )
                    else:
                        sigma = float(self.config.random_scale) * finite_bound_scale(lower, upper, fallback=1.0)
                        random_values[index] = defaults[index] + rng.normal(loc=0.0, scale=sigma)
                return ControlSchedule(problem.parameterization, problem.parameterization.clip(random_values))
            return ControlSchedule(problem.parameterization, problem.parameterization.clip(np.asarray(self.config.initial_guess, dtype=float)))
        if isinstance(initial_schedule, ControlSchedule):
            return initial_schedule.clipped()
        return ControlSchedule(problem.parameterization, problem.parameterization.clip(np.asarray(initial_schedule, dtype=float)))

    def solve(self, problem: ControlProblem, *, initial_schedule: ControlSchedule | np.ndarray | None = None) -> ControlResult:
        prepared_objectives = tuple(_prepare_objective(problem, objective) for objective in problem.objectives)
        prepared_leakage_penalties = _prepare_leakage_penalties(problem)
        schedule0 = self._initial_schedule(problem, initial_schedule)

        bounds = tuple(problem.parameterization.bounds())
        scale_vector = np.asarray(
            [finite_bound_scale(float(lower), float(upper), fallback=1.0) for lower, upper in bounds],
            dtype=float,
        )
        scaled_bounds = tuple(
            (float(lower) / max(scale, 1.0e-18), float(upper) / max(scale, 1.0e-18))
            for (lower, upper), scale in zip(bounds, scale_vector, strict=True)
        )

        last_vector: np.ndarray | None = None
        last_evaluation: Any = None
        history: list[GrapeIterationRecord] = []
        start_time = time.perf_counter()
        evaluation_counter = 0
        engine = str(self.config.engine).lower()
        jax_device = self.config.jax_device
        jax_cache: dict[int, Any] = {}

        def evaluate(vector: np.ndarray):
            nonlocal last_vector, last_evaluation, evaluation_counter
            if last_vector is not None and np.array_equal(vector, last_vector):
                return last_evaluation

            evaluation_counter += 1
            parameter_vector = np.asarray(vector, dtype=float).reshape(-1) * scale_vector
            schedule = ControlSchedule.from_flattened(problem.parameterization, parameter_vector).clipped()
            evaluation = _evaluate_schedule(
                problem,
                schedule,
                prepared_objectives,
                prepared_leakage_penalties,
                apply_hardware=bool(self.config.apply_hardware_in_forward_model),
                engine=engine,
                jax_device=jax_device,
                _jax_cache=jax_cache,
            )
            last_vector = np.array(vector, copy=True)
            last_evaluation = evaluation
            if evaluation_counter == 1 or evaluation_counter % int(self.config.history_every) == 0:
                history.append(
                    GrapeIterationRecord(
                        evaluation=evaluation_counter,
                        objective=float(evaluation.objective),
                        gradient_norm=float(np.linalg.norm(np.asarray(evaluation.gradient, dtype=float).reshape(-1))),
                        elapsed_s=float(time.perf_counter() - start_time),
                        metrics=dict(evaluation.metrics),
                    )
                )
            return evaluation

        def objective_function(vector: np.ndarray) -> float:
            return float(evaluate(vector).objective)

        def gradient_function(vector: np.ndarray) -> np.ndarray:
            return np.asarray(evaluate(vector).gradient, dtype=float).reshape(-1) * scale_vector

        options = dict(self.config.scipy_options)
        options.setdefault("maxiter", int(self.config.maxiter))
        if str(self.config.optimizer_method).upper() == "L-BFGS-B":
            options.setdefault("ftol", float(self.config.ftol))
            options.setdefault("gtol", float(self.config.gtol))

        optimizer_result = minimize(
            objective_function,
            schedule0.flattened() / scale_vector,
            method=str(self.config.optimizer_method),
            jac=gradient_function if bool(self.config.use_gradients) else None,
            bounds=scaled_bounds,
            options=options,
        )

        final_schedule = ControlSchedule.from_flattened(problem.parameterization, optimizer_result.x * scale_vector).clipped()
        final_evaluation = _evaluate_schedule(
            problem,
            final_schedule,
            prepared_objectives,
            prepared_leakage_penalties,
            apply_hardware=bool(self.config.apply_hardware_in_forward_model),
            engine=engine,
            jax_device=jax_device,
            _jax_cache=jax_cache,
        )
        export_resolution = apply_control_pipeline(problem, final_schedule, apply_hardware=True).resolved
        final_metrics = dict(final_evaluation.metrics)
        if problem.hardware_model is None:
            final_metrics.setdefault("nominal_command_fidelity", float(final_evaluation.metrics.get("nominal_fidelity", np.nan)))
            final_metrics.setdefault("nominal_physical_fidelity", float(final_evaluation.metrics.get("nominal_fidelity", np.nan)))
            final_metrics.setdefault("objective_command_reference", float(final_evaluation.objective))
            final_metrics.setdefault("objective_physical_reference", float(final_evaluation.objective))
        else:
            if bool(self.config.apply_hardware_in_forward_model):
                final_metrics["nominal_physical_fidelity"] = float(final_evaluation.metrics.get("nominal_fidelity", np.nan))
                final_metrics["objective_physical_reference"] = float(final_evaluation.objective)
                final_metrics.setdefault("nominal_command_fidelity", float("nan"))
                final_metrics.setdefault("objective_command_reference", float("nan"))
            else:
                final_metrics["nominal_command_fidelity"] = float(final_evaluation.metrics.get("nominal_fidelity", np.nan))
                final_metrics["objective_command_reference"] = float(final_evaluation.objective)
                final_metrics["nominal_physical_fidelity"] = float("nan")
                final_metrics["objective_physical_reference"] = float("nan")
        if problem.hardware_model is not None and bool(self.config.report_command_reference):
            if bool(self.config.apply_hardware_in_forward_model):
                command_reference = _evaluate_schedule(
                    problem,
                    final_schedule,
                    prepared_objectives,
                    prepared_leakage_penalties,
                    apply_hardware=False,
                    engine=engine,
                    jax_device=jax_device,
                    _jax_cache=jax_cache,
                )
                final_metrics["nominal_command_fidelity"] = float(command_reference.metrics.get("nominal_fidelity", np.nan))
                final_metrics["objective_command_reference"] = float(command_reference.objective)
                final_metrics["objective_physical_reference"] = float(final_evaluation.objective)
            else:
                physical_reference = _evaluate_schedule(
                    problem,
                    final_schedule,
                    prepared_objectives,
                    prepared_leakage_penalties,
                    apply_hardware=True,
                    engine=engine,
                    jax_device=jax_device,
                    _jax_cache=jax_cache,
                )
                final_metrics["nominal_command_fidelity"] = float(final_evaluation.metrics.get("nominal_fidelity", np.nan))
                final_metrics["objective_command_reference"] = float(final_evaluation.objective)
                final_metrics["nominal_physical_fidelity"] = float(physical_reference.metrics.get("nominal_fidelity", np.nan))
                final_metrics["objective_physical_reference"] = float(physical_reference.objective)

        final_metrics["hardware_forward_model_applied"] = bool(self.config.apply_hardware_in_forward_model and problem.hardware_model is not None)
        final_metrics["optimizer_uses_gradients"] = bool(self.config.use_gradients)
        if not history or history[-1].evaluation != evaluation_counter:
            history.append(
                GrapeIterationRecord(
                    evaluation=max(evaluation_counter, 1),
                    objective=float(final_evaluation.objective),
                    gradient_norm=float(np.linalg.norm(np.asarray(final_evaluation.gradient, dtype=float).reshape(-1))),
                    elapsed_s=float(time.perf_counter() - start_time),
                    metrics=dict(final_evaluation.metrics),
                )
            )

        return ControlResult(
            success=bool(optimizer_result.success),
            message=str(optimizer_result.message),
            schedule=final_schedule,
            objective_value=float(final_evaluation.objective),
            metrics=final_metrics,
            system_metrics=tuple(final_evaluation.system_metrics),
            history=history,
            nominal_final_unitary=final_evaluation.nominal_final_unitary,
            optimizer_summary={
                "method": str(self.config.optimizer_method),
                "nit": int(getattr(optimizer_result, "nit", 0) or 0),
                "nfev": int(getattr(optimizer_result, "nfev", 0) or 0),
                "njev": int(getattr(optimizer_result, "njev", 0) or 0),
                "status": int(getattr(optimizer_result, "status", 0) or 0),
                "variable_scaling": "bound_based",
            },
            command_values=np.asarray(export_resolution.command_values, dtype=float),
            physical_values=np.asarray(export_resolution.physical_values, dtype=float),
            time_boundaries_s=np.asarray(export_resolution.time_boundaries_s, dtype=float),
            parameterization_metrics=dict(export_resolution.parameterization_metrics),
            hardware_metrics=dict(export_resolution.hardware_metrics),
            hardware_reports=tuple(
                {"name": report.name, "metrics": dict(report.metrics)} for report in export_resolution.hardware_reports
            ),
            backend="structured-control",
        )


def solve_structured_control(
    problem: ControlProblem,
    *,
    config: StructuredControlConfig | None = None,
    initial_schedule: ControlSchedule | np.ndarray | None = None,
) -> ControlResult:
    return StructuredControlSolver(config=config).solve(problem, initial_schedule=initial_schedule)


def build_structured_control_problem_from_model(
    model: Any,
    *,
    frame: Any,
    time_grid: PiecewiseConstantTimeGrid,
    channel_specs: Sequence[ModelControlChannelSpec],
    structured_channels: Sequence[StructuredControlChannel],
    objectives: Sequence[Any],
    penalties: Sequence[Any] = (),
    ensemble_members: Sequence[Any] = (),
    ensemble_aggregate: str = "mean",
    hardware_model: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> ControlProblem:
    from .problems import build_control_terms_from_model

    control_terms = build_control_terms_from_model(model, channel_specs)
    parameterization = StructuredPulseParameterization(
        time_grid=time_grid,
        control_terms=control_terms,
        channels=tuple(structured_channels),
    )
    return build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=tuple(channel_specs),
        objectives=tuple(objectives),
        penalties=tuple(penalties),
        ensemble_members=tuple(ensemble_members),
        ensemble_aggregate=str(ensemble_aggregate),
        parameterization=parameterization,
        hardware_model=hardware_model,
        metadata={} if metadata is None else dict(metadata),
    )


def _complex_export_channels(control_terms: tuple[Any, ...], waveform_values: np.ndarray) -> dict[str, np.ndarray]:
    data = np.asarray(waveform_values, dtype=float)
    channels: dict[str, np.ndarray] = {}
    for term_index, term in enumerate(control_terms):
        if term.export_channel is None:
            continue
        contribution = np.asarray(data[term_index, :], dtype=np.complex128)
        if str(term.quadrature).upper() == "Q":
            contribution = 1j * contribution
        channels.setdefault(str(term.export_channel), np.zeros(data.shape[1], dtype=np.complex128))
        channels[str(term.export_channel)] += contribution
    return channels


@dataclass(frozen=True)
class StructuredControlArtifacts:
    directory: Path
    result_json: Path
    parameters_csv: Path
    waveforms_csv: Path
    history_csv: Path
    waveform_plot: Path
    spectrum_plot: Path
    history_plot: Path


def save_structured_control_artifacts(
    problem: ControlProblem,
    result: ControlResult,
    directory: str | Path,
) -> StructuredControlArtifacts:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_json = result.save(output_dir / "result.json")
    parameters_csv = output_dir / "parameters.csv"
    waveforms_csv = output_dir / "waveforms.csv"
    history_csv = output_dir / "history.csv"
    waveform_plot = output_dir / "waveforms.png"
    spectrum_plot = output_dir / "spectrum.png"
    history_plot = output_dir / "optimization_history.png"

    if not isinstance(problem.parameterization, StructuredPulseParameterization):
        raise TypeError("save_structured_control_artifacts requires a StructuredPulseParameterization.")

    with parameters_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "value", "lower_bound", "upper_bound", "default", "units", "description"],
        )
        writer.writeheader()
        for record in problem.parameterization.parameter_records(result.schedule.values):
            writer.writerow(record)

    command_values = np.asarray(result.command_values, dtype=float)
    physical_values = np.asarray(result.physical_values, dtype=float)
    time_midpoints = np.asarray(problem.time_grid.midpoints_s(), dtype=float)
    fieldnames = ["time_midpoint_s"]
    for term in problem.control_terms:
        fieldnames.append(f"command_{term.name}")
        fieldnames.append(f"physical_{term.name}")
    with waveforms_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for time_index, midpoint in enumerate(time_midpoints):
            row: dict[str, Any] = {"time_midpoint_s": float(midpoint)}
            for control_index, term in enumerate(problem.control_terms):
                row[f"command_{term.name}"] = float(command_values[control_index, time_index])
                row[f"physical_{term.name}"] = float(physical_values[control_index, time_index])
            writer.writerow(row)

    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["evaluation", "objective", "gradient_norm", "elapsed_s", "nominal_fidelity"],
        )
        writer.writeheader()
        for record in result.history:
            writer.writerow(
                {
                    "evaluation": int(record.evaluation),
                    "objective": float(record.objective),
                    "gradient_norm": float(record.gradient_norm),
                    "elapsed_s": float(record.elapsed_s),
                    "nominal_fidelity": float(record.metrics.get("nominal_fidelity", np.nan)),
                }
            )

    boundaries = np.asarray(problem.time_grid.boundaries_s(), dtype=float)
    figure, axes = _make_agg_subplots(problem.n_controls, figsize=(10, 2.6 * max(problem.n_controls, 1)), sharex=True)
    for control_index, axis in enumerate(axes):
        axis.step(boundaries[:-1], command_values[control_index, :], where="post", label="command", linewidth=1.8)
        axis.step(boundaries[:-1], physical_values[control_index, :], where="post", label="physical", linewidth=1.8)
        axis.set_ylabel(problem.control_terms[control_index].name)
        axis.grid(alpha=0.25)
        if control_index == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    figure.suptitle("Structured-control waveforms")
    figure.tight_layout()
    figure.savefig(waveform_plot, dpi=180)

    command_channels = _complex_export_channels(tuple(problem.control_terms), command_values)
    physical_channels = _complex_export_channels(tuple(problem.control_terms), physical_values)
    dt = float(np.mean(np.asarray(problem.time_grid.step_durations_s, dtype=float)))
    freq_axis = np.fft.fftfreq(problem.time_grid.steps, d=dt)
    positive = freq_axis >= 0.0
    figure, axes = _make_agg_subplots(max(len(command_channels), 1), figsize=(10, 3.0 * max(len(command_channels), 1)), sharex=True)
    for axis, channel_name in zip(axes, sorted(command_channels), strict=False):
        command_fft = np.fft.fft(command_channels[channel_name])
        physical_fft = np.fft.fft(physical_channels.get(channel_name, command_channels[channel_name]))
        axis.plot(freq_axis[positive], np.abs(command_fft[positive]), label="command spectrum", linewidth=1.8)
        axis.plot(freq_axis[positive], np.abs(physical_fft[positive]), label="physical spectrum", linewidth=1.8)
        axis.set_ylabel(channel_name)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("frequency (Hz)")
    figure.suptitle("Structured-control spectra")
    figure.tight_layout()
    figure.savefig(spectrum_plot, dpi=180)

    figure, axes = _make_agg_subplots(2, figsize=(10, 6), sharex=True)
    evaluations = [int(record.evaluation) for record in result.history]
    objectives = [float(record.objective) for record in result.history]
    fidelities = [float(record.metrics.get("nominal_fidelity", np.nan)) for record in result.history]
    axes[0].plot(evaluations, objectives, marker="o", linewidth=1.8)
    axes[0].set_ylabel("objective")
    axes[0].grid(alpha=0.25)
    axes[1].plot(evaluations, fidelities, marker="o", linewidth=1.8)
    axes[1].set_xlabel("evaluation")
    axes[1].set_ylabel("nominal fidelity")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Optimization progression")
    figure.tight_layout()
    figure.savefig(history_plot, dpi=180)

    return StructuredControlArtifacts(
        directory=output_dir,
        result_json=result_json,
        parameters_csv=parameters_csv,
        waveforms_csv=waveforms_csv,
        history_csv=history_csv,
        waveform_plot=waveform_plot,
        spectrum_plot=spectrum_plot,
        history_plot=history_plot,
    )


__all__ = [
    "PulseParameterSpec",
    "StructuredPulseFamily",
    "GaussianDragPulseFamily",
    "FourierSeriesPulseFamily",
    "StructuredControlChannel",
    "StructuredPulseParameterization",
    "StructuredControlConfig",
    "StructuredControlSolver",
    "StructuredControlArtifacts",
    "build_structured_control_problem_from_model",
    "save_structured_control_artifacts",
    "solve_structured_control",
]