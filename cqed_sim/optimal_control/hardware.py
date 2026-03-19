from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .parameterizations import ControlSchedule, PiecewiseConstantTimeGrid
    from .problems import ControlProblem, ControlTerm


def selected_control_indices(
    control_terms: tuple["ControlTerm", ...],
    *,
    control_names: tuple[str, ...] = (),
    export_channels: tuple[str, ...] = (),
) -> tuple[int, ...]:
    name_filter = tuple(str(value) for value in control_names)
    channel_filter = tuple(str(value) for value in export_channels)
    selected: list[int] = []
    for index, term in enumerate(control_terms):
        if name_filter and str(term.name) not in name_filter:
            continue
        if channel_filter:
            if term.export_channel is None or str(term.export_channel) not in channel_filter:
                continue
        selected.append(index)
    return tuple(selected)


def selected_iq_pairs(
    control_terms: tuple["ControlTerm", ...],
    *,
    control_names: tuple[str, ...] = (),
    export_channels: tuple[str, ...] = (),
) -> tuple[tuple[str, int, int], ...]:
    name_filter = tuple(str(value) for value in control_names)
    channel_filter = tuple(str(value) for value in export_channels)
    grouped: dict[str, dict[str, int]] = {}
    for index, term in enumerate(control_terms):
        if term.quadrature not in {"I", "Q"} or term.export_channel is None:
            continue
        if name_filter and str(term.name) not in name_filter:
            continue
        channel = str(term.export_channel)
        if channel_filter and channel not in channel_filter:
            continue
        channel_terms = grouped.setdefault(channel, {})
        if term.quadrature in channel_terms:
            raise ValueError(f"Export channel '{channel}' defines multiple {term.quadrature} quadratures.")
        channel_terms[term.quadrature] = index
    pairs: list[tuple[str, int, int]] = []
    for channel, quadratures in grouped.items():
        if "I" in quadratures and "Q" in quadratures:
            pairs.append((channel, int(quadratures["I"]), int(quadratures["Q"])))
    return tuple(pairs)


def _waveform_metrics(prefix: str, values: np.ndarray, time_grid: "PiecewiseConstantTimeGrid") -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    metrics = {
        f"{prefix}_max_abs_amplitude": float(np.max(np.abs(data))) if data.size else 0.0,
        f"{prefix}_rms_amplitude": float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0,
        f"{prefix}_max_slew": 0.0,
        f"{prefix}_rms_slew": 0.0,
    }
    if data.shape[1] >= 2:
        midpoints = np.asarray(time_grid.midpoints_s(), dtype=float)
        dt = np.maximum(np.diff(midpoints), 1.0e-18)
        slew = np.diff(data, axis=1) / dt[None, :]
        metrics[f"{prefix}_max_slew"] = float(np.max(np.abs(slew))) if slew.size else 0.0
        metrics[f"{prefix}_rms_slew"] = float(np.sqrt(np.mean(np.square(slew)))) if slew.size else 0.0
    return metrics


@dataclass(frozen=True)
class HardwareMapReport:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedControlWaveforms:
    parameter_values: np.ndarray
    command_values: np.ndarray
    physical_values: np.ndarray
    time_boundaries_s: np.ndarray
    parameterization_metrics: dict[str, Any] = field(default_factory=dict)
    hardware_reports: tuple[HardwareMapReport, ...] = ()
    hardware_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppliedControlWaveforms:
    resolved: ResolvedControlWaveforms
    pullback_command: Callable[[np.ndarray], np.ndarray]
    pullback_physical: Callable[[np.ndarray], np.ndarray]


@dataclass
class _AppliedHardwareMap:
    values: np.ndarray
    pullback: Callable[[np.ndarray], np.ndarray]
    report: HardwareMapReport


class HardwareMap(ABC):
    @abstractmethod
    def apply(
        self,
        values: np.ndarray,
        *,
        control_terms: tuple["ControlTerm", ...],
        time_grid: "PiecewiseConstantTimeGrid",
    ) -> _AppliedHardwareMap:
        raise NotImplementedError


@dataclass(frozen=True)
class HardwareModel:
    maps: tuple[HardwareMap, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "maps", tuple(self.maps))

    def apply(
        self,
        command_values: np.ndarray,
        *,
        control_terms: tuple["ControlTerm", ...],
        time_grid: "PiecewiseConstantTimeGrid",
    ) -> tuple[np.ndarray, tuple[HardwareMapReport, ...], dict[str, Any], Callable[[np.ndarray], np.ndarray]]:
        current = np.asarray(command_values, dtype=float)
        pullbacks: list[Callable[[np.ndarray], np.ndarray]] = []
        reports: list[HardwareMapReport] = []

        for hardware_map in self.maps:
            applied = hardware_map.apply(current, control_terms=control_terms, time_grid=time_grid)
            current = np.asarray(applied.values, dtype=float)
            pullbacks.append(applied.pullback)
            reports.append(applied.report)

        def pullback(gradient_physical: np.ndarray) -> np.ndarray:
            gradient = np.asarray(gradient_physical, dtype=float)
            for fn in reversed(pullbacks):
                gradient = fn(gradient)
            return gradient

        metrics: dict[str, Any] = {
            "hardware_active": bool(self.maps),
            "hardware_map_count": int(len(self.maps)),
            **_waveform_metrics("command", command_values, time_grid),
            **_waveform_metrics("physical", current, time_grid),
        }
        return current, tuple(reports), metrics, pullback


def resolve_control_schedule(
    problem: "ControlProblem",
    schedule: "ControlSchedule",
    *,
    apply_hardware: bool = True,
) -> ResolvedControlWaveforms:
    return apply_control_pipeline(problem, schedule, apply_hardware=apply_hardware).resolved


def apply_control_pipeline(
    problem: "ControlProblem",
    schedule: "ControlSchedule",
    *,
    apply_hardware: bool = True,
) -> AppliedControlWaveforms:
    parameter_values = np.asarray(schedule.values, dtype=float)
    command_values = np.asarray(problem.parameterization.command_values(parameter_values), dtype=float)
    parameterization_metrics = problem.parameterization.parameterization_metrics(parameter_values, command_values)

    def pullback_command(gradient_command: np.ndarray) -> np.ndarray:
        return np.asarray(
            problem.parameterization.pullback(
                np.asarray(gradient_command, dtype=float),
                parameter_values,
                command_values=command_values,
            ),
            dtype=float,
        )

    hardware_reports: tuple[HardwareMapReport, ...] = ()
    hardware_metrics: dict[str, Any] = {
        "hardware_active": False,
        "hardware_map_count": 0,
        **_waveform_metrics("command", command_values, problem.time_grid),
        **_waveform_metrics("physical", command_values, problem.time_grid),
    }
    physical_values = np.array(command_values, copy=True)
    pullback_physical = pullback_command

    if apply_hardware and problem.hardware_model is not None:
        physical_values, hardware_reports, hardware_metrics, hardware_pullback = problem.hardware_model.apply(
            command_values,
            control_terms=problem.control_terms,
            time_grid=problem.time_grid,
        )

        def pullback_physical(gradient_physical: np.ndarray) -> np.ndarray:
            return pullback_command(hardware_pullback(np.asarray(gradient_physical, dtype=float)))

    resolved = ResolvedControlWaveforms(
        parameter_values=np.asarray(parameter_values, dtype=float),
        command_values=np.asarray(command_values, dtype=float),
        physical_values=np.asarray(physical_values, dtype=float),
        time_boundaries_s=np.asarray(problem.time_grid.boundaries_s(), dtype=float),
        parameterization_metrics=dict(parameterization_metrics),
        hardware_reports=tuple(hardware_reports),
        hardware_metrics=dict(hardware_metrics),
    )
    return AppliedControlWaveforms(
        resolved=resolved,
        pullback_command=pullback_command,
        pullback_physical=pullback_physical,
    )


def _first_order_lowpass_forward(x: np.ndarray, step_durations_s: np.ndarray, cutoff_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.asarray(x, dtype=float), np.asarray(x, dtype=float)
    tau = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alphas = np.ones(x.shape[0], dtype=float)
    if x.shape[0] > 1:
        alphas[1:] = np.asarray(step_durations_s[:-1], dtype=float) / (tau + np.asarray(step_durations_s[:-1], dtype=float))
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for index in range(1, x.shape[0]):
        alpha = float(alphas[index])
        y[index] = y[index - 1] + alpha * (x[index] - y[index - 1])
    return y, alphas


def _first_order_lowpass_pullback(gradient_y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    gradient = np.asarray(gradient_y, dtype=float)
    if gradient.size == 0:
        return gradient
    lambdas = np.array(gradient, copy=True)
    for index in range(gradient.shape[0] - 2, -1, -1):
        lambdas[index] += (1.0 - float(alphas[index + 1])) * lambdas[index + 1]
    gradient_x = np.zeros_like(gradient, dtype=float)
    gradient_x[0] = lambdas[0]
    if gradient.shape[0] > 1:
        gradient_x[1:] = alphas[1:] * lambdas[1:]
    return gradient_x


@dataclass(frozen=True)
class FirstOrderLowPassHardwareMap(HardwareMap):
    cutoff_hz: float
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if float(self.cutoff_hz) <= 0.0:
            raise ValueError("FirstOrderLowPassHardwareMap.cutoff_hz must be positive.")

    def apply(
        self,
        values: np.ndarray,
        *,
        control_terms: tuple["ControlTerm", ...],
        time_grid: "PiecewiseConstantTimeGrid",
    ) -> _AppliedHardwareMap:
        data = np.asarray(values, dtype=float)
        selected = selected_control_indices(
            control_terms,
            control_names=tuple(self.control_names),
            export_channels=tuple(self.export_channels),
        )
        durations = np.asarray(time_grid.step_durations_s, dtype=float)
        output = np.array(data, copy=True)
        alpha_cache: dict[int, np.ndarray] = {}
        for index in selected:
            filtered, alphas = _first_order_lowpass_forward(output[index], durations, float(self.cutoff_hz))
            output[index] = filtered
            alpha_cache[int(index)] = alphas

        def pullback(gradient_output: np.ndarray) -> np.ndarray:
            gradient = np.asarray(gradient_output, dtype=float)
            result = np.array(gradient, copy=True)
            for index in selected:
                result[index] = _first_order_lowpass_pullback(gradient[index], alpha_cache[int(index)])
            return result

        delta = output - data
        return _AppliedHardwareMap(
            values=output,
            pullback=pullback,
            report=HardwareMapReport(
                name=type(self).__name__,
                metrics={
                    "cutoff_hz": float(self.cutoff_hz),
                    "selected_controls": [str(control_terms[index].name) for index in selected],
                    "max_abs_delta": float(np.max(np.abs(delta[selected, :]))) if selected else 0.0,
                },
            ),
        )


def _boundary_window(steps: int, ramp_slices: int, *, apply_start: bool, apply_end: bool) -> np.ndarray:
    window = np.ones(int(steps), dtype=float)
    if int(steps) <= 0 or int(ramp_slices) <= 0:
        return window
    span = min(int(ramp_slices), int(steps))
    if span == 1:
        ramp = np.zeros(1, dtype=float)
    else:
        ramp = np.sin(0.5 * np.pi * np.arange(span, dtype=float) / float(span - 1))
    if apply_start:
        window[:span] *= ramp
    if apply_end:
        window[-span:] *= ramp[::-1]
    return window


@dataclass(frozen=True)
class BoundaryWindowHardwareMap(HardwareMap):
    ramp_slices: int = 1
    apply_start: bool = True
    apply_end: bool = True
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.ramp_slices) < 1:
            raise ValueError("BoundaryWindowHardwareMap.ramp_slices must be at least 1.")
        if not bool(self.apply_start) and not bool(self.apply_end):
            raise ValueError("BoundaryWindowHardwareMap must apply to the start, the end, or both.")

    def apply(
        self,
        values: np.ndarray,
        *,
        control_terms: tuple["ControlTerm", ...],
        time_grid: "PiecewiseConstantTimeGrid",
    ) -> _AppliedHardwareMap:
        data = np.asarray(values, dtype=float)
        selected = selected_control_indices(
            control_terms,
            control_names=tuple(self.control_names),
            export_channels=tuple(self.export_channels),
        )
        window = _boundary_window(
            time_grid.steps,
            int(self.ramp_slices),
            apply_start=bool(self.apply_start),
            apply_end=bool(self.apply_end),
        )
        output = np.array(data, copy=True)
        if selected:
            output[np.asarray(selected, dtype=int), :] *= window[None, :]

        def pullback(gradient_output: np.ndarray) -> np.ndarray:
            gradient = np.asarray(gradient_output, dtype=float)
            result = np.array(gradient, copy=True)
            if selected:
                result[np.asarray(selected, dtype=int), :] *= window[None, :]
            return result

        start_value = float(np.max(np.abs(output[np.asarray(selected, dtype=int), 0]))) if selected else 0.0
        end_value = float(np.max(np.abs(output[np.asarray(selected, dtype=int), -1]))) if selected else 0.0
        return _AppliedHardwareMap(
            values=output,
            pullback=pullback,
            report=HardwareMapReport(
                name=type(self).__name__,
                metrics={
                    "ramp_slices": int(self.ramp_slices),
                    "apply_start": bool(self.apply_start),
                    "apply_end": bool(self.apply_end),
                    "selected_controls": [str(control_terms[index].name) for index in selected],
                    "boundary_start_max_abs": start_value,
                    "boundary_end_max_abs": end_value,
                },
            ),
        )


@dataclass(frozen=True)
class SmoothIQRadiusLimitHardwareMap(HardwareMap):
    amplitude_max: float
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if float(self.amplitude_max) <= 0.0:
            raise ValueError("SmoothIQRadiusLimitHardwareMap.amplitude_max must be positive.")
        if float(self.epsilon) <= 0.0:
            raise ValueError("SmoothIQRadiusLimitHardwareMap.epsilon must be positive.")

    def apply(
        self,
        values: np.ndarray,
        *,
        control_terms: tuple["ControlTerm", ...],
        time_grid: "PiecewiseConstantTimeGrid",
    ) -> _AppliedHardwareMap:
        _ = time_grid
        data = np.asarray(values, dtype=float)
        pairs = selected_iq_pairs(
            control_terms,
            control_names=tuple(self.control_names),
            export_channels=tuple(self.export_channels),
        )
        output = np.array(data, copy=True)
        pair_cache: dict[str, dict[str, np.ndarray]] = {}
        amplitude_max = float(self.amplitude_max)
        epsilon = float(self.epsilon)

        for channel, i_index, q_index in pairs:
            i_values = np.asarray(data[i_index, :], dtype=float)
            q_values = np.asarray(data[q_index, :], dtype=float)
            radius = np.sqrt(np.square(i_values) + np.square(q_values))
            scale = np.ones_like(radius, dtype=float)
            mask = radius > epsilon
            z = np.zeros_like(radius, dtype=float)
            z[mask] = radius[mask] / amplitude_max
            scale[mask] = np.tanh(z[mask]) / np.maximum(z[mask], epsilon)
            output[i_index, :] = scale * i_values
            output[q_index, :] = scale * q_values
            pair_cache[channel] = {
                "i_values": i_values,
                "q_values": q_values,
                "radius": radius,
                "scale": scale,
            }

        def pullback(gradient_output: np.ndarray) -> np.ndarray:
            gradient = np.asarray(gradient_output, dtype=float)
            result = np.array(gradient, copy=True)
            for channel, i_index, q_index in pairs:
                cached = pair_cache[channel]
                i_values = cached["i_values"]
                q_values = cached["q_values"]
                radius = cached["radius"]
                scale = cached["scale"]
                grad_i = np.asarray(gradient[i_index, :], dtype=float)
                grad_q = np.asarray(gradient[q_index, :], dtype=float)
                dot = grad_i * i_values + grad_q * q_values
                phi_prime = np.zeros_like(radius, dtype=float)
                mask = radius > epsilon
                if np.any(mask):
                    z = radius[mask] / amplitude_max
                    sech_sq = 1.0 / np.cosh(z) ** 2
                    phi_prime[mask] = (z * sech_sq - np.tanh(z)) / (amplitude_max * np.maximum(np.square(z), epsilon))
                correction = np.zeros_like(radius, dtype=float)
                correction[mask] = (phi_prime[mask] / radius[mask]) * dot[mask]
                result[i_index, :] = scale * grad_i + correction * i_values
                result[q_index, :] = scale * grad_q + correction * q_values
            return result

        metrics: dict[str, Any] = {
            "amplitude_max": amplitude_max,
            "selected_channels": [channel for channel, _i, _q in pairs],
            "max_command_radius": 0.0,
            "max_physical_radius": 0.0,
            "clipping_fraction": 0.0,
        }
        if pairs:
            command_radii = []
            physical_radii = []
            clipped_counts = []
            total_counts = []
            for channel, i_index, q_index in pairs:
                _ = channel
                command_radius = np.sqrt(np.square(data[i_index, :]) + np.square(data[q_index, :]))
                physical_radius = np.sqrt(np.square(output[i_index, :]) + np.square(output[q_index, :]))
                command_radii.append(float(np.max(command_radius)))
                physical_radii.append(float(np.max(physical_radius)))
                clipped_counts.append(int(np.count_nonzero(command_radius > amplitude_max + 1.0e-15)))
                total_counts.append(int(command_radius.size))
            metrics["max_command_radius"] = float(max(command_radii))
            metrics["max_physical_radius"] = float(max(physical_radii))
            metrics["clipping_fraction"] = float(np.sum(clipped_counts) / max(np.sum(total_counts), 1))

        return _AppliedHardwareMap(
            values=output,
            pullback=pullback,
            report=HardwareMapReport(name=type(self).__name__, metrics=metrics),
        )


@dataclass(frozen=True)
class QuantizationHardwareMap(HardwareMap):
    """DAC quantization model with N-bit resolution.

    Quantizes the selected control waveform to ``2 ** n_bits`` discrete levels
    uniformly spanning each control term's ``amplitude_bounds``.  Finite amplitude
    bounds are required on every selected control.

    The gradient uses the **straight-through estimator** (identity pullback), so the
    optimizer can descend on continuous Fourier / held-sample parameters while the
    forward model reflects the discretised physical waveform.

    To use this as a *validation-only* step without affecting GRAPE optimization, set
    ``GrapeConfig(apply_hardware_in_forward_model=False)`` and include this map in the
    :class:`HardwareModel`; the quantized waveform will then appear only in the
    reported ``physical_values`` diagnostics.

    Args:
        n_bits: DAC resolution in bits (≥ 1).  Produces ``2 ** n_bits`` levels.
        control_names: Restrict to these control names (empty = all controls).
        export_channels: Restrict to these export channels (empty = all channels).
    """

    n_bits: int
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.n_bits) < 1:
            raise ValueError("QuantizationHardwareMap.n_bits must be at least 1.")

    def apply(
        self,
        values: np.ndarray,
        *,
        control_terms: tuple["ControlTerm", ...],
        time_grid: "PiecewiseConstantTimeGrid",
    ) -> _AppliedHardwareMap:
        _ = time_grid
        data = np.asarray(values, dtype=float)
        selected = selected_control_indices(
            control_terms,
            control_names=tuple(self.control_names),
            export_channels=tuple(self.export_channels),
        )
        n_levels = int(2 ** int(self.n_bits))
        output = np.array(data, copy=True)
        max_errors: list[float] = []

        for index in selected:
            lower = float(control_terms[index].amplitude_bounds[0])
            upper = float(control_terms[index].amplitude_bounds[1])
            if not (np.isfinite(lower) and np.isfinite(upper)):
                raise ValueError(
                    f"QuantizationHardwareMap requires finite amplitude_bounds on all selected controls; "
                    f"control '{control_terms[index].name}' has bounds ({lower}, {upper})."
                )
            if abs(upper - lower) < 1.0e-18:
                continue
            step = (upper - lower) / (n_levels - 1)
            quantized = np.round((data[index] - lower) / step) * step + lower
            quantized = np.clip(quantized, lower, upper)
            max_errors.append(float(np.max(np.abs(quantized - data[index]))))
            output[index] = quantized

        def pullback(gradient_output: np.ndarray) -> np.ndarray:
            # Straight-through estimator: gradient passes unchanged through quantisation.
            return np.asarray(gradient_output, dtype=float)

        return _AppliedHardwareMap(
            values=output,
            pullback=pullback,
            report=HardwareMapReport(
                name=type(self).__name__,
                metrics={
                    "n_bits": int(self.n_bits),
                    "n_levels": int(n_levels),
                    "selected_controls": [str(control_terms[index].name) for index in selected],
                    "max_quantization_error": float(max(max_errors)) if max_errors else 0.0,
                },
            ),
        )


def _fir_forward(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal FIR: y[n] = sum_{l=0}^{L-1} h[l] * x[n-l], x[n]=0 for n<0."""
    N = x.shape[0]
    y = np.zeros_like(x, dtype=float)
    for l, coeff in enumerate(h):
        if N - l > 0:
            y[l:] += float(coeff) * x[:N - l]
    return y


def _fir_pullback(dy: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Transposed FIR (cross-correlation): dx[m] = sum_{l} h[l] * dy[m+l]."""
    N = dy.shape[0]
    dx = np.zeros_like(dy, dtype=float)
    for l, coeff in enumerate(h):
        if N - l > 0:
            dx[:N - l] += float(coeff) * dy[l:]
    return dx


@dataclass(frozen=True)
class FIRHardwareMap(HardwareMap):
    """General causal finite-impulse-response (FIR) filter on control waveforms.

    Computes the causal convolution:

        y[n] = sum_{l=0}^{L-1} kernel[l] * x[n-l],   x[n] = 0 for n < 0

    Gradients flow through the transposed convolution (cross-correlation with ``kernel``),
    which is the exact adjoint of the forward operation.

    **Pre-emphasis use case:** to partially compensate a first-order low-pass hardware
    filter with time constant *tau* and step *dt*, use the high-pass kernel
    ``[1 + tau/dt, -tau/dt]``.  This pre-distorts the command so that the cascaded
    response (pre-emphasis followed by the hardware filter) better approximates unity.

    Args:
        kernel: FIR filter tap weights ``h[0], h[1], ..., h[L-1]`` as a tuple.
        control_names: Restrict to these control names (empty = all controls).
        export_channels: Restrict to these export channels (empty = all channels).
    """

    kernel: tuple[float, ...]
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kernel:
            raise ValueError("FIRHardwareMap.kernel must be non-empty.")
        object.__setattr__(self, "kernel", tuple(float(v) for v in self.kernel))

    def apply(
        self,
        values: np.ndarray,
        *,
        control_terms: tuple["ControlTerm", ...],
        time_grid: "PiecewiseConstantTimeGrid",
    ) -> _AppliedHardwareMap:
        _ = time_grid
        data = np.asarray(values, dtype=float)
        selected = selected_control_indices(
            control_terms,
            control_names=tuple(self.control_names),
            export_channels=tuple(self.export_channels),
        )
        h = np.asarray(self.kernel, dtype=float)
        output = np.array(data, copy=True)
        for index in selected:
            output[index] = _fir_forward(np.asarray(data[index], dtype=float), h)

        def pullback(gradient_output: np.ndarray) -> np.ndarray:
            grad = np.asarray(gradient_output, dtype=float)
            result = np.array(grad, copy=True)
            for index in selected:
                result[index] = _fir_pullback(np.asarray(grad[index], dtype=float), h)
            return result

        delta = output - data
        return _AppliedHardwareMap(
            values=output,
            pullback=pullback,
            report=HardwareMapReport(
                name=type(self).__name__,
                metrics={
                    "kernel_length": int(len(h)),
                    "kernel": list(self.kernel),
                    "selected_controls": [str(control_terms[index].name) for index in selected],
                    "max_abs_delta": float(np.max(np.abs(delta[np.asarray(selected, dtype=int), :]))) if selected else 0.0,
                },
            ),
        )


__all__ = [
    "HardwareMapReport",
    "ResolvedControlWaveforms",
    "HardwareMap",
    "HardwareModel",
    "FirstOrderLowPassHardwareMap",
    "BoundaryWindowHardwareMap",
    "SmoothIQRadiusLimitHardwareMap",
    "QuantizationHardwareMap",
    "FIRHardwareMap",
    "selected_control_indices",
    "selected_iq_pairs",
    "resolve_control_schedule",
]