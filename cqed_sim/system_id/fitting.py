from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from cqed_sim.calibration_targets.common import CalibrationResult

from .calibration_hooks import CalibrationEvidence
from .priors import FixedPrior, NormalPrior


CalibrationEvidenceCategory = Literal["model", "noise", "measurement", "hardware"]
PerParameterFloat = float | Mapping[str, float]
ParameterBounds = Mapping[str, tuple[float | None, float | None]]


def _coerce_trace_axes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_name: str,
    y_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError(f"{x_name} and {y_name} must have the same length.")
    if x_values.size < 3:
        raise ValueError(f"{x_name} and {y_name} must contain at least three samples.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError(f"{x_name} and {y_name} must be finite.")
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    if np.any(np.diff(x_values) <= 0.0):
        raise ValueError(f"{x_name} values must be strictly increasing.")
    return x_values, y_values


def _clip_initial_guess(
    p0: tuple[float, ...],
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
) -> tuple[float, ...]:
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    guess = np.asarray(p0, dtype=float)
    eps = np.maximum(1.0e-15, 1.0e-9 * np.maximum(1.0, np.abs(guess)))
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    if np.any(finite_lower):
        guess[finite_lower] = np.maximum(guess[finite_lower], lower[finite_lower] + eps[finite_lower])
    if np.any(finite_upper):
        guess[finite_upper] = np.minimum(guess[finite_upper], upper[finite_upper] - eps[finite_upper])
    return tuple(float(value) for value in guess)


def _curve_fit_sigmas(pcov: np.ndarray, size: int) -> np.ndarray:
    if pcov.size == 0:
        return np.zeros(size, dtype=float)
    diag = np.diag(np.asarray(pcov, dtype=float))
    diag = np.where(np.isfinite(diag) & (diag >= 0.0), diag, 0.0)
    return np.sqrt(diag)


def _fit_with_curve_fit(
    model,
    x: np.ndarray,
    y: np.ndarray,
    *,
    p0: tuple[float, ...],
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    clipped_p0 = _clip_initial_guess(p0, bounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(
            model,
            x,
            y,
            p0=clipped_p0,
            bounds=bounds,
            maxfev=20000,
        )
    return np.asarray(popt, dtype=float), _curve_fit_sigmas(np.asarray(pcov, dtype=float), len(clipped_p0))


def _edge_baseline(y: np.ndarray) -> float:
    edge_count = max(2, min(10, y.size // 8 if y.size >= 8 else 2))
    edge_values = np.concatenate((y[:edge_count], y[-edge_count:]))
    return float(np.median(edge_values))


def _estimate_decay_constant(x: np.ndarray, y: np.ndarray, offset: float) -> float:
    amplitude = np.abs(y - float(offset))
    max_amplitude = float(np.max(amplitude))
    if max_amplitude <= 1.0e-15:
        return float(max(x[-1] - x[0], np.diff(x).min()))
    threshold = max_amplitude / np.e
    below = np.flatnonzero(amplitude <= threshold)
    if below.size:
        tau = float(x[int(below[0])] - x[0])
        if tau > 0.0:
            return tau
    span = float(x[-1] - x[0])
    spacing = float(np.median(np.diff(x)))
    return max(span / 3.0, spacing)


def _guess_angular_frequency(x: np.ndarray, y: np.ndarray) -> float:
    spacing = np.diff(x)
    if spacing.size == 0:
        return 0.0
    step = float(np.median(spacing))
    if step <= 0.0 or np.max(np.abs(spacing - step)) > 1.0e-3 * step:
        return 0.0
    centered = np.asarray(y, dtype=float) - float(np.mean(y))
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=step)
    if freqs.size <= 1:
        return 0.0
    index = int(np.argmax(np.abs(spectrum[1:])) + 1)
    if index <= 0:
        return 0.0
    return float(2.0 * np.pi * freqs[index])


def _estimate_lorentzian_width(x: np.ndarray, y: np.ndarray, baseline: float, peak_index: int) -> float:
    residual = np.abs(np.asarray(y, dtype=float) - float(baseline))
    peak_height = float(residual[peak_index])
    if peak_height <= 1.0e-15:
        return float(max((x[-1] - x[0]) / 20.0, np.median(np.diff(x))))
    half_height = peak_height / 2.0
    left = peak_index
    while left > 0 and residual[left] >= half_height:
        left -= 1
    right = peak_index
    while right < residual.size - 1 and residual[right] >= half_height:
        right += 1
    if left == peak_index or right == peak_index:
        return float(max((x[-1] - x[0]) / 20.0, np.median(np.diff(x))))
    return float(max((x[right] - x[left]) / 2.0, np.median(np.diff(x))))


def _exponential_trace(x: np.ndarray, amplitude: float, tau: float, offset: float) -> np.ndarray:
    return float(offset) + float(amplitude) * np.exp(-np.asarray(x, dtype=float) / float(tau))


def _ramsey_trace(x: np.ndarray, amplitude: float, detuning: float, t2_star: float, offset: float, phase: float) -> np.ndarray:
    x_values = np.asarray(x, dtype=float)
    return float(offset) + float(amplitude) * np.exp(-x_values / float(t2_star)) * np.cos(float(detuning) * x_values + float(phase))


def _rabi_trace(x: np.ndarray, amplitude: float, omega_scale: float, offset: float, phase: float, duration: float) -> np.ndarray:
    x_values = np.asarray(x, dtype=float)
    angle = 0.5 * float(omega_scale) * x_values * float(duration) + float(phase)
    return float(offset) + float(amplitude) * np.sin(angle) ** 2


def _lorentzian_trace(x: np.ndarray, amplitude: float, center: float, linewidth: float, offset: float) -> np.ndarray:
    gamma = max(float(linewidth), 1.0e-15)
    x_values = np.asarray(x, dtype=float)
    return float(offset) + float(amplitude) / (1.0 + ((x_values - float(center)) / gamma) ** 2)


def _fit_exponential_result(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_key: str,
    lifetime_key: str,
    fit_model: str,
) -> CalibrationResult:
    offset_guess = _edge_baseline(y)
    amplitude_guess = float(y[0] - offset_guess)
    if abs(amplitude_guess) <= 1.0e-12:
        amplitude_guess = float(np.max(y) - np.min(y))
    tau_guess = _estimate_decay_constant(x, y, offset_guess)
    scale = max(1.0, float(np.max(np.abs(y))) * 2.0)
    span = float(x[-1] - x[0])
    spacing = float(np.median(np.diff(x)))
    popt, sigma = _fit_with_curve_fit(
        _exponential_trace,
        x,
        y,
        p0=(amplitude_guess, tau_guess, offset_guess),
        bounds=(
            (-scale, max(spacing * 1.0e-3, 1.0e-15), -scale),
            (scale, max(span * 100.0, spacing * 10.0), scale),
        ),
    )
    return CalibrationResult(
        fitted_parameters={
            lifetime_key: float(popt[1]),
            "amplitude": float(popt[0]),
            "offset": float(popt[2]),
        },
        uncertainties={
            lifetime_key: float(sigma[1]),
            "amplitude": float(sigma[0]),
            "offset": float(sigma[2]),
        },
        raw_data={x_key: x, "excited_population": y},
        metadata={"fit_model": fit_model},
    )


def fit_t1_trace(delays: np.ndarray, excited_population: np.ndarray) -> CalibrationResult:
    """Fit an offset exponential decay to a measured T1 trace."""

    x_values, y_values = _coerce_trace_axes(delays, excited_population, x_name="delays", y_name="excited_population")
    return _fit_exponential_result(
        x_values,
        y_values,
        x_key="delays",
        lifetime_key="t1",
        fit_model="offset_exponential_decay",
    )


def fit_t2_echo_trace(delays: np.ndarray, excited_population: np.ndarray) -> CalibrationResult:
    """Fit an offset exponential decay to a measured Hahn-echo trace."""

    x_values, y_values = _coerce_trace_axes(delays, excited_population, x_name="delays", y_name="excited_population")
    return _fit_exponential_result(
        x_values,
        y_values,
        x_key="delays",
        lifetime_key="t2_echo",
        fit_model="offset_echo_decay",
    )


def fit_ramsey_trace(delays: np.ndarray, excited_population: np.ndarray) -> CalibrationResult:
    """Fit a decaying cosine Ramsey trace with offset and phase."""

    x_values, y_values = _coerce_trace_axes(delays, excited_population, x_name="delays", y_name="excited_population")
    span = float(x_values[-1] - x_values[0])
    spacing = float(np.median(np.diff(x_values)))
    amplitude_guess = max(float(np.max(y_values) - np.min(y_values)) / 2.0, 1.0e-6)
    offset_guess = float(np.mean(y_values))
    detuning_guess = _guess_angular_frequency(x_values, y_values)
    if abs(detuning_guess) <= 1.0e-12:
        detuning_guess = 2.0 * np.pi / max(span, spacing)
    t2_guess = max(_estimate_decay_constant(x_values, y_values, offset_guess), spacing)
    scale = max(1.0, float(np.max(np.abs(y_values))) * 2.0)
    detuning_bound = max(abs(detuning_guess) * 50.0, 200.0 * np.pi / max(span, spacing))
    popt, sigma = _fit_with_curve_fit(
        _ramsey_trace,
        x_values,
        y_values,
        p0=(amplitude_guess, detuning_guess, t2_guess, offset_guess, 0.0),
        bounds=(
            (-scale, -detuning_bound, max(spacing * 1.0e-3, 1.0e-15), -scale, -2.0 * np.pi),
            (scale, detuning_bound, max(span * 100.0, spacing * 10.0), scale, 2.0 * np.pi),
        ),
    )
    return CalibrationResult(
        fitted_parameters={
            "delta_omega": float(popt[1]),
            "t2_star": float(popt[2]),
            "amplitude": float(popt[0]),
            "offset": float(popt[3]),
            "phase": float(popt[4]),
        },
        uncertainties={
            "delta_omega": float(sigma[1]),
            "t2_star": float(sigma[2]),
            "amplitude": float(sigma[0]),
            "offset": float(sigma[3]),
            "phase": float(sigma[4]),
        },
        raw_data={"delays": x_values, "excited_population": y_values},
        metadata={"fit_model": "offset_decaying_cosine"},
    )


def fit_rabi_trace(
    amplitudes: np.ndarray,
    excited_population: np.ndarray,
    *,
    duration: float,
) -> CalibrationResult:
    """Fit a phase-shifted sin-squared Rabi trace with offset."""

    if float(duration) <= 0.0:
        raise ValueError("duration must be positive.")
    x_values, y_values = _coerce_trace_axes(amplitudes, excited_population, x_name="amplitudes", y_name="excited_population")
    span = float(x_values[-1] - x_values[0])
    spacing = float(np.median(np.diff(x_values)))
    amplitude_guess = max(float(np.max(y_values) - np.min(y_values)), 1.0e-6)
    offset_guess = float(np.min(y_values))
    angular_frequency_guess = _guess_angular_frequency(x_values, y_values)
    if angular_frequency_guess <= 1.0e-12:
        angular_frequency_guess = np.pi / max(span, spacing)
    omega_scale_guess = max(angular_frequency_guess / float(duration), 1.0e-12)
    scale = max(1.0, float(np.max(np.abs(y_values))) * 2.0)
    omega_scale_bound = max(omega_scale_guess * 50.0, 100.0 * np.pi / max(span * float(duration), spacing * float(duration)))
    popt, sigma = _fit_with_curve_fit(
        lambda x_axis, amplitude, omega_scale, offset, phase: _rabi_trace(x_axis, amplitude, omega_scale, offset, phase, duration),
        x_values,
        y_values,
        p0=(amplitude_guess, omega_scale_guess, offset_guess, 0.0),
        bounds=(
            (-scale, 1.0e-15, -scale, -2.0 * np.pi),
            (scale, omega_scale_bound, scale, 2.0 * np.pi),
        ),
    )
    return CalibrationResult(
        fitted_parameters={
            "omega_scale": float(popt[1]),
            "duration": float(duration),
            "amplitude": float(popt[0]),
            "offset": float(popt[2]),
            "phase": float(popt[3]),
        },
        uncertainties={
            "omega_scale": float(sigma[1]),
            "amplitude": float(sigma[0]),
            "offset": float(sigma[2]),
            "phase": float(sigma[3]),
        },
        raw_data={"amplitudes": x_values, "excited_population": y_values},
        metadata={"fit_model": "offset_sin_squared", "duration": float(duration)},
    )


def fit_spectroscopy_trace(drive_frequencies: np.ndarray, response: np.ndarray) -> CalibrationResult:
    """Fit a single Lorentzian peak or dip from a measured spectroscopy trace."""

    x_values, y_values = _coerce_trace_axes(drive_frequencies, response, x_name="drive_frequencies", y_name="response")
    baseline_guess = _edge_baseline(y_values)
    peak_index = int(np.argmax(np.abs(y_values - baseline_guess)))
    amplitude_guess = float(y_values[peak_index] - baseline_guess)
    if abs(amplitude_guess) <= 1.0e-12:
        amplitude_guess = float(np.max(y_values) - np.min(y_values))
    center_guess = float(x_values[peak_index])
    linewidth_guess = _estimate_lorentzian_width(x_values, y_values, baseline_guess, peak_index)
    span = float(x_values[-1] - x_values[0])
    spacing = float(np.median(np.diff(x_values)))
    scale = max(1.0, float(np.max(np.abs(y_values))) * 2.0)
    popt, sigma = _fit_with_curve_fit(
        _lorentzian_trace,
        x_values,
        y_values,
        p0=(amplitude_guess, center_guess, linewidth_guess, baseline_guess),
        bounds=(
            (-scale, float(x_values[0]), max(spacing * 0.5, 1.0e-15), -scale),
            (scale, float(x_values[-1]), max(span * 10.0, spacing * 10.0), scale),
        ),
    )
    return CalibrationResult(
        fitted_parameters={
            "omega_peak": float(popt[1]),
            "linewidth": float(popt[2]),
            "amplitude": float(popt[0]),
            "offset": float(popt[3]),
        },
        uncertainties={
            "omega_peak": float(sigma[1]),
            "linewidth": float(sigma[2]),
            "amplitude": float(sigma[0]),
            "offset": float(sigma[3]),
        },
        raw_data={"drive_frequencies": x_values, "response": y_values},
        metadata={"fit_model": "single_lorentzian_peak"},
    )


def prior_from_fit(
    value: float,
    sigma: float,
    *,
    low: float | None = None,
    high: float | None = None,
    sigma_scale: float = 1.0,
    min_sigma: float = 0.0,
) -> FixedPrior | NormalPrior:
    """Convert a fit estimate and uncertainty into a bounded prior."""

    resolved_value = float(value)
    if low is not None:
        resolved_value = max(float(low), resolved_value)
    if high is not None:
        resolved_value = min(float(high), resolved_value)
    resolved_sigma = abs(float(sigma)) if np.isfinite(sigma) else 0.0
    resolved_sigma = max(resolved_sigma * float(sigma_scale), float(min_sigma))
    if resolved_sigma <= 0.0:
        return FixedPrior(value=resolved_value)
    return NormalPrior(mean=resolved_value, sigma=resolved_sigma, low=low, high=high)


def _resolve_per_parameter_float(values: PerParameterFloat, key: str, default: float) -> float:
    if isinstance(values, Mapping):
        return float(values.get(key, default))
    return float(values)


def evidence_from_fit(
    result: CalibrationResult,
    *,
    category: CalibrationEvidenceCategory,
    parameter_map: Mapping[str, str] | None = None,
    channel: str | None = None,
    bounds: ParameterBounds | None = None,
    sigma_scale: PerParameterFloat = 1.0,
    min_sigma: PerParameterFloat = 0.0,
    notes: Mapping[str, Any] | None = None,
) -> CalibrationEvidence:
    """Map selected fitted parameters into a CalibrationEvidence category."""

    if category == "hardware" and not channel:
        raise ValueError("channel is required when category='hardware'.")
    if category != "hardware" and channel is not None:
        raise ValueError("channel is only valid when category='hardware'.")

    resolved_map = dict(parameter_map) if parameter_map is not None else {
        key: key for key in result.fitted_parameters
    }
    priors = {}
    for source_key, target_key in resolved_map.items():
        if source_key not in result.fitted_parameters:
            raise KeyError(f"Unknown fitted parameter '{source_key}'.")
        lower, upper = (None, None) if bounds is None else bounds.get(target_key, (None, None))
        priors[target_key] = prior_from_fit(
            result.fitted_parameters[source_key],
            result.uncertainties.get(source_key, 0.0),
            low=lower,
            high=upper,
            sigma_scale=_resolve_per_parameter_float(sigma_scale, target_key, 1.0),
            min_sigma=_resolve_per_parameter_float(min_sigma, target_key, 0.0),
        )

    fit_source = {
        "category": category,
        "parameter_map": dict(resolved_map),
        "fitted_parameters": {key: float(result.fitted_parameters[key]) for key in resolved_map},
        "fit_metadata": dict(result.metadata),
    }
    if channel is not None:
        fit_source["channel"] = str(channel)

    resolved_notes = dict(notes or {})
    existing_sources = list(resolved_notes.get("fit_sources", []))
    existing_sources.append(fit_source)
    resolved_notes["fit_sources"] = existing_sources

    if category == "model":
        return CalibrationEvidence(model_posteriors=priors, notes=resolved_notes)
    if category == "noise":
        return CalibrationEvidence(noise_posteriors=priors, notes=resolved_notes)
    if category == "measurement":
        return CalibrationEvidence(measurement_posteriors=priors, notes=resolved_notes)
    return CalibrationEvidence(hardware_posteriors={str(channel): priors}, notes=resolved_notes)


def _merge_mapping_group(
    mappings: list[Mapping[str, Any]],
    *,
    group_name: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key in merged:
                raise ValueError(f"Duplicate {group_name} posterior for '{key}'.")
            merged[key] = value
    return merged


def merge_calibration_evidence(
    *evidences: CalibrationEvidence,
    notes: Mapping[str, Any] | None = None,
) -> CalibrationEvidence:
    """Merge multiple evidence blocks while rejecting duplicate posterior keys."""

    model_posteriors = _merge_mapping_group(
        [evidence.model_posteriors for evidence in evidences],
        group_name="model",
    )
    noise_posteriors = _merge_mapping_group(
        [evidence.noise_posteriors for evidence in evidences],
        group_name="noise",
    )
    measurement_posteriors = _merge_mapping_group(
        [evidence.measurement_posteriors for evidence in evidences],
        group_name="measurement",
    )

    hardware_posteriors: dict[str, dict[str, Any]] = {}
    for evidence in evidences:
        for channel, channel_posteriors in evidence.hardware_posteriors.items():
            merged_channel = hardware_posteriors.setdefault(channel, {})
            for key, value in channel_posteriors.items():
                if key in merged_channel:
                    raise ValueError(f"Duplicate hardware posterior for '{channel}.{key}'.")
                merged_channel[key] = value

    merged_notes: dict[str, Any] = {}
    fit_sources: list[Any] = []
    for evidence in evidences:
        for key, value in evidence.notes.items():
            if key == "fit_sources":
                fit_sources.extend(value if isinstance(value, list) else [value])
                continue
            merged_notes[key] = value
    for key, value in dict(notes or {}).items():
        if key == "fit_sources":
            fit_sources.extend(value if isinstance(value, list) else [value])
            continue
        merged_notes[key] = value
    if fit_sources:
        merged_notes["fit_sources"] = fit_sources

    return CalibrationEvidence(
        model_posteriors=model_posteriors,
        noise_posteriors=noise_posteriors,
        hardware_posteriors=hardware_posteriors,
        measurement_posteriors=measurement_posteriors,
        notes=merged_notes,
    )


__all__ = [
    "CalibrationEvidenceCategory",
    "fit_rabi_trace",
    "fit_ramsey_trace",
    "fit_spectroscopy_trace",
    "fit_t1_trace",
    "fit_t2_echo_trace",
    "prior_from_fit",
    "evidence_from_fit",
    "merge_calibration_evidence",
]