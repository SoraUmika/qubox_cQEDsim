from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.constants import Boltzmann as K_B
from scipy.optimize import minimize
from scipy.special import erfcinv

from cqed_sim.core import FrameSpec
from cqed_sim.measurement import (
    ReadoutChain,
    StrongReadoutMixingSpec,
    build_strong_readout_disturbance,
    estimate_dispersive_critical_photon_number,
)
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import NoiseSpec, SimulationConfig, prepare_simulation

from .hardware import HardwareModel
from .parameterizations import PiecewiseConstantTimeGrid
from .problems import ControlTerm
from .readout_emptying import (
    ReadoutEmptyingConstraints,
    ReadoutEmptyingReplay,
    ReadoutEmptyingResult,
    ReadoutEmptyingSpec,
    _embed_basis,
    _free_segment_mask,
    _metrics_from_replay,
    _segment_edges_s,
    apply_phase_chirp,
    build_emptying_constraint_matrix,
    build_kerr_phase_correction,
    build_readout_emptying_parameterization,
    compute_emptying_null_space,
    evaluate_readout_emptying_with_chain,
    export_readout_emptying_to_pulse,
    replay_kerr_readout_branches,
    replay_linear_readout_branches,
    synthesize_readout_emptying_pulse,
)


def _default_compiler_dt(segment_edges_s: np.ndarray) -> float:
    durations = np.diff(np.asarray(segment_edges_s, dtype=float))
    if durations.size == 0:
        return 1.0e-9
    return float(min(2.0e-9, max(float(np.min(durations)) / 8.0, 1.0e-10)))


def _resolved_scale_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError(f"{name} must contain at least one scale value.")
    if any(value <= 0.0 for value in resolved):
        raise ValueError(f"{name} must contain only positive scale values.")
    return resolved


def _copy_spec(spec: ReadoutEmptyingSpec, **updates: Any) -> ReadoutEmptyingSpec:
    return replace(spec, **updates)


def _updated_spec_for_edges(spec: ReadoutEmptyingSpec, segment_edges_s: np.ndarray) -> ReadoutEmptyingSpec:
    edges = np.asarray(segment_edges_s, dtype=float)
    return _copy_spec(
        spec,
        tau=float(edges[-1] - edges[0]),
        segment_times=tuple(float(value) for value in edges),
        n_segments=int(edges.size - 1),
    )


def _compose_result(
    spec: ReadoutEmptyingSpec,
    constraints: ReadoutEmptyingConstraints | None,
    segment_edges_s: np.ndarray,
    segment_amplitudes: np.ndarray,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ReadoutEmptyingResult:
    edges = np.asarray(segment_edges_s, dtype=float)
    amplitudes = np.asarray(segment_amplitudes, dtype=np.complex128)
    resolved_spec = _updated_spec_for_edges(spec, edges)
    linear_replay = replay_linear_readout_branches(resolved_spec, amplitudes)
    nonlinear_replay = (
        replay_kerr_readout_branches(resolved_spec, amplitudes)
        if abs(float(resolved_spec.kerr)) > 1.0e-18
        else linear_replay
    )
    metrics = _metrics_from_replay(
        nonlinear_replay,
        segment_edges_s=edges,
        segment_amplitudes=amplitudes,
    )
    composed_diagnostics: dict[str, Any] = {
        "linear_metrics": _metrics_from_replay(
            linear_replay,
            segment_edges_s=edges,
            segment_amplitudes=amplitudes,
        ),
        "active_metrics": dict(metrics),
    }
    if nonlinear_replay is not linear_replay:
        composed_diagnostics["nonlinear_metrics"] = dict(metrics)
    if diagnostics is not None:
        composed_diagnostics.update(dict(diagnostics))
    return ReadoutEmptyingResult(
        spec=resolved_spec,
        constraints=constraints,
        segment_amplitudes=np.asarray(amplitudes, dtype=np.complex128),
        segment_edges_s=np.asarray(edges, dtype=float),
        time_grid_s=np.asarray(nonlinear_replay.time_grid_s, dtype=float),
        command_waveform=np.asarray(nonlinear_replay.command_waveform, dtype=np.complex128),
        trajectories={key: np.asarray(value, dtype=np.complex128) for key, value in nonlinear_replay.trajectories.items()},
        final_alpha={key: complex(value) for key, value in nonlinear_replay.final_alpha.items()},
        final_n={key: float(value) for key, value in nonlinear_replay.final_n.items()},
        metrics=dict(metrics),
        diagnostics=composed_diagnostics,
    )


def _square_pulse_result(reference: ReadoutEmptyingResult) -> ReadoutEmptyingResult:
    amplitude = float(reference.metrics["waveform_l2_norm"] / np.sqrt(max(reference.spec.tau, 1.0e-18)))
    segment_amplitudes = np.full(int(reference.spec.n_segments), amplitude, dtype=np.complex128)
    return _compose_result(
        reference.spec,
        reference.constraints,
        reference.segment_edges_s,
        segment_amplitudes,
        diagnostics={
            "source": "square_baseline",
            "matched_waveform_l2_norm": float(reference.metrics["waveform_l2_norm"]),
        },
    )


def _command_waveform_segments(result: ReadoutEmptyingResult) -> np.ndarray:
    return np.asarray(result.segment_amplitudes, dtype=np.complex128)


def _hardware_controls(channel: str = "readout") -> tuple[ControlTerm, ControlTerm]:
    zero = np.zeros((1, 1), dtype=np.complex128)
    return (
        ControlTerm(
            name=f"{channel}_I",
            operator=zero,
            export_channel=str(channel),
            drive_target=str(channel),
            quadrature="I",
        ),
        ControlTerm(
            name=f"{channel}_Q",
            operator=zero,
            export_channel=str(channel),
            drive_target=str(channel),
            quadrature="Q",
        ),
    )


def _apply_hardware_model(
    result: ReadoutEmptyingResult,
    hardware_model: HardwareModel | None,
    *,
    channel: str = "readout",
) -> tuple[ReadoutEmptyingResult, dict[str, Any]]:
    if hardware_model is None:
        return result, {
            "hardware_active": False,
            "command_values": np.vstack([result.segment_amplitudes.real, result.segment_amplitudes.imag]).astype(float, copy=False),
            "physical_values": np.vstack([result.segment_amplitudes.real, result.segment_amplitudes.imag]).astype(float, copy=False),
            "hardware_metrics": {
                "hardware_active": False,
                "hardware_map_count": 0,
                "command_physical_rms_delta": 0.0,
                "command_physical_peak_delta": 0.0,
            },
            "hardware_reports": (),
        }

    command_values = np.vstack([result.segment_amplitudes.real, result.segment_amplitudes.imag]).astype(float, copy=False)
    control_terms = _hardware_controls(channel)
    time_grid = PiecewiseConstantTimeGrid(step_durations_s=tuple(np.diff(np.asarray(result.segment_edges_s, dtype=float))))
    physical_values, reports, metrics, _ = hardware_model.apply(
        command_values,
        control_terms=control_terms,
        time_grid=time_grid,
    )
    delta = np.asarray(physical_values, dtype=float) - np.asarray(command_values, dtype=float)
    enriched_metrics = {
        **dict(metrics),
        "command_physical_rms_delta": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "command_physical_peak_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }
    physical_segments = np.asarray(physical_values[0] + 1j * physical_values[1], dtype=np.complex128)
    physical_result = _compose_result(
        result.spec,
        result.constraints,
        result.segment_edges_s,
        physical_segments,
        diagnostics={
            "source": "hardware_replayed",
            "command_segment_amplitudes": np.asarray(result.segment_amplitudes, dtype=np.complex128),
            "hardware_metrics": dict(enriched_metrics),
            "hardware_reports": tuple({"name": report.name, "metrics": dict(report.metrics)} for report in reports),
        },
    )
    return physical_result, {
        "hardware_active": True,
        "command_values": np.asarray(command_values, dtype=float),
        "physical_values": np.asarray(physical_values, dtype=float),
        "hardware_metrics": dict(enriched_metrics),
        "hardware_reports": tuple({"name": report.name, "metrics": dict(report.metrics)} for report in reports),
    }


def _drive_frequency_from_reference(
    spec: ReadoutEmptyingSpec,
    *,
    measurement_chain: ReadoutChain | None = None,
    readout_model: Any | None = None,
) -> float | None:
    if measurement_chain is not None:
        return float(measurement_chain.resonator.omega_r + 0.5 * spec.chi + spec.detuning_center)
    if readout_model is None:
        return None
    omega_r = None
    if hasattr(readout_model, "omega_r"):
        omega_r = float(getattr(readout_model, "omega_r"))
    elif hasattr(readout_model, "as_universal_model"):
        universal = readout_model.as_universal_model()
        for mode in getattr(universal, "bosonic_modes", ()):
            aliases = set(getattr(mode, "aliases", ())) | {getattr(mode, "label", "")}
            if "readout" in aliases:
                omega_r = float(mode.omega)
                break
    if omega_r is None:
        return None
    return float(omega_r + 0.5 * spec.chi + spec.detuning_center)


def _clone_measurement_chain_with_noise_temperature(chain: ReadoutChain, noise_temperature: float) -> ReadoutChain:
    return replace(
        chain,
        amplifier=replace(chain.amplifier, noise_temperature=float(max(noise_temperature, 0.0))),
    )


def _sigma_for_target_overlap_error(center_distance: float, target_error: float) -> float:
    distance = float(max(center_distance, 0.0))
    error = float(target_error)
    if distance <= 0.0 or error <= 0.0:
        return 0.0
    if error >= 0.5:
        return float("inf")
    scale = float(erfcinv(2.0 * error))
    if abs(scale) <= 1.0e-18:
        return float("inf")
    return float(distance / (2.0 * np.sqrt(2.0) * scale))


def _noise_temperature_for_integrated_sigma(chain: ReadoutChain, sigma: float, *, dt: float, duration: float) -> float:
    if float(sigma) <= 0.0:
        return 0.0
    n_samples = max(1, int(np.ceil(float(duration) / float(dt))))
    impedance = float(chain.amplifier.impedance_ohm)
    if float(dt) <= 0.0 or impedance <= 0.0:
        return float(chain.amplifier.noise_temperature)
    instantaneous_sigma = float(sigma) * np.sqrt(float(n_samples))
    return float((instantaneous_sigma * instantaneous_sigma * float(dt)) / (2.0 * K_B * impedance))


def _resolved_measurement_chain(
    reference_result: ReadoutEmptyingResult | None,
    config: "ReadoutEmptyingVerificationConfig | ReadoutEmptyingRefinementConfig",
) -> tuple[ReadoutChain | None, dict[str, Any]]:
    chain = config.measurement_chain
    if chain is None:
        return None, {"mode": "disabled"}

    mode = str(getattr(config, "measurement_noise_mode", "as_provided")).lower()
    if mode == "as_provided" or reference_result is None:
        return chain, {
            "mode": "as_provided",
            "noise_temperature": float(chain.amplifier.noise_temperature),
            "noise_std": float(chain.integrated_noise_sigma(duration=float(reference_result.spec.tau), dt=float(chain.dt)))
            if reference_result is not None
            else float(chain.integrated_noise_sigma()),
        }

    if mode != "calibrated_target_error":
        raise ValueError(f"Unsupported measurement_noise_mode '{mode}'.")

    zero_noise_chain = _clone_measurement_chain_with_noise_temperature(chain, 0.0)
    zero_noise_measurement = evaluate_readout_emptying_with_chain(reference_result, zero_noise_chain, shots_per_branch=0)
    center_distance = float(zero_noise_measurement["metrics"].get("measurement_chain_center_distance", 0.0))
    resolved_dt = float(chain.dt)
    resolved_duration = float(reference_result.spec.tau)
    target_sigma = _sigma_for_target_overlap_error(
        center_distance,
        float(getattr(config, "measurement_target_square_error", 0.10)),
    )
    target_temperature = _noise_temperature_for_integrated_sigma(
        chain,
        target_sigma,
        dt=resolved_dt,
        duration=resolved_duration,
    )
    resolved_temperature = max(
        float(target_temperature),
        float(getattr(config, "measurement_min_noise_temperature", 0.0)),
    )
    calibrated_chain = _clone_measurement_chain_with_noise_temperature(chain, resolved_temperature)
    return calibrated_chain, {
        "mode": mode,
        "reference_label": "square",
        "target_square_error": float(getattr(config, "measurement_target_square_error", 0.10)),
        "center_distance": float(center_distance),
        "target_noise_std": float(target_sigma),
        "noise_temperature": float(resolved_temperature),
        "noise_std": float(calibrated_chain.integrated_noise_sigma(duration=resolved_duration, dt=resolved_dt)),
    }


def _ringdown_metrics(
    result: ReadoutEmptyingResult,
    *,
    threshold_photons: float,
) -> dict[str, float]:
    threshold = float(threshold_photons)
    if threshold <= 0.0:
        raise ValueError("ringdown_threshold_photons must be positive.")
    final_n = {str(label): float(value) for label, value in result.final_n.items()}
    if not final_n:
        return {
            "post_pulse_residual_integral": 0.0,
            "ringdown_tail_energy": 0.0,
            "ringdown_time_to_threshold": 0.0,
        }
    kappa = float(result.spec.kappa)
    worst_label = max(final_n, key=final_n.get)
    worst_final = float(final_n[worst_label])
    if kappa <= 0.0:
        time_to_threshold = float(0.0 if worst_final <= threshold else np.inf)
        post_integral = float(0.0 if worst_final <= 0.0 else np.inf)
        tail_energy = float(0.0 if worst_final <= threshold else np.inf)
    else:
        time_to_threshold = float(max(np.log(max(worst_final, threshold) / threshold), 0.0) / kappa)
        post_integral = float(worst_final / kappa)
        tail_energy = float(max(worst_final - threshold, 0.0) / kappa)
    metrics = {
        "post_pulse_residual_integral": float(post_integral),
        "ringdown_tail_energy": float(tail_energy),
        "ringdown_time_to_threshold": float(time_to_threshold),
    }
    for label, value in final_n.items():
        if kappa <= 0.0:
            label_time = float(0.0 if value <= threshold else np.inf)
            label_integral = float(0.0 if value <= 0.0 else np.inf)
            label_tail = float(0.0 if value <= threshold else np.inf)
        else:
            label_time = float(max(np.log(max(value, threshold) / threshold), 0.0) / kappa)
            label_integral = float(value / kappa)
            label_tail = float(max(value - threshold, 0.0) / kappa)
        metrics[f"post_pulse_residual_integral_{label}"] = float(label_integral)
        metrics[f"ringdown_tail_energy_{label}"] = float(label_tail)
        metrics[f"ringdown_time_to_threshold_{label}"] = float(label_time)
    metrics["ringdown_worst_final_residual"] = float(worst_final)
    return metrics


def _resolved_strong_readout_spec(
    config: "ReadoutEmptyingVerificationConfig | ReadoutEmptyingRefinementConfig",
    *,
    result: ReadoutEmptyingResult,
    measurement_chain: ReadoutChain | None,
    readout_model: Any | None,
) -> StrongReadoutMixingSpec | None:
    spec = getattr(config, "strong_readout_spec", None)
    if spec is None or measurement_chain is None:
        return None
    if spec.n_crit is not None:
        return spec
    if readout_model is None:
        return spec
    omega_q = getattr(readout_model, "omega_q", None)
    omega_r = getattr(readout_model, "omega_r", None)
    alpha = getattr(readout_model, "alpha", None)
    if omega_q is None or omega_r is None or alpha is None:
        return spec
    n_crit = estimate_dispersive_critical_photon_number(
        omega_q=float(omega_q),
        omega_mode=float(omega_r),
        alpha=float(alpha),
        chi=float(result.spec.chi),
        g=float(measurement_chain.resonator.g),
    )
    return replace(spec, n_crit=float(n_crit))


def _strong_readout_summary(
    result: ReadoutEmptyingResult,
    *,
    measurement_chain: ReadoutChain | None,
    readout_model: Any | None,
    compiler_dt_s: float | None,
    drive_frequency: float | None,
    config: "ReadoutEmptyingVerificationConfig | ReadoutEmptyingRefinementConfig",
) -> dict[str, Any]:
    resolved_spec = _resolved_strong_readout_spec(
        config,
        result=result,
        measurement_chain=measurement_chain,
        readout_model=readout_model,
    )
    if resolved_spec is None or measurement_chain is None:
        return {"metrics": {}}

    dt = float(measurement_chain.dt if compiler_dt_s is None else compiler_dt_s)
    pulse = export_readout_emptying_to_pulse(result, channel="readout", carrier=0.0)
    compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=float(result.spec.tau))
    drive_envelope = np.asarray(compiled.channels["readout"].baseband[:-1], dtype=np.complex128)
    disturbance = build_strong_readout_disturbance(
        measurement_chain.resonator,
        drive_envelope,
        dt=dt,
        spec=resolved_spec,
        duration=float(result.spec.tau),
        drive_frequency=drive_frequency,
        chi=float(result.spec.chi),
    )
    ge_power = float(np.sum(np.abs(disturbance.ge_envelope) ** 2) * dt)
    ef_power = float(np.sum(np.abs(disturbance.ef_envelope) ** 2) * dt)
    higher_power = float(
        sum(float(np.sum(np.abs(envelope) ** 2) * dt) for envelope in disturbance.higher_envelopes.values())
    )
    metrics = {
        "strong_readout_peak_activation": float(disturbance.peak_activation),
        "strong_readout_peak_mean_occupancy": float(disturbance.peak_mean_occupancy),
        "strong_readout_ge_power_integral": float(ge_power),
        "strong_readout_ef_power_integral": float(ef_power),
        "strong_readout_disturbance_proxy": float(ge_power + ef_power + higher_power),
    }
    return {
        "metrics": metrics,
        "disturbance": disturbance,
        "resolved_spec": resolved_spec,
        "dt": float(dt),
    }


def _resolved_frame(frame: FrameSpec, *, drive_frequency: float | None) -> FrameSpec:
    if drive_frequency is None or abs(float(frame.omega_r_frame)) > 1.0e-18:
        return frame
    return replace(frame, omega_r_frame=float(drive_frequency))


def _default_lindblad_observables(model: Any) -> dict[str, Any]:
    observables = {
        "a_r": model.readout_annihilation(),
        "n_r": model.readout_number(),
        "P_g": model.transmon_level_projector(0),
        "P_e": model.transmon_level_projector(1),
    }
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims", ()))
    if dims and dims[0] >= 3:
        observables["P_f"] = model.transmon_level_projector(2)
    return observables


def _qubit_initial_state(model: Any, level: int):
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims", ()))
    if not dims:
        raise ValueError("readout_model must expose subsystem_dims.")
    levels = (int(level),) + tuple(0 for _ in dims[1:])
    return model.basis_state(*levels)


def _lindblad_validation(
    result: ReadoutEmptyingResult,
    *,
    readout_model: Any,
    frame: FrameSpec,
    noise: NoiseSpec | None,
    compiler_dt_s: float | None,
    max_step_s: float | None,
    simulation_config: SimulationConfig | None,
    drive_frequency: float | None,
) -> dict[str, Any]:
    pulse = export_readout_emptying_to_pulse(result, channel="readout", carrier=0.0)
    compiled = SequenceCompiler(
        dt=_default_compiler_dt(result.segment_edges_s) if compiler_dt_s is None else float(compiler_dt_s)
    ).compile([pulse], t_end=float(result.spec.tau))
    resolved_frame = _resolved_frame(frame, drive_frequency=drive_frequency)
    if simulation_config is None:
        solver_config = SimulationConfig(frame=resolved_frame, max_step=max_step_s)
    else:
        solver_config = replace(simulation_config, frame=resolved_frame)
        if max_step_s is not None and simulation_config.max_step is None:
            solver_config = replace(solver_config, max_step=max_step_s)
    session = prepare_simulation(
        readout_model,
        compiled,
        {"readout": "readout"},
        config=solver_config,
        noise=noise,
        e_ops=_default_lindblad_observables(readout_model),
    )
    simulations = {
        "g": session.run(_qubit_initial_state(readout_model, 0)),
        "e": session.run(_qubit_initial_state(readout_model, 1)),
    }
    tlist = np.asarray(compiled.tlist, dtype=float)
    readout_kappa = float(
        result.spec.kappa
        if noise is None or noise.kappa_readout is None or noise.kappa_readout <= 0.0
        else noise.kappa_readout
    )
    alpha = {label: np.asarray(run.expectations["a_r"], dtype=np.complex128) for label, run in simulations.items()}
    photons = {label: np.asarray(run.expectations["n_r"], dtype=float) for label, run in simulations.items()}
    output = {label: np.sqrt(max(readout_kappa, 0.0)) * values for label, values in alpha.items()}
    diff = output["e"] - output["g"]
    separation = float(np.trapezoid(np.abs(diff) ** 2, x=tlist))
    final_pg_g = float(np.real(simulations["g"].expectations["P_g"][-1]))
    final_pe_g = float(np.real(simulations["g"].expectations["P_e"][-1]))
    final_pg_e = float(np.real(simulations["e"].expectations["P_g"][-1]))
    final_pe_e = float(np.real(simulations["e"].expectations["P_e"][-1]))
    final_pf_e = 0.0
    if "P_f" in simulations["e"].expectations:
        final_pf_e = float(np.real(simulations["e"].expectations["P_f"][-1]))
    else:
        final_pf_e = float(max(0.0, 1.0 - final_pg_e - final_pe_e))
    metrics = {
        "lindblad_output_separation": float(separation),
        "lindblad_final_residual_g": float(photons["g"][-1]),
        "lindblad_final_residual_e": float(photons["e"][-1]),
        "lindblad_max_final_residual": float(max(photons["g"][-1], photons["e"][-1])),
        "lindblad_peak_photons_g": float(np.max(photons["g"])),
        "lindblad_peak_photons_e": float(np.max(photons["e"])),
        "g_to_e_probability": float(final_pe_g),
        "g_population_retained": float(final_pg_g),
        "e_to_g_probability": float(final_pg_e),
        "e_population_retained": float(final_pe_e),
        "e_to_f_probability": float(final_pf_e),
        "background_relaxation_total": float(final_pe_g + final_pg_e + final_pf_e),
        "non_qnd_total": float(final_pe_g + final_pg_e + final_pf_e),
        "compiler_dt_s": float(compiled.dt),
    }
    return {
        "metrics": metrics,
        "time_grid_s": tlist,
        "alpha": alpha,
        "photon_numbers": photons,
        "output_field": output,
        "expectations": {label: run.expectations for label, run in simulations.items()},
    }


def _measurement_summary(
    result: ReadoutEmptyingResult,
    chain: ReadoutChain | None,
    *,
    shots_per_branch: int,
    seed: int | None,
) -> dict[str, Any]:
    if chain is None:
        return {"metrics": {}}
    return evaluate_readout_emptying_with_chain(
        result,
        chain,
        shots_per_branch=int(shots_per_branch),
        seed=seed,
    )


def _robustness_result_for_scale(
    result: ReadoutEmptyingResult,
    *,
    scale_kind: str,
    scale: float,
    measurement_chain: ReadoutChain | None,
) -> dict[str, float]:
    base_spec = result.spec
    base_edges = np.asarray(result.segment_edges_s, dtype=float)
    base_segments = np.asarray(result.segment_amplitudes, dtype=np.complex128)
    if scale_kind == "chi":
        variant_spec = _copy_spec(base_spec, chi=float(base_spec.chi) * float(scale))
        variant_edges = base_edges
        variant_segments = base_segments
    elif scale_kind == "kappa":
        variant_spec = _copy_spec(base_spec, kappa=float(base_spec.kappa) * float(scale))
        variant_edges = base_edges
        variant_segments = base_segments
    elif scale_kind == "kerr":
        variant_spec = _copy_spec(base_spec, kerr=float(base_spec.kerr) * float(scale))
        variant_edges = base_edges
        variant_segments = base_segments
    elif scale_kind == "amplitude":
        variant_spec = base_spec
        variant_edges = base_edges
        variant_segments = float(scale) * base_segments
    elif scale_kind == "timing":
        variant_edges = np.asarray(base_edges, dtype=float) * float(scale)
        variant_spec = _updated_spec_for_edges(base_spec, variant_edges)
        variant_segments = base_segments
    else:
        raise ValueError(f"Unsupported robustness scale kind '{scale_kind}'.")
    variant = _compose_result(
        variant_spec,
        result.constraints,
        variant_edges,
        variant_segments,
        diagnostics={"source": f"robustness_{scale_kind}", "scale": float(scale)},
    )
    metrics = dict(variant.metrics)
    if measurement_chain is not None:
        measurement = _measurement_summary(variant, measurement_chain, shots_per_branch=0, seed=None)
        metrics.update(measurement.get("metrics", {}))
    return metrics


def _robustness_summary(
    command_result: ReadoutEmptyingResult,
    nominal_result: ReadoutEmptyingResult,
    config: "ReadoutEmptyingVerificationConfig",
    *,
    measurement_chain: ReadoutChain | None = None,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    overall_residuals: list[float] = []
    overall_separations: list[float] = []
    for scale_kind, scales in (
        ("chi", config.chi_scales),
        ("kappa", config.kappa_scales),
        ("kerr", config.kerr_scales),
        ("amplitude", config.amplitude_scales),
        ("timing", config.timing_scales),
    ):
        records: list[dict[str, float]] = []
        for scale in scales:
            metrics = _robustness_result_for_scale(
                nominal_result,
                scale_kind=scale_kind,
                scale=float(scale),
                measurement_chain=measurement_chain,
            )
            records.append(
                {
                    "scale": float(scale),
                    "max_final_residual_photons": float(metrics["max_final_residual_photons"]),
                    "integrated_branch_separation": float(metrics["integrated_branch_separation"]),
                    "measurement_chain_separation": float(metrics.get("measurement_chain_separation", np.nan)),
                }
            )
        overall_residuals.extend(record["max_final_residual_photons"] for record in records)
        overall_separations.extend(record["integrated_branch_separation"] for record in records)
        summaries[scale_kind] = {
            "records": records,
            "worst_residual": float(max(record["max_final_residual_photons"] for record in records)),
            "best_residual": float(min(record["max_final_residual_photons"] for record in records)),
            "worst_separation": float(min(record["integrated_branch_separation"] for record in records)),
            "best_separation": float(max(record["integrated_branch_separation"] for record in records)),
        }

    hardware_variants: dict[str, Any] = {}
    for label, model in dict(config.hardware_variants).items():
        variant_result, variant_hardware = _apply_hardware_model(command_result, model)
        hardware_variants[str(label)] = {
            "metrics": dict(variant_result.metrics),
            "hardware_metrics": dict(variant_hardware["hardware_metrics"]),
        }
        overall_residuals.append(float(variant_result.metrics["max_final_residual_photons"]))
        overall_separations.append(float(variant_result.metrics["integrated_branch_separation"]))
    summaries["hardware_variants"] = hardware_variants
    summaries["overall_worst_residual"] = float(max(overall_residuals, default=0.0))
    summaries["overall_worst_separation"] = float(min(overall_separations, default=0.0))
    return summaries


def _readout_basis_data(
    spec: ReadoutEmptyingSpec,
    constraints: ReadoutEmptyingConstraints | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    resolved_constraints = ReadoutEmptyingConstraints() if constraints is None else constraints
    constraint_matrix = build_emptying_constraint_matrix(spec)
    free_mask = _free_segment_mask(spec, resolved_constraints)
    if spec.allow_complex_segments:
        reduced_matrix = np.asarray(constraint_matrix[:, free_mask], dtype=np.complex128)
        basis_free = compute_emptying_null_space(reduced_matrix)
        return _embed_basis(spec.n_segments, free_mask, basis_free), free_mask, False
    reduced_complex = np.asarray(constraint_matrix[:, free_mask], dtype=np.complex128)
    reduced_matrix = np.vstack([reduced_complex.real, reduced_complex.imag])
    basis_free = compute_emptying_null_space(reduced_matrix).astype(float, copy=False)
    return _embed_basis(spec.n_segments, free_mask, basis_free), free_mask, True


@dataclass(frozen=True)
class ReadoutEmptyingVerificationConfig:
    measurement_chain: ReadoutChain | None = None
    hardware_model: HardwareModel | None = None
    readout_model: Any | None = None
    frame: FrameSpec = field(default_factory=FrameSpec)
    noise: NoiseSpec | None = None
    compiler_dt_s: float | None = None
    max_step_s: float | None = None
    simulation_config: SimulationConfig | None = None
    shots_per_branch: int = 128
    seed: int | None = None
    include_square_baseline: bool = True
    include_kerr_corrected_baseline: bool = True
    measurement_noise_mode: str = "calibrated_target_error"
    measurement_target_square_error: float = 0.10
    measurement_min_noise_temperature: float = 0.0
    strong_readout_spec: StrongReadoutMixingSpec | None = field(default_factory=StrongReadoutMixingSpec)
    ringdown_threshold_photons: float = 1.0e-2
    chi_scales: tuple[float, ...] = (0.97, 1.0, 1.03)
    kappa_scales: tuple[float, ...] = (0.97, 1.0, 1.03)
    kerr_scales: tuple[float, ...] = (0.9, 1.0, 1.1)
    amplitude_scales: tuple[float, ...] = (0.98, 1.0, 1.02)
    timing_scales: tuple[float, ...] = (0.98, 1.0, 1.02)
    hardware_variants: Mapping[str, HardwareModel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shots_per_branch", int(self.shots_per_branch))
        if int(self.shots_per_branch) < 0:
            raise ValueError("shots_per_branch must be nonnegative.")
        measurement_noise_mode = str(self.measurement_noise_mode).lower()
        if measurement_noise_mode not in {"as_provided", "calibrated_target_error"}:
            raise ValueError("measurement_noise_mode must be 'as_provided' or 'calibrated_target_error'.")
        if not 0.0 <= float(self.measurement_target_square_error) < 0.5:
            raise ValueError("measurement_target_square_error must be in [0, 0.5).")
        if float(self.measurement_min_noise_temperature) < 0.0:
            raise ValueError("measurement_min_noise_temperature must be nonnegative.")
        if float(self.ringdown_threshold_photons) <= 0.0:
            raise ValueError("ringdown_threshold_photons must be positive.")
        if self.compiler_dt_s is not None and float(self.compiler_dt_s) <= 0.0:
            raise ValueError("compiler_dt_s must be positive when provided.")
        if self.max_step_s is not None and float(self.max_step_s) <= 0.0:
            raise ValueError("max_step_s must be positive when provided.")
        object.__setattr__(self, "measurement_noise_mode", measurement_noise_mode)
        object.__setattr__(self, "measurement_target_square_error", float(self.measurement_target_square_error))
        object.__setattr__(self, "measurement_min_noise_temperature", float(self.measurement_min_noise_temperature))
        object.__setattr__(self, "ringdown_threshold_photons", float(self.ringdown_threshold_photons))
        object.__setattr__(self, "chi_scales", _resolved_scale_tuple(self.chi_scales, name="chi_scales"))
        object.__setattr__(self, "kappa_scales", _resolved_scale_tuple(self.kappa_scales, name="kappa_scales"))
        object.__setattr__(self, "kerr_scales", _resolved_scale_tuple(self.kerr_scales, name="kerr_scales"))
        object.__setattr__(
            self,
            "amplitude_scales",
            _resolved_scale_tuple(self.amplitude_scales, name="amplitude_scales"),
        )
        object.__setattr__(self, "timing_scales", _resolved_scale_tuple(self.timing_scales, name="timing_scales"))
        object.__setattr__(self, "hardware_variants", dict(self.hardware_variants))


@dataclass
class ReadoutEmptyingVerificationReport:
    config: ReadoutEmptyingVerificationConfig
    baseline_results: dict[str, ReadoutEmptyingResult]
    baseline_metrics: dict[str, dict[str, float]]
    measurement_metrics: dict[str, dict[str, float]]
    disturbance_metrics: dict[str, dict[str, float]]
    ringdown_metrics: dict[str, dict[str, float]]
    lindblad_metrics: dict[str, dict[str, float]]
    hardware_metrics: dict[str, dict[str, Any]]
    robustness: dict[str, dict[str, Any]]
    comparison_table: dict[str, dict[str, float]]
    artifacts: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadoutEmptyingRefinementConfig:
    maxiter: int = 24
    method: str = "Powell"
    seed: int | None = None
    measurement_chain: ReadoutChain | None = None
    hardware_model: HardwareModel | None = None
    readout_model: Any | None = None
    frame: FrameSpec = field(default_factory=FrameSpec)
    noise: NoiseSpec | None = None
    compiler_dt_s: float | None = None
    max_step_s: float | None = None
    simulation_config: SimulationConfig | None = None
    shots_per_branch: int = 64
    measurement_noise_mode: str = "calibrated_target_error"
    measurement_target_square_error: float = 0.10
    measurement_min_noise_temperature: float = 0.0
    strong_readout_spec: StrongReadoutMixingSpec | None = field(default_factory=StrongReadoutMixingSpec)
    ringdown_threshold_photons: float = 1.0e-2
    objective_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "residual": 1.0,
            "separation": 1.0,
            "measurement": 1.0,
            "leakage": 0.5,
            "robustness": 0.25,
            "bandwidth": 0.25,
        }
    )
    chi_uncertainty: float = 0.03
    kappa_uncertainty: float = 0.03
    kerr_uncertainty: float = 0.10
    amplitude_scale_uncertainty: float = 0.02
    timing_scale_uncertainty: float = 0.02
    hardware_variants: Mapping[str, HardwareModel] = field(default_factory=dict)
    allow_chirp_scale: bool = True
    allow_segment_duration_scaling: bool = False
    allow_endpoint_ramps: bool = False
    duration_log_bound: float = 0.2
    chirp_scale_bounds: tuple[float, float] = (0.0, 2.0)
    endpoint_ramp_bounds: tuple[float, float] = (0.5, 1.0)
    null_coordinate_bound_scale: float = 2.0
    build_verification_report: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "maxiter", int(self.maxiter))
        if int(self.maxiter) <= 0:
            raise ValueError("maxiter must be positive.")
        if float(self.duration_log_bound) <= 0.0:
            raise ValueError("duration_log_bound must be positive.")
        if float(self.null_coordinate_bound_scale) <= 0.0:
            raise ValueError("null_coordinate_bound_scale must be positive.")
        measurement_noise_mode = str(self.measurement_noise_mode).lower()
        if measurement_noise_mode not in {"as_provided", "calibrated_target_error"}:
            raise ValueError("measurement_noise_mode must be 'as_provided' or 'calibrated_target_error'.")
        if not 0.0 <= float(self.measurement_target_square_error) < 0.5:
            raise ValueError("measurement_target_square_error must be in [0, 0.5).")
        if float(self.measurement_min_noise_temperature) < 0.0:
            raise ValueError("measurement_min_noise_temperature must be nonnegative.")
        if float(self.ringdown_threshold_photons) <= 0.0:
            raise ValueError("ringdown_threshold_photons must be positive.")
        if self.compiler_dt_s is not None and float(self.compiler_dt_s) <= 0.0:
            raise ValueError("compiler_dt_s must be positive when provided.")
        if self.max_step_s is not None and float(self.max_step_s) <= 0.0:
            raise ValueError("max_step_s must be positive when provided.")
        if any(float(value) < 0.0 for value in self.objective_weights.values()):
            raise ValueError("objective_weights must be nonnegative.")
        if float(self.chi_uncertainty) < 0.0 or float(self.kappa_uncertainty) < 0.0 or float(self.kerr_uncertainty) < 0.0:
            raise ValueError("parameter uncertainties must be nonnegative.")
        if float(self.amplitude_scale_uncertainty) < 0.0 or float(self.timing_scale_uncertainty) < 0.0:
            raise ValueError("scale uncertainties must be nonnegative.")
        object.__setattr__(self, "hardware_variants", dict(self.hardware_variants))
        object.__setattr__(self, "measurement_noise_mode", measurement_noise_mode)
        object.__setattr__(self, "measurement_target_square_error", float(self.measurement_target_square_error))
        object.__setattr__(self, "measurement_min_noise_temperature", float(self.measurement_min_noise_temperature))
        object.__setattr__(self, "ringdown_threshold_photons", float(self.ringdown_threshold_photons))
        object.__setattr__(self, "objective_weights", {str(key): float(value) for key, value in self.objective_weights.items()})


@dataclass
class ReadoutEmptyingRefinementResult:
    seed_result: ReadoutEmptyingResult
    refined_result: ReadoutEmptyingResult
    success: bool
    message: str
    objective_value: float
    initial_objective_value: float
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    initial_parameter_values: np.ndarray
    metrics: dict[str, float]
    history: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    verification_report: ReadoutEmptyingVerificationReport | None = None


def verify_readout_emptying_pulse(
    result: ReadoutEmptyingResult,
    config: ReadoutEmptyingVerificationConfig | None = None,
    *,
    comparison_results: Mapping[str, ReadoutEmptyingResult] | None = None,
) -> ReadoutEmptyingVerificationReport:
    resolved_config = ReadoutEmptyingVerificationConfig() if config is None else config
    candidates: dict[str, ReadoutEmptyingResult] = {}
    base_spec = result.spec
    base_constraints = result.constraints

    if base_spec.include_kerr_phase_correction and abs(float(base_spec.kerr)) > 1.0e-18:
        candidates["kerr_corrected"] = result
        analytic_spec = _copy_spec(base_spec, include_kerr_phase_correction=False)
        candidates["analytic_seed"] = synthesize_readout_emptying_pulse(analytic_spec, base_constraints)
    else:
        candidates["analytic_seed"] = result
        if resolved_config.include_kerr_corrected_baseline and abs(float(base_spec.kerr)) > 1.0e-18:
            corrected_spec = _copy_spec(base_spec, include_kerr_phase_correction=True)
            candidates["kerr_corrected"] = synthesize_readout_emptying_pulse(corrected_spec, base_constraints)

    if resolved_config.include_square_baseline:
        reference = candidates.get("analytic_seed", result)
        candidates["square"] = _square_pulse_result(reference)

    if comparison_results is not None:
        for label, comparison in dict(comparison_results).items():
            candidates[str(label)] = comparison

    baseline_results: dict[str, ReadoutEmptyingResult] = {}
    baseline_metrics: dict[str, dict[str, float]] = {}
    measurement_metrics: dict[str, dict[str, float]] = {}
    disturbance_metrics: dict[str, dict[str, float]] = {}
    ringdown_metrics: dict[str, dict[str, float]] = {}
    lindblad_metrics: dict[str, dict[str, float]] = {}
    hardware_metrics: dict[str, dict[str, Any]] = {}
    robustness: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "command_results": {},
        "measurement": {},
        "disturbance": {},
        "ringdown": {},
        "lindblad": {},
        "hardware": {},
    }
    drive_frequency = _drive_frequency_from_reference(
        base_spec,
        measurement_chain=resolved_config.measurement_chain,
        readout_model=resolved_config.readout_model,
    )

    for label, candidate in candidates.items():
        nominal_result, hardware_summary = _apply_hardware_model(candidate, resolved_config.hardware_model)
        baseline_results[label] = nominal_result
        baseline_metrics[label] = dict(nominal_result.metrics)
        hardware_metrics[label] = dict(hardware_summary["hardware_metrics"])
        diagnostics["command_results"][label] = candidate
        diagnostics["hardware"][label] = hardware_summary

    measurement_reference = baseline_results.get("square")
    resolved_measurement_chain, measurement_calibration = _resolved_measurement_chain(
        measurement_reference,
        resolved_config,
    )
    diagnostics["measurement_calibration"] = measurement_calibration

    for label, nominal_result in baseline_results.items():
        measurement = _measurement_summary(
            nominal_result,
            resolved_measurement_chain,
            shots_per_branch=resolved_config.shots_per_branch,
            seed=None if resolved_config.seed is None else int(resolved_config.seed) + len(measurement_metrics),
        )
        measurement_metrics[label] = {
            key: float(value)
            for key, value in measurement.get("metrics", {}).items()
            if isinstance(value, (int, float, np.floating))
        }
        disturbance = _strong_readout_summary(
            nominal_result,
            measurement_chain=resolved_measurement_chain,
            readout_model=resolved_config.readout_model,
            compiler_dt_s=resolved_config.compiler_dt_s,
            drive_frequency=drive_frequency,
            config=resolved_config,
        )
        disturbance_metrics[label] = {
            key: float(value)
            for key, value in disturbance.get("metrics", {}).items()
            if isinstance(value, (int, float, np.floating))
        }
        ringdown = {
            "metrics": _ringdown_metrics(
                nominal_result,
                threshold_photons=resolved_config.ringdown_threshold_photons,
            )
        }
        ringdown_metrics[label] = dict(ringdown["metrics"])
        diagnostics["measurement"][label] = measurement
        diagnostics["disturbance"][label] = disturbance
        diagnostics["ringdown"][label] = ringdown
        if resolved_config.readout_model is not None:
            lindblad = _lindblad_validation(
                nominal_result,
                readout_model=resolved_config.readout_model,
                frame=resolved_config.frame,
                noise=resolved_config.noise,
                compiler_dt_s=resolved_config.compiler_dt_s,
                max_step_s=resolved_config.max_step_s,
                simulation_config=resolved_config.simulation_config,
                drive_frequency=drive_frequency,
            )
            lindblad_metrics[label] = dict(lindblad["metrics"])
            diagnostics["lindblad"][label] = lindblad
        else:
            lindblad_metrics[label] = {}
        robustness[label] = _robustness_summary(
            diagnostics["command_results"][label],
            nominal_result,
            resolved_config,
            measurement_chain=resolved_measurement_chain,
        )

    comparison_table: dict[str, dict[str, float]] = {}
    for label, metrics in baseline_metrics.items():
        comparison = dict(metrics)
        comparison.update(measurement_metrics.get(label, {}))
        comparison.update(disturbance_metrics.get(label, {}))
        comparison.update(ringdown_metrics.get(label, {}))
        comparison.update(lindblad_metrics.get(label, {}))
        comparison["robustness_worst_residual"] = float(robustness[label]["overall_worst_residual"])
        comparison["robustness_worst_separation"] = float(robustness[label]["overall_worst_separation"])
        comparison["hardware_command_physical_rms_delta"] = float(hardware_metrics[label]["command_physical_rms_delta"])
        comparison_table[label] = comparison

    return ReadoutEmptyingVerificationReport(
        config=resolved_config,
        baseline_results=baseline_results,
        baseline_metrics=baseline_metrics,
        measurement_metrics=measurement_metrics,
        disturbance_metrics=disturbance_metrics,
        ringdown_metrics=ringdown_metrics,
        lindblad_metrics=lindblad_metrics,
        hardware_metrics=hardware_metrics,
        robustness=robustness,
        comparison_table=comparison_table,
        diagnostics=diagnostics,
    )


def refine_readout_emptying_pulse(
    pulse_init: ReadoutEmptyingResult,
    config: ReadoutEmptyingRefinementConfig | None = None,
) -> ReadoutEmptyingRefinementResult:
    resolved_config = ReadoutEmptyingRefinementConfig() if config is None else config
    constraints = pulse_init.constraints
    base_spec = _copy_spec(pulse_init.spec, include_kerr_phase_correction=False)
    base_result = synthesize_readout_emptying_pulse(base_spec, constraints)
    parameterization = build_readout_emptying_parameterization(base_spec, constraints)
    base_basis = np.asarray(base_result.diagnostics["null_space_basis"], dtype=np.complex128)
    free_mask = np.asarray(base_result.diagnostics["free_segment_mask"], dtype=bool)
    initial_values = np.asarray(parameterization.zero_array(), dtype=float)
    parameter_names = [spec.name for spec in parameterization.parameter_specs]
    bounds = list(parameterization.bounds())
    defaults = np.asarray(initial_values, dtype=float)
    for index, (lower, upper) in enumerate(bounds):
        if not np.isfinite(lower) or not np.isfinite(upper):
            center = float(defaults[index])
            span = max(abs(center), 1.0) * float(resolved_config.null_coordinate_bound_scale)
            bounds[index] = (center - span, center + span)

    chirp_default = 1.0 if pulse_init.spec.include_kerr_phase_correction and abs(float(base_spec.kerr)) > 1.0e-18 else 0.0
    if resolved_config.allow_chirp_scale and abs(float(base_spec.kerr)) > 1.0e-18:
        parameter_names.append("chirp_scale")
        bounds.append(tuple(float(value) for value in resolved_config.chirp_scale_bounds))
        initial_values = np.concatenate([initial_values, np.array([chirp_default], dtype=float)])

    if resolved_config.allow_segment_duration_scaling:
        for index in range(int(base_spec.n_segments)):
            parameter_names.append(f"duration_log_scale_{index}")
            bounds.append((-float(resolved_config.duration_log_bound), float(resolved_config.duration_log_bound)))
            initial_values = np.concatenate([initial_values, np.array([0.0], dtype=float)])

    if resolved_config.allow_endpoint_ramps:
        for name in ("endpoint_ramp_start", "endpoint_ramp_end"):
            parameter_names.append(name)
            bounds.append(tuple(float(value) for value in resolved_config.endpoint_ramp_bounds))
            initial_values = np.concatenate([initial_values, np.array([1.0], dtype=float)])

    nominal_variants = {
        "chi": (1.0 - float(resolved_config.chi_uncertainty), 1.0 + float(resolved_config.chi_uncertainty)),
        "kappa": (1.0 - float(resolved_config.kappa_uncertainty), 1.0 + float(resolved_config.kappa_uncertainty)),
        "kerr": (1.0 - float(resolved_config.kerr_uncertainty), 1.0 + float(resolved_config.kerr_uncertainty)),
        "amplitude": (1.0 - float(resolved_config.amplitude_scale_uncertainty), 1.0 + float(resolved_config.amplitude_scale_uncertainty)),
        "timing": (1.0 - float(resolved_config.timing_scale_uncertainty), 1.0 + float(resolved_config.timing_scale_uncertainty)),
    }
    square_reference_result = _apply_hardware_model(_square_pulse_result(base_result), resolved_config.hardware_model)[0]
    resolved_measurement_chain, measurement_calibration = _resolved_measurement_chain(square_reference_result, resolved_config)

    def unpack(values: np.ndarray) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
        cursor = 0
        null_values = np.asarray(values[: len(parameterization.parameter_specs)], dtype=float)
        cursor += len(parameterization.parameter_specs)
        chirp_scale = chirp_default
        if resolved_config.allow_chirp_scale and abs(float(base_spec.kerr)) > 1.0e-18:
            chirp_scale = float(values[cursor])
            cursor += 1
        duration_scales = None
        if resolved_config.allow_segment_duration_scaling:
            duration_scales = np.asarray(values[cursor : cursor + int(base_spec.n_segments)], dtype=float)
            cursor += int(base_spec.n_segments)
        ramps = None
        if resolved_config.allow_endpoint_ramps:
            ramps = np.asarray(values[cursor : cursor + 2], dtype=float)
        return null_values, chirp_scale, duration_scales, ramps

    def build_candidate(values: np.ndarray) -> tuple[ReadoutEmptyingResult, dict[str, Any]]:
        null_values, chirp_scale, duration_scales, ramps = unpack(values)
        edges = np.asarray(_segment_edges_s(base_spec), dtype=float)
        candidate_spec = base_spec
        if duration_scales is not None:
            base_durations = np.diff(edges)
            weights = base_durations * np.exp(np.asarray(duration_scales, dtype=float))
            weights = weights / np.sum(weights)
            durations = float(base_spec.tau) * weights
            edges = np.concatenate([[0.0], np.cumsum(durations)])
            candidate_spec = _updated_spec_for_edges(base_spec, edges)

        basis, mask, real_only = _readout_basis_data(candidate_spec, constraints)
        n_coords = int(basis.shape[1])
        if real_only:
            coords = np.asarray(null_values[:n_coords], dtype=float).astype(np.complex128)
        else:
            half = n_coords
            coords_data = np.asarray(null_values, dtype=float)
            if coords_data.size < 2 * half:
                raise ValueError("Callable null-space vector is shorter than the resolved basis dimension.")
            coords = coords_data[:half] + 1j * coords_data[half : 2 * half]
        segments = np.asarray(basis @ coords, dtype=np.complex128)
        if real_only:
            segments = np.asarray(segments.real, dtype=np.complex128)
        linear_segments = np.asarray(segments, dtype=np.complex128)
        chirp_diagnostics: dict[str, Any] = {}
        if abs(float(candidate_spec.kerr)) > 1.0e-18 and abs(float(chirp_scale)) > 1.0e-12:
            linear_replay = replay_linear_readout_branches(candidate_spec, linear_segments)
            phase = build_kerr_phase_correction(candidate_spec, linear_replay)
            segments = apply_phase_chirp(
                linear_segments,
                edges,
                float(chirp_scale) * np.asarray(phase["phase_rad"], dtype=float),
                linear_replay.time_grid_s,
            )
            chirp_diagnostics = {
                "chirp_scale": float(chirp_scale),
                "phase_rad": np.asarray(phase["phase_rad"], dtype=float),
            }
        if ramps is not None:
            window = np.ones_like(segments, dtype=float)
            if window.size:
                window[0] *= float(ramps[0])
                window[-1] *= float(ramps[1])
            segments = np.asarray(segments * window, dtype=np.complex128)

        command_result = _compose_result(
            candidate_spec,
            constraints,
            edges,
            segments,
            diagnostics={
                "source": "refinement_candidate_command",
                "resolved_free_segment_mask": mask,
                "resolved_basis_dimension": int(basis.shape[1]),
                "linear_segment_amplitudes": np.asarray(linear_segments, dtype=np.complex128),
                "chirp": chirp_diagnostics,
            },
        )
        physical_result, hardware_summary = _apply_hardware_model(command_result, resolved_config.hardware_model)
        return physical_result, {
            "command_result": command_result,
            "hardware_summary": hardware_summary,
        }

    def candidate_summary(values: np.ndarray) -> tuple[float, dict[str, Any], ReadoutEmptyingResult, dict[str, Any]]:
        candidate_result, diagnostics = build_candidate(values)
        measurement = _measurement_summary(
            candidate_result,
            resolved_measurement_chain,
            shots_per_branch=resolved_config.shots_per_branch,
            seed=resolved_config.seed,
        )
        measurement_metrics = {
            key: float(value)
            for key, value in measurement.get("metrics", {}).items()
            if isinstance(value, (int, float, np.floating))
        }
        drive_frequency = _drive_frequency_from_reference(
            candidate_result.spec,
            measurement_chain=resolved_measurement_chain,
            readout_model=resolved_config.readout_model,
        )
        lindblad_metrics: dict[str, float] = {}
        lindblad_data: dict[str, Any] | None = None
        if resolved_config.readout_model is not None:
            lindblad_data = _lindblad_validation(
                candidate_result,
                readout_model=resolved_config.readout_model,
                frame=resolved_config.frame,
                noise=resolved_config.noise,
                compiler_dt_s=resolved_config.compiler_dt_s,
                max_step_s=resolved_config.max_step_s,
                simulation_config=resolved_config.simulation_config,
                drive_frequency=drive_frequency,
            )
            lindblad_metrics = dict(lindblad_data["metrics"])

        disturbance = _strong_readout_summary(
            candidate_result,
            measurement_chain=resolved_measurement_chain,
            readout_model=resolved_config.readout_model,
            compiler_dt_s=resolved_config.compiler_dt_s,
            drive_frequency=drive_frequency,
            config=resolved_config,
        )
        disturbance_metrics = {
            key: float(value)
            for key, value in disturbance.get("metrics", {}).items()
            if isinstance(value, (int, float, np.floating))
        }
        ringdown_metrics = _ringdown_metrics(
            candidate_result,
            threshold_photons=resolved_config.ringdown_threshold_photons,
        )

        uncertainty_config = ReadoutEmptyingVerificationConfig(
            measurement_chain=resolved_measurement_chain,
            hardware_model=resolved_config.hardware_model,
            readout_model=None,
            frame=resolved_config.frame,
            noise=resolved_config.noise,
            compiler_dt_s=resolved_config.compiler_dt_s,
            max_step_s=resolved_config.max_step_s,
            simulation_config=resolved_config.simulation_config,
            shots_per_branch=0,
            measurement_noise_mode="as_provided",
            measurement_target_square_error=resolved_config.measurement_target_square_error,
            measurement_min_noise_temperature=resolved_config.measurement_min_noise_temperature,
            strong_readout_spec=resolved_config.strong_readout_spec,
            ringdown_threshold_photons=resolved_config.ringdown_threshold_photons,
            chi_scales=nominal_variants["chi"],
            kappa_scales=nominal_variants["kappa"],
            kerr_scales=nominal_variants["kerr"],
            amplitude_scales=nominal_variants["amplitude"],
            timing_scales=nominal_variants["timing"],
            hardware_variants=resolved_config.hardware_variants,
        )
        robustness = _robustness_summary(
            diagnostics["command_result"],
            candidate_result,
            uncertainty_config,
            measurement_chain=resolved_measurement_chain,
        )

        summary = {
            "residual": float(candidate_result.metrics["max_final_residual_photons"]),
            "separation": float(
                lindblad_metrics.get(
                    "lindblad_output_separation",
                    measurement_metrics.get("measurement_chain_separation", candidate_result.metrics["integrated_branch_separation"]),
                )
            ),
            "measurement_error": float(
                measurement_metrics.get("measurement_chain_gaussian_overlap_error", 0.0)
            ),
            "leakage": float(disturbance_metrics.get("strong_readout_disturbance_proxy", 0.0)),
            "robustness": float(robustness["overall_worst_residual"]),
            "bandwidth": float(diagnostics["hardware_summary"]["hardware_metrics"]["command_physical_rms_delta"]),
            "ringdown_time_to_threshold": float(ringdown_metrics["ringdown_time_to_threshold"]),
            "lindblad_output_separation": float(
                lindblad_metrics.get(
                    "lindblad_output_separation",
                    measurement_metrics.get("measurement_chain_separation", candidate_result.metrics["integrated_branch_separation"]),
                )
            ),
        }
        metadata = {
            "measurement": measurement,
            "measurement_metrics": measurement_metrics,
            "measurement_calibration": measurement_calibration,
            "disturbance": disturbance,
            "disturbance_metrics": disturbance_metrics,
            "ringdown_metrics": ringdown_metrics,
            "lindblad": lindblad_data,
            "lindblad_metrics": lindblad_metrics,
            "robustness": robustness,
            **diagnostics,
        }
        return 0.0, summary, candidate_result, metadata

    _, initial_summary, initial_result, initial_metadata = candidate_summary(initial_values)
    references = {
        "residual": max(initial_summary["residual"], 1.0e-6),
        "separation": max(initial_summary["separation"], 1.0e-12),
        "measurement_error": max(initial_summary["measurement_error"], 1.0e-3),
        "leakage": max(initial_summary["leakage"], 1.0e-6),
        "robustness": max(initial_summary["robustness"], 1.0e-6),
        "bandwidth": max(initial_summary["bandwidth"], 1.0e-6),
    }
    history: list[dict[str, Any]] = []
    objective_weights = dict(resolved_config.objective_weights)

    def objective(values: np.ndarray) -> float:
        clipped = np.asarray(
            [np.clip(float(value), float(lower), float(upper)) for value, (lower, upper) in zip(values, bounds, strict=True)],
            dtype=float,
        )
        _, summary, _, _ = candidate_summary(clipped)
        objective_value = 0.0
        objective_value += objective_weights.get("residual", 0.0) * summary["residual"] / references["residual"]
        objective_value -= objective_weights.get("separation", 0.0) * summary["separation"] / references["separation"]
        objective_value += objective_weights.get("measurement", 0.0) * summary["measurement_error"] / references["measurement_error"]
        objective_value += objective_weights.get("leakage", 0.0) * summary["leakage"] / references["leakage"]
        objective_value += objective_weights.get("robustness", 0.0) * summary["robustness"] / references["robustness"]
        objective_value += objective_weights.get("bandwidth", 0.0) * summary["bandwidth"] / references["bandwidth"]
        history.append(
            {
                "evaluation": int(len(history)),
                "objective": float(objective_value),
                **{key: float(value) for key, value in summary.items()},
            }
        )
        return float(objective_value)

    initial_objective = float(objective(initial_values))
    optimizer_result = minimize(
        objective,
        np.asarray(initial_values, dtype=float),
        method=str(resolved_config.method),
        bounds=bounds,
        options={"maxiter": int(resolved_config.maxiter)},
    )
    candidate_values = np.asarray(optimizer_result.x, dtype=float)
    candidate_objective = float(objective(candidate_values))
    _, candidate_summary_metrics, candidate_result, candidate_metadata = candidate_summary(candidate_values)

    if candidate_objective < initial_objective - 1.0e-9:
        accepted_result = candidate_result
        accepted_values = candidate_values
        accepted_metadata = candidate_metadata
        accepted_objective = candidate_objective
        accepted_summary = candidate_summary_metrics
        accepted_message = str(optimizer_result.message)
        success = bool(optimizer_result.success)
    else:
        accepted_result = initial_result
        accepted_values = np.asarray(initial_values, dtype=float)
        accepted_metadata = initial_metadata
        accepted_objective = initial_objective
        accepted_summary = initial_summary
        accepted_message = "Refinement kept the initial seed because no lower-cost candidate was found."
        success = bool(optimizer_result.success)

    verification_report = None
    if resolved_config.build_verification_report:
        verification_report = verify_readout_emptying_pulse(
            pulse_init,
            ReadoutEmptyingVerificationConfig(
                measurement_chain=resolved_measurement_chain,
                hardware_model=resolved_config.hardware_model,
                readout_model=resolved_config.readout_model,
                frame=resolved_config.frame,
                noise=resolved_config.noise,
                compiler_dt_s=resolved_config.compiler_dt_s,
                max_step_s=resolved_config.max_step_s,
                simulation_config=resolved_config.simulation_config,
                shots_per_branch=resolved_config.shots_per_branch,
                seed=resolved_config.seed,
                measurement_noise_mode="as_provided",
                measurement_target_square_error=resolved_config.measurement_target_square_error,
                measurement_min_noise_temperature=resolved_config.measurement_min_noise_temperature,
                strong_readout_spec=resolved_config.strong_readout_spec,
                ringdown_threshold_photons=resolved_config.ringdown_threshold_photons,
                hardware_variants=resolved_config.hardware_variants,
            ),
            comparison_results={"refined": accepted_result},
        )

    return ReadoutEmptyingRefinementResult(
        seed_result=pulse_init,
        refined_result=accepted_result,
        success=success,
        message=str(accepted_message),
        objective_value=float(accepted_objective),
        initial_objective_value=float(initial_objective),
        parameter_names=tuple(str(name) for name in parameter_names),
        parameter_values=np.asarray(accepted_values, dtype=float),
        initial_parameter_values=np.asarray(initial_values, dtype=float),
        metrics={
            "objective_improvement": float(initial_objective - accepted_objective),
            "final_residual": float(accepted_summary["residual"]),
            "final_separation": float(accepted_summary["separation"]),
            "final_measurement_error": float(accepted_summary["measurement_error"]),
            "final_leakage": float(accepted_summary["leakage"]),
            "final_robustness": float(accepted_summary["robustness"]),
            "final_bandwidth_penalty": float(accepted_summary["bandwidth"]),
            "final_ringdown_time_to_threshold": float(accepted_summary["ringdown_time_to_threshold"]),
            "final_lindblad_output_separation": float(accepted_summary["lindblad_output_separation"]),
        },
        history=history,
        diagnostics={
            "seed_summary": initial_summary,
            "accepted_summary": accepted_summary,
            "accepted_metadata": accepted_metadata,
            "measurement_calibration": measurement_calibration,
            "optimizer_summary": {
                "success": bool(optimizer_result.success),
                "message": str(optimizer_result.message),
                "nit": None if getattr(optimizer_result, "nit", None) is None else int(optimizer_result.nit),
                "nfev": None if getattr(optimizer_result, "nfev", None) is None else int(optimizer_result.nfev),
            },
            "reference_scales": references,
            "bounds": [tuple(float(value) for value in bound) for bound in bounds],
        },
        verification_report=verification_report,
    )


__all__ = [
    "ReadoutEmptyingVerificationConfig",
    "ReadoutEmptyingVerificationReport",
    "ReadoutEmptyingRefinementConfig",
    "ReadoutEmptyingRefinementResult",
    "verify_readout_emptying_pulse",
    "refine_readout_emptying_pulse",
]
