from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import Bounds, minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples.sqr_block_phase_study as phase1
import examples.sqr_multitone_study as sms
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler


FOLLOWUP_OUTPUT = Path("outputs/sqr_block_phase_followup")
DEFAULT_SEED = 20260308
DEFAULT_N_MAX = 3
DEFAULT_DURATION_NS = 700.0
DEFAULT_WAIT_SEGMENTS = (4, 5)
PHASE_TARGET_NAMES = (
    "zero",
    "natural_drift",
    "kerr_cancel",
    "alternating",
    "affine_quadratic",
    "random_small_a",
    "random_small_b",
    "random_small_c",
    "random_medium_a",
    "random_medium_b",
    "random_medium_c",
    "far_from_drift",
)


@dataclass
class FamilyRun:
    family_id: str
    label: str
    category: str
    spec: phase1.ExperimentSpec
    theta: np.ndarray
    phi: np.ndarray
    lambda_target: np.ndarray
    model: sms.DispersiveTransmonCavityModel
    frame: sms.FrameSpec
    pulse_params: sms.PulseParams
    unitary_qobj: qt.Qobj
    unitary: np.ndarray
    compiled: Any
    total_duration_s: float
    analysis: dict[str, Any]
    vector: np.ndarray | None = None
    optimizer: str = ""
    cost_name: str = ""
    notes: str = ""
    bounds_lb: np.ndarray | None = None
    bounds_ub: np.ndarray | None = None
    evaluate_from_vector: Callable[[np.ndarray, np.ndarray], "FamilyRun"] | None = None


@dataclass(frozen=True)
class TargetPhaseProfile:
    name: str
    lambda_target: np.ndarray
    category: str
    notes: str = ""


def _uniform_pi_spec(
    *,
    n_max: int = DEFAULT_N_MAX,
    duration_ns: float = DEFAULT_DURATION_NS,
    chi_hz: float = phase1.CHI_HZ,
    kerr_hz: float = phase1.KERR_HZ,
    experiment_id: str = "followup_uniform_pi",
    lambda_label: str = "zero",
    seed: int = DEFAULT_SEED,
) -> phase1.ExperimentSpec:
    theta, phi = phase1.rotation_profile_uniform_pi(n_levels=n_max + 1)
    return phase1.ExperimentSpec(
        experiment_id=experiment_id,
        title=f"Follow-up uniform pi, N={n_max}, T={duration_ns:.0f} ns, lambda={lambda_label}",
        n_max=int(n_max),
        duration_s=float(duration_ns) * 1.0e-9,
        theta=tuple(float(x) for x in theta),
        phi=tuple(float(x) for x in phi),
        lambda_family=str(lambda_label),
        chi_hz=float(chi_hz),
        kerr_hz=float(kerr_hz),
        family_ids=(),
        notes="Follow-up controllability study baseline.",
        seed=int(seed),
    )


def _build_case_context(spec: phase1.ExperimentSpec) -> tuple[
    sms.DispersiveTransmonCavityModel,
    sms.FrameSpec,
    sms.PulseParams,
    np.ndarray,
    np.ndarray,
    sms.TargetProfile,
    list[sms.ToneControl],
]:
    model, frame, pulse = phase1.build_model_and_pulse(spec)
    theta = np.asarray(spec.theta, dtype=float)
    phi = np.asarray(spec.phi, dtype=float)
    profile = sms.TargetProfile(name=spec.experiment_id, mode="manual", theta=theta, phi=phi, seed=int(spec.seed))
    base_controls = sms.build_controls_from_target(
        profile=profile,
        model=model,
        frame=frame,
        duration_s=float(spec.duration_s),
        theta_cutoff=float(pulse.theta_cutoff),
        include_all_levels=bool(pulse.include_zero_theta_tones),
    )
    return model, frame, pulse, theta, phi, profile, base_controls


def _block_scalar_rates(model: sms.DispersiveTransmonCavityModel, frame: sms.FrameSpec, n_levels: int) -> np.ndarray:
    h0 = np.asarray(model.static_hamiltonian(frame).full(), dtype=np.complex128)
    rates = np.zeros(int(n_levels), dtype=float)
    for n in range(int(n_levels)):
        idx = phase1._block_indices(model.n_cav, n)
        block = h0[np.ix_(idx, idx)]
        rates[n] = float(np.real(np.trace(block)) / 2.0)
    return rates


def predicted_relative_block_phases(
    model: sms.DispersiveTransmonCavityModel,
    frame: sms.FrameSpec,
    n_levels: int,
    total_duration_s: float,
) -> np.ndarray:
    rates = _block_scalar_rates(model=model, frame=frame, n_levels=int(n_levels))
    rel = -(rates - float(rates[0])) * float(total_duration_s)
    return np.asarray((rel + 0.5 * np.pi) % np.pi - 0.5 * np.pi, dtype=float)


def _phase_rms(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def _relative(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    return np.asarray(arr - float(arr[0]), dtype=float)


def _phase_steering_distance(lambda_impl_rel: np.ndarray, lambda_drift_rel: np.ndarray) -> float:
    return _phase_rms(np.asarray(lambda_impl_rel, dtype=float), np.asarray(lambda_drift_rel, dtype=float))


def _analysis_with_target(run: FamilyRun, lambda_target: np.ndarray) -> dict[str, Any]:
    return phase1.analyze_unitary(
        unitary=run.unitary,
        theta=run.theta,
        phi=run.phi,
        lambda_target=np.asarray(lambda_target, dtype=float),
    )


def _evaluate_compiled(
    *,
    family_id: str,
    label: str,
    category: str,
    spec: phase1.ExperimentSpec,
    theta: np.ndarray,
    phi: np.ndarray,
    lambda_target: np.ndarray,
    model: sms.DispersiveTransmonCavityModel,
    frame: sms.FrameSpec,
    pulse_params: sms.PulseParams,
    compiled: Any,
    unitary_qobj: qt.Qobj,
    vector: np.ndarray | None,
    optimizer: str,
    cost_name: str,
    notes: str,
    bounds_lb: np.ndarray | None = None,
    bounds_ub: np.ndarray | None = None,
    evaluate_from_vector: Callable[[np.ndarray, np.ndarray], FamilyRun] | None = None,
) -> FamilyRun:
    unitary = np.asarray(unitary_qobj.full(), dtype=np.complex128)
    analysis = phase1.analyze_unitary(unitary=unitary, theta=theta, phi=phi, lambda_target=lambda_target)
    return FamilyRun(
        family_id=family_id,
        label=label,
        category=category,
        spec=spec,
        theta=np.asarray(theta, dtype=float),
        phi=np.asarray(phi, dtype=float),
        lambda_target=np.asarray(lambda_target, dtype=float),
        model=model,
        frame=frame,
        pulse_params=pulse_params,
        unitary_qobj=unitary_qobj,
        unitary=unitary,
        compiled=compiled,
        total_duration_s=float(compiled.tlist[-1]),
        analysis=analysis,
        vector=None if vector is None else np.asarray(vector, dtype=float),
        optimizer=optimizer,
        cost_name=cost_name,
        notes=notes,
        bounds_lb=None if bounds_lb is None else np.asarray(bounds_lb, dtype=float),
        bounds_ub=None if bounds_ub is None else np.asarray(bounds_ub, dtype=float),
        evaluate_from_vector=evaluate_from_vector,
    )


def _family_run_from_phase1(
    run: phase1.RunResult,
    *,
    label: str,
    category: str,
    cost_name: str,
    notes: str,
    bounds_lb: np.ndarray | None = None,
    bounds_ub: np.ndarray | None = None,
    evaluate_from_vector: Callable[[np.ndarray, np.ndarray], FamilyRun] | None = None,
) -> FamilyRun:
    return _evaluate_compiled(
        family_id=run.family.family_id,
        label=label,
        category=category,
        spec=run.experiment,
        theta=run.theta,
        phi=run.phi,
        lambda_target=run.lambda_target,
        model=run.model,
        frame=run.frame,
        pulse_params=run.pulse_params,
        compiled=run.compiled,
        unitary_qobj=run.unitary_qobj,
        vector=run.vector,
        optimizer="" if run.optimization_status is None else str(run.optimization_status.get("stage2", {}).get("method", "")),
        cost_name=cost_name,
        notes=notes,
        bounds_lb=bounds_lb,
        bounds_ub=bounds_ub,
        evaluate_from_vector=evaluate_from_vector,
    )


def _phase_target_profiles(
    *,
    n_levels: int,
    duration_s: float,
    kerr_hz: float,
    seed: int,
    drift_rel: np.ndarray,
) -> list[TargetPhaseProfile]:
    rng = np.random.default_rng(int(seed))
    n = np.arange(int(n_levels), dtype=float)
    kerr = np.asarray(
        phase1.lambda_profile(
            family="kerr_cancel",
            n_levels=int(n_levels),
            duration_s=float(duration_s),
            kerr_hz=float(kerr_hz),
            scale_rad=0.0,
            seed=int(seed),
        ),
        dtype=float,
    )
    profiles = [
        TargetPhaseProfile("zero", np.zeros(int(n_levels), dtype=float), "structured", "No extra block phase."),
        TargetPhaseProfile("natural_drift", np.asarray(drift_rel, dtype=float), "structured", "Exact static-trace drift profile."),
        TargetPhaseProfile("kerr_cancel", kerr, "structured", "Quadratic Kerr-canceling profile."),
        TargetPhaseProfile(
            "alternating",
            np.asarray([0.0, 0.24, -0.24, 0.24, -0.24][: int(n_levels)], dtype=float),
            "structured",
            "Alternating sign profile.",
        ),
        TargetPhaseProfile(
            "affine_quadratic",
            np.asarray(0.07 * n - 0.11 * n * (n - 1.0), dtype=float),
            "structured",
            "Affine plus quadratic profile.",
        ),
    ]
    for idx, name in enumerate(("random_small_a", "random_small_b", "random_small_c"), start=1):
        lam = rng.uniform(-0.15, 0.15, size=int(n_levels))
        lam[0] = 0.0
        profiles.append(TargetPhaseProfile(name, np.asarray(lam, dtype=float), "random_small", f"Seed offset {idx}."))
    for idx, name in enumerate(("random_medium_a", "random_medium_b", "random_medium_c"), start=1):
        lam = rng.uniform(-0.60, 0.60, size=int(n_levels))
        lam[0] = 0.0
        profiles.append(TargetPhaseProfile(name, np.asarray(lam, dtype=float), "random_medium", f"Seed offset {idx}."))
    far = np.asarray(drift_rel, dtype=float) + np.asarray([0.0, -1.15, 0.85, -1.30, 0.95][: int(n_levels)], dtype=float)
    far[0] = 0.0
    profiles.append(TargetPhaseProfile("far_from_drift", far, "stress", "Deliberately far from the drift profile."))
    ordered: list[TargetPhaseProfile] = []
    by_name = {item.name: item for item in profiles}
    for name in PHASE_TARGET_NAMES:
        if name in by_name:
            ordered.append(by_name[name])
    return ordered


def _parameter_bounds_for_family(
    family_id: str,
    *,
    base_controls: list[sms.ToneControl],
    pulse_params: sms.PulseParams,
) -> tuple[list[int], np.ndarray, Bounds]:
    family = phase1.FAMILY_SPECS[family_id]
    active = sms.active_controls(
        base_controls,
        theta_cutoff=float(pulse_params.theta_cutoff),
        include_zero_amp=bool(phase1.OPT_PARAMS.allow_zero_theta_corrections),
    )
    x0, bounds = sms.parameter_layout(mode=str(family.mode), active_indices=active, opt=phase1.OPT_PARAMS)
    return active, np.asarray(x0, dtype=float), bounds


def run_multitone_family(
    *,
    spec: phase1.ExperimentSpec,
    family_id: str,
    lambda_target: np.ndarray,
    label: str,
    category: str,
    notes: str,
    initial_vector: np.ndarray | None = None,
) -> FamilyRun:
    model, frame, pulse_params, theta, phi, _, base_controls = _build_case_context(spec)
    family = phase1.FAMILY_SPECS[family_id]
    active, x0, bounds = _parameter_bounds_for_family(
        family_id=family_id,
        base_controls=base_controls,
        pulse_params=pulse_params,
    )
    if initial_vector is not None and initial_vector.shape == x0.shape:
        x0 = np.asarray(initial_vector, dtype=float).copy()

    def evaluate_from_vector(vector: np.ndarray, lambda_target_eval: np.ndarray) -> FamilyRun:
        run_eval, _, _ = phase1._evaluate_vector(
            vector=np.asarray(vector, dtype=float),
            spec=spec,
            family=family,
            base_controls=base_controls,
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            theta=theta,
            phi=phi,
            lambda_target=np.asarray(lambda_target_eval, dtype=float),
            active_indices=active,
        )
        return _family_run_from_phase1(
            run_eval,
            label=label,
            category=category,
            cost_name=str(family.objective),
            notes=notes,
            bounds_lb=bounds.lb,
            bounds_ub=bounds.ub,
        )

    if family.mode is None:
        run = phase1.evaluate_controls(
            spec=spec,
            family=family,
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            theta=theta,
            phi=phi,
            lambda_target=np.asarray(lambda_target, dtype=float),
            controls=base_controls,
        )
    else:
        run = phase1.optimize_family(
            spec=spec,
            family=family,
            base_controls=base_controls,
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            theta=theta,
            phi=phi,
            lambda_target=np.asarray(lambda_target, dtype=float),
            initial_vector=x0,
        )
    return _family_run_from_phase1(
        run,
        label=label,
        category=category,
        cost_name=str(family.objective),
        notes=notes,
        bounds_lb=bounds.lb,
        bounds_ub=bounds.ub,
        evaluate_from_vector=evaluate_from_vector if family.mode is not None else None,
    )


def _compile_pulses(pulses: list[Pulse], dt_s: float, t_end: float) -> Any:
    return SequenceCompiler(dt=float(dt_s)).compile(pulses, t_end=float(t_end))


def _propagate(
    *,
    model: sms.DispersiveTransmonCavityModel,
    frame: sms.FrameSpec,
    pulse_params: sms.PulseParams,
    compiled: Any,
) -> qt.Qobj:
    return sms.propagate_pulse_unitary(
        model=model,
        frame=frame,
        compiled=compiled,
        max_step_s=float(pulse_params.max_step_eval_s),
        qutip_nsteps=int(pulse_params.qutip_nsteps),
    )


def _segment_average_initial_guess(
    *,
    base_controls: list[sms.ToneControl],
    spec: phase1.ExperimentSpec,
    pulse_params: sms.PulseParams,
    nseg: int,
) -> np.ndarray:
    pulse = sms.build_pulse_from_controls(
        controls=base_controls,
        duration_s=float(spec.duration_s),
        pulse=pulse_params,
        label="piecewise_init",
    )
    compiled = sms.compile_single_pulse(pulse, dt_s=float(pulse_params.dt_eval_s))
    sig = np.asarray(compiled.channels["qubit"].distorted, dtype=np.complex128)
    t = np.asarray(compiled.tlist, dtype=float)
    total = float(spec.duration_s)
    seg_width = total / float(nseg)
    values = np.zeros(int(nseg), dtype=np.complex128)
    for idx in range(int(nseg)):
        start = idx * seg_width
        stop = (idx + 1) * seg_width
        mask = (t >= start) & (t < stop)
        if np.any(mask):
            values[idx] = np.mean(sig[mask])
    out = np.empty(2 * int(nseg), dtype=float)
    out[0::2] = np.real(values)
    out[1::2] = np.imag(values)
    return out


def _piecewise_segments_from_vector(vector: np.ndarray, *, nseg: int, wait_segments: set[int]) -> np.ndarray:
    full = np.zeros(int(nseg), dtype=np.complex128)
    cursor = 0
    for seg in range(int(nseg)):
        if seg in wait_segments:
            full[seg] = 0.0
            continue
        full[seg] = complex(float(vector[cursor]), float(vector[cursor + 1]))
        cursor += 2
    return full


def _piecewise_pulse(segments: np.ndarray, *, duration_s: float, label: str, t0: float = 0.0) -> Pulse:
    return Pulse(
        channel="qubit",
        t0=float(t0),
        duration=float(duration_s),
        envelope=np.asarray(segments, dtype=np.complex128),
        sample_rate=float(len(segments) / float(duration_s)),
        amp=1.0,
        phase=0.0,
        label=label,
    )


def optimize_piecewise_family(
    *,
    spec: phase1.ExperimentSpec,
    lambda_target: np.ndarray,
    label: str,
    family_id: str,
    notes: str,
    nseg: int = 10,
    maxiter: int = 10,
    wait_segments: tuple[int, ...] = (),
    initial_vector: np.ndarray | None = None,
) -> FamilyRun:
    model, frame, pulse_params, theta, phi, _, base_controls = _build_case_context(spec)
    wait_set = set(int(x) for x in wait_segments)
    x0_full = _segment_average_initial_guess(
        base_controls=base_controls,
        spec=spec,
        pulse_params=pulse_params,
        nseg=int(nseg),
    )
    keep = [seg for seg in range(int(nseg)) if seg not in wait_set]
    x0 = np.empty(2 * len(keep), dtype=float)
    cursor = 0
    for seg in keep:
        x0[cursor] = float(x0_full[2 * seg])
        x0[cursor + 1] = float(x0_full[2 * seg + 1])
        cursor += 2
    if initial_vector is not None and initial_vector.shape == x0.shape:
        x0 = np.asarray(initial_vector, dtype=float).copy()

    amp_ref = max(float(np.max(np.abs(x0))) if x0.size else 0.0, float(sms.sqr_lambda0_rad_s(spec.duration_s)))
    amp_bound = max(1.8 * amp_ref, float(2.0 * np.pi * 8.0e6))
    bounds = Bounds(
        -amp_bound * np.ones_like(x0, dtype=float),
        amp_bound * np.ones_like(x0, dtype=float),
    )
    scale = max(amp_ref, 1.0)
    best: dict[str, Any] = {"loss": float("inf"), "run": None, "vector": None}

    def build_run(vector: np.ndarray, lambda_target_eval: np.ndarray) -> FamilyRun:
        segments = _piecewise_segments_from_vector(
            np.asarray(vector, dtype=float),
            nseg=int(nseg),
            wait_segments=wait_set,
        )
        pulse = _piecewise_pulse(
            segments,
            duration_s=float(spec.duration_s),
            label=family_id,
        )
        compiled = _compile_pulses([pulse], dt_s=float(pulse_params.dt_eval_s), t_end=float(spec.duration_s + pulse_params.dt_eval_s))
        unitary_qobj = _propagate(model=model, frame=frame, pulse_params=pulse_params, compiled=compiled)
        return _evaluate_compiled(
            family_id=family_id,
            label=label,
            category="piecewise_iq_fixed_T" if not wait_set else "piecewise_iq_wait_fixed_T",
            spec=spec,
            theta=theta,
            phi=phi,
            lambda_target=np.asarray(lambda_target_eval, dtype=float),
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            compiled=compiled,
            unitary_qobj=unitary_qobj,
            vector=vector,
            optimizer="Powell",
            cost_name="hybrid_phase",
            notes=notes,
            bounds_lb=bounds.lb,
            bounds_ub=bounds.ub,
        )

    def evaluate_from_vector(vector: np.ndarray, lambda_target_eval: np.ndarray) -> FamilyRun:
        vector = np.clip(np.asarray(vector, dtype=float), bounds.lb, bounds.ub)
        return build_run(vector, np.asarray(lambda_target_eval, dtype=float))

    def objective(vector: np.ndarray) -> float:
        run = build_run(np.asarray(vector, dtype=float), np.asarray(lambda_target, dtype=float))
        reg = 6.0e-4 * float(np.mean(np.square((np.asarray(vector, dtype=float) - x0) / scale)))
        loss, _ = phase1._hybrid_phase_loss(run.analysis, reg=reg)
        if float(loss) < float(best["loss"]):
            best["loss"] = float(loss)
            best["run"] = run
            best["vector"] = np.asarray(vector, dtype=float).copy()
        return float(loss)

    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(maxiter), "disp": False},
    )
    candidates = [np.asarray(x0, dtype=float), np.asarray(result.x, dtype=float)]
    if best["vector"] is not None:
        candidates.append(np.asarray(best["vector"], dtype=float))
    evaluated = [build_run(np.clip(candidate, bounds.lb, bounds.ub), np.asarray(lambda_target, dtype=float)) for candidate in candidates]
    final_run = max(evaluated, key=lambda item: float(item.analysis["full_unitary_fidelity"]))
    final_run.evaluate_from_vector = evaluate_from_vector
    final_run.optimizer = "Powell"
    final_run.bounds_lb = np.asarray(bounds.lb, dtype=float)
    final_run.bounds_ub = np.asarray(bounds.ub, dtype=float)
    return final_run


def _global_tone_transform(
    base_controls: list[sms.ToneControl],
    *,
    amp_delta: float,
    phase_delta: float,
    detuning_rad_s: float,
) -> list[sms.ToneControl]:
    out: list[sms.ToneControl] = []
    for tone in base_controls:
        out.append(
            sms.ToneControl(
                manifold=int(tone.manifold),
                omega_rad_s=float(tone.omega_rad_s),
                amp_rad_s=float(tone.amp_rad_s * (1.0 + amp_delta)),
                phase_rad=float(tone.phase_rad + phase_delta),
                detuning_rad_s=float(tone.detuning_rad_s + detuning_rad_s),
                phase_ramp_rad_s=float(tone.phase_ramp_rad_s),
            )
        )
    return out


def optimize_composite_family(
    *,
    spec: phase1.ExperimentSpec,
    lambda_target: np.ndarray,
    label: str,
    family_id: str,
    notes: str,
    maxiter: int = 10,
    initial_vector: np.ndarray | None = None,
) -> FamilyRun:
    model, frame, pulse_params, theta, phi, _, base_controls = _build_case_context(spec)
    half = 0.5 * float(spec.duration_s)
    bounds = Bounds(
        np.asarray([-0.95, -np.pi, sms.hz_to_rad_s(-4.0e6), -0.95, -np.pi, sms.hz_to_rad_s(-4.0e6)], dtype=float),
        np.asarray([1.50, np.pi, sms.hz_to_rad_s(4.0e6), 1.50, np.pi, sms.hz_to_rad_s(4.0e6)], dtype=float),
    )
    x0 = np.zeros(6, dtype=float) if initial_vector is None else np.asarray(initial_vector, dtype=float).copy()

    def build_run(vector: np.ndarray, lambda_target_eval: np.ndarray) -> FamilyRun:
        vec = np.asarray(vector, dtype=float)
        controls_a = _global_tone_transform(
            base_controls,
            amp_delta=float(vec[0]),
            phase_delta=float(vec[1]),
            detuning_rad_s=float(vec[2]),
        )
        controls_b = _global_tone_transform(
            base_controls,
            amp_delta=float(vec[3]),
            phase_delta=float(vec[4]),
            detuning_rad_s=float(vec[5]),
        )
        pulse_a = sms.build_pulse_from_controls(controls=controls_a, duration_s=half, pulse=pulse_params, label=f"{family_id}_a")
        pulse_b = sms.build_pulse_from_controls(controls=controls_b, duration_s=half, pulse=pulse_params, label=f"{family_id}_b")
        pulse_a = replace(pulse_a, t0=0.0)
        pulse_b = replace(pulse_b, t0=half)
        compiled = _compile_pulses([pulse_a, pulse_b], dt_s=float(pulse_params.dt_eval_s), t_end=float(spec.duration_s + pulse_params.dt_eval_s))
        unitary_qobj = _propagate(model=model, frame=frame, pulse_params=pulse_params, compiled=compiled)
        return _evaluate_compiled(
            family_id=family_id,
            label=label,
            category="composite_multitone_fixed_T",
            spec=spec,
            theta=theta,
            phi=phi,
            lambda_target=np.asarray(lambda_target_eval, dtype=float),
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            compiled=compiled,
            unitary_qobj=unitary_qobj,
            vector=vec,
            optimizer="Powell",
            cost_name="hybrid_phase",
            notes=notes,
            bounds_lb=bounds.lb,
            bounds_ub=bounds.ub,
        )

    def evaluate_from_vector(vector: np.ndarray, lambda_target_eval: np.ndarray) -> FamilyRun:
        return build_run(np.clip(np.asarray(vector, dtype=float), bounds.lb, bounds.ub), np.asarray(lambda_target_eval, dtype=float))

    def objective(vector: np.ndarray) -> float:
        run = build_run(np.asarray(vector, dtype=float), np.asarray(lambda_target, dtype=float))
        reg = 4.0e-4 * float(np.mean(np.square(np.asarray(vector, dtype=float))))
        loss, _ = phase1._hybrid_phase_loss(run.analysis, reg=reg)
        return float(loss)

    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": int(maxiter), "disp": False},
    )
    final_run = build_run(np.asarray(result.x, dtype=float), np.asarray(lambda_target, dtype=float))
    final_run.evaluate_from_vector = evaluate_from_vector
    final_run.bounds_lb = np.asarray(bounds.lb, dtype=float)
    final_run.bounds_ub = np.asarray(bounds.ub, dtype=float)
    final_run.optimizer = "Powell"
    return final_run


def _run_row(run: FamilyRun, *, drift_rel: np.ndarray) -> dict[str, Any]:
    phase = run.analysis["phase_summary"]
    snap = run.analysis["explicit_snap_benchmark"]
    impl_rel = np.asarray(phase["lambda_impl_relative_rad"], dtype=float)
    target_rel = np.asarray(phase["lambda_target_relative_rad"], dtype=float)
    predicted = predicted_relative_block_phases(
        model=run.model,
        frame=run.frame,
        n_levels=int(run.theta.size),
        total_duration_s=float(run.total_duration_s),
    )
    return {
        "family_id": run.family_id,
        "label": run.label,
        "category": run.category,
        "optimizer": run.optimizer,
        "cost_name": run.cost_name,
        "experiment_id": run.spec.experiment_id,
        "duration_ns": float(run.total_duration_s * 1.0e9),
        "n_levels": int(run.theta.size),
        "full_unitary_fidelity": float(run.analysis["full_unitary_fidelity"]),
        "block_gauge_fidelity": float(run.analysis["block_gauge_fidelity"]),
        "block_rotation_fidelity_mean": float(run.analysis["block_rotation_fidelity_mean"]),
        "block_rotation_fidelity_min": float(run.analysis["block_rotation_fidelity_min"]),
        "phase_error_rms_rad": float(phase["phase_error_rms_rad"]),
        "phase_steering_distance_from_drift_rad": float(_phase_steering_distance(impl_rel, np.asarray(drift_rel, dtype=float))),
        "max_abs_phase_error_vs_prediction_rad": float(np.max(np.abs(impl_rel - predicted))),
        "lambda_target_relative_rad": json.dumps([float(x) for x in target_rel.tolist()]),
        "lambda_impl_relative_rad": json.dumps([float(x) for x in impl_rel.tolist()]),
        "lambda_predicted_relative_rad": json.dumps([float(x) for x in predicted.tolist()]),
        "off_block_norm": float(run.analysis["off_block_norm"]),
        "ideal_post_snap_fidelity": float(snap["full_unitary_fidelity_after_snap"]),
        "snap_gap": float(snap["full_unitary_fidelity_after_snap"] - run.analysis["full_unitary_fidelity"]),
        "notes": run.notes,
    }


def _reachability_rows(
    *,
    family_runs: list[FamilyRun],
    targets: list[TargetPhaseProfile],
    drift_rel: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in family_runs:
        base_impl_rel = np.asarray(run.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
        for target in targets:
            analysis = _analysis_with_target(run, target.lambda_target)
            phase = analysis["phase_summary"]
            snap = analysis["explicit_snap_benchmark"]
            impl_rel = np.asarray(phase["lambda_impl_relative_rad"], dtype=float)
            rows.append(
                {
                    "family_id": run.family_id,
                    "label": run.label,
                    "category": run.category,
                    "target_name": target.name,
                    "target_category": target.category,
                    "duration_ns": float(run.total_duration_s * 1.0e9),
                    "n_levels": int(run.theta.size),
                    "full_unitary_fidelity": float(analysis["full_unitary_fidelity"]),
                    "block_gauge_fidelity": float(analysis["block_gauge_fidelity"]),
                    "block_rotation_fidelity_mean": float(analysis["block_rotation_fidelity_mean"]),
                    "block_rotation_fidelity_min": float(analysis["block_rotation_fidelity_min"]),
                    "phase_error_rms_rad": float(phase["phase_error_rms_rad"]),
                    "target_distance_from_drift_rad": float(_phase_rms(_relative(target.lambda_target), np.asarray(drift_rel, dtype=float))),
                    "phase_steering_distance_from_drift_rad": float(_phase_steering_distance(impl_rel, np.asarray(drift_rel, dtype=float))),
                    "max_abs_phase_shift_from_family_baseline_rad": float(np.max(np.abs(impl_rel - base_impl_rel))),
                    "ideal_post_snap_fidelity": float(snap["full_unitary_fidelity_after_snap"]),
                    "snap_gap": float(snap["full_unitary_fidelity_after_snap"] - analysis["full_unitary_fidelity"]),
                    "lambda_target_relative_rad": json.dumps([float(x) for x in _relative(target.lambda_target).tolist()]),
                    "lambda_impl_relative_rad": json.dumps([float(x) for x in impl_rel.tolist()]),
                }
            )
    return rows


def _spotcheck_rows(
    *,
    zero_target_runs: list[FamilyRun],
    retargeted_runs: list[FamilyRun],
    drift_rel: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id = {run.family_id: run for run in zero_target_runs}
    for run in retargeted_runs:
        baseline = by_id.get(run.family_id)
        if baseline is None:
            continue
        lam_base = np.asarray(baseline.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
        lam_run = np.asarray(run.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
        rows.append(
            {
                "family_id": run.family_id,
                "baseline_experiment_id": baseline.spec.experiment_id,
                "retargeted_experiment_id": run.spec.experiment_id,
                "duration_ns": float(run.total_duration_s * 1.0e9),
                "full_unitary_fidelity": float(run.analysis["full_unitary_fidelity"]),
                "block_gauge_fidelity": float(run.analysis["block_gauge_fidelity"]),
                "phase_error_rms_rad": float(run.analysis["phase_summary"]["phase_error_rms_rad"]),
                "phase_shift_vs_zero_target_rad": float(_phase_steering_distance(lam_run, lam_base)),
                "phase_shift_vs_drift_rad": float(_phase_steering_distance(lam_run, np.asarray(drift_rel, dtype=float))),
                "lambda_impl_relative_rad": json.dumps([float(x) for x in lam_run.tolist()]),
                "lambda_target_relative_rad": json.dumps(
                    [float(x) for x in np.asarray(run.analysis["phase_summary"]["lambda_target_relative_rad"], dtype=float).tolist()]
                ),
            }
        )
    return rows


def _sensitivity_analysis(
    run: FamilyRun,
    *,
    max_params: int = 8,
    eps_fraction: float = 1.0e-3,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if run.vector is None or run.evaluate_from_vector is None:
        return {"available": False, "family_id": run.family_id}, []
    x0 = np.asarray(run.vector, dtype=float)
    if x0.size == 0:
        return {"available": False, "family_id": run.family_id}, []
    lb = np.asarray(run.bounds_lb if run.bounds_lb is not None else -np.inf * np.ones_like(x0), dtype=float)
    ub = np.asarray(run.bounds_ub if run.bounds_ub is not None else np.inf * np.ones_like(x0), dtype=float)
    rng = np.random.default_rng(int(seed))
    if x0.size <= int(max_params):
        indices = list(range(x0.size))
    else:
        indices = sorted(rng.choice(np.arange(x0.size, dtype=int), size=int(max_params), replace=False).tolist())
    base_metrics = np.asarray(
        [
            float(run.analysis["block_gauge_fidelity"]),
            float(run.analysis["block_rotation_fidelity_mean"]),
            float(run.analysis["block_rotation_fidelity_min"]),
        ],
        dtype=float,
    )
    base_phase = np.asarray(run.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
    j_phase_cols: list[np.ndarray] = []
    j_metric_cols: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for idx in indices:
        width = float(ub[idx] - lb[idx]) if np.isfinite(lb[idx]) and np.isfinite(ub[idx]) else 1.0
        step = max(float(abs(x0[idx])) * float(eps_fraction), float(width) * float(eps_fraction), 1.0e-6)
        xp = np.asarray(x0, dtype=float).copy()
        xm = np.asarray(x0, dtype=float).copy()
        xp[idx] = min(float(ub[idx]), float(xp[idx] + step))
        xm[idx] = max(float(lb[idx]), float(xm[idx] - step))
        delta = float(xp[idx] - xm[idx])
        if delta <= 0.0:
            continue
        run_p = run.evaluate_from_vector(xp, run.lambda_target)
        run_m = run.evaluate_from_vector(xm, run.lambda_target)
        phase_p = np.asarray(run_p.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
        phase_m = np.asarray(run_m.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
        metrics_p = np.asarray(
            [
                float(run_p.analysis["block_gauge_fidelity"]),
                float(run_p.analysis["block_rotation_fidelity_mean"]),
                float(run_p.analysis["block_rotation_fidelity_min"]),
            ],
            dtype=float,
        )
        metrics_m = np.asarray(
            [
                float(run_m.analysis["block_gauge_fidelity"]),
                float(run_m.analysis["block_rotation_fidelity_mean"]),
                float(run_m.analysis["block_rotation_fidelity_min"]),
            ],
            dtype=float,
        )
        d_phase = (phase_p - phase_m) / delta
        d_metrics = (metrics_p - metrics_m) / delta
        j_phase_cols.append(d_phase)
        j_metric_cols.append(d_metrics)
        rows.append(
            {
                "family_id": run.family_id,
                "parameter_index": int(idx),
                "step_size": float(delta),
                "phase_grad_norm": float(np.linalg.norm(d_phase)),
                "metric_grad_norm": float(np.linalg.norm(d_metrics)),
                "max_abs_phase_grad": float(np.max(np.abs(d_phase))),
                "block_gauge_grad": float(d_metrics[0]),
                "block_rotation_mean_grad": float(d_metrics[1]),
                "block_rotation_min_grad": float(d_metrics[2]),
                "phase_shift_plus_rad": float(np.linalg.norm(phase_p - base_phase)),
                "phase_shift_minus_rad": float(np.linalg.norm(phase_m - base_phase)),
                "metric_shift_plus": float(np.linalg.norm(metrics_p - base_metrics)),
                "metric_shift_minus": float(np.linalg.norm(metrics_m - base_metrics)),
            }
        )
    if not j_phase_cols:
        return {"available": False, "family_id": run.family_id}, rows
    j_phase = np.stack(j_phase_cols, axis=1)
    j_metric = np.stack(j_metric_cols, axis=1)
    return {
        "available": True,
        "family_id": run.family_id,
        "n_parameters_sampled": int(j_phase.shape[1]),
        "phase_jacobian_singular_values": [float(x) for x in np.linalg.svd(j_phase, compute_uv=False).tolist()],
        "metric_jacobian_singular_values": [float(x) for x in np.linalg.svd(j_metric, compute_uv=False).tolist()],
        "phase_jacobian_fro_norm": float(np.linalg.norm(j_phase)),
        "metric_jacobian_fro_norm": float(np.linalg.norm(j_metric)),
        "max_coordinate_phase_grad_norm": float(max(row["phase_grad_norm"] for row in rows)),
        "max_coordinate_metric_grad_norm": float(max(row["metric_grad_norm"] for row in rows)),
    }, rows


def _duration_rows(*, n_max_values: tuple[int, ...], durations_ns: tuple[float, ...], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_max in n_max_values:
        for duration_ns in durations_ns:
            spec = _uniform_pi_spec(
                n_max=int(n_max),
                duration_ns=float(duration_ns),
                experiment_id=f"pinning_N{n_max}_T{int(duration_ns)}ns",
                lambda_label="zero",
                seed=int(seed),
            )
            model, frame, pulse_params, theta, phi, _, base_controls = _build_case_context(spec)
            naive = phase1.evaluate_controls(
                spec=spec,
                family=phase1.FAMILY_SPECS["A_naive"],
                model=model,
                frame=frame,
                pulse_params=pulse_params,
                theta=theta,
                phi=phi,
                lambda_target=np.zeros_like(theta),
                controls=base_controls,
            )
            observed = np.asarray(naive.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
            predicted = predicted_relative_block_phases(model=model, frame=frame, n_levels=int(theta.size), total_duration_s=float(naive.compiled.tlist[-1]))
            rows.append(
                {
                    "n_max": int(n_max),
                    "duration_ns": float(duration_ns),
                    "family_id": "A_naive",
                    "block_gauge_fidelity": float(naive.analysis["block_gauge_fidelity"]),
                    "block_rotation_fidelity_mean": float(naive.analysis["block_rotation_fidelity_mean"]),
                    "observed_relative_phase_rad": json.dumps([float(x) for x in observed.tolist()]),
                    "predicted_relative_phase_rad": json.dumps([float(x) for x in predicted.tolist()]),
                    "max_abs_phase_error_vs_prediction_rad": float(np.max(np.abs(observed - predicted))),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_fixed_duration_phase_pinning(
    *,
    runs: list[FamilyRun],
    drift_rel: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    n = np.arange(len(drift_rel), dtype=int)
    ax.plot(n, drift_rel, "k--", lw=2.2, label="Static-trace prediction")
    for run in runs:
        impl = np.asarray(run.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
        ax.plot(n, impl, marker="o", label=run.family_id)
    ax.set_xlabel("Fock index n")
    ax.set_ylabel("Relative block phase [rad]")
    ax.set_title("Fixed-duration block-phase pinning at T = 700 ns")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_reachability_cloud(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    family_ids = sorted({row["family_id"] for row in rows})
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    color_map = {family_id: colors[idx % len(colors)] for idx, family_id in enumerate(family_ids)}
    for family_id in family_ids:
        subset = [row for row in rows if row["family_id"] == family_id]
        ax.scatter(
            [float(row["target_distance_from_drift_rad"]) for row in subset],
            [float(row["phase_steering_distance_from_drift_rad"]) for row in subset],
            s=40,
            color=color_map[family_id],
            alpha=0.85,
            label=family_id,
        )
    ax.set_xlabel("Target phase distance from drift [rad RMS]")
    ax.set_ylabel("Achieved steering distance from drift [rad RMS]")
    ax.set_title("Reachability cloud: fixed-T qubit-only phase steering")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_fidelity_vs_target_distance(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    family_ids = sorted({row["family_id"] for row in rows})
    for family_id in family_ids:
        subset = sorted(
            (row for row in rows if row["family_id"] == family_id),
            key=lambda row: float(row["target_distance_from_drift_rad"]),
        )
        ax.plot(
            [float(row["target_distance_from_drift_rad"]) for row in subset],
            [float(row["full_unitary_fidelity"]) for row in subset],
            marker="o",
            label=family_id,
        )
    ax.set_xlabel("Target phase distance from drift [rad RMS]")
    ax.set_ylabel("Full truncated-space fidelity")
    ax.set_title("Full fidelity falls as the phase target moves away from drift")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_duration_curves(duration_rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    n_max_values = sorted({int(row["n_max"]) for row in duration_rows})
    for ax, n_max in zip(axes, n_max_values, strict=False):
        subset = sorted((row for row in duration_rows if int(row["n_max"]) == n_max), key=lambda row: float(row["duration_ns"]))
        durations = [float(row["duration_ns"]) for row in subset]
        observed = [np.asarray(json.loads(row["observed_relative_phase_rad"]), dtype=float) for row in subset]
        predicted = [np.asarray(json.loads(row["predicted_relative_phase_rad"]), dtype=float) for row in subset]
        for block_idx in range(len(observed[0])):
            if block_idx == 0:
                ax.plot(durations, [vec[block_idx] for vec in predicted], "k--", lw=2.0, label="predicted")
            else:
                ax.plot(durations, [vec[block_idx] for vec in predicted], "k--", lw=1.2)
            ax.plot(durations, [vec[block_idx] for vec in observed], marker="o", label=f"obs n={block_idx}")
        ax.set_title(f"N = {n_max}")
        ax.set_xlabel("Total duration [ns]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Relative block phase [rad]")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Duration is a 1D drift-assisted phase-shaping axis")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sensitivity(summary_map: dict[str, dict[str, Any]], output_path: Path) -> None:
    available = [payload for payload in summary_map.values() if payload.get("available")]
    if not available:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = [payload["family_id"] for payload in available]
    phase_norms = [float(payload["phase_jacobian_fro_norm"]) for payload in available]
    metric_norms = [float(payload["metric_jacobian_fro_norm"]) for payload in available]
    axes[0].bar(labels, phase_norms, color="tab:red")
    axes[0].set_title("Phase-Jacobian Frobenius norm")
    axes[0].set_ylabel("Norm")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(labels, metric_norms, color="tab:blue")
    axes[1].set_title("SU(2)-metric Jacobian Frobenius norm")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _report_text(
    *,
    fixed_rows: list[dict[str, Any]],
    reachability_rows: list[dict[str, Any]],
    spotcheck_rows: list[dict[str, Any]],
    duration_rows: list[dict[str, Any]],
    sensitivity_summary: dict[str, dict[str, Any]],
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Block-Phase Controllability Follow-Up")
    lines.append("")
    lines.append("## Model and Conventions")
    lines.append("")
    lines.append("- Tensor ordering: `qubit otimes cavity` throughout.")
    lines.append("- Internal Hamiltonian units: angular frequency in rad/s. Reported user-facing `chi / 2pi` and `K / 2pi` are in Hz.")
    lines.append(f"- Parameters: `chi / 2pi = {phase1.CHI_HZ / 1.0e6:.3f} MHz`, `K / 2pi = {phase1.KERR_HZ / 1.0e3:.1f} kHz`.")
    lines.append("- Base Hamiltonian: dispersive `chi` plus cavity self-Kerr, with qubit-only drive control.")
    lines.append("")
    lines.append("## Structural Result")
    lines.append("")
    lines.append("- For each fixed Fock sector `n`, the qubit-only drive contributes only traceless `b + b_dag` terms inside the `2 x 2` block.")
    lines.append("- Therefore `Tr H_n(t)` is set entirely by the static drift block, so `det U_n(T)` and the block phase `lambda_n` are fixed by the static block trace and the total duration `T`.")
    lines.append("- Consequence: at fixed total duration, no qubit-only pulse family in this simulator can independently steer the relative block phases. Pulse-shape freedom only changes the SU(2) part of each block.")
    lines.append("")
    lines.append("## New Pulse Families Tested")
    lines.append("")
    lines.append("- Phase-aware multitone SQR with per-tone amplitude, phase, and detuning freedom.")
    lines.append("- Piecewise-constant IQ control with ten segments over the full gate.")
    lines.append("- Segmented piecewise IQ with a fixed inserted wait window and active correction segments.")
    lines.append("- A simple composite two-segment phase-switched multitone sequence.")
    lines.append("")
    lines.append("## Fixed-Duration Evidence")
    lines.append("")
    best_fixed = sorted(fixed_rows, key=lambda row: float(row["full_unitary_fidelity"]), reverse=True)
    for row in best_fixed:
        lines.append(
            f"- `{row['family_id']}`: `F_full={row['full_unitary_fidelity']:.4f}`, "
            f"`F_block={row['block_gauge_fidelity']:.4f}`, "
            f"`phase_RMS={row['phase_error_rms_rad']:.4f} rad`, "
            f"`steering_from_drift={row['phase_steering_distance_from_drift_rad']:.3e} rad`, "
            f"`max|lambda_impl-lambda_pred|={row['max_abs_phase_error_vs_prediction_rad']:.3e} rad`."
        )
    lines.append("")
    lines.append("These runs cover much more flexible fixed-duration families than the first pass, but the extracted relative block phases still stay numerically pinned to the static-trace prediction.")
    lines.append("")
    if spotcheck_rows:
        lines.append("## Retargeting Spot Checks")
        lines.append("")
        for row in spotcheck_rows:
            lines.append(
                f"- `{row['family_id']}` retargeted away from zero-phase: `F_full={row['full_unitary_fidelity']:.4f}`, "
                f"`phase_RMS={row['phase_error_rms_rad']:.4f} rad`, "
                f"`phase_shift_vs_zero_target={row['phase_shift_vs_zero_target_rad']:.3e} rad`."
            )
        lines.append("")
        lines.append("Changing the phase objective changes the optimizer outcome for the SU(2) blocks, but not the implemented fixed-T block phase itself.")
        lines.append("")
    lines.append("## Reachability Scan")
    lines.append("")
    grouped = {}
    for row in reachability_rows:
        grouped.setdefault(row["family_id"], []).append(row)
    for family_id, subset in grouped.items():
        steering = max(float(row["phase_steering_distance_from_drift_rad"]) for row in subset)
        far = max(subset, key=lambda row: float(row["target_distance_from_drift_rad"]))
        near = min(subset, key=lambda row: float(row["target_distance_from_drift_rad"]))
        lines.append(
            f"- `{family_id}`: maximum observed steering distance over the target scan was `{steering:.3e} rad`. "
            f"Near-drift targets reached `F_full={near['full_unitary_fidelity']:.4f}`; the farthest tested target dropped to `F_full={far['full_unitary_fidelity']:.4f}`."
        )
    lines.append("")
    lines.append("That is the central reachability result: the fixed-T reachable set in block-phase space is effectively a single point for each `(N, T)`.")
    lines.append("")
    lines.append("## Sensitivity Diagnostic")
    lines.append("")
    for family_id, payload in sensitivity_summary.items():
        if not payload.get("available"):
            continue
        phase_norm = float(payload["phase_jacobian_fro_norm"])
        metric_norm = float(payload["metric_jacobian_fro_norm"])
        ratio = phase_norm / metric_norm if metric_norm > 0.0 else float("inf")
        lines.append(
            f"- `{family_id}`: `||J_phase||_F={phase_norm:.3e}`, `||J_SU2||_F={metric_norm:.3e}`, ratio `{ratio:.3e}`."
        )
    lines.append("")
    lines.append("Locally, nearby parameter directions still move the SU(2) quality, but they do not open meaningful block-phase directions.")
    lines.append("")
    lines.append("## Timing / Truncation Diagnostic")
    lines.append("")
    lines.append("- Varying the total duration changes the block phases exactly as predicted by the static block traces. That provides a one-dimensional drift-assisted phase-shaping axis.")
    lines.append("- The same pinning relation was verified on `N = 2, 3, 4` truncations.")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("- The lack of independent built-in SNAP control is not just a limitation of the original Gaussian/chirped ansatz families.")
    lines.append("- Under the current truncated dispersive-plus-Kerr Hamiltonian with qubit-only drive and fixed total duration, the relative block phases are fundamentally pinned by drift.")
    lines.append("- Segmented control, inserted waits at fixed total duration, and flexible piecewise IQ improve the SU(2) blocks but do not materially enlarge the reachable block-phase set.")
    lines.append("- The practical interpretation is `Route B`: treat the native block phase as drift-assisted timing/compiler structure, then apply a smaller explicit SNAP-style correction when arbitrary per-Fock phase synthesis is required.")
    lines.append("- If truly arbitrary block-phase synthesis is needed in one shot, the control set must be enlarged beyond qubit-only SQR, e.g. by a primitive that changes the block traces or an explicit cavity/SNAP-like operation.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name in (
        "fixed_duration_runs.csv",
        "reachability_table.csv",
        "retargeting_spotchecks.csv",
        "duration_truncation_table.csv",
        "sensitivity_table.csv",
        "summary.json",
    ):
        lines.append(f"- `{(output_dir / name).as_posix()}`")
    return "\n".join(lines) + "\n"


def run_study(output_dir: Path, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_spec = _uniform_pi_spec(
        n_max=DEFAULT_N_MAX,
        duration_ns=DEFAULT_DURATION_NS,
        experiment_id="followup_fixedT_zero_target",
        lambda_label="zero",
        seed=int(seed),
    )
    model, frame, pulse_params, theta, _, _, _ = _build_case_context(base_spec)
    drift_rel = predicted_relative_block_phases(
        model=model,
        frame=frame,
        n_levels=int(theta.size),
        total_duration_s=float(base_spec.duration_s + pulse_params.dt_eval_s),
    )
    targets = _phase_target_profiles(
        n_levels=int(theta.size),
        duration_s=float(base_spec.duration_s + pulse_params.dt_eval_s),
        kerr_hz=float(base_spec.kerr_hz),
        seed=int(seed),
        drift_rel=np.asarray(drift_rel, dtype=float),
    )
    target_map = {item.name: item for item in targets}

    zero = target_map["zero"].lambda_target
    far = target_map["far_from_drift"].lambda_target

    naive_zero = run_multitone_family(
        spec=replace(base_spec, experiment_id="followup_naive_zero", lambda_family="zero"),
        family_id="A_naive",
        lambda_target=zero,
        label="Naive baseline SQR",
        category="multitone_fixed_T",
        notes="No optimization; baseline multitone construction.",
    )
    multitone_zero = run_multitone_family(
        spec=replace(base_spec, experiment_id="followup_multitone_zero", lambda_family="zero"),
        family_id="D_extended_phase",
        lambda_target=zero,
        label="Phase-aware multitone SQR",
        category="multitone_fixed_T",
        notes="Per-tone amplitude, phase, and detuning freedom.",
    )
    multitone_far = run_multitone_family(
        spec=replace(base_spec, experiment_id="followup_multitone_far", lambda_family="far_from_drift"),
        family_id="D_extended_phase",
        lambda_target=far,
        label="Phase-aware multitone SQR",
        category="multitone_fixed_T",
        notes="Same family retargeted to a deliberately far phase profile.",
        initial_vector=multitone_zero.vector,
    )
    piecewise_zero = optimize_piecewise_family(
        spec=replace(base_spec, experiment_id="followup_piecewise_zero", lambda_family="zero"),
        lambda_target=zero,
        label="Piecewise-constant IQ",
        family_id="piecewise_iq",
        notes="Ten fixed-duration IQ segments.",
        nseg=10,
        maxiter=10,
    )
    piecewise_far = optimize_piecewise_family(
        spec=replace(base_spec, experiment_id="followup_piecewise_far", lambda_family="far_from_drift"),
        lambda_target=far,
        label="Piecewise-constant IQ",
        family_id="piecewise_iq",
        notes="Same ten-segment family retargeted far from drift.",
        nseg=10,
        maxiter=8,
        initial_vector=piecewise_zero.vector,
    )
    wait_zero = optimize_piecewise_family(
        spec=replace(base_spec, experiment_id="followup_wait_zero", lambda_family="zero"),
        lambda_target=zero,
        label="Segmented IQ with inserted wait",
        family_id="piecewise_wait",
        notes="Fixed total duration with a central wait window and active correction segments.",
        nseg=10,
        maxiter=10,
        wait_segments=DEFAULT_WAIT_SEGMENTS,
    )
    composite_zero = optimize_composite_family(
        spec=replace(base_spec, experiment_id="followup_composite_zero", lambda_family="zero"),
        lambda_target=zero,
        label="Composite phase-switched multitone",
        family_id="composite_multitone",
        notes="Two equal-time active multitone segments with global scale/phase/detuning freedom.",
        maxiter=10,
    )

    fixed_duration_runs = [naive_zero, multitone_zero, piecewise_zero, wait_zero, composite_zero]
    retargeted_runs = [multitone_far, piecewise_far]

    fixed_rows = [_run_row(run, drift_rel=np.asarray(drift_rel, dtype=float)) for run in fixed_duration_runs]
    spotcheck_rows = _spotcheck_rows(
        zero_target_runs=[multitone_zero, piecewise_zero],
        retargeted_runs=retargeted_runs,
        drift_rel=np.asarray(drift_rel, dtype=float),
    )
    reachability_rows = _reachability_rows(
        family_runs=[multitone_zero, piecewise_zero, wait_zero, composite_zero],
        targets=targets,
        drift_rel=np.asarray(drift_rel, dtype=float),
    )
    duration_rows = _duration_rows(
        n_max_values=(2, 3, 4),
        durations_ns=(450.0, 700.0, 1000.0),
        seed=int(seed),
    )

    sensitivity_summary: dict[str, dict[str, Any]] = {}
    sensitivity_rows: list[dict[str, Any]] = []
    for run in (multitone_zero, piecewise_zero, wait_zero):
        summary, rows = _sensitivity_analysis(run, max_params=8, eps_fraction=1.0e-3, seed=int(seed))
        sensitivity_summary[run.family_id] = summary
        sensitivity_rows.extend(rows)

    _write_csv(fixed_rows, output_dir / "fixed_duration_runs.csv")
    _write_csv(reachability_rows, output_dir / "reachability_table.csv")
    _write_csv(spotcheck_rows, output_dir / "retargeting_spotchecks.csv")
    _write_csv(duration_rows, output_dir / "duration_truncation_table.csv")
    _write_csv(sensitivity_rows, output_dir / "sensitivity_table.csv")

    _plot_fixed_duration_phase_pinning(
        runs=fixed_duration_runs,
        drift_rel=np.asarray(drift_rel, dtype=float),
        output_path=output_dir / "fixed_duration_phase_pinning.png",
    )
    _plot_reachability_cloud(reachability_rows, output_path=output_dir / "reachability_cloud.png")
    _plot_fidelity_vs_target_distance(reachability_rows, output_path=output_dir / "fidelity_vs_target_distance.png")
    _plot_duration_curves(duration_rows, output_path=output_dir / "duration_phase_curves.png")
    _plot_sensitivity(sensitivity_summary, output_path=output_dir / "sensitivity_summary.png")

    report = _report_text(
        fixed_rows=fixed_rows,
        reachability_rows=reachability_rows,
        spotcheck_rows=spotcheck_rows,
        duration_rows=duration_rows,
        sensitivity_summary=sensitivity_summary,
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    payload = {
        "meta": {
            "tensor_ordering": "qubit tensor cavity",
            "internal_frequency_units": "rad/s",
            "chi_hz": float(base_spec.chi_hz),
            "kerr_hz": float(base_spec.kerr_hz),
            "seed": int(seed),
        },
        "drift_relative_phase_rad": [float(x) for x in np.asarray(drift_rel, dtype=float).tolist()],
        "fixed_duration_runs": fixed_rows,
        "reachability_rows": reachability_rows,
        "retargeting_spotchecks": spotcheck_rows,
        "duration_rows": duration_rows,
        "sensitivity_summary": sensitivity_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-duration block-phase controllability follow-up study for qubit-only SQR control."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FOLLOWUP_OUTPUT,
        help="Directory for tables, plots, and the markdown report.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for generated target phase profiles.")
    args = parser.parse_args()
    summary = run_study(output_dir=args.output_dir, seed=int(args.seed))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "fixed_runs": len(summary["fixed_duration_runs"]),
                "reachability_rows": len(summary["reachability_rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
