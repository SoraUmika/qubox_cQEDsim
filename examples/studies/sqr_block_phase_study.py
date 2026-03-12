from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples.studies.sqr_multitone_study as sms
from cqed_sim.core.conventions import qubit_cavity_block_indices
from cqed_sim.sim.extractors import conditioned_bloch_xyz


matplotlib.use("Agg")


CHI_HZ = -2.84e6
KERR_HZ = -30.0e3
DEFAULT_DURATION_NS = 700.0
DEFAULT_SIGMA_FRACTION = 1.0 / 6.0
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    label: str
    mode: str | None
    objective: str
    phase_aware: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    n_max: int
    duration_s: float
    theta: tuple[float, ...]
    phi: tuple[float, ...]
    lambda_family: str
    lambda_scale_rad: float = 0.0
    seed: int = DEFAULT_SEED
    chi_hz: float = CHI_HZ
    kerr_hz: float = KERR_HZ
    family_ids: tuple[str, ...] = ("A_naive", "B_extended_rot", "C_chirp_rot", "D_chirp_phase")
    include_extended_phase: bool = False
    notes: str = ""


@dataclass
class RunResult:
    experiment: ExperimentSpec
    family: FamilySpec
    theta: np.ndarray
    phi: np.ndarray
    lambda_target: np.ndarray
    model: sms.DispersiveTransmonCavityModel
    frame: sms.FrameSpec
    pulse_params: sms.PulseParams
    controls: list[sms.ToneControl]
    pulse: Any
    compiled: Any
    unitary_qobj: qt.Qobj
    unitary: np.ndarray
    analysis: dict[str, Any]
    vector: np.ndarray | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    optimization_status: dict[str, Any] | None = None


FAMILY_SPECS: dict[str, FamilySpec] = {
    "A_naive": FamilySpec(
        family_id="A_naive",
        label="Construction A: naive baseline SQR",
        mode=None,
        objective="baseline",
        phase_aware=False,
    ),
    "B_extended_rot": FamilySpec(
        family_id="B_extended_rot",
        label="Construction B: enlarged per-tone ansatz, rotation-only cost",
        mode=sms.MODE_EXTENDED,
        objective="rotation_only",
        phase_aware=False,
    ),
    "C_chirp_rot": FamilySpec(
        family_id="C_chirp_rot",
        label="Construction C: chirped/loop-capable ansatz, rotation-only cost",
        mode=sms.MODE_CHIRP,
        objective="rotation_only",
        phase_aware=False,
    ),
    "D_extended_phase": FamilySpec(
        family_id="D_extended_phase",
        label="Construction D: extended ansatz with explicit block-phase objective",
        mode=sms.MODE_EXTENDED,
        objective="hybrid_phase",
        phase_aware=True,
    ),
    "D_chirp_phase": FamilySpec(
        family_id="D_chirp_phase",
        label="Construction D: chirped ansatz with explicit block-phase objective",
        mode=sms.MODE_CHIRP,
        objective="hybrid_phase",
        phase_aware=True,
    ),
}


OPT_PARAMS = sms.OptimizationParams(
    method_stage1="Powell",
    method_stage2="L-BFGS-B",
    maxiter_stage1_basic=12,
    maxiter_stage2_basic=20,
    maxiter_stage1_extended=6,
    maxiter_stage2_extended=10,
    maxiter_stage1_chirp=8,
    maxiter_stage2_chirp=12,
    amp_delta_bounds=(-1.0, 1.0),
    allow_zero_theta_corrections=True,
    phase_delta_bounds=(-np.pi, np.pi),
    detuning_hz_bounds=(-3.2e6, 3.2e6),
    phase_ramp_hz_bounds=(-2.4e6, 2.4e6),
    reg_amp=6.0e-4,
    reg_phase=4.0e-4,
    reg_detuning=2.0e-4,
    reg_phase_ramp=1.5e-4,
)


def hz_to_rad_s(value_hz: float) -> float:
    return float(2.0 * np.pi * value_hz)


def wrap_pi(values: np.ndarray | float) -> np.ndarray | float:
    return sms.wrap_pi(values)


def _target_rotation_matrix(theta: float, phi: float) -> np.ndarray:
    return np.asarray(sms.qubit_rotation_xy(float(theta), float(phi)).full(), dtype=np.complex128)


def _block_indices(n_cav: int, n: int) -> tuple[int, int]:
    return qubit_cavity_block_indices(n_cav, n)


def _extract_block(unitary: np.ndarray, n_cav: int, n: int) -> np.ndarray:
    idx = _block_indices(n_cav, n)
    return np.asarray(unitary[np.ix_(idx, idx)], dtype=np.complex128)


def _off_block_norm(unitary: np.ndarray, n_cav: int) -> float:
    matrix = np.asarray(unitary, dtype=np.complex128)
    mask = np.zeros_like(matrix, dtype=bool)
    for n in range(int(n_cav)):
        idx = _block_indices(n_cav, n)
        mask[np.ix_(idx, idx)] = True
    return float(np.linalg.norm(matrix[~mask]))


def _build_target_unitary(theta: np.ndarray, phi: np.ndarray, lambda_phase: np.ndarray) -> np.ndarray:
    n_cav = int(theta.size)
    full = np.zeros((2 * n_cav, 2 * n_cav), dtype=np.complex128)
    for n in range(n_cav):
        idx = _block_indices(n_cav, n)
        block = np.exp(1j * float(lambda_phase[n])) * _target_rotation_matrix(float(theta[n]), float(phi[n]))
        full[np.ix_(idx, idx)] = block
    return full


def _relative_phases(phases: np.ndarray) -> np.ndarray:
    phases = np.asarray(phases, dtype=float)
    if phases.size == 0:
        return phases
    rel = phases - float(phases[0])
    return np.unwrap(rel)


def _phase_errors(lambda_impl: np.ndarray, lambda_target: np.ndarray) -> dict[str, Any]:
    impl_rel = _relative_phases(lambda_impl)
    target_rel = _relative_phases(lambda_target)
    err = impl_rel - target_rel
    x = np.arange(err.size, dtype=float)
    design = np.stack([np.ones_like(x), x], axis=1)
    coeff, *_ = np.linalg.lstsq(design, err, rcond=None)
    affine = design @ coeff
    affine_resid = err - affine
    return {
        "lambda_impl_relative_rad": impl_rel.tolist(),
        "lambda_target_relative_rad": target_rel.tolist(),
        "phase_error_relative_rad": err.tolist(),
        "phase_error_rms_rad": float(np.sqrt(np.mean(err**2))) if err.size else 0.0,
        "phase_error_max_abs_rad": float(np.max(np.abs(err))) if err.size else 0.0,
        "phase_error_affine_rms_rad": float(np.sqrt(np.mean(affine_resid**2))) if err.size else 0.0,
        "phase_error_affine_fit": {
            "offset_rad": float(coeff[0]),
            "slope_rad_per_n": float(coeff[1]),
        },
    }


def _canonical_block_phase(block: np.ndarray) -> float:
    det = np.linalg.det(np.asarray(block, dtype=np.complex128))
    return float(0.5 * np.angle(det))


def _block_gauge_fidelity(target_blocks: list[np.ndarray], actual_blocks: list[np.ndarray]) -> float:
    accum = 0.0
    for target, actual in zip(target_blocks, actual_blocks, strict=True):
        accum += float(abs(np.trace(np.asarray(target).conj().T @ np.asarray(actual))))
    denom = 2.0 * max(len(target_blocks), 1)
    return float((accum / denom) ** 2)


def analyze_unitary(unitary: np.ndarray, theta: np.ndarray, phi: np.ndarray, lambda_target: np.ndarray) -> dict[str, Any]:
    n_cav = int(theta.size)
    target_blocks: list[np.ndarray] = []
    actual_blocks: list[np.ndarray] = []
    block_rows: list[dict[str, Any]] = []
    lambda_impl = np.zeros(n_cav, dtype=float)
    rotation_fids: list[float] = []
    full_block_fids: list[float] = []
    pre_z_values: list[float] = []
    post_z_values: list[float] = []

    for n in range(n_cav):
        target_rot = sms.normalize_unitary(_target_rotation_matrix(float(theta[n]), float(phi[n])))
        target_block = np.exp(1j * float(lambda_target[n])) * target_rot
        actual_block = _extract_block(unitary, n_cav=n_cav, n=n)
        lambda_n = _canonical_block_phase(actual_block)
        lambda_impl[n] = lambda_n
        actual_su2 = np.exp(-1j * lambda_n) * actual_block
        actual_su2 = sms.normalize_unitary(actual_su2)
        target_blocks.append(target_block)
        actual_blocks.append(actual_block)

        rotation_fid = sms.process_fidelity(target_rot, actual_su2)
        full_block_fid = sms.process_fidelity(target_block, actual_block)
        mismatch = sms.normalize_unitary(target_rot.conj().T @ actual_su2)
        decomp = sms.decompose_z_rxy_z(mismatch)
        overlap = complex(np.trace(target_rot.conj().T @ actual_block))
        lambda_overlap = float(np.angle(overlap)) if abs(overlap) > 1.0e-12 else float("nan")
        theta_actual, nx, ny, nz, phi_actual = sms.rotation_axis_parameters(actual_su2)

        rotation_fids.append(float(rotation_fid))
        full_block_fids.append(float(full_block_fid))
        pre_z_values.append(float(decomp["alpha_rad"]))
        post_z_values.append(float(decomp["beta_rad"]))
        block_rows.append(
            {
                "n": int(n),
                "theta_target_rad": float(theta[n]),
                "phi_target_rad": float(phi[n]),
                "lambda_target_rad": float(lambda_target[n]),
                "lambda_impl_rad": float(lambda_n),
                "lambda_overlap_to_target_rot_rad": float(lambda_overlap),
                "rotation_process_fidelity": float(rotation_fid),
                "full_block_process_fidelity": float(full_block_fid),
                "theta_impl_rad": float(theta_actual),
                "axis_x": float(nx),
                "axis_y": float(ny),
                "axis_z": float(nz),
                "phi_impl_rad": float(phi_actual),
                "residual_pre_z_rad": float(decomp["alpha_rad"]),
                "residual_post_z_rad": float(decomp["beta_rad"]),
                "residual_fit_error": float(decomp["fit_error"]),
            }
        )

    target_unitary = _build_target_unitary(theta=theta, phi=phi, lambda_phase=lambda_target)
    d = float(target_unitary.shape[0])
    full_overlap = np.trace(target_unitary.conj().T @ np.asarray(unitary, dtype=np.complex128))
    full_fidelity = float(np.abs(full_overlap) ** 2 / (d * d))
    rotation_blocks = [
        np.exp(1j * 0.0) * sms.normalize_unitary(_target_rotation_matrix(float(theta[n]), float(phi[n])))
        for n in range(n_cav)
    ]
    block_gauge_fid = _block_gauge_fidelity(rotation_blocks, actual_blocks)
    off_block = _off_block_norm(unitary, n_cav=n_cav)
    phase_summary = _phase_errors(lambda_impl=lambda_impl, lambda_target=lambda_target)

    snap_diag = np.eye(2 * n_cav, dtype=np.complex128)
    snap_correction = []
    for n, (target_block, actual_block) in enumerate(zip(target_blocks, actual_blocks, strict=True)):
        overlap = complex(np.trace(target_block.conj().T @ actual_block))
        correction = 0.0 if abs(overlap) <= 1.0e-12 else -float(np.angle(overlap))
        snap_correction.append(correction)
        idx = _block_indices(n_cav, n)
        snap_diag[idx[0], idx[0]] = np.exp(1j * correction)
        snap_diag[idx[1], idx[1]] = np.exp(1j * correction)
    corrected = snap_diag @ np.asarray(unitary, dtype=np.complex128)
    corrected_overlap = np.trace(target_unitary.conj().T @ corrected)
    corrected_fidelity = float(np.abs(corrected_overlap) ** 2 / (d * d))

    return {
        "n_cav": int(n_cav),
        "full_unitary_fidelity": float(full_fidelity),
        "block_gauge_fidelity": float(block_gauge_fid),
        "block_rotation_fidelity_mean": float(np.mean(rotation_fids)),
        "block_rotation_fidelity_min": float(np.min(rotation_fids)),
        "block_full_fidelity_mean": float(np.mean(full_block_fids)),
        "block_full_fidelity_min": float(np.min(full_block_fids)),
        "residual_pre_z_rms_rad": float(np.sqrt(np.mean(np.square(pre_z_values)))),
        "residual_post_z_rms_rad": float(np.sqrt(np.mean(np.square(post_z_values)))),
        "off_block_norm": float(off_block),
        "phase_summary": phase_summary,
        "explicit_snap_benchmark": {
            "snap_correction_rad": [float(x) for x in snap_correction],
            "full_unitary_fidelity_after_snap": float(corrected_fidelity),
            "improvement": float(corrected_fidelity - full_fidelity),
        },
        "block_rows": block_rows,
    }


def rotation_profile_uniform_pi(n_levels: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(int(n_levels), np.pi, dtype=float), np.zeros(int(n_levels), dtype=float)


def rotation_profile_structured(n_levels: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.array([np.pi, 0.75 * np.pi, -0.5 * np.pi, 0.35 * np.pi, -0.25 * np.pi], dtype=float)
    phi = np.array([0.0, np.pi / 2.0, np.pi / 4.0, np.pi / 3.0, -np.pi / 5.0], dtype=float)
    theta = theta[: int(n_levels)].copy()
    phi = phi[: int(n_levels)].copy()
    return theta, phi


def lambda_profile(
    family: str,
    n_levels: int,
    duration_s: float,
    kerr_hz: float,
    scale_rad: float,
    seed: int,
) -> np.ndarray:
    n = np.arange(int(n_levels), dtype=float)
    if family == "zero":
        return np.zeros(int(n_levels), dtype=float)
    if family == "kerr_cancel":
        kerr_rad_s = hz_to_rad_s(float(kerr_hz))
        return -(0.5 * kerr_rad_s * n * (n - 1.0) * float(duration_s))
    if family == "quadratic_small":
        return float(scale_rad) * n * (n - 1.0)
    if family == "alternating_small":
        return float(scale_rad) * np.asarray([0.0 if i == 0 else ((-1.0) ** i) for i in range(int(n_levels))], dtype=float)
    if family == "random_small":
        rng = np.random.default_rng(int(seed))
        lam = rng.uniform(-float(scale_rad), float(scale_rad), size=int(n_levels))
        lam[0] = 0.0
        return lam
    raise ValueError(f"Unsupported lambda family '{family}'.")


def build_model_and_pulse(spec: ExperimentSpec) -> tuple[sms.DispersiveTransmonCavityModel, sms.FrameSpec, sms.PulseParams]:
    system = sms.SystemParams(
        n_max=int(spec.n_max),
        omega_c_hz=0.0,
        omega_q_hz=0.0,
        qubit_alpha_hz=0.0,
        chi_nominal_hz=float(spec.chi_hz),
        chi_easy_hz=float(spec.chi_hz),
        chi_hard_hz=float(spec.chi_hz),
        chi2_hz=0.0,
        chi3_hz=0.0,
        kerr_hz=float(spec.kerr_hz),
        use_rotating_frame=True,
    )
    model, frame = sms.build_model_and_frame(system, chi_hz=float(spec.chi_hz))
    pulse = sms.PulseParams(
        duration_nominal_s=float(spec.duration_s),
        sigma_fraction=DEFAULT_SIGMA_FRACTION,
        envelope_kind="gaussian",
        theta_cutoff=1.0e-9,
        include_zero_theta_tones=True,
        dt_eval_s=2.0e-9,
        dt_opt_s=4.0e-9,
        max_step_eval_s=2.0e-9,
        max_step_opt_s=4.0e-9,
        qutip_nsteps=150000,
    )
    return model, frame, pulse


def evaluate_controls(
    *,
    spec: ExperimentSpec,
    family: FamilySpec,
    model: sms.DispersiveTransmonCavityModel,
    frame: sms.FrameSpec,
    pulse_params: sms.PulseParams,
    theta: np.ndarray,
    phi: np.ndarray,
    lambda_target: np.ndarray,
    controls: list[sms.ToneControl],
    vector: np.ndarray | None = None,
    history: list[dict[str, float]] | None = None,
    optimization_status: dict[str, Any] | None = None,
) -> RunResult:
    pulse = sms.build_pulse_from_controls(
        controls=controls,
        duration_s=float(spec.duration_s),
        pulse=pulse_params,
        label=family.family_id,
    )
    compiled = sms.compile_single_pulse(pulse, dt_s=float(pulse_params.dt_eval_s))
    unitary_qobj = sms.propagate_pulse_unitary(
        model=model,
        frame=frame,
        compiled=compiled,
        max_step_s=float(pulse_params.max_step_eval_s),
        qutip_nsteps=int(pulse_params.qutip_nsteps),
    )
    unitary = np.asarray(unitary_qobj.full(), dtype=np.complex128)
    analysis = analyze_unitary(unitary=unitary, theta=theta, phi=phi, lambda_target=lambda_target)
    return RunResult(
        experiment=spec,
        family=family,
        theta=np.asarray(theta, dtype=float),
        phi=np.asarray(phi, dtype=float),
        lambda_target=np.asarray(lambda_target, dtype=float),
        model=model,
        frame=frame,
        pulse_params=pulse_params,
        controls=controls,
        pulse=pulse,
        compiled=compiled,
        unitary_qobj=unitary_qobj,
        unitary=unitary,
        analysis=analysis,
        vector=None if vector is None else np.asarray(vector, dtype=float),
        history=[] if history is None else history,
        optimization_status=optimization_status,
    )


def _rotation_only_loss(analysis: dict[str, Any], reg: float) -> tuple[float, dict[str, float]]:
    worst_shortfall = max(0.0, 0.995 - float(analysis["block_rotation_fidelity_min"]))
    terms = {
        "rotation_mean": float(1.0 - analysis["block_rotation_fidelity_mean"]),
        "rotation_worst": float(2.0 * worst_shortfall * worst_shortfall),
        "off_block": float(0.05 * analysis["off_block_norm"] ** 2),
        "pre_z": float(0.08 * analysis["residual_pre_z_rms_rad"] ** 2),
        "post_z": float(0.08 * analysis["residual_post_z_rms_rad"] ** 2),
        "regularization": float(reg),
    }
    return float(sum(terms.values())), terms


def _hybrid_phase_loss(analysis: dict[str, Any], reg: float) -> tuple[float, dict[str, float]]:
    phase_rms = float(analysis["phase_summary"]["phase_error_rms_rad"])
    worst_shortfall = max(0.0, 0.992 - float(analysis["block_full_fidelity_min"]))
    terms = {
        "full_unitary": float(1.0 - analysis["full_unitary_fidelity"]),
        "rotation_mean": float(0.40 * (1.0 - analysis["block_rotation_fidelity_mean"])),
        "phase_rms": float(0.22 * phase_rms * phase_rms),
        "worst_block": float(0.55 * worst_shortfall * worst_shortfall),
        "off_block": float(0.05 * analysis["off_block_norm"] ** 2),
        "regularization": float(reg),
    }
    return float(sum(terms.values())), terms


def _evaluate_vector(
    vector: np.ndarray,
    *,
    spec: ExperimentSpec,
    family: FamilySpec,
    base_controls: list[sms.ToneControl],
    model: sms.DispersiveTransmonCavityModel,
    frame: sms.FrameSpec,
    pulse_params: sms.PulseParams,
    theta: np.ndarray,
    phi: np.ndarray,
    lambda_target: np.ndarray,
    active_indices: list[int],
) -> tuple[RunResult, float, dict[str, float]]:
    controls = sms.controls_from_vector(
        base=base_controls,
        active_indices=active_indices,
        mode=str(family.mode),
        vector=np.asarray(vector, dtype=float),
        duration_s=float(spec.duration_s),
    )
    run = evaluate_controls(
        spec=spec,
        family=family,
        model=model,
        frame=frame,
        pulse_params=pulse_params,
        theta=theta,
        phi=phi,
        lambda_target=lambda_target,
        controls=controls,
        vector=vector,
    )
    reg = sms.regularization_cost(str(family.mode), np.asarray(vector, dtype=float), OPT_PARAMS)
    if family.objective == "rotation_only":
        loss, terms = _rotation_only_loss(run.analysis, reg=reg)
    elif family.objective == "hybrid_phase":
        loss, terms = _hybrid_phase_loss(run.analysis, reg=reg)
    else:
        raise ValueError(f"Unsupported optimization objective '{family.objective}'.")
    return run, float(loss), terms


def optimize_family(
    *,
    spec: ExperimentSpec,
    family: FamilySpec,
    base_controls: list[sms.ToneControl],
    model: sms.DispersiveTransmonCavityModel,
    frame: sms.FrameSpec,
    pulse_params: sms.PulseParams,
    theta: np.ndarray,
    phi: np.ndarray,
    lambda_target: np.ndarray,
    initial_vector: np.ndarray | None = None,
) -> RunResult:
    active = sms.active_controls(
        base_controls,
        theta_cutoff=float(pulse_params.theta_cutoff),
        include_zero_amp=bool(OPT_PARAMS.allow_zero_theta_corrections),
    )
    x0, bounds = sms.parameter_layout(mode=str(family.mode), active_indices=active, opt=OPT_PARAMS)
    if initial_vector is not None and initial_vector.shape == x0.shape:
        x0 = np.asarray(initial_vector, dtype=float).copy()
    history: list[dict[str, float]] = []
    best: dict[str, Any] = {"loss": float("inf"), "run": None, "vector": None, "terms": None}

    def objective(vector: np.ndarray) -> float:
        run, loss, terms = _evaluate_vector(
            vector=np.asarray(vector, dtype=float),
            spec=spec,
            family=family,
            base_controls=base_controls,
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            theta=theta,
            phi=phi,
            lambda_target=lambda_target,
            active_indices=active,
        )
        row = {
            "eval": float(len(history)),
            "loss_total": float(loss),
            "full_unitary_fidelity": float(run.analysis["full_unitary_fidelity"]),
            "block_gauge_fidelity": float(run.analysis["block_gauge_fidelity"]),
            "block_rotation_fidelity_mean": float(run.analysis["block_rotation_fidelity_mean"]),
            "block_full_fidelity_mean": float(run.analysis["block_full_fidelity_mean"]),
            "phase_error_rms_rad": float(run.analysis["phase_summary"]["phase_error_rms_rad"]),
            "phase_error_affine_rms_rad": float(run.analysis["phase_summary"]["phase_error_affine_rms_rad"]),
            "off_block_norm": float(run.analysis["off_block_norm"]),
        }
        for key, value in terms.items():
            row[f"loss_{key}"] = float(value)
        history.append(row)
        if float(loss) < float(best["loss"]):
            best["loss"] = float(loss)
            best["run"] = run
            best["vector"] = np.asarray(vector, dtype=float).copy()
            best["terms"] = dict(terms)
        return float(loss)

    if family.mode == sms.MODE_EXTENDED:
        maxiter_stage1 = int(OPT_PARAMS.maxiter_stage1_extended)
        maxiter_stage2 = int(OPT_PARAMS.maxiter_stage2_extended)
    else:
        maxiter_stage1 = int(OPT_PARAMS.maxiter_stage1_chirp)
        maxiter_stage2 = int(OPT_PARAMS.maxiter_stage2_chirp)

    stage1 = minimize(
        objective,
        x0=x0,
        method=str(OPT_PARAMS.method_stage1),
        bounds=bounds,
        options={"maxiter": maxiter_stage1, "disp": False},
    )
    stage2 = minimize(
        objective,
        x0=np.asarray(stage1.x, dtype=float),
        method=str(OPT_PARAMS.method_stage2),
        bounds=bounds,
        options={"maxiter": maxiter_stage2},
    )

    candidates = [np.asarray(x0, dtype=float), np.asarray(stage1.x, dtype=float), np.asarray(stage2.x, dtype=float)]
    if best["vector"] is not None:
        candidates.append(np.asarray(best["vector"], dtype=float))

    rng = np.random.default_rng(int(spec.seed))
    best_vec = np.asarray(best["vector"] if best["vector"] is not None else stage2.x, dtype=float)
    for _ in range(1):
        trial = best_vec + rng.normal(scale=0.10, size=best_vec.shape)
        trial = np.clip(trial, bounds.lb, bounds.ub)
        local = minimize(
            objective,
            x0=trial,
            method=str(OPT_PARAMS.method_stage2),
            bounds=bounds,
            options={"maxiter": max(8, maxiter_stage2 // 2)},
        )
        candidates.append(np.asarray(local.x, dtype=float))

    final_runs: list[tuple[RunResult, float, dict[str, float], np.ndarray]] = []
    for candidate in candidates:
        run, loss, terms = _evaluate_vector(
            vector=np.asarray(candidate, dtype=float),
            spec=spec,
            family=family,
            base_controls=base_controls,
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            theta=theta,
            phi=phi,
            lambda_target=lambda_target,
            active_indices=active,
        )
        final_runs.append((run, float(loss), terms, np.asarray(candidate, dtype=float)))

    final_run, final_loss, final_terms, final_vector = min(final_runs, key=lambda item: float(item[1]))
    final_run.history = history
    final_run.vector = np.asarray(final_vector, dtype=float)
    final_run.optimization_status = {
        "family": family.family_id,
        "mode": family.mode,
        "objective": family.objective,
        "active_manifolds": [int(base_controls[idx].manifold) for idx in active],
        "stage1": {
            "method": str(OPT_PARAMS.method_stage1),
            "success": bool(stage1.success),
            "message": str(stage1.message),
            "fun": float(stage1.fun),
            "nit": int(getattr(stage1, "nit", 0)),
            "nfev": int(getattr(stage1, "nfev", 0)),
        },
        "stage2": {
            "method": str(OPT_PARAMS.method_stage2),
            "success": bool(stage2.success),
            "message": str(stage2.message),
            "fun": float(stage2.fun),
            "nit": int(getattr(stage2, "nit", 0)),
            "nfev": int(getattr(stage2, "nfev", 0)),
        },
        "best_loss": float(final_loss),
        "loss_terms_final": {key: float(value) for key, value in final_terms.items()},
        "n_objective_calls": int(len(history)),
    }
    return final_run


def _family_row(run: RunResult) -> dict[str, Any]:
    phase = run.analysis["phase_summary"]
    snap = run.analysis["explicit_snap_benchmark"]
    return {
        "experiment_id": run.experiment.experiment_id,
        "title": run.experiment.title,
        "family_id": run.family.family_id,
        "family_label": run.family.label,
        "n_max": int(run.experiment.n_max),
        "n_levels": int(run.theta.size),
        "duration_ns": float(run.experiment.duration_s * 1.0e9),
        "chi_hz": float(run.experiment.chi_hz),
        "kerr_hz": float(run.experiment.kerr_hz),
        "lambda_family": str(run.experiment.lambda_family),
        "full_unitary_fidelity": float(run.analysis["full_unitary_fidelity"]),
        "block_gauge_fidelity": float(run.analysis["block_gauge_fidelity"]),
        "block_rotation_fidelity_mean": float(run.analysis["block_rotation_fidelity_mean"]),
        "block_rotation_fidelity_min": float(run.analysis["block_rotation_fidelity_min"]),
        "block_full_fidelity_mean": float(run.analysis["block_full_fidelity_mean"]),
        "block_full_fidelity_min": float(run.analysis["block_full_fidelity_min"]),
        "phase_error_rms_rad": float(phase["phase_error_rms_rad"]),
        "phase_error_affine_rms_rad": float(phase["phase_error_affine_rms_rad"]),
        "phase_error_max_abs_rad": float(phase["phase_error_max_abs_rad"]),
        "residual_pre_z_rms_rad": float(run.analysis["residual_pre_z_rms_rad"]),
        "residual_post_z_rms_rad": float(run.analysis["residual_post_z_rms_rad"]),
        "off_block_norm": float(run.analysis["off_block_norm"]),
        "full_fidelity_after_ideal_snap": float(snap["full_unitary_fidelity_after_snap"]),
        "snap_benchmark_improvement": float(snap["improvement"]),
    }


def _block_rows(run: RunResult) -> list[dict[str, Any]]:
    rows = []
    for row in run.analysis["block_rows"]:
        out = dict(row)
        out["experiment_id"] = run.experiment.experiment_id
        out["family_id"] = run.family.family_id
        out["n_max"] = int(run.experiment.n_max)
        out["duration_ns"] = float(run.experiment.duration_s * 1.0e9)
        out["lambda_family"] = str(run.experiment.lambda_family)
        rows.append(out)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_fidelity_vs_truncation(rows: list[dict[str, Any]], output_path: Path, experiment_prefix: str) -> None:
    filtered = [row for row in rows if row["experiment_id"].startswith(experiment_prefix)]
    families = ["A_naive", "B_extended_rot", "C_chirp_rot", "D_chirp_phase"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for family in families:
        subset = sorted((row for row in filtered if row["family_id"] == family), key=lambda row: int(row["n_max"]))
        if not subset:
            continue
        ax.plot(
            [int(row["n_max"]) for row in subset],
            [float(row["full_unitary_fidelity"]) for row in subset],
            "o-",
            label=family,
        )
    ax.set_xlabel("Truncation N")
    ax.set_ylabel("Full truncated-space fidelity")
    ax.set_title("Full-unitary fidelity vs truncation size")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_fidelity_vs_duration(rows: list[dict[str, Any]], output_path: Path, experiment_ids: list[str]) -> None:
    filtered = [row for row in rows if row["experiment_id"] in set(experiment_ids)]
    families = ["A_naive", "C_chirp_rot", "D_chirp_phase"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for family in families:
        subset = sorted((row for row in filtered if row["family_id"] == family), key=lambda row: float(row["duration_ns"]))
        if not subset:
            continue
        ax.plot(
            [float(row["duration_ns"]) for row in subset],
            [float(row["full_unitary_fidelity"]) for row in subset],
            "o-",
            label=family,
        )
    ax.set_xlabel("Gate duration [ns]")
    ax.set_ylabel("Full truncated-space fidelity")
    ax.set_title("Fidelity vs gate duration")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_phase_profile(run_map: dict[str, RunResult], output_path: Path, title: str) -> None:
    if not run_map:
        return
    sample = next(iter(run_map.values()))
    n = np.arange(sample.theta.size, dtype=int)
    target = np.asarray(sample.analysis["phase_summary"]["lambda_target_relative_rad"], dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(n, target, "k--", linewidth=2.0, label="target")
    for family_id, run in run_map.items():
        impl = np.asarray(run.analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float)
        ax.plot(n, impl, "o-", label=family_id)
    ax.set_xlabel("Fock n")
    ax.set_ylabel("Relative block phase [rad]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_conditional_bloch_loops(run: RunResult, output_path: Path, levels: list[int]) -> None:
    initial = run.model.basis_state(0, 0)
    result = sms.simulate_sequence(
        run.model,
        run.compiled,
        initial,
        {"qubit": "qubit"},
        config=sms.SimulationConfig(frame=run.frame, max_step=float(run.pulse_params.max_step_eval_s), store_states=True),
    )
    if result.states is None:
        raise RuntimeError("Expected stored states for Bloch trajectory plot.")
    fig, axes = plt.subplots(1, len(levels), figsize=(4.0 * len(levels), 4.0), sharex=True, sharey=True)
    if len(levels) == 1:
        axes = [axes]
    for ax, n in zip(axes, levels, strict=False):
        xs = []
        ys = []
        zs = []
        for state in result.states:
            x, y, z, prob, valid = conditioned_bloch_xyz(state, n=int(n), fallback="nan")
            if valid and prob > 1.0e-8:
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
            else:
                xs.append(np.nan)
                ys.append(np.nan)
                zs.append(np.nan)
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        zs_arr = np.asarray(zs, dtype=float)
        ax.plot(xs_arr, ys_arr, color="tab:blue", linewidth=1.6)
        valid_idx = np.where(np.isfinite(xs_arr) & np.isfinite(ys_arr))[0]
        if valid_idx.size:
            start = int(valid_idx[0])
            stop = int(valid_idx[-1])
            ax.scatter([xs_arr[start]], [ys_arr[start]], color="tab:green", label="start", zorder=3)
            ax.scatter([xs_arr[stop]], [ys_arr[stop]], color="tab:red", label="end", zorder=3)
            ax.text(0.03, 0.05, f"z_end={zs_arr[stop]:+.3f}", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"n={int(n)}")
        ax.set_xlabel("<sigma_x>")
        ax.set_ylabel("<sigma_y>")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run.family.family_id}: conditioned Bloch loops")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _serialize_run(run: RunResult) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_id": run.experiment.experiment_id,
            "title": run.experiment.title,
            "n_max": int(run.experiment.n_max),
            "duration_s": float(run.experiment.duration_s),
            "theta": [float(x) for x in run.theta],
            "phi": [float(x) for x in run.phi],
            "lambda_family": str(run.experiment.lambda_family),
            "notes": str(run.experiment.notes),
            "chi_hz": float(run.experiment.chi_hz),
            "kerr_hz": float(run.experiment.kerr_hz),
        },
        "family": {
            "family_id": run.family.family_id,
            "label": run.family.label,
            "mode": run.family.mode,
            "objective": run.family.objective,
            "phase_aware": bool(run.family.phase_aware),
        },
        "controls": [tone.as_dict() for tone in run.controls],
        "analysis": run.analysis,
        "history": run.history,
        "optimization_status": run.optimization_status,
        "vector": None if run.vector is None else [float(x) for x in np.asarray(run.vector, dtype=float)],
    }


def _uniform_pi_experiment(n_max: int, lambda_family: str, duration_ns: float, notes: str = "") -> ExperimentSpec:
    theta, phi = rotation_profile_uniform_pi(n_levels=n_max + 1)
    return ExperimentSpec(
        experiment_id=f"uniform_pi_{lambda_family}_N{n_max}_T{int(duration_ns)}ns",
        title=f"Uniform pi target, {lambda_family} phase, N={n_max}",
        n_max=int(n_max),
        duration_s=float(duration_ns) * 1.0e-9,
        theta=tuple(float(x) for x in theta),
        phi=tuple(float(x) for x in phi),
        lambda_family=str(lambda_family),
        include_extended_phase=(lambda_family != "zero" and n_max >= 3),
        notes=notes,
    )


def _structured_experiment(n_max: int, lambda_family: str, scale_rad: float = 0.0) -> ExperimentSpec:
    theta, phi = rotation_profile_structured(n_levels=n_max + 1)
    return ExperimentSpec(
        experiment_id=f"structured_{lambda_family}_N{n_max}_T700ns",
        title=f"Structured rotation target, {lambda_family} phase, N={n_max}",
        n_max=int(n_max),
        duration_s=700.0e-9,
        theta=tuple(float(x) for x in theta),
        phi=tuple(float(x) for x in phi),
        lambda_family=str(lambda_family),
        lambda_scale_rad=float(scale_rad),
        include_extended_phase=False,
        family_ids=("A_naive", "C_chirp_rot", "D_chirp_phase"),
        notes="Higher-complexity rotation profile beyond uniform pi.",
    )


def build_experiments() -> list[ExperimentSpec]:
    experiments = [
        _uniform_pi_experiment(2, "zero", DEFAULT_DURATION_NS, notes="Experiment 1 baseline."),
        _uniform_pi_experiment(3, "zero", DEFAULT_DURATION_NS, notes="Experiment 1 baseline."),
        _uniform_pi_experiment(4, "zero", DEFAULT_DURATION_NS, notes="Experiment 1 baseline."),
        _uniform_pi_experiment(2, "kerr_cancel", DEFAULT_DURATION_NS, notes="Experiment 4 Kerr-cancel target."),
        _uniform_pi_experiment(3, "kerr_cancel", DEFAULT_DURATION_NS, notes="Experiment 4 Kerr-cancel target."),
        _uniform_pi_experiment(4, "kerr_cancel", DEFAULT_DURATION_NS, notes="Experiment 4 Kerr-cancel target."),
        _uniform_pi_experiment(3, "kerr_cancel", 450.0, notes="Duration scan."),
        _uniform_pi_experiment(3, "kerr_cancel", 1000.0, notes="Duration scan."),
        _structured_experiment(3, "zero"),
        _structured_experiment(3, "quadratic_small", scale_rad=0.08),
        _structured_experiment(3, "alternating_small", scale_rad=0.16),
        _structured_experiment(3, "random_small", scale_rad=0.12),
        ExperimentSpec(
            experiment_id="uniform_pi_zero_N3_T700ns_kerr0",
            title="Uniform pi target, zero phase, N=3, Kerr off",
            n_max=3,
            duration_s=700.0e-9,
            theta=tuple(float(x) for x in rotation_profile_uniform_pi(4)[0]),
            phi=tuple(float(x) for x in rotation_profile_uniform_pi(4)[1]),
            lambda_family="zero",
            kerr_hz=0.0,
            family_ids=("A_naive", "D_extended_phase"),
            notes="Ablation: cavity Kerr disabled.",
        ),
        ExperimentSpec(
            experiment_id="uniform_pi_zero_N3_T700ns_chi_low",
            title="Uniform pi target, zero phase, N=3, reduced chi",
            n_max=3,
            duration_s=700.0e-9,
            theta=tuple(float(x) for x in rotation_profile_uniform_pi(4)[0]),
            phi=tuple(float(x) for x in rotation_profile_uniform_pi(4)[1]),
            lambda_family="zero",
            chi_hz=0.85 * CHI_HZ,
            family_ids=("D_extended_phase",),
            notes="Ablation: chi reduced by 15%.",
        ),
        ExperimentSpec(
            experiment_id="uniform_pi_zero_N3_T700ns_chi_high",
            title="Uniform pi target, zero phase, N=3, increased chi",
            n_max=3,
            duration_s=700.0e-9,
            theta=tuple(float(x) for x in rotation_profile_uniform_pi(4)[0]),
            phi=tuple(float(x) for x in rotation_profile_uniform_pi(4)[1]),
            lambda_family="zero",
            chi_hz=1.15 * CHI_HZ,
            family_ids=("D_extended_phase",),
            notes="Ablation: chi increased by 15%.",
        ),
    ]

    for spec in experiments:
        if spec.experiment_id == "uniform_pi_zero_N3_T700ns":
            object.__setattr__(
                spec,
                "family_ids",
                ("A_naive", "B_extended_rot", "C_chirp_rot", "D_extended_phase", "D_chirp_phase"),
            )
        elif spec.experiment_id == "uniform_pi_kerr_cancel_N3_T700ns":
            object.__setattr__(
                spec,
                "family_ids",
                ("A_naive", "B_extended_rot", "C_chirp_rot", "D_extended_phase", "D_chirp_phase"),
            )
        elif spec.experiment_id.endswith("N2_T700ns") or spec.experiment_id.endswith("N4_T700ns"):
            object.__setattr__(spec, "family_ids", ("A_naive", "D_extended_phase"))
        elif spec.experiment_id in {"uniform_pi_kerr_cancel_N3_T450ns", "uniform_pi_kerr_cancel_N3_T1000ns"}:
            object.__setattr__(spec, "family_ids", ("A_naive", "D_extended_phase"))
        elif spec.experiment_id.startswith("structured_"):
            object.__setattr__(spec, "family_ids", ("A_naive", "D_extended_phase"))
    return experiments


def run_experiment(spec: ExperimentSpec) -> dict[str, RunResult]:
    model, frame, pulse_params = build_model_and_pulse(spec)
    theta = np.asarray(spec.theta, dtype=float)
    phi = np.asarray(spec.phi, dtype=float)
    lambda_target = lambda_profile(
        family=str(spec.lambda_family),
        n_levels=int(theta.size),
        duration_s=float(spec.duration_s),
        kerr_hz=float(spec.kerr_hz),
        scale_rad=float(spec.lambda_scale_rad),
        seed=int(spec.seed),
    )
    profile = sms.TargetProfile(name=spec.experiment_id, mode="manual", theta=theta, phi=phi, seed=int(spec.seed))
    base_controls = sms.build_controls_from_target(
        profile=profile,
        model=model,
        frame=frame,
        duration_s=float(spec.duration_s),
        theta_cutoff=float(pulse_params.theta_cutoff),
        include_all_levels=bool(pulse_params.include_zero_theta_tones),
    )

    run_map: dict[str, RunResult] = {}
    for family_id in spec.family_ids:
        family = FAMILY_SPECS[family_id]
        if family.mode is None:
            run_map[family_id] = evaluate_controls(
                spec=spec,
                family=family,
                model=model,
                frame=frame,
                pulse_params=pulse_params,
                theta=theta,
                phi=phi,
                lambda_target=lambda_target,
                controls=base_controls,
            )
        elif family.objective == "rotation_only":
            run_map[family_id] = optimize_family(
                spec=spec,
                family=family,
                base_controls=base_controls,
                model=model,
                frame=frame,
                pulse_params=pulse_params,
                theta=theta,
                phi=phi,
                lambda_target=lambda_target,
            )
        else:
            init = None
            if family.mode == sms.MODE_EXTENDED and "B_extended_rot" in run_map:
                init = run_map["B_extended_rot"].vector
            if family.mode == sms.MODE_CHIRP and "C_chirp_rot" in run_map:
                init = run_map["C_chirp_rot"].vector
            run_map[family_id] = optimize_family(
                spec=spec,
                family=family,
                base_controls=base_controls,
                model=model,
                frame=frame,
                pulse_params=pulse_params,
                theta=theta,
                phi=phi,
                lambda_target=lambda_target,
                initial_vector=init,
            )

    if spec.include_extended_phase:
        family = FAMILY_SPECS["D_extended_phase"]
        init = run_map.get("B_extended_rot")
        run_map["D_extended_phase"] = optimize_family(
            spec=spec,
            family=family,
            base_controls=base_controls,
            model=model,
            frame=frame,
            pulse_params=pulse_params,
            theta=theta,
            phi=phi,
            lambda_target=lambda_target,
            initial_vector=None if init is None else init.vector,
        )
    return run_map


def _report_text(*, rows: list[dict[str, Any]], output_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# SQR Built-In SNAP-like Phase Proof-of-Concept")
    lines.append("")
    lines.append("## Conventions and Model")
    lines.append("")
    lines.append("- Tensor ordering: `qubit otimes cavity`, with block indices selected via `qubit_cavity_block_indices(...)`.")
    lines.append("- Internal simulator frequency units are rad/s. User-facing chi and Kerr values are reported in Hz.")
    lines.append("- Hamiltonian terms enabled in this study: dispersive chi and cavity Kerr; `chi2 = chi3 = alpha = 0` for the initial proof-of-concept model.")
    lines.append("- Drive family: multitone Gaussian qubit drive using the existing SQR tone builder and the existing QuTiP propagator path.")
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")

    def best_row(prefix: str, family_id: str) -> dict[str, Any] | None:
        subset = [row for row in rows if row["experiment_id"].startswith(prefix) and row["family_id"] == family_id]
        if not subset:
            return None
        return max(subset, key=lambda row: float(row["full_unitary_fidelity"]))

    def exact_row(experiment_id: str, family_id: str) -> dict[str, Any] | None:
        for row in rows:
            if row["experiment_id"] == experiment_id and row["family_id"] == family_id:
                return row
        return None

    zero_naive = best_row("uniform_pi_zero", "A_naive")
    zero_phase = max(
        [row for row in rows if row["experiment_id"].startswith("uniform_pi_zero") and row["family_id"] in {"D_extended_phase", "D_chirp_phase"}],
        key=lambda row: float(row["full_unitary_fidelity"]),
        default=None,
    )
    kerr_rot = best_row("uniform_pi_kerr_cancel", "C_chirp_rot")
    kerr_phase = max(
        [row for row in rows if row["experiment_id"].startswith("uniform_pi_kerr_cancel") and row["family_id"] in {"D_extended_phase", "D_chirp_phase"}],
        key=lambda row: float(row["full_unitary_fidelity"]),
        default=None,
    )
    kerr_ext = best_row("uniform_pi_kerr_cancel", "D_extended_phase")
    kerr_chirp = best_row("uniform_pi_kerr_cancel", "D_chirp_phase")
    structured = [row for row in rows if row["experiment_id"].startswith("structured_")]
    zero_n3_naive = exact_row("uniform_pi_zero_N3_T700ns", "A_naive")
    zero_n3_phase = exact_row("uniform_pi_zero_N3_T700ns", "D_chirp_phase")
    kerr_n3_rot = exact_row("uniform_pi_kerr_cancel_N3_T700ns", "C_chirp_rot")
    kerr_n3_phase = exact_row("uniform_pi_kerr_cancel_N3_T700ns", "D_chirp_phase")

    if zero_naive is not None:
        lines.append(
            "- The naive baseline can keep the block-gauge fidelity above the true full-unitary fidelity, which confirms that endpoint-style rotation agreement is not sufficient for the stronger joint-unitary target."
        )
        lines.append(
            "  Representative naive zero-phase row: "
            + f"`F_full={zero_naive['full_unitary_fidelity']:.4f}`, "
            + f"`F_block-gauge={zero_naive['block_gauge_fidelity']:.4f}`, "
            + f"`phase_rms={zero_naive['phase_error_rms_rad']:.4f} rad`."
        )

    if zero_n3_naive is not None and zero_n3_phase is not None:
        lines.append(
            "- For the zero-phase target, the phase-aware optimizer improves the block SU(2) action, but it does not materially change the extracted per-Fock phase profile. The gap to the ideal post-SNAP benchmark therefore remains phase-limited."
        )
        lines.append(
            "  N=3 zero-phase, naive vs phase-aware: "
            + f"`F_full={zero_n3_naive['full_unitary_fidelity']:.4f} -> {zero_n3_phase['full_unitary_fidelity']:.4f}`, "
            + f"`phase_rms={zero_n3_naive['phase_error_rms_rad']:.4f} -> {zero_n3_phase['phase_error_rms_rad']:.4f} rad`, "
            + f"`ideal-post-SNAP benchmark={zero_n3_phase['full_fidelity_after_ideal_snap']:.4f}`."
        )

    if kerr_n3_rot is not None and kerr_n3_phase is not None:
        lines.append(
            "- For the Kerr-cancel target at 700 ns, the natural drift-induced phase profile already lies close to the Kerr-like target. The phase-aware optimizer then improves the full fidelity mainly by fixing the SU(2) blocks, not by independently steering the block phases."
        )
        lines.append(
            "  Chirped rotation-only vs phase-aware: "
            + f"`F_full={kerr_n3_rot['full_unitary_fidelity']:.4f} -> {kerr_n3_phase['full_unitary_fidelity']:.4f}`, "
            + f"`phase_rms={kerr_n3_rot['phase_error_rms_rad']:.4f} -> {kerr_n3_phase['phase_error_rms_rad']:.4f} rad`."
        )

    if kerr_ext is not None and kerr_chirp is not None:
        lines.append(
            "- Comparing phase-aware extended detuning control to phase-aware chirped control indicates whether extra loop-capable freedom matters."
        )
        lines.append(
            "  "
            + f"`extended F_full={kerr_ext['full_unitary_fidelity']:.4f}, phase_rms={kerr_ext['phase_error_rms_rad']:.4f} rad`; "
            + f"`chirped F_full={kerr_chirp['full_unitary_fidelity']:.4f}, phase_rms={kerr_chirp['phase_error_rms_rad']:.4f} rad`."
        )

    if structured:
        best_structured = max(structured, key=lambda row: float(row["full_unitary_fidelity"]))
        lines.append(
            "- As the target rotations become more structured in `(theta_n, phi_n)`, the reachable fidelity drops relative to the uniform-pi cases. That is consistent with the larger simultaneous burden of matching both per-block SU(2) content and block phases."
        )
        lines.append(
            "  Best structured-target row: "
            + f"`{best_structured['family_id']}` with `F_full={best_structured['full_unitary_fidelity']:.4f}`."
        )

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Main summary table: `{(output_dir / 'summary_table.csv').as_posix()}`")
    lines.append(f"- Per-block table: `{(output_dir / 'block_table.csv').as_posix()}`")
    lines.append(f"- Full JSON payload: `{(output_dir / 'summary.json').as_posix()}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if kerr_phase is not None:
        gap = float(kerr_phase["full_fidelity_after_ideal_snap"] - kerr_phase["full_unitary_fidelity"])
        lines.append(
            "On small truncated spaces, the evidence supports only a qualified and limited proof of concept."
        )
        lines.append(
            "The multitone SQR family can realize the desired conditional rotations reasonably well, and for certain durations it can exploit the natural chi/Kerr drift to land close to a useful Kerr-like block-phase pattern."
        )
        lines.append(
            "But across fixed `(N, T)` studies the extracted relative block phases are essentially unchanged across the naive, extended, chirped, and phase-aware constructions. That means the built-in phase is mostly inherited from the drift, not independently controllable."
        )
        lines.append(
            f"The ideal explicit-SNAP benchmark still closes the remaining gap (`Delta F ~= {gap:.4f}` on the best Kerr-target row), so a separate SNAP-like correction remains the cleaner route for arbitrary phase profiles."
        )
    else:
        lines.append("Read the summary table directly if the Kerr-target comparison row is absent.")
    return "\n".join(lines) + "\n"


def run_study(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments()
    all_rows: list[dict[str, Any]] = []
    all_block_rows: list[dict[str, Any]] = []
    detailed: dict[str, dict[str, RunResult]] = {}

    for spec in experiments:
        print(f"[study] running {spec.experiment_id}")
        run_map = run_experiment(spec)
        detailed[spec.experiment_id] = run_map
        for run in run_map.values():
            all_rows.append(_family_row(run))
            all_block_rows.extend(_block_rows(run))

    _write_csv(all_rows, output_dir / "summary_table.csv")
    _write_csv(all_block_rows, output_dir / "block_table.csv")

    zero_phase_runs = {
        family_id: run
        for family_id, run in detailed["uniform_pi_zero_N4_T700ns"].items()
        if family_id in {"A_naive", "B_extended_rot", "C_chirp_rot", "D_chirp_phase"}
    }
    _plot_phase_profile(
        zero_phase_runs,
        output_path=output_dir / "phase_profile_zero_phase_N4.png",
        title="N=4 zero-phase target: achieved vs target block phase",
    )

    kerr_phase_runs = {
        family_id: run
        for family_id, run in detailed["uniform_pi_kerr_cancel_N4_T700ns"].items()
        if family_id in {"A_naive", "B_extended_rot", "C_chirp_rot", "D_extended_phase", "D_chirp_phase"}
    }
    _plot_phase_profile(
        kerr_phase_runs,
        output_path=output_dir / "phase_profile_kerr_target_N4.png",
        title="N=4 Kerr-cancel target: achieved vs target block phase",
    )

    _plot_fidelity_vs_truncation(
        rows=all_rows,
        output_path=output_dir / "fidelity_vs_truncation_zero_phase.png",
        experiment_prefix="uniform_pi_zero",
    )
    _plot_fidelity_vs_truncation(
        rows=all_rows,
        output_path=output_dir / "fidelity_vs_truncation_kerr_target.png",
        experiment_prefix="uniform_pi_kerr_cancel",
    )
    _plot_fidelity_vs_duration(
        rows=all_rows,
        output_path=output_dir / "fidelity_vs_duration_kerr_target_N3.png",
        experiment_ids=[
            "uniform_pi_kerr_cancel_N3_T450ns",
            "uniform_pi_kerr_cancel_N3_T700ns",
            "uniform_pi_kerr_cancel_N3_T1000ns",
        ],
    )

    trajectory_run = detailed["uniform_pi_kerr_cancel_N3_T700ns"]["D_extended_phase"]
    _plot_conditional_bloch_loops(
        trajectory_run,
        output_path=output_dir / "conditional_bloch_loops_kerr_target_N3.png",
        levels=[0, 1, 2],
    )

    report = _report_text(rows=all_rows, output_dir=output_dir)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    payload = {
        "meta": {
            "chi_hz": CHI_HZ,
            "kerr_hz": KERR_HZ,
            "default_duration_ns": DEFAULT_DURATION_NS,
            "tensor_ordering": "qubit tensor cavity",
            "internal_frequency_units": "rad/s",
        },
        "summary_rows": all_rows,
        "block_rows": all_block_rows,
        "experiments": {
            experiment_id: {family_id: _serialize_run(run) for family_id, run in run_map.items()}
            for experiment_id, run_map in detailed.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SQR built-in SNAP-like block-phase proof-of-concept study.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sqr_block_phase_study"),
        help="Directory where tables, figures, and the markdown report will be written.",
    )
    args = parser.parse_args()
    summary = run_study(output_dir=args.output_dir)
    print("Saved study outputs to", args.output_dir)
    print("Generated", len(summary["summary_rows"]), "summary rows.")


if __name__ == "__main__":
    main()
