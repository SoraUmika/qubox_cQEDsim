from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.calibration.sqr import (
    GuardedBenchmarkResult,
    RandomSQRTarget,
    calibrate_guarded_sqr_target,
    evaluate_guarded_sqr_target,
    extract_multitone_effective_qubit_unitary,
)
from cqed_sim.core.ideal_gates import qubit_rotation_xy
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.unitary_synthesis.metrics import subspace_unitary_fidelity
from cqed_sim.unitary_synthesis.progress import (
    CompositeReporter,
    HistoryReporter,
    JupyterLiveReporter,
    NullReporter,
    ProgressReporter,
    save_history_csv,
    save_history_json,
)


TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _coerce_numeric_tuple(values: Any, cast: type[int] | type[float]) -> tuple[int, ...] | tuple[float, ...]:
    if isinstance(values, (int, float, np.integer, np.floating)):
        return (cast(values),)
    return tuple(cast(x) for x in values)


@dataclass(frozen=True)
class TargetCase:
    name: str
    description: str
    target: RandomSQRTarget
    phase: str
    composite_with_ideal_pi: bool = False


@dataclass(frozen=True)
class SQRSpeedLimitConfig:
    seed: int = 1234
    n_match: int = 2
    guard_levels: int = 1
    chi_hz: float = -2.84e6
    chi2_hz: float = 0.0
    chi3_hz: float = 0.0
    kerr_hz: float = 0.0
    kerr2_hz: float = 0.0
    omega_q_hz: float = 0.0
    omega_c_hz: float = 0.0
    qubit_alpha_hz: float = 0.0
    use_rotating_frame: bool = True
    qb_t1_relax_ns: float | None = None
    qb_t2_ramsey_ns: float | None = None
    qb_t2_echo_ns: float | None = None
    t2_source: str = "ramsey"
    cavity_kappa_1_per_s: float = 0.0
    durations_ns: tuple[int, ...] = (50, 75, 100, 150, 200, 300, 500, 750, 1000)
    sigma_fractions: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
    multistart: int = 2
    parallel_enabled: bool = False
    parallel_n_jobs: int = 1
    dt_s: float = 2.0e-9
    max_step_s: float = 2.0e-9
    optimizer_maxiter_stage1: int = 6
    optimizer_maxiter_stage2: int = 8
    d_lambda_bounds: tuple[float, float] = (-0.5, 0.5)
    d_alpha_bounds: tuple[float, float] = (-np.pi, np.pi)
    d_omega_hz_bounds: tuple[float, float] = (-2.0e6, 2.0e6)
    lambda_guard: float = 0.10
    weight_mode: str = "uniform"
    fidelity_thresholds: tuple[float, ...] = (0.99, 0.999, 0.9999)
    representative_duration_ns: int = 200
    output_root: Path = Path("outputs/analysis/sqr_speedlimit_multitone_gaussian")
    report_path: Path = Path("cqed_sim/analysis/reports/sqr_speedlimit_report.md")
    progress_every: int = 1
    qutip_nsteps_sqr_calibration: int = 100000

    def __post_init__(self) -> None:
        object.__setattr__(self, "durations_ns", _coerce_numeric_tuple(self.durations_ns, int))
        object.__setattr__(self, "sigma_fractions", _coerce_numeric_tuple(self.sigma_fractions, float))
        object.__setattr__(self, "fidelity_thresholds", _coerce_numeric_tuple(self.fidelity_thresholds, float))

    @property
    def logical_n(self) -> int:
        return int(self.n_match + 1)

    @property
    def total_levels(self) -> int:
        return int(self.logical_n + self.guard_levels)

    @property
    def cavity_fock_cutoff(self) -> int:
        return int(self.total_levels - 1)

    def output_dir(self) -> Path:
        return Path(self.output_root) / f"seed_{int(self.seed)}"


def hz_to_rad_s(value_hz: float) -> float:
    return float(2.0 * np.pi * value_hz)


def rad_s_to_hz(value_rad_s: float) -> float:
    return float(value_rad_s / (2.0 * np.pi))


def wrap_pi(value: float | np.ndarray) -> float | np.ndarray:
    wrapped = (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(value) == 0:
        return float(np.asarray(wrapped))
    return wrapped


def normalize_unitary(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    det = np.linalg.det(matrix)
    if abs(det) > 1.0e-15:
        matrix = matrix * np.exp(-0.5j * np.angle(det))
    return matrix


def rotation_axis_parameters(unitary: np.ndarray) -> tuple[float, float]:
    u = normalize_unitary(np.asarray(unitary, dtype=np.complex128))
    theta = float(2.0 * np.arccos(np.clip(np.real(np.trace(u) / 2.0), -1.0, 1.0)))
    if theta < 1.0e-12:
        return 0.0, 0.0
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sin_half = float(np.sin(theta / 2.0))
    nx = float(np.real(1.0j * np.trace(sx @ u) / (2.0 * sin_half)))
    ny = float(np.real(1.0j * np.trace(sy @ u) / (2.0 * sin_half)))
    phi = float(np.mod(np.arctan2(ny, nx), 2.0 * np.pi))
    return theta, phi


def _initial_drive_amplitude(theta_target: float, duration_s: float, sigma_fraction: float, tlist: np.ndarray) -> float:
    env = np.asarray(gaussian_envelope(np.asarray(tlist, dtype=float) / max(float(duration_s), 1.0e-18), sigma=sigma_fraction), dtype=np.complex128)
    area = float(TRAPEZOID(np.real(env), np.asarray(tlist, dtype=float)))
    if abs(area) < 1.0e-15:
        return 0.0
    return float(theta_target / area)


def _conditional_detuning_rad_s(n: int, config: Mapping[str, Any]) -> float:
    detuning_hz = (
        float(config.get("st_chi_hz", 0.0)) * int(n)
        + float(config.get("st_chi2_hz", 0.0)) * (int(n) * (int(n) - 1))
        + float(config.get("st_chi3_hz", 0.0)) * (int(n) * (int(n) - 1) * (int(n) - 2))
    )
    return hz_to_rad_s(detuning_hz)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
    return path


def _manual_target(
    *,
    name: str,
    description: str,
    theta_logical: Sequence[float],
    phi_logical: Sequence[float],
    config: SQRSpeedLimitConfig,
    phase: str,
    composite_with_ideal_pi: bool = False,
) -> TargetCase:
    theta = tuple(float(x) for x in theta_logical) + tuple(0.0 for _ in range(int(config.guard_levels)))
    phi = tuple(float(x) for x in phi_logical) + tuple(0.0 for _ in range(int(config.guard_levels)))
    target = RandomSQRTarget(
        target_id=str(name),
        target_class=str(phase),
        logical_n=int(config.logical_n),
        guard_levels=int(config.guard_levels),
        theta=theta,
        phi=phi,
    )
    return TargetCase(
        name=str(name),
        description=str(description),
        target=target,
        phase=str(phase),
        composite_with_ideal_pi=bool(composite_with_ideal_pi),
    )


def build_default_target_cases(config: SQRSpeedLimitConfig) -> dict[str, list[TargetCase]]:
    logical_n = int(config.logical_n)
    rng = np.random.default_rng(int(config.seed) + 991)

    phase1 = [
        _manual_target(
            name="phase1_selective_flip_n0",
            description="Selective pattern: only n=0 is flipped by the SQR. The tested gate is U_test = U_pi_ideal @ U_SQR.",
            theta_logical=[np.pi] + [0.0] * max(logical_n - 1, 0),
            phi_logical=[0.0] * logical_n,
            config=config,
            phase="phase1",
            composite_with_ideal_pi=True,
        )
    ]

    alt_theta = [np.pi if (n % 2 == 0) else 0.0 for n in range(logical_n)]
    alt_phi = [0.0 for _ in range(logical_n)]
    mixed_theta = [np.pi / 2.0 if (n % 2 == 0) else np.pi for n in range(logical_n)]
    mixed_phi = [0.0, np.pi / 2.0, np.pi / 3.0][:logical_n]
    cluster_theta = [np.pi / 2.0, np.pi, np.pi / 2.0][:logical_n]
    cluster_phi = [0.0, np.pi / 2.0, np.pi / 2.0][:logical_n]
    random_theta = rng.uniform(-0.92 * np.pi, 0.92 * np.pi, size=logical_n)
    random_phi = rng.uniform(0.0, 2.0 * np.pi, size=logical_n)

    phase2 = [
        _manual_target(
            name="phase2_alternating_flips",
            description="Alternating selective flips across Fock levels.",
            theta_logical=alt_theta,
            phi_logical=alt_phi,
            config=config,
            phase="phase2",
        ),
        _manual_target(
            name="phase2_mixed_angles",
            description="Mixed pi/2 and pi rotations with nontrivial XY phases.",
            theta_logical=mixed_theta,
            phi_logical=mixed_phi,
            config=config,
            phase="phase2",
        ),
        _manual_target(
            name="phase2_seeded_random",
            description="Deterministic seeded random target over the matched subspace.",
            theta_logical=random_theta,
            phi_logical=random_phi,
            config=config,
            phase="phase2",
        ),
        _manual_target(
            name="phase2_cluster_like",
            description="Cluster-relevant heuristic block pattern on n <= n_match.",
            theta_logical=cluster_theta,
            phi_logical=cluster_phi,
            config=config,
            phase="phase2",
        ),
    ]
    return {"phase1": phase1, "phase2": phase2}


def _base_calibration_config(study: SQRSpeedLimitConfig, duration_s: float, sigma_fraction: float) -> dict[str, Any]:
    return {
        "duration_sqr_s": float(duration_s),
        "sqr_sigma_fraction": float(sigma_fraction),
        "dt_s": float(study.dt_s),
        "max_step_s": float(study.max_step_s),
        "omega_q_hz": float(study.omega_q_hz),
        "omega_c_hz": float(study.omega_c_hz),
        "qubit_alpha_hz": float(study.qubit_alpha_hz),
        "st_chi_hz": float(study.chi_hz),
        "st_chi2_hz": float(study.chi2_hz),
        "st_chi3_hz": float(study.chi3_hz),
        "st_K_hz": float(study.kerr_hz),
        "st_K2_hz": float(study.kerr2_hz),
        "use_rotating_frame": bool(study.use_rotating_frame),
        "cavity_fock_cutoff": int(study.cavity_fock_cutoff),
        "n_cav_dim": int(study.total_levels),
        "max_n_cal": int(study.cavity_fock_cutoff),
        "sqr_theta_cutoff": 1.0e-10,
        "allow_zero_theta_corrections": True,
        "qb_T1_relax_ns": None if study.qb_t1_relax_ns is None else float(study.qb_t1_relax_ns),
        "qb_T2_ramsey_ns": None if study.qb_t2_ramsey_ns is None else float(study.qb_t2_ramsey_ns),
        "qb_T2_echo_ns": None if study.qb_t2_echo_ns is None else float(study.qb_t2_echo_ns),
        "t2_source": str(study.t2_source),
        "cavity_kappa_1_per_s": float(study.cavity_kappa_1_per_s),
        "optimizer_method_stage1": "Powell",
        "optimizer_method_stage2": "L-BFGS-B",
        "optimizer_maxiter_stage1": int(study.optimizer_maxiter_stage1),
        "optimizer_maxiter_stage2": int(study.optimizer_maxiter_stage2),
        "d_lambda_bounds": tuple(float(x) for x in study.d_lambda_bounds),
        "d_alpha_bounds": tuple(float(x) for x in study.d_alpha_bounds),
        "d_omega_hz_bounds": tuple(float(x) for x in study.d_omega_hz_bounds),
        "regularization_lambda": 1.0e-6,
        "regularization_alpha": 1.0e-6,
        "regularization_omega": 1.0e-18,
        "qutip_nsteps_sqr_calibration": int(study.qutip_nsteps_sqr_calibration),
    }


def _study_uses_dissipation(study: SQRSpeedLimitConfig) -> bool:
    return any(
        value not in (None, 0, 0.0)
        for value in (
            study.qb_t1_relax_ns,
            study.qb_t2_ramsey_ns,
            study.qb_t2_echo_ns,
            study.cavity_kappa_1_per_s,
        )
    )


def _benchmark_bounds(config: Mapping[str, Any], n_active: int) -> tuple[np.ndarray, np.ndarray]:
    d_lambda_bounds = tuple(config.get("d_lambda_bounds", (-0.5, 0.5)))
    d_alpha_bounds = tuple(config.get("d_alpha_bounds", (-np.pi, np.pi)))
    d_omega_hz_bounds = tuple(config.get("d_omega_hz_bounds", (-2.0e6, 2.0e6)))
    lower = np.asarray(
        [float(d_lambda_bounds[0]), float(d_alpha_bounds[0]), hz_to_rad_s(float(d_omega_hz_bounds[0]))] * int(n_active),
        dtype=float,
    )
    upper = np.asarray(
        [float(d_lambda_bounds[1]), float(d_alpha_bounds[1]), hz_to_rad_s(float(d_omega_hz_bounds[1]))] * int(n_active),
        dtype=float,
    )
    return lower, upper


def _active_levels(target: RandomSQRTarget) -> list[int]:
    return [n for n in range(int(target.logical_n))]


def _initial_vector_for_start(
    *,
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    seed: int,
    start_index: int,
) -> np.ndarray:
    active = _active_levels(target)
    if int(start_index) == 0:
        return np.zeros(3 * len(active), dtype=float)
    lower, upper = _benchmark_bounds(config, len(active))
    rng = np.random.default_rng(int(seed))
    vec = rng.uniform(lower, upper)
    vec[0::3] *= 0.35
    vec[1::3] *= 0.35
    vec[2::3] *= 0.20
    return np.asarray(vec, dtype=float)


def _correction_map(calibration: GuardedBenchmarkResult) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for n, (dl, da, dw) in enumerate(
        zip(
            calibration.calibration.d_lambda,
            calibration.calibration.d_alpha,
            calibration.calibration.d_omega_rad_s,
            strict=False,
        )
    ):
        out[int(n)] = (float(dl), float(da), float(dw))
    return out


def _logical_block_unitaries(
    *,
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    result: GuardedBenchmarkResult,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    corrections = _correction_map(result)
    actual_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for n in range(int(target.logical_n)):
        actual, _ = extract_multitone_effective_qubit_unitary(n, target, config, corrections=corrections)
        actual_blocks.append(normalize_unitary(np.asarray(actual, dtype=np.complex128)))
        target_blocks.append(
            normalize_unitary(np.asarray(qubit_rotation_xy(float(target.theta[n]), float(target.phi[n])).full(), dtype=np.complex128))
        )
    return actual_blocks, target_blocks


def _per_n_rows_from_manifolds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "n": int(row["n"]),
                "theta_target_rad": float(row["theta_target"]),
                "phi_target_rad": float(row["phi_target"]),
                "theta_achieved_rad": float(row.get("achieved_theta", float("nan"))),
                "phi_achieved_rad": float(row.get("achieved_phi", float("nan"))),
                "theta_error_rad": float(row.get("achieved_theta", float("nan")) - float(row["theta_target"])),
                "phi_error_rad": float(wrap_pi(float(row.get("achieved_phi", float("nan")) - float(row["phi_target"])))),
                "block_process_fidelity": float(row["process_fidelity"]),
            }
        )
    return out


def _blocks_to_qb_first(blocks: Sequence[np.ndarray]) -> np.ndarray:
    n_levels = int(len(blocks))
    out = np.zeros((2 * n_levels, 2 * n_levels), dtype=np.complex128)
    for n, block in enumerate(blocks):
        idx = [int(n), int(n_levels + n)]
        out[np.ix_(idx, idx)] = np.asarray(block, dtype=np.complex128)
    return out


def _ideal_pi_qb_first(n_levels: int) -> np.ndarray:
    rx = np.asarray(qubit_rotation_xy(np.pi, 0.0).full(), dtype=np.complex128)
    return np.kron(rx, np.eye(int(n_levels), dtype=np.complex128))


def _point_metrics(
    *,
    case: TargetCase,
    config: Mapping[str, Any],
    result: GuardedBenchmarkResult,
) -> dict[str, Any]:
    simulation_mode = str(result.per_manifold[0].get("simulation_mode", "unitary")) if result.per_manifold else "unitary"
    if simulation_mode == "channel":
        return {
            "subspace_fidelity": float(result.logical_fidelity),
            "logical_fidelity_weighted": float(result.logical_fidelity),
            "leakage_average": 0.0,
            "leakage_worst": 0.0,
            "guard_selectivity_error": float(result.epsilon_guard),
            "actual_unitary_qb_first": None,
            "target_unitary_qb_first": None,
            "actual_sqr_unitary_qb_first": None,
            "target_sqr_unitary_qb_first": None,
            "per_n_rows": _per_n_rows_from_manifolds(result.per_manifold[: int(case.target.logical_n)]),
            "simulation_mode": "channel",
            "execution_path": result.metadata.get("execution_path", "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.mesolve"),
        }

    actual_blocks, target_blocks = _logical_block_unitaries(target=case.target, config=config, result=result)
    actual_unitary = _blocks_to_qb_first(actual_blocks)
    target_unitary = _blocks_to_qb_first(target_blocks)
    if bool(case.composite_with_ideal_pi):
        pi_fast = _ideal_pi_qb_first(len(actual_blocks))
        actual_eval = pi_fast @ actual_unitary
        target_eval = pi_fast @ target_unitary
    else:
        actual_eval = actual_unitary
        target_eval = target_unitary

    per_n_rows: list[dict[str, Any]] = []
    for n, (actual_block, target_block) in enumerate(zip(actual_blocks, target_blocks, strict=False)):
        theta_actual, phi_actual = rotation_axis_parameters(actual_block)
        theta_target, phi_target = rotation_axis_parameters(target_block)
        tr = np.trace(target_block.conj().T @ actual_block)
        block_fidelity = float(np.abs(tr) ** 2 / 4.0)
        per_n_rows.append(
            {
                "n": int(n),
                "theta_target_rad": float(theta_target),
                "phi_target_rad": float(phi_target),
                "theta_achieved_rad": float(theta_actual),
                "phi_achieved_rad": float(phi_actual),
                "theta_error_rad": float(theta_actual - theta_target),
                "phi_error_rad": float(wrap_pi(phi_actual - phi_target)),
                "block_process_fidelity": float(block_fidelity),
            }
        )

    return {
        "subspace_fidelity": float(subspace_unitary_fidelity(actual_eval, target_eval, gauge="global")),
        "logical_fidelity_weighted": float(result.logical_fidelity),
        "leakage_average": 0.0,
        "leakage_worst": 0.0,
        "guard_selectivity_error": float(result.epsilon_guard),
        "actual_unitary_qb_first": actual_eval,
        "target_unitary_qb_first": target_eval,
        "actual_sqr_unitary_qb_first": actual_unitary,
        "target_sqr_unitary_qb_first": target_unitary,
        "per_n_rows": per_n_rows,
        "simulation_mode": "unitary",
        "execution_path": result.metadata.get("execution_path", "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator"),
    }


def evaluate_nominal_case(
    *,
    case: TargetCase,
    study: SQRSpeedLimitConfig,
    duration_s: float,
    sigma_fraction: float,
) -> dict[str, Any]:
    calib_cfg = _base_calibration_config(study, duration_s=float(duration_s), sigma_fraction=float(sigma_fraction))
    if _study_uses_dissipation(study):
        nominal_eval = evaluate_guarded_sqr_target(
            target=case.target,
            config=calib_cfg,
            corrections={},
            lambda_guard=float(study.lambda_guard),
            weight_mode=str(study.weight_mode),
        )
        metrics = {
            "subspace_fidelity": float(nominal_eval["logical_fidelity"]),
            "leakage_average": 0.0,
            "leakage_worst": 0.0,
            "guard_selectivity_error": float(nominal_eval["epsilon_guard"]),
            "actual_unitary_qb_first": None,
            "target_unitary_qb_first": None,
            "actual_sqr_unitary_qb_first": None,
            "target_sqr_unitary_qb_first": None,
            "per_n_rows": _per_n_rows_from_manifolds(nominal_eval["per_manifold"][: int(case.target.logical_n)]),
            "simulation_mode": str(nominal_eval.get("simulation_mode", "channel")),
            "execution_path": str(nominal_eval.get("execution_path", "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.mesolve")),
        }
        return {
            "calibration_config": calib_cfg,
            "subspace_fidelity": float(metrics["subspace_fidelity"]),
            "leakage_average": float(metrics["leakage_average"]),
            "leakage_worst": float(metrics["leakage_worst"]),
            "guard_selectivity_error": float(metrics["guard_selectivity_error"]),
            "actual_unitary_qb_first": metrics["actual_unitary_qb_first"],
            "target_unitary_qb_first": metrics["target_unitary_qb_first"],
            "actual_sqr_unitary_qb_first": metrics["actual_sqr_unitary_qb_first"],
            "target_sqr_unitary_qb_first": metrics["target_sqr_unitary_qb_first"],
            "waveform": _waveform_from_corrections(
                target=case.target,
                config=calib_cfg,
                corrections={},
                dt_s=float(study.dt_s),
            ),
            "per_n_rows": metrics["per_n_rows"],
            "simulation_mode": str(metrics["simulation_mode"]),
            "execution_path": str(metrics["execution_path"]),
        }

    actual_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for n in range(int(case.target.logical_n)):
        actual, _ = extract_multitone_effective_qubit_unitary(n, case.target, calib_cfg, corrections={})
        actual_blocks.append(normalize_unitary(np.asarray(actual, dtype=np.complex128)))
        target_blocks.append(
            normalize_unitary(np.asarray(qubit_rotation_xy(float(case.target.theta[n]), float(case.target.phi[n])).full(), dtype=np.complex128))
        )
    actual_unitary = _blocks_to_qb_first(actual_blocks)
    target_unitary = _blocks_to_qb_first(target_blocks)
    if bool(case.composite_with_ideal_pi):
        pi_fast = _ideal_pi_qb_first(len(actual_blocks))
        actual_eval = pi_fast @ actual_unitary
        target_eval = pi_fast @ target_unitary
    else:
        actual_eval = actual_unitary
        target_eval = target_unitary
    per_n_rows: list[dict[str, Any]] = []
    for n, (actual_block, target_block) in enumerate(zip(actual_blocks, target_blocks, strict=False)):
        theta_actual, phi_actual = rotation_axis_parameters(actual_block)
        theta_target, phi_target = rotation_axis_parameters(target_block)
        tr = np.trace(target_block.conj().T @ actual_block)
        per_n_rows.append(
            {
                "n": int(n),
                "theta_target_rad": float(theta_target),
                "phi_target_rad": float(phi_target),
                "theta_achieved_rad": float(theta_actual),
                "phi_achieved_rad": float(phi_actual),
                "theta_error_rad": float(theta_actual - theta_target),
                "phi_error_rad": float(wrap_pi(phi_actual - phi_target)),
                "block_process_fidelity": float(np.abs(tr) ** 2 / 4.0),
            }
        )
    return {
        "calibration_config": calib_cfg,
        "subspace_fidelity": float(subspace_unitary_fidelity(actual_eval, target_eval, gauge="global")),
        "leakage_average": 0.0,
        "leakage_worst": 0.0,
        "guard_selectivity_error": 0.0,
        "actual_unitary_qb_first": actual_eval,
        "target_unitary_qb_first": target_eval,
        "actual_sqr_unitary_qb_first": actual_unitary,
        "target_sqr_unitary_qb_first": target_unitary,
        "waveform": _waveform_from_corrections(
            target=case.target,
            config=calib_cfg,
            corrections={},
            dt_s=float(study.dt_s),
        ),
        "per_n_rows": per_n_rows,
        "simulation_mode": "unitary",
        "execution_path": "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator",
    }


def _run_sweep_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    case = payload["case"]
    study = payload["study"]
    duration_s = float(payload["duration_s"])
    sigma_fraction = float(payload["sigma_fraction"])
    sigma_index = int(payload["sigma_index"])
    start_index = int(payload["start_index"])
    calib_cfg = _base_calibration_config(study, duration_s=duration_s, sigma_fraction=sigma_fraction)
    run_seed = int(study.seed + 100000 * (sigma_index + 1) + 997 * start_index + int(round(duration_s * 1.0e9)))
    init = _initial_vector_for_start(target=case.target, config=calib_cfg, seed=run_seed, start_index=start_index)
    history = HistoryReporter()
    run_id = f"{case.name}_T{int(round(duration_s * 1.0e9)):04d}ns_sigma{str(sigma_fraction).replace('.', 'p')}_start{start_index:02d}"
    result = calibrate_guarded_sqr_target(
        target=case.target,
        config=calib_cfg,
        lambda_guard=float(study.lambda_guard),
        weight_mode=str(study.weight_mode),
        initial_vector=init,
        reporter=history,
        progress_every=int(study.progress_every),
        run_id=run_id,
        backend_label="pulse",
    )
    metrics = _point_metrics(case=case, config=calib_cfg, result=result)
    return {
        "run_id": run_id,
        "seed": int(run_seed),
        "sigma_fraction": float(sigma_fraction),
        "start_index": int(start_index),
        "calibration_config": calib_cfg,
        "result": result,
        "metrics": metrics,
        "history": list(history.events),
    }


def _waveform_from_corrections(
    *,
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    corrections: Mapping[int, tuple[float, float, float]] | None = None,
    dt_s: float | None = None,
) -> dict[str, Any]:
    correction_map = {} if corrections is None else {int(k): tuple(float(v) for v in values) for k, values in corrections.items()}
    duration_s = float(config["duration_sqr_s"])
    sigma_fraction = float(config["sqr_sigma_fraction"])
    dt = float(config["dt_s"] if dt_s is None else dt_s)
    n_steps = max(2, int(math.ceil(duration_s / max(dt, 1.0e-18))) + 1)
    tlist = np.linspace(0.0, duration_s, n_steps, dtype=float)
    env = np.asarray(gaussian_envelope(tlist / max(duration_s, 1.0e-18), sigma=sigma_fraction), dtype=np.complex128)
    coeff = np.zeros_like(tlist, dtype=np.complex128)
    tones: list[dict[str, Any]] = []
    for n in range(int(target.total_levels)):
        theta = float(target.theta[n])
        phi = float(target.phi[n])
        base_amp = _initial_drive_amplitude(theta, duration_s, sigma_fraction, tlist)
        d_lambda, d_alpha, d_omega = correction_map.get(int(n), (0.0, 0.0, 0.0))
        amp = float(base_amp + d_lambda * (np.pi / (2.0 * duration_s)))
        omega = float(_conditional_detuning_rad_s(int(n), config) + d_omega)
        phase = float(phi + d_alpha)
        tone = amp * env * np.exp(1j * phase) * np.exp(1j * omega * tlist)
        coeff += tone
        if abs(amp) > 1.0e-12 or abs(phase) > 1.0e-12 or abs(omega) > 1.0e-12:
            tones.append(
                {
                    "n": int(n),
                    "amp_rad_s": float(amp),
                    "phase_rad": float(phase),
                    "omega_rad_s": float(omega),
                    "omega_hz": float(rad_s_to_hz(omega)),
                }
            )
    return {
        "tlist_s": tlist,
        "coeff": coeff,
        "i": np.real(coeff),
        "q": np.imag(coeff),
        "magnitude": np.abs(coeff),
        "phase": np.angle(coeff),
        "tones": tones,
        "duration_s": duration_s,
        "sigma_fraction": sigma_fraction,
    }


def _waveform_from_result(
    *,
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    result: GuardedBenchmarkResult,
    dt_s: float | None = None,
) -> dict[str, Any]:
    return _waveform_from_corrections(
        target=target,
        config=config,
        corrections=_correction_map(result),
        dt_s=dt_s,
    )


def _plot_waveform(waveform: Mapping[str, Any], output_path: Path, title: str) -> Path:
    t_ns = np.asarray(waveform["tlist_s"], dtype=float) * 1.0e9
    coeff = np.asarray(waveform["coeff"], dtype=np.complex128)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.8), sharex=True)
    axes[0].plot(t_ns, coeff.real, color="tab:blue", linewidth=1.6)
    axes[0].set_ylabel("I(t) [rad/s]")
    axes[1].plot(t_ns, coeff.imag, color="tab:orange", linewidth=1.6)
    axes[1].set_ylabel("Q(t) [rad/s]")
    axes[2].plot(t_ns, np.abs(coeff), color="tab:green", linewidth=1.6)
    axes[2].set_ylabel("|Omega(t)|")
    axes[2].set_xlabel("Time [ns]")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _plot_spectrum(waveform: Mapping[str, Any], output_path: Path, title: str) -> Path:
    coeff = np.asarray(waveform["coeff"], dtype=np.complex128)
    tlist = np.asarray(waveform["tlist_s"], dtype=float)
    if coeff.size < 2:
        return output_path
    dt = float(tlist[1] - tlist[0])
    freq_hz = np.fft.fftshift(np.fft.fftfreq(coeff.size, d=dt))
    spec = np.fft.fftshift(np.fft.fft(coeff))
    norm = max(float(np.max(np.abs(spec))), 1.0e-18)
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(freq_hz / 1.0e6, 20.0 * np.log10(np.abs(spec) / norm + 1.0e-15), color="tab:purple")
    for tone in waveform.get("tones", []):
        f_mhz = float(tone["omega_hz"]) / 1.0e6
        ax.axvline(f_mhz, color="0.55", linestyle="--", linewidth=0.8)
        ax.text(f_mhz, -7.5, f"n={tone['n']}", rotation=90, va="bottom", ha="right", fontsize=7)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Normalized spectrum [dB]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _plot_per_n_rotations(per_n_rows: Sequence[Mapping[str, Any]], output_path: Path, title: str) -> Path:
    n_values = np.asarray([int(row["n"]) for row in per_n_rows], dtype=int)
    theta_target = np.asarray([float(row["theta_target_rad"]) for row in per_n_rows], dtype=float)
    theta_actual = np.asarray([float(row["theta_achieved_rad"]) for row in per_n_rows], dtype=float)
    phi_target = np.asarray([float(row["phi_target_rad"]) for row in per_n_rows], dtype=float)
    phi_actual = np.asarray([float(row["phi_achieved_rad"]) for row in per_n_rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(n_values, theta_target, "o-", color="black", label="target")
    axes[0].plot(n_values, theta_actual, "s--", color="tab:blue", label="achieved")
    axes[0].set_xlabel("Fock n")
    axes[0].set_ylabel("theta [rad]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(n_values, phi_target, "o-", color="black", label="target")
    axes[1].plot(n_values, phi_actual, "s--", color="tab:orange", label="achieved")
    axes[1].set_xlabel("Fock n")
    axes[1].set_ylabel("phi [rad]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _plot_case_curves(case_name: str, rows: Sequence[Mapping[str, Any]], output_path: Path) -> Path:
    duration_ns = np.asarray([float(row["duration_s"]) * 1.0e9 for row in rows], dtype=float)
    fidelity = np.asarray([float(row["subspace_fidelity"]) for row in rows], dtype=float)
    leakage = np.asarray([float(row["leakage_worst"]) for row in rows], dtype=float)
    guard = np.asarray([float(row["guard_selectivity_error"]) for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(duration_ns, fidelity, "o-", color="tab:blue")
    axes[0].set_xlabel("T_SQR [ns]")
    axes[0].set_ylabel("Subspace fidelity")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].set_title(f"{case_name}: fidelity")
    axes[1].plot(duration_ns, leakage, "o-", color="tab:red", label="leakage")
    axes[1].plot(duration_ns, guard, "s--", color="tab:green", label="guard selectivity")
    axes[1].set_xlabel("T_SQR [ns]")
    axes[1].set_ylabel("Error metric")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[1].set_title(f"{case_name}: leakage / guard proxy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _threshold_summary(rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: float(row["duration_s"]))
    for threshold in thresholds:
        reached = next((row for row in ordered if float(row["subspace_fidelity"]) >= float(threshold)), None)
        out.append(
            {
                "threshold": float(threshold),
                "min_duration_s": None if reached is None else float(reached["duration_s"]),
                "min_duration_ns": None if reached is None else float(reached["duration_s"]) * 1.0e9,
            }
        )
    return out


def run_speedlimit_sweep_point(
    *,
    case: TargetCase,
    duration_s: float,
    study: SQRSpeedLimitConfig,
    sigma_fractions: Sequence[float] | None = None,
    multistart: int | None = None,
    reporter: ProgressReporter | None = None,
    point_output_dir: Path | None = None,
) -> dict[str, Any]:
    sigma_values = _coerce_numeric_tuple(study.sigma_fractions if sigma_fractions is None else sigma_fractions, float)
    n_starts = int(study.multistart if multistart is None else multistart)
    combined_history: list[dict[str, Any]] = []
    history_by_run: dict[str, list[dict[str, Any]]] = {}
    start_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_score = float("inf")
    candidate_payloads = [
        {
            "case": case,
            "study": study,
            "duration_s": float(duration_s),
            "sigma_fraction": float(sigma_fraction),
            "sigma_index": int(sigma_index),
            "start_index": int(start_index),
        }
        for sigma_index, sigma_fraction in enumerate(sigma_values)
        for start_index in range(max(1, n_starts))
    ]

    can_parallel = bool(study.parallel_enabled) and int(study.parallel_n_jobs) > 1 and len(candidate_payloads) > 1 and reporter is None
    if can_parallel:
        with ProcessPoolExecutor(max_workers=int(study.parallel_n_jobs)) as executor:
            candidate_results = list(executor.map(_run_sweep_candidate, candidate_payloads))
    else:
        candidate_results = []
        live_reporter = reporter or NullReporter()
        for payload in candidate_payloads:
            calib_cfg = _base_calibration_config(study, duration_s=float(payload["duration_s"]), sigma_fraction=float(payload["sigma_fraction"]))
            run_seed = int(study.seed + 100000 * (int(payload["sigma_index"]) + 1) + 997 * int(payload["start_index"]) + int(round(float(payload["duration_s"]) * 1.0e9)))
            init = _initial_vector_for_start(target=case.target, config=calib_cfg, seed=run_seed, start_index=int(payload["start_index"]))
            history = HistoryReporter()
            run_id = f"{case.name}_T{int(round(float(payload['duration_s']) * 1.0e9)):04d}ns_sigma{str(payload['sigma_fraction']).replace('.', 'p')}_start{int(payload['start_index']):02d}"
            run_reporter = CompositeReporter([history, live_reporter])
            result = calibrate_guarded_sqr_target(
                target=case.target,
                config=calib_cfg,
                lambda_guard=float(study.lambda_guard),
                weight_mode=str(study.weight_mode),
                initial_vector=init,
                reporter=run_reporter,
                progress_every=int(study.progress_every),
                run_id=run_id,
                backend_label="pulse",
            )
            candidate_results.append(
                {
                    "run_id": run_id,
                    "seed": int(run_seed),
                    "sigma_fraction": float(payload["sigma_fraction"]),
                    "start_index": int(payload["start_index"]),
                    "calibration_config": calib_cfg,
                    "result": result,
                    "metrics": _point_metrics(case=case, config=calib_cfg, result=result),
                    "history": list(history.events),
                }
            )

    for worker_result in candidate_results:
        score = float(1.0 - worker_result["metrics"]["subspace_fidelity"])
        history_by_run[worker_result["run_id"]] = list(worker_result["history"])
        combined_history.extend(worker_result["history"])
        start_row = {
            "run_id": worker_result["run_id"],
            "seed": int(worker_result["seed"]),
            "sigma_fraction": float(worker_result["sigma_fraction"]),
            "start_index": int(worker_result["start_index"]),
            "subspace_fidelity": float(worker_result["metrics"]["subspace_fidelity"]),
            "logical_fidelity_weighted": float(worker_result["metrics"]["logical_fidelity_weighted"]),
            "guard_selectivity_error": float(worker_result["metrics"]["guard_selectivity_error"]),
            "objective_loss_total": float(worker_result["result"].loss_total),
        }
        start_rows.append(start_row)
        if score < best_score:
            best_score = score
            best_payload = worker_result

    if best_payload is None:
        raise RuntimeError("No sweep candidates were evaluated.")

    waveform = _waveform_from_result(
        target=case.target,
        config=best_payload["calibration_config"],
        result=best_payload["result"],
        dt_s=float(study.dt_s),
    )
    point_summary = {
        "case_name": case.name,
        "description": case.description,
        "phase": case.phase,
        "duration_s": float(duration_s),
        "duration_ns": float(duration_s) * 1.0e9,
        "sigma_fraction": float(best_payload["sigma_fraction"]),
        "selected_run_id": best_payload["run_id"],
        "selected_seed": int(best_payload["seed"]),
        "selected_start_index": int(best_payload["start_index"]),
        "subspace_fidelity": float(best_payload["metrics"]["subspace_fidelity"]),
        "logical_fidelity_weighted": float(best_payload["metrics"]["logical_fidelity_weighted"]),
        "leakage_average": float(best_payload["metrics"]["leakage_average"]),
        "leakage_worst": float(best_payload["metrics"]["leakage_worst"]),
        "guard_selectivity_error": float(best_payload["metrics"]["guard_selectivity_error"]),
        "objective_loss_total": float(best_payload["result"].loss_total),
        "n_active_runs": int(len(start_rows)),
        "calibration_summary": best_payload["result"].calibration.to_dict(),
        "per_manifold": list(best_payload["result"].per_manifold),
        "per_n_rows": list(best_payload["metrics"]["per_n_rows"]),
        "history": list(best_payload["history"]),
        "history_by_run": history_by_run,
        "all_start_rows": start_rows,
        "actual_unitary_qb_first": best_payload["metrics"]["actual_unitary_qb_first"],
        "target_unitary_qb_first": best_payload["metrics"]["target_unitary_qb_first"],
        "actual_sqr_unitary_qb_first": best_payload["metrics"]["actual_sqr_unitary_qb_first"],
        "target_sqr_unitary_qb_first": best_payload["metrics"]["target_sqr_unitary_qb_first"],
        "simulation_mode": best_payload["metrics"].get("simulation_mode", "unitary"),
        "execution_path": best_payload["metrics"].get("execution_path", "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator"),
        "parallel_used": bool(can_parallel),
        "parallel_n_jobs": int(study.parallel_n_jobs if can_parallel else 1),
        "waveform": waveform,
    }

    if point_output_dir is not None:
        point_output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(point_output_dir / "summary.json", point_summary)
        save_history_json(combined_history, point_output_dir / "history.json")
        save_history_csv(combined_history, point_output_dir / "history.csv")

    return point_summary


def _render_markdown_report(study: SQRSpeedLimitConfig, summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# SQR Speed-Limit Report")
    lines.append("")
    lines.append("## Study Definition")
    lines.append(f"- Seed: `{int(study.seed)}`")
    lines.append(f"- Matched subspace: `n = 0..{int(study.n_match)}` with `logical_n = {int(study.logical_n)}`")
    lines.append(f"- Guard levels: `{int(study.guard_levels)}`")
    lines.append(f"- Dispersive parameter: `chi = {float(study.chi_hz) / 1.0e6:.6f} MHz`")
    lines.append(
        "- Phase-1 ordering convention: "
        + "`U_test = U_pi_ideal @ U_SQR`, meaning the selective SQR acts first on the state and the ideal fast pi is applied after it."
    )
    lines.append(
        "- Leakage note: in the minimal dispersive qubit-drive-only model used for the main sweep, photon number is conserved, so subspace leakage for `n<=n_match` probe states is identically zero. "
        + "The reported guard metric is therefore a selectivity proxy on out-of-support manifolds, not literal dynamical leakage."
    )
    lines.append("")
    lines.append("## Phase 1")
    for case_payload in summary["phase1"]:
        lines.append(f"### {case_payload['case_name']}")
        lines.append(f"- Description: {case_payload['description']}")
        for row in case_payload["thresholds"]:
            if row["min_duration_ns"] is None:
                lines.append(f"  - `F >= {row['threshold']:.4f}`: not reached")
            else:
                lines.append(f"  - `F >= {row['threshold']:.4f}`: `{row['min_duration_ns']:.1f} ns`")
        best = max(case_payload["sweep"], key=lambda row: float(row["subspace_fidelity"]))
        lines.append(
            f"- Best point: `T = {best['duration_ns']:.1f} ns`, "
            + f"`sigma = {best['sigma_fraction']:.2f}`, "
            + f"`F_subspace = {best['subspace_fidelity']:.6f}`, "
            + f"`guard = {best['guard_selectivity_error']:.3e}`"
        )
        lines.append("")
    lines.append("## Phase 2")
    for case_payload in summary["phase2"]:
        lines.append(f"### {case_payload['case_name']}")
        lines.append(f"- Description: {case_payload['description']}")
        for row in case_payload["thresholds"]:
            if row["min_duration_ns"] is None:
                lines.append(f"  - `F >= {row['threshold']:.4f}`: not reached")
            else:
                lines.append(f"  - `F >= {row['threshold']:.4f}`: `{row['min_duration_ns']:.1f} ns`")
        lines.append("")
    lines.append("## Configuration Snapshot")
    lines.append("```json")
    lines.append(json.dumps(_jsonify(asdict(study)), indent=2))
    lines.append("```")
    return "\n".join(lines)


def run_speedlimit_study(
    study: SQRSpeedLimitConfig | None = None,
    *,
    output_dir: Path | None = None,
    reporter: ProgressReporter | None = None,
    phase1_cases: Sequence[TargetCase] | None = None,
    phase2_cases: Sequence[TargetCase] | None = None,
    durations_ns: Sequence[int] | None = None,
) -> dict[str, Any]:
    study = SQRSpeedLimitConfig() if study is None else study
    out_dir = Path(study.output_dir() if output_dir is None else output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_default_target_cases(study)
    phase1_cases = list(cases["phase1"] if phase1_cases is None else phase1_cases)
    phase2_cases = list(cases["phase2"] if phase2_cases is None else phase2_cases)
    durations_ns = tuple(int(x) for x in (study.durations_ns if durations_ns is None else durations_ns))

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "phase1": [],
        "phase2": [],
    }
    progress_reporter = reporter or NullReporter()

    for phase_key, phase_cases in (("phase1", phase1_cases), ("phase2", phase2_cases)):
        for case in phase_cases:
            case_dir = out_dir / phase_key / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            sweep_rows: list[dict[str, Any]] = []
            point_payloads: dict[int, dict[str, Any]] = {}
            for duration_ns in durations_ns:
                point_dir = case_dir / f"T_{int(duration_ns):04d}ns"
                point = run_speedlimit_sweep_point(
                    case=case,
                    duration_s=float(duration_ns) * 1.0e-9,
                    study=study,
                    reporter=progress_reporter,
                    point_output_dir=point_dir,
                )
                point_payloads[int(duration_ns)] = point
                sweep_rows.append(
                    {
                        "duration_s": float(point["duration_s"]),
                        "duration_ns": float(point["duration_ns"]),
                        "sigma_fraction": float(point["sigma_fraction"]),
                        "subspace_fidelity": float(point["subspace_fidelity"]),
                        "logical_fidelity_weighted": float(point["logical_fidelity_weighted"]),
                        "leakage_average": float(point["leakage_average"]),
                        "leakage_worst": float(point["leakage_worst"]),
                        "guard_selectivity_error": float(point["guard_selectivity_error"]),
                        "selected_run_id": point["selected_run_id"],
                        "summary_path": str(point_dir / "summary.json"),
                    }
                )
            thresholds = _threshold_summary(sweep_rows, study.fidelity_thresholds)
            best_point = max(sweep_rows, key=lambda row: float(row["subspace_fidelity"]))
            representative_ns = min(durations_ns, key=lambda value: abs(int(value) - int(study.representative_duration_ns)))
            selected_duration = int(best_point["duration_ns"] if best_point["subspace_fidelity"] >= 0.99 else representative_ns)
            selected_payload = point_payloads[int(selected_duration)]
            _plot_case_curves(case.name, sweep_rows, case_dir / "fidelity_leakage.png")
            _plot_waveform(selected_payload["waveform"], case_dir / "selected_waveform.png", title=f"{case.name}: waveform")
            _plot_spectrum(selected_payload["waveform"], case_dir / "selected_spectrum.png", title=f"{case.name}: spectrum")
            _plot_per_n_rotations(selected_payload["per_n_rows"], case_dir / "selected_per_n.png", title=f"{case.name}: per-n blocks")
            case_payload = {
                "case_name": case.name,
                "description": case.description,
                "phase": phase_key,
                "sweep": sweep_rows,
                "thresholds": thresholds,
                "selected_visualization_duration_ns": float(selected_duration),
                "curve_path": str(case_dir / "fidelity_leakage.png"),
                "waveform_path": str(case_dir / "selected_waveform.png"),
                "spectrum_path": str(case_dir / "selected_spectrum.png"),
                "per_n_path": str(case_dir / "selected_per_n.png"),
            }
            summary[phase_key].append(case_payload)
            _write_json(case_dir / "case_summary.json", case_payload)

    report_text = _render_markdown_report(study, summary)
    study.report_path.parent.mkdir(parents=True, exist_ok=True)
    study.report_path.write_text(report_text, encoding="utf-8")
    summary["report_path"] = str(study.report_path)
    _write_json(out_dir / "study_summary.json", summary)
    return summary


def _build_notebook_live_reporter() -> JupyterLiveReporter:
    return JupyterLiveReporter(what="objective_total", fidelity_what="metrics.fidelity_subspace", print_every=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SQR multitone Gaussian speed-limit study.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic random seed.")
    parser.add_argument("--n-match", type=int, default=2, help="Maximum matched Fock level.")
    parser.add_argument("--multistart", type=int, default=2, help="Multistart count per sigma point.")
    parser.add_argument(
        "--durations-ns",
        type=str,
        default="50,75,100,150,200,300,500,750,1000",
        help="Comma-separated SQR durations in ns.",
    )
    parser.add_argument(
        "--sigma-fractions",
        type=str,
        default="0.15,0.2,0.25,0.3",
        help="Comma-separated Gaussian sigma/T values.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a shorter smoke sweep for quick validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    durations = tuple(int(x.strip()) for x in str(args.durations_ns).split(",") if x.strip())
    sigma_fractions = tuple(float(x.strip()) for x in str(args.sigma_fractions).split(",") if x.strip())
    study = SQRSpeedLimitConfig(
        seed=int(args.seed),
        n_match=int(args.n_match),
        multistart=int(args.multistart),
        durations_ns=(50, 100, 200) if bool(args.smoke) else durations,
        sigma_fractions=(0.15, 0.25) if bool(args.smoke) else sigma_fractions,
    )
    summary = run_speedlimit_study(study, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "report_path": summary["report_path"],
                "phase1_cases": [row["case_name"] for row in summary["phase1"]],
                "phase2_cases": [row["case_name"] for row in summary["phase2"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()