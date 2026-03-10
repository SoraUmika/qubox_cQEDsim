from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import qutip as qt
from scipy.optimize import Bounds, minimize

from cqed_sim.core.conventions import qubit_cavity_block_indices
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.ideal_gates import qubit_rotation_xy
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import Gate, SQRGate
from cqed_sim.pulses.calibration import build_sqr_tone_specs, pad_sqr_angles, sqr_lambda0_rad_s
from cqed_sim.pulses.envelopes import MultitoneTone, gaussian_envelope, multitone_gaussian_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.noise import NoiseSpec, collapse_operators
from cqed_sim.sim.runner import hamiltonian_time_slices
from cqed_sim.unitary_synthesis.progress import NullReporter, ProgressEvent, ProgressReporter


def default_calibration_config_keys() -> tuple[str, ...]:
    return (
        "duration_sqr_s",
        "sqr_sigma_fraction",
        "dt_s",
        "max_step_s",
        "omega_q_hz",
        "omega_c_hz",
        "qubit_alpha_hz",
        "st_chi_hz",
        "st_chi2_hz",
        "st_chi3_hz",
        "st_K_hz",
        "st_K2_hz",
        "fock_fqs_hz",
        "use_rotating_frame",
        "max_n_cal",
        "sqr_theta_cutoff",
        "allow_zero_theta_corrections",
        "qb_T1_relax_ns",
        "qb_T2_ramsey_ns",
        "qb_T2_echo_ns",
        "t2_source",
        "cavity_kappa_1_per_s",
        "optimizer_method_stage1",
        "optimizer_method_stage2",
        "d_lambda_bounds",
        "d_alpha_bounds",
        "d_omega_hz_bounds",
        "regularization_lambda",
        "regularization_alpha",
        "regularization_omega",
    )


@dataclass
class SQRLevelCalibration:
    n: int
    theta_target: float
    phi_target: float
    skipped: bool
    initial_params: tuple[float, float, float]
    optimized_params: tuple[float, float, float]
    initial_loss: float
    optimized_loss: float
    process_fidelity: float
    success_stage1: bool = False
    success_stage2: bool = False
    message_stage1: str = ""
    message_stage2: str = ""


@dataclass
class SQRCalibrationResult:
    sqr_name: str
    max_n: int
    d_lambda: list[float]
    d_alpha: list[float]
    d_omega_rad_s: list[float]
    theta_target: list[float]
    phi_target: list[float]
    initial_loss: list[float]
    optimized_loss: list[float]
    levels: list[SQRLevelCalibration] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def d_omega_hz(self) -> list[float]:
        return [float(value / (2.0 * np.pi)) for value in self.d_omega_rad_s]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sqr_name": self.sqr_name,
            "max_n": int(self.max_n),
            "d_lambda": [float(x) for x in self.d_lambda],
            "d_alpha": [float(x) for x in self.d_alpha],
            "d_omega_rad_s": [float(x) for x in self.d_omega_rad_s],
            "d_omega_hz": self.d_omega_hz,
            "theta_target": [float(x) for x in self.theta_target],
            "phi_target": [float(x) for x in self.phi_target],
            "initial_loss": [float(x) for x in self.initial_loss],
            "optimized_loss": [float(x) for x in self.optimized_loss],
            "levels": [asdict(level) for level in self.levels],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SQRCalibrationResult":
        levels = [SQRLevelCalibration(**level) for level in payload.get("levels", [])]
        d_omega_rad_s = payload.get("d_omega_rad_s")
        if d_omega_rad_s is None:
            d_omega_hz = payload.get("d_omega_hz", [])
            d_omega_rad_s = [2.0 * np.pi * float(value) for value in d_omega_hz]
        return cls(
            sqr_name=str(payload["sqr_name"]),
            max_n=int(payload["max_n"]),
            d_lambda=[float(x) for x in payload.get("d_lambda", [])],
            d_alpha=[float(x) for x in payload.get("d_alpha", [])],
            d_omega_rad_s=[float(x) for x in d_omega_rad_s],
            theta_target=[float(x) for x in payload.get("theta_target", [])],
            phi_target=[float(x) for x in payload.get("phi_target", [])],
            initial_loss=[float(x) for x in payload.get("initial_loss", [])],
            optimized_loss=[float(x) for x in payload.get("optimized_loss", [])],
            levels=levels,
            metadata=dict(payload.get("metadata", {})),
        )

    def correction_for_n(self, n: int) -> tuple[float, float, float]:
        if n < len(self.d_lambda):
            return self.d_lambda[n], self.d_alpha[n], self.d_omega_rad_s[n]
        return 0.0, 0.0, 0.0

    def improvement_summary(self) -> dict[str, Any]:
        improved = [level.n for level in self.levels if level.optimized_loss < level.initial_loss]
        mean_initial = float(np.mean(self.initial_loss)) if self.initial_loss else float("nan")
        mean_optimized = float(np.mean(self.optimized_loss)) if self.optimized_loss else float("nan")
        return {
            "sqr_name": self.sqr_name,
            "max_n": self.max_n,
            "improved_levels": improved,
            "mean_initial_loss": mean_initial,
            "mean_optimized_loss": mean_optimized,
            "mean_loss_reduction": float(mean_initial - mean_optimized),
        }


@dataclass(frozen=True)
class RandomSQRTarget:
    target_id: str
    target_class: str
    logical_n: int
    guard_levels: int
    theta: tuple[float, ...]
    phi: tuple[float, ...]

    @property
    def total_levels(self) -> int:
        return int(self.logical_n + self.guard_levels)

    def as_gate(self) -> SQRGate:
        return SQRGate(index=0, name=self.target_id, theta=self.theta, phi=self.phi)


@dataclass
class GuardedBenchmarkResult:
    target_id: str
    target_class: str
    duration_s: float
    logical_n: int
    guard_levels: int
    lambda_guard: float
    weight_mode: str
    poisson_alpha: complex | None
    logical_fidelity: float
    epsilon_guard: float
    loss_total: float
    success: bool
    converged: bool
    iterations: int
    objective_evaluations: int
    calibration: SQRCalibrationResult
    per_manifold: list[dict[str, Any]]
    convergence_trace: list[dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "T": float(self.duration_s),
            "target_id": self.target_id,
            "target_class": self.target_class,
            "F_logical": float(self.logical_fidelity),
            "epsilon_guard": float(self.epsilon_guard),
            "L": float(self.loss_total),
            "iters": int(self.iterations),
            "objective_evaluations": int(self.objective_evaluations),
            "converged": bool(self.converged),
            "success": bool(self.success),
            "lambda_guard": float(self.lambda_guard),
            "weight_mode": self.weight_mode,
            "simulation_mode": str(self.metadata.get("simulation_mode", "unitary")),
            "execution_path": str(
                self.metadata.get(
                    "execution_path",
                    "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator",
                )
            ),
        }


def extract_sqr_gates(gates: list[Gate]) -> list[SQRGate]:
    return [gate for gate in gates if isinstance(gate, SQRGate)]


def select_sqr_gate(sqr_gates: list[SQRGate], index: int = 0, name: str | None = None) -> SQRGate:
    if not sqr_gates:
        raise ValueError("No SQR gates available for selection.")
    if name is not None:
        for gate in sqr_gates:
            if gate.name == name:
                return gate
        raise KeyError(f"SQR gate named '{name}' was not found.")
    if index < 0 or index >= len(sqr_gates):
        raise IndexError(f"SQR gate index {index} is out of range for {len(sqr_gates)} gates.")
    return sqr_gates[index]


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe or "unnamed_sqr"


def _relevant_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in default_calibration_config_keys()}


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(_relevant_config_snapshot(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def calibration_cache_path(gate: SQRGate, config: Mapping[str, Any], cache_dir: str | Path | None = None) -> Path:
    base = Path("." if cache_dir is None else cache_dir)
    return base / f"sqr_{_safe_name(gate.name)}.json"


def _build_time_grid(duration_s: float, dt_s: float) -> np.ndarray:
    n_steps = max(2, int(math.ceil(duration_s / dt_s)) + 1)
    return np.linspace(0.0, duration_s, n_steps, dtype=float)


def _gaussian_area(duration_s: float, sigma_fraction: float, tlist: np.ndarray | None = None) -> float:
    grid = _build_time_grid(duration_s, duration_s / 4096.0) if tlist is None else np.asarray(tlist, dtype=float)
    t_rel = grid / duration_s
    env = np.asarray(gaussian_envelope(t_rel, sigma=sigma_fraction), dtype=np.complex128)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(np.real(env), grid))


def _conditional_detuning_rad_s(n: int, config: Mapping[str, Any]) -> float:
    detuning_hz = (
        float(config.get("st_chi_hz", 0.0)) * n
        + float(config.get("st_chi2_hz", 0.0)) * (n * (n - 1))
        + float(config.get("st_chi3_hz", 0.0)) * (n * (n - 1) * (n - 2))
    )
    return float(2.0 * np.pi * detuning_hz)


def _initial_drive_amplitude(theta_target: float, duration_s: float, sigma_fraction: float, tlist: np.ndarray) -> float:
    area = _gaussian_area(duration_s, sigma_fraction, tlist=tlist)
    if abs(area) < 1.0e-15:
        return 0.0
    return float(theta_target / area)


def _drive_components(
    theta_target: float,
    phi_target: float,
    duration_s: float,
    sigma_fraction: float,
    tlist: np.ndarray,
    d_lambda: float,
    d_alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    t_rel = tlist / duration_s
    env = np.asarray(gaussian_envelope(t_rel, sigma=sigma_fraction), dtype=np.complex128)
    base_amp = _initial_drive_amplitude(theta_target, duration_s, sigma_fraction, tlist)
    # Lab-aligned additive amplitude correction in normalized lambda0 units:
    #   amp = theta/(2T) + lambda0*d_lambda_norm
    lam0 = sqr_lambda0_rad_s(duration_s)
    omega = (base_amp + float(d_lambda) * lam0) * np.real(env)
    phi = float(phi_target + d_alpha)
    return omega * np.cos(phi), omega * np.sin(phi), float(base_amp)


def _normalized_unitary(matrix: np.ndarray) -> np.ndarray:
    det = np.linalg.det(matrix)
    if abs(det) > 1.0e-15:
        matrix = matrix * np.exp(-0.5j * np.angle(det))
    return matrix


def _project_to_unitary(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.complex128))
    return _normalized_unitary(u @ vh)


def _benchmark_simulation_config(config: Mapping[str, Any], total_levels: int) -> dict[str, Any]:
    return {
        "omega_c_hz": float(config.get("omega_c_hz", 0.0)),
        "omega_q_hz": float(config.get("omega_q_hz", 0.0)),
        "qubit_alpha_hz": float(config.get("qubit_alpha_hz", 0.0)),
        "st_chi_hz": float(config.get("st_chi_hz", 0.0)),
        "st_chi2_hz": float(config.get("st_chi2_hz", 0.0)),
        "st_chi3_hz": float(config.get("st_chi3_hz", 0.0)),
        "st_K_hz": float(config.get("st_K_hz", 0.0)),
        "st_K2_hz": float(config.get("st_K2_hz", 0.0)),
        "fock_fqs_hz": None if config.get("fock_fqs_hz") is None else [float(value) for value in config.get("fock_fqs_hz")],
        "n_cav_dim": int(config.get("n_cav_dim", total_levels)),
        "use_rotating_frame": bool(config.get("use_rotating_frame", True)),
        "qb_T1_relax_ns": None if config.get("qb_T1_relax_ns") is None else float(config.get("qb_T1_relax_ns")),
        "qb_T2_ramsey_ns": None if config.get("qb_T2_ramsey_ns") is None else float(config.get("qb_T2_ramsey_ns")),
        "qb_T2_echo_ns": None if config.get("qb_T2_echo_ns") is None else float(config.get("qb_T2_echo_ns")),
        "t2_source": str(config.get("t2_source", "ramsey")),
        "cavity_kappa_1_per_s": float(config.get("cavity_kappa_1_per_s", 0.0)),
    }


def _hz_to_rad_s(hz: float) -> float:
    return float(2.0 * np.pi * hz)


def _ns_to_s(ns: float | None) -> float | None:
    return None if ns is None else float(ns) * 1.0e-9


def _choose_t2_ns(config: Mapping[str, Any]) -> float | None:
    source = str(config.get("t2_source", "ramsey")).lower()
    if source == "echo":
        value = config.get("qb_T2_echo_ns")
        return None if value is None else float(value)
    if source != "ramsey":
        raise ValueError(f"Unsupported t2_source '{config.get('t2_source')}'.")
    value = config.get("qb_T2_ramsey_ns")
    return None if value is None else float(value)


def _derive_tphi_seconds(t1_ns: float | None, t2_ns: float | None) -> float | None:
    if t2_ns is None:
        return None
    t2_s = _ns_to_s(t2_ns)
    if t1_ns is None:
        return t2_s
    t1_s = _ns_to_s(t1_ns)
    inv_tphi = max(0.0, 1.0 / t2_s - 1.0 / (2.0 * t1_s))
    return None if inv_tphi <= 0.0 else 1.0 / inv_tphi


def _build_benchmark_model(config: Mapping[str, Any]) -> DispersiveTransmonCavityModel:
    chi_higher = tuple(
        value
        for value in (
            _hz_to_rad_s(float(config.get("st_chi2_hz", 0.0))),
            _hz_to_rad_s(float(config.get("st_chi3_hz", 0.0))),
        )
        if value != 0.0
    )
    kerr_higher = tuple(value for value in (_hz_to_rad_s(float(config.get("st_K2_hz", 0.0))),) if value != 0.0)
    return DispersiveTransmonCavityModel(
        omega_c=_hz_to_rad_s(float(config.get("omega_c_hz", 0.0))),
        omega_q=_hz_to_rad_s(float(config.get("omega_q_hz", 0.0))),
        alpha=_hz_to_rad_s(float(config.get("qubit_alpha_hz", 0.0))),
        chi=_hz_to_rad_s(float(config.get("st_chi_hz", 0.0))),
        chi_higher=chi_higher,
        kerr=_hz_to_rad_s(float(config.get("st_K_hz", 0.0))),
        kerr_higher=kerr_higher,
        n_cav=int(config["n_cav_dim"]),
        n_tr=2,
    )


def _build_benchmark_frame(model: DispersiveTransmonCavityModel, config: Mapping[str, Any]) -> FrameSpec:
    if bool(config.get("use_rotating_frame", True)):
        return FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return FrameSpec()


def _build_benchmark_noise(config: Mapping[str, Any]) -> NoiseSpec | None:
    if not _noise_enabled(config):
        return None
    t1_s = _ns_to_s(float(config["qb_T1_relax_ns"])) if config.get("qb_T1_relax_ns") is not None else None
    tphi_s = _derive_tphi_seconds(config.get("qb_T1_relax_ns"), _choose_t2_ns(config))
    kappa = float(config.get("cavity_kappa_1_per_s", 0.0))
    if t1_s is None and tphi_s is None and kappa <= 0.0:
        return None
    return NoiseSpec(t1=t1_s, tphi=tphi_s, kappa=kappa if kappa > 0.0 else None)


def _noise_enabled(config: Mapping[str, Any]) -> bool:
    return any(
        value not in (None, 0, 0.0)
        for value in (
            config.get("qb_T1_relax_ns"),
            config.get("qb_T2_ramsey_ns"),
            config.get("qb_T2_echo_ns"),
            config.get("cavity_kappa_1_per_s", 0.0),
        )
    )


def _solver_options(config: Mapping[str, Any]) -> dict[str, Any]:
    options = {
        "atol": 1.0e-8,
        "rtol": 1.0e-7,
        "store_states": True,
        "nsteps": int(config.get("qutip_nsteps_sqr_calibration", 100000)),
    }
    if float(config.get("max_step_s", 0.0)) > 0.0:
        options["max_step"] = float(config["max_step_s"])
    return options


def _vectorize_operator(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.complex128).reshape((-1,), order="F")


def _devectorize_operator(vector: np.ndarray, dim: int) -> np.ndarray:
    return np.asarray(vector, dtype=np.complex128).reshape((int(dim), int(dim)), order="F")


def _bloch_from_density_matrix(rho: np.ndarray) -> tuple[float, float, float]:
    rho = np.asarray(rho, dtype=np.complex128)
    return (
        2.0 * float(np.real(rho[0, 1])),
        2.0 * float(np.imag(rho[0, 1])),
        float(np.real(rho[0, 0] - rho[1, 1])),
    )


def _build_multitone_simulation(
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    corrections: Mapping[int, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    sim_config = _benchmark_simulation_config(config, int(target.total_levels))
    model = _build_benchmark_model(sim_config)
    frame = _build_benchmark_frame(model, sim_config)
    correction_map = {} if corrections is None else {int(k): tuple(float(x) for x in values) for k, values in corrections.items()}
    d_lambda_values = [0.0] * int(target.total_levels)
    for level, (d_lambda, _d_alpha, _d_omega) in correction_map.items():
        if 0 <= int(level) < len(d_lambda_values):
            d_lambda_values[int(level)] = float(d_lambda)
    raw_tones = build_sqr_tone_specs(
        model=model,
        frame=frame,
        theta_values=list(target.theta),
        phi_values=list(target.phi),
        duration_s=float(config["duration_sqr_s"]),
        d_lambda_values=d_lambda_values,
        fock_fqs_hz=sim_config.get("fock_fqs_hz"),
        include_all_levels=bool(config.get("allow_zero_theta_corrections", True)),
        tone_cutoff=float(config.get("sqr_theta_cutoff", 1.0e-10)),
    )
    tone_specs: list[MultitoneTone] = []
    for tone in raw_tones:
        _d_lambda, d_alpha, d_omega = correction_map.get(int(tone.manifold), (0.0, 0.0, 0.0))
        tone_specs.append(
            MultitoneTone(
                manifold=int(tone.manifold),
                omega_rad_s=float(tone.omega_rad_s + d_omega),
                amp_rad_s=float(tone.amp_rad_s),
                phase_rad=float(tone.phase_rad + d_alpha),
            )
        )

    duration_s = float(config["duration_sqr_s"])
    sigma_fraction = float(config["sqr_sigma_fraction"])

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return multitone_gaussian_envelope(
            t_rel,
            duration_s=duration_s,
            sigma_fraction=sigma_fraction,
            tone_specs=tone_specs,
        )

    pulse = Pulse("qubit", 0.0, duration_s, envelope, amp=1.0, phase=0.0, label=target.target_id)
    compiled = SequenceCompiler(dt=float(config["dt_s"])).compile([pulse], t_end=duration_s + float(config["dt_s"]))
    drive_ops = {"qubit": "qubit"}
    hamiltonian = hamiltonian_time_slices(model, compiled, drive_ops, frame=frame)
    noise = _build_benchmark_noise(sim_config)
    c_ops = collapse_operators(model, noise)
    return {
        "model": model,
        "frame": frame,
        "compiled": compiled,
        "hamiltonian": hamiltonian,
        "c_ops": c_ops,
        "tone_specs": [tone.as_dict() for tone in tone_specs],
        "active_levels": [int(tone.manifold) for tone in tone_specs],
        "noise_enabled": bool(c_ops),
        "config_snapshot": sim_config,
    }


def _final_full_unitary(prepared: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
    propagators = qt.propagator(
        prepared["hamiltonian"],
        prepared["compiled"].tlist,
        options=_solver_options(config),
        tlist=prepared["compiled"].tlist,
    )
    final = propagators[-1] if isinstance(propagators, list) else propagators
    return np.asarray(final.full(), dtype=np.complex128)


def _project_manifold_unitary(prepared: Mapping[str, Any], manifold_n: int, config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if prepared["c_ops"]:
        raise ValueError("Dissipative configurations require channel extraction, not unitary extraction.")
    full = _final_full_unitary(prepared, config)
    indices = qubit_cavity_block_indices(int(prepared["model"].n_cav), int(manifold_n))
    block = np.asarray(full[np.ix_(indices, indices)], dtype=np.complex128)
    return _normalized_unitary(block), {
        "tlist": np.asarray(prepared["compiled"].tlist, dtype=float),
        "active_levels": list(prepared["active_levels"]),
        "tone_specs": list(prepared["tone_specs"]),
        "noise_enabled": False,
        "simulation_path": "cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator",
    }


def _evolve_with_shared_solver(
    prepared: Mapping[str, Any],
    initial_state: qt.Qobj,
    config: Mapping[str, Any],
) -> qt.Qobj:
    options = _solver_options(config)
    if prepared["c_ops"] or initial_state.isoper:
        result = qt.mesolve(
            prepared["hamiltonian"],
            initial_state,
            prepared["compiled"].tlist,
            c_ops=prepared["c_ops"],
            e_ops=[],
            options=options,
        )
    else:
        result = qt.sesolve(
            prepared["hamiltonian"],
            initial_state,
            prepared["compiled"].tlist,
            e_ops=[],
            options=options,
        )
    return result.states[-1]


def _project_manifold_channel(prepared: Mapping[str, Any], manifold_n: int, config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    model = prepared["model"]
    indices = qubit_cavity_block_indices(int(model.n_cav), int(manifold_n))
    basis_states = [model.basis_state(0, int(manifold_n)), model.basis_state(1, int(manifold_n))]
    superoperator = np.zeros((4, 4), dtype=np.complex128)
    for column, (bra_state, ket_state) in enumerate(
        (
            (basis_states[0], basis_states[0]),
            (basis_states[0], basis_states[1]),
            (basis_states[1], basis_states[0]),
            (basis_states[1], basis_states[1]),
        )
    ):
        final = _evolve_with_shared_solver(prepared, bra_state * ket_state.dag(), config)
        matrix = np.asarray(final.full(), dtype=np.complex128)
        block = np.asarray(matrix[np.ix_(indices, indices)], dtype=np.complex128)
        superoperator[:, column] = _vectorize_operator(block)
    return superoperator, {
        "tlist": np.asarray(prepared["compiled"].tlist, dtype=float),
        "active_levels": list(prepared["active_levels"]),
        "tone_specs": list(prepared["tone_specs"]),
        "noise_enabled": bool(prepared["c_ops"]),
        "simulation_path": "cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.mesolve",
    }


def extract_multitone_effective_qubit_channel(
    manifold_n: int,
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    corrections: Mapping[int, tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    prepared = _build_multitone_simulation(target, config, corrections=corrections)
    return _project_manifold_channel(prepared, manifold_n, config)


def _final_unitary_from_hamiltonian(
    hamiltonian: Any,
    tlist: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    options = {"atol": 1.0e-8, "rtol": 1.0e-7, "nsteps": int(config.get("qutip_nsteps_sqr_calibration", 100000))}
    if float(config.get("max_step_s", 0.0)) > 0.0:
        options["max_step"] = float(config["max_step_s"])
    propagators = qt.propagator(hamiltonian, tlist, options=options, tlist=tlist)
    final = propagators[-1] if isinstance(propagators, list) else propagators
    return _normalized_unitary(np.asarray(final.full(), dtype=np.complex128))


def extract_effective_qubit_unitary(
    n: int,
    theta_target: float,
    phi_target: float,
    config: Mapping[str, Any],
    d_lambda: float = 0.0,
    d_alpha: float = 0.0,
    d_omega_rad_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    duration_s = float(config["duration_sqr_s"])
    tlist = _build_time_grid(duration_s, float(config["dt_s"]))
    sigma_fraction = float(config["sqr_sigma_fraction"])
    omega_x, omega_y, base_amp = _drive_components(
        theta_target=theta_target,
        phi_target=phi_target,
        duration_s=duration_s,
        sigma_fraction=sigma_fraction,
        tlist=tlist,
        d_lambda=d_lambda,
        d_alpha=d_alpha,
    )
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    detuning = _conditional_detuning_rad_s(n, config) + float(d_omega_rad_s)
    h = [
        0.5 * detuning * sz,
        [0.5 * sx, omega_x],
        [0.5 * sy, omega_y],
    ]
    matrix = _final_unitary_from_hamiltonian(h, tlist, config)
    return matrix, {
        "tlist": tlist,
        "omega_x": omega_x,
        "omega_y": omega_y,
        "base_amp_rad_s": base_amp,
        "detuning_rad_s": detuning,
    }


def target_qubit_unitary(theta_target: float, phi_target: float) -> np.ndarray:
    return np.asarray(qubit_rotation_xy(float(theta_target), float(phi_target)).full(), dtype=np.complex128)


def conditional_process_fidelity(target_unitary: np.ndarray, simulated_unitary: np.ndarray) -> float:
    target = np.asarray(target_unitary, dtype=np.complex128)
    simulated = np.asarray(simulated_unitary, dtype=np.complex128)
    if simulated.shape == target.shape:
        overlap = np.trace(target.conj().T @ simulated)
        return float(np.clip(np.abs(overlap) ** 2 / 4.0, 0.0, 1.0))
    if simulated.shape == (target.shape[0] ** 2, target.shape[1] ** 2):
        target_super = np.kron(target.conj(), target)
        fidelity = np.trace(target_super.conj().T @ simulated) / float(target.shape[0] ** 2)
        return float(np.clip(np.real_if_close(fidelity).real, 0.0, 1.0))
    raise ValueError(
        f"Unsupported simulated operator shape {simulated.shape}; expected {target.shape} or {(target.shape[0] ** 2, target.shape[1] ** 2)}."
    )


def _objective_regularization(config: Mapping[str, Any], d_lambda: float, d_alpha: float, d_omega_rad_s: float) -> float:
    eta_lambda = float(config.get("regularization_lambda", 0.0))
    eta_alpha = float(config.get("regularization_alpha", 0.0))
    eta_omega = float(config.get("regularization_omega", 0.0))
    return float(eta_lambda * (d_lambda**2) + eta_alpha * (d_alpha**2) + eta_omega * (d_omega_rad_s**2))


def conditional_loss(
    params: np.ndarray,
    n: int,
    theta_target: float,
    phi_target: float,
    config: Mapping[str, Any],
) -> float:
    d_lambda, d_alpha, d_omega_rad_s = map(float, params)
    simulated, _ = extract_effective_qubit_unitary(
        n=n,
        theta_target=theta_target,
        phi_target=phi_target,
        config=config,
        d_lambda=d_lambda,
        d_alpha=d_alpha,
        d_omega_rad_s=d_omega_rad_s,
    )
    target = target_qubit_unitary(theta_target, phi_target)
    fidelity = conditional_process_fidelity(target, simulated)
    return float(1.0 - fidelity + _objective_regularization(config, d_lambda, d_alpha, d_omega_rad_s))


def _optimizer_bounds(config: Mapping[str, Any]) -> Bounds:
    d_lambda_bounds = tuple(config.get("d_lambda_bounds", (-0.5, 0.5)))
    d_alpha_bounds = tuple(config.get("d_alpha_bounds", (-np.pi, np.pi)))
    d_omega_hz_bounds = tuple(config.get("d_omega_hz_bounds", (-2.0e6, 2.0e6)))
    lower = [float(d_lambda_bounds[0]), float(d_alpha_bounds[0]), float(2.0 * np.pi * d_omega_hz_bounds[0])]
    upper = [float(d_lambda_bounds[1]), float(d_alpha_bounds[1]), float(2.0 * np.pi * d_omega_hz_bounds[1])]
    return Bounds(lower, upper)


def _calibration_max_n(gate: SQRGate, config: Mapping[str, Any]) -> int:
    max_n_requested = int(config.get("max_n_cal", int(config["cavity_fock_cutoff"])))
    theta = np.asarray(gate.theta, dtype=float)
    phi = np.asarray(gate.phi, dtype=float)
    return int(min(max_n_requested, int(config["cavity_fock_cutoff"]), theta.size - 1, phi.size - 1))


def evaluate_sqr_gate_levels(
    gate: SQRGate,
    config: Mapping[str, Any],
    corrections: SQRCalibrationResult | None = None,
) -> list[dict[str, Any]]:
    max_n = _calibration_max_n(gate, config)
    theta, phi = pad_sqr_angles(gate.theta, gate.phi, int(config["n_cav_dim"]))
    rows = []
    for n in range(max_n + 1):
        d_lambda, d_alpha, d_omega = (0.0, 0.0, 0.0) if corrections is None else corrections.correction_for_n(n)
        simulated, extra = extract_effective_qubit_unitary(
            n=n,
            theta_target=float(theta[n]),
            phi_target=float(phi[n]),
            config=config,
            d_lambda=d_lambda,
            d_alpha=d_alpha,
            d_omega_rad_s=d_omega,
        )
        target = target_qubit_unitary(float(theta[n]), float(phi[n]))
        fidelity = conditional_process_fidelity(target, simulated)
        rows.append(
            {
                "n": n,
                "theta_target": float(theta[n]),
                "phi_target": float(phi[n]),
                "d_lambda": float(d_lambda),
                "d_alpha": float(d_alpha),
                "d_omega_rad_s": float(d_omega),
                "d_omega_hz": float(d_omega / (2.0 * np.pi)),
                "process_fidelity": float(fidelity),
                "loss": float(1.0 - fidelity),
                **extra,
            }
        )
    return rows


def calibrate_sqr_gate(gate: SQRGate, config: Mapping[str, Any]) -> SQRCalibrationResult:
    theta, phi = pad_sqr_angles(gate.theta, gate.phi, int(config["n_cav_dim"]))
    max_n = _calibration_max_n(gate, config)
    theta_cutoff = float(config.get("sqr_theta_cutoff", 1.0e-10))
    allow_zero_theta_corrections = bool(config.get("allow_zero_theta_corrections", True))
    bounds = _optimizer_bounds(config)
    method_stage1 = str(config.get("optimizer_method_stage1", "Powell"))
    method_stage2 = str(config.get("optimizer_method_stage2", "L-BFGS-B"))

    d_lambda = [0.0] * (max_n + 1)
    d_alpha = [0.0] * (max_n + 1)
    d_omega_rad_s = [0.0] * (max_n + 1)
    initial_loss = [0.0] * (max_n + 1)
    optimized_loss = [0.0] * (max_n + 1)
    levels: list[SQRLevelCalibration] = []

    for n in range(max_n + 1):
        theta_n = float(theta[n])
        phi_n = float(phi[n])
        x0 = np.zeros(3, dtype=float)
        loss0 = conditional_loss(x0, n=n, theta_target=theta_n, phi_target=phi_n, config=config)
        initial_loss[n] = float(loss0)
        if (not allow_zero_theta_corrections) and abs(theta_n) < theta_cutoff:
            levels.append(
                SQRLevelCalibration(
                    n=n,
                    theta_target=theta_n,
                    phi_target=phi_n,
                    skipped=True,
                    initial_params=(0.0, 0.0, 0.0),
                    optimized_params=(0.0, 0.0, 0.0),
                    initial_loss=float(loss0),
                    optimized_loss=float(loss0),
                    process_fidelity=float(1.0 - loss0),
                )
            )
            optimized_loss[n] = float(loss0)
            continue

        objective = lambda x: conditional_loss(x, n=n, theta_target=theta_n, phi_target=phi_n, config=config)
        stage1 = minimize(
            objective,
            x0=x0,
            method=method_stage1,
            bounds=bounds,
            options={"maxiter": int(config.get("optimizer_maxiter_stage1", 60)), "disp": False},
        )
        stage2 = minimize(
            objective,
            x0=np.asarray(stage1.x, dtype=float),
            method=method_stage2,
            bounds=bounds,
            options={"maxiter": int(config.get("optimizer_maxiter_stage2", 80))},
        )
        best_x = np.asarray(stage2.x if stage2.fun <= stage1.fun else stage1.x, dtype=float)
        best_loss = float(min(stage1.fun, stage2.fun))
        simulated = extract_effective_qubit_unitary(
            n=n,
            theta_target=theta_n,
            phi_target=phi_n,
            config=config,
            d_lambda=float(best_x[0]),
            d_alpha=float(best_x[1]),
            d_omega_rad_s=float(best_x[2]),
        )[0]
        target = target_qubit_unitary(theta_n, phi_n)
        fidelity = conditional_process_fidelity(target, simulated)

        d_lambda[n] = float(best_x[0])
        d_alpha[n] = float(best_x[1])
        d_omega_rad_s[n] = float(best_x[2])
        optimized_loss[n] = float(best_loss)
        levels.append(
            SQRLevelCalibration(
                n=n,
                theta_target=theta_n,
                phi_target=phi_n,
                skipped=False,
                initial_params=(0.0, 0.0, 0.0),
                optimized_params=(float(best_x[0]), float(best_x[1]), float(best_x[2])),
                initial_loss=float(loss0),
                optimized_loss=float(best_loss),
                process_fidelity=float(fidelity),
                success_stage1=bool(stage1.success),
                success_stage2=bool(stage2.success),
                message_stage1=str(stage1.message),
                message_stage2=str(stage2.message),
            )
        )

    return SQRCalibrationResult(
        sqr_name=gate.name,
        max_n=max_n,
        d_lambda=d_lambda,
        d_alpha=d_alpha,
        d_omega_rad_s=d_omega_rad_s,
        theta_target=[float(theta[n]) for n in range(max_n + 1)],
        phi_target=[float(phi[n]) for n in range(max_n + 1)],
        initial_loss=initial_loss,
        optimized_loss=optimized_loss,
        levels=levels,
        metadata={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objective_type": "process_infidelity_plus_regularization",
            "amplitude_correction_convention": "amp = theta/(2T) + lambda0*d_lambda_norm",
            "config_snapshot": _relevant_config_snapshot(config),
            "config_hash": _config_hash(config),
            "optimizer_method_stage1": str(config.get("optimizer_method_stage1", "Powell")),
            "optimizer_method_stage2": str(config.get("optimizer_method_stage2", "L-BFGS-B")),
        },
    )


def export_calibration_result(
    result: SQRCalibrationResult,
    output_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if config is not None:
        payload.setdefault("metadata", {})
        payload["metadata"]["config_snapshot"] = _relevant_config_snapshot(config)
        payload["metadata"]["config_hash"] = _config_hash(config)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_calibration_result(path: str | Path) -> SQRCalibrationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SQRCalibrationResult.from_dict(payload)


def _cache_is_valid(result: SQRCalibrationResult, gate: SQRGate, config: Mapping[str, Any]) -> bool:
    if result.sqr_name != gate.name:
        return False
    metadata = result.metadata or {}
    if metadata.get("config_hash") == _config_hash(config):
        return True
    snapshot = metadata.get("config_snapshot", {})
    return (
        snapshot.get("duration_sqr_s") == config.get("duration_sqr_s")
        and snapshot.get("max_n_cal") == config.get("max_n_cal")
    )


def load_or_calibrate_sqr_gate(
    gate: SQRGate,
    config: Mapping[str, Any],
    cache_dir: str | Path | None = None,
) -> SQRCalibrationResult:
    cache_path = calibration_cache_path(gate, config, cache_dir=cache_dir or config.get("calibration_cache_dir", "calibrations"))
    force_recompute = bool(config.get("calibration_force_recompute", False))
    if cache_path.exists() and not force_recompute:
        cached = load_calibration_result(cache_path)
        if _cache_is_valid(cached, gate, config):
            cached.metadata = dict(cached.metadata)
            cached.metadata["cache_hit"] = True
            cached.metadata["cache_path"] = str(cache_path)
            return cached
    result = calibrate_sqr_gate(gate, config)
    result.metadata = dict(result.metadata)
    result.metadata["cache_hit"] = False
    result.metadata["cache_path"] = str(cache_path)
    export_calibration_result(result, cache_path, config=config)
    return result


def calibrate_all_sqr_gates(
    gates: list[Gate],
    config: Mapping[str, Any],
    cache_dir: str | Path | None = None,
) -> dict[str, SQRCalibrationResult]:
    results: dict[str, SQRCalibrationResult] = {}
    for gate in extract_sqr_gates(gates):
        if gate.name not in results:
            results[gate.name] = load_or_calibrate_sqr_gate(gate, config, cache_dir=cache_dir)
    return results


def _wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.mod(phi, 2.0 * np.pi)


def _scale_theta(values: np.ndarray, theta_max: float, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs < 1.0e-12:
        return np.zeros_like(values)
    scale = float(rng.uniform(0.35, 1.0)) * float(theta_max) / max_abs
    return scale * values


def _sample_random_target_family(
    rng: np.random.Generator,
    logical_n: int,
    target_class: str,
    theta_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(logical_n, dtype=float)
    if target_class == "iid":
        theta = rng.uniform(-theta_max, theta_max, size=logical_n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=logical_n)
        return theta, phi
    if target_class == "smooth":
        raw_theta = (
            rng.normal() * np.cos(2.0 * np.pi * n / max(logical_n, 1))
            + rng.normal() * np.sin(2.0 * np.pi * n / max(logical_n, 1))
            + 0.4 * rng.normal() * np.cos(4.0 * np.pi * n / max(logical_n, 1))
        )
        raw_phi = (
            rng.normal() * np.cos(2.0 * np.pi * n / max(logical_n, 1))
            + rng.normal() * np.sin(2.0 * np.pi * n / max(logical_n, 1))
        )
        theta = _scale_theta(raw_theta, theta_max, rng)
        phi = _wrap_phi(np.pi + raw_phi)
        return theta, phi
    if target_class == "hard":
        raw_theta = ((-1.0) ** np.arange(logical_n)) * rng.uniform(0.45 * theta_max, theta_max, size=logical_n)
        theta = raw_theta + rng.normal(scale=0.1 * theta_max, size=logical_n)
        theta = np.clip(theta, -theta_max, theta_max)
        phi = _wrap_phi(np.linspace(0.0, 2.0 * np.pi, logical_n, endpoint=False) + rng.normal(scale=0.5, size=logical_n))
        return theta, phi
    if target_class == "sparse":
        theta = np.zeros(logical_n, dtype=float)
        phi = np.zeros(logical_n, dtype=float)
        n_active = max(1, min(logical_n, int(rng.integers(1, max(2, logical_n // 2 + 1)))))
        active = np.sort(rng.choice(logical_n, size=n_active, replace=False))
        theta[active] = rng.uniform(-theta_max, theta_max, size=n_active)
        phi[active] = rng.uniform(0.0, 2.0 * np.pi, size=n_active)
        return theta, phi
    raise ValueError(f"Unsupported target_class '{target_class}'.")


def generate_random_sqr_targets(
    logical_n: int,
    guard_levels: int,
    n_targets_per_class: int,
    seed: int,
    target_classes: Sequence[str] = ("iid", "smooth", "hard", "sparse"),
    theta_max: float = np.pi,
) -> list[RandomSQRTarget]:
    rng = np.random.default_rng(int(seed))
    targets: list[RandomSQRTarget] = []
    counter = 0
    for target_class in target_classes:
        for _ in range(int(n_targets_per_class)):
            theta_logical, phi_logical = _sample_random_target_family(
                rng=rng,
                logical_n=int(logical_n),
                target_class=str(target_class),
                theta_max=float(theta_max),
            )
            total_levels = int(logical_n + guard_levels)
            theta = np.zeros(total_levels, dtype=float)
            phi = np.zeros(total_levels, dtype=float)
            theta[:logical_n] = theta_logical
            phi[:logical_n] = phi_logical
            target_id = f"random_{target_class}_{counter:03d}"
            targets.append(
                RandomSQRTarget(
                    target_id=target_id,
                    target_class=str(target_class),
                    logical_n=int(logical_n),
                    guard_levels=int(guard_levels),
                    theta=tuple(float(value) for value in theta),
                    phi=tuple(float(value) for value in phi),
                )
            )
            counter += 1
    return targets


def _logical_weights(
    logical_n: int,
    weight_mode: str = "uniform",
    poisson_alpha: complex | None = None,
) -> np.ndarray:
    n = np.arange(int(logical_n), dtype=float)
    if weight_mode == "uniform":
        weights = np.ones(int(logical_n), dtype=float)
    elif weight_mode == "poisson":
        alpha = 0.0 if poisson_alpha is None else complex(poisson_alpha)
        mean = float(abs(alpha) ** 2)
        weights = np.exp(-mean) * np.power(mean, n) / np.asarray([math.factorial(int(k)) for k in n], dtype=float)
    else:
        raise ValueError(f"Unsupported weight_mode '{weight_mode}'.")
    total = float(np.sum(weights))
    return weights / total if total > 0.0 else np.ones(int(logical_n), dtype=float) / float(logical_n)


def _bloch_from_unitary_on_ground(unitary: np.ndarray) -> tuple[float, float, float]:
    ket = np.asarray(unitary[:, 0], dtype=np.complex128).reshape((2, 1))
    rho = ket @ ket.conj().T
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return (
        float(np.real(np.trace(rho @ sx))),
        2.0 * float(np.imag(rho[0, 1])),
        float(np.real(np.trace(rho @ sz))),
    )


def _unitary_rotation_parameters(unitary: np.ndarray) -> tuple[float, float, float]:
    u = _normalized_unitary(np.asarray(unitary, dtype=np.complex128))
    cos_half = float(np.clip(np.real(np.trace(u) / 2.0), -1.0, 1.0))
    theta = float(2.0 * np.arccos(cos_half))
    if theta < 1.0e-12:
        return 0.0, 0.0, 0.0
    sin_half = math.sin(theta / 2.0)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    nx = float(np.real(1.0j * np.trace(sx @ u) / (2.0 * sin_half)))
    ny = float(np.real(1.0j * np.trace(sy @ u) / (2.0 * sin_half)))
    nz = float(np.real(1.0j * np.trace(sz @ u) / (2.0 * sin_half)))
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm > 1.0e-12:
        nx /= norm
        ny /= norm
        nz /= norm
    phi = float(np.mod(np.arctan2(ny, nx), 2.0 * np.pi))
    return theta, phi, nz


def _benchmark_active_levels(
    target: RandomSQRTarget,
    theta_cutoff: float,
    include_zero_theta_levels: bool = True,
) -> list[int]:
    if bool(include_zero_theta_levels):
        return [n for n in range(int(target.logical_n))]
    return [n for n in range(int(target.logical_n)) if abs(float(target.theta[n])) >= float(theta_cutoff)]


def _corrections_from_vector(active_levels: list[int], x: np.ndarray) -> dict[int, tuple[float, float, float]]:
    vector = np.asarray(x, dtype=float).reshape((-1,))
    out: dict[int, tuple[float, float, float]] = {}
    for idx, level in enumerate(active_levels):
        base = 3 * idx
        out[int(level)] = (float(vector[base]), float(vector[base + 1]), float(vector[base + 2]))
    return out


def _full_correction_arrays(
    active_levels: list[int],
    params_map: Mapping[int, tuple[float, float, float]],
    total_levels: int,
) -> tuple[list[float], list[float], list[float]]:
    d_lambda = [0.0] * int(total_levels)
    d_alpha = [0.0] * int(total_levels)
    d_omega = [0.0] * int(total_levels)
    for level in active_levels:
        dl, da, dw = params_map.get(int(level), (0.0, 0.0, 0.0))
        d_lambda[int(level)] = float(dl)
        d_alpha[int(level)] = float(da)
        d_omega[int(level)] = float(dw)
    return d_lambda, d_alpha, d_omega


def _benchmark_bounds(config: Mapping[str, Any], n_active: int) -> Bounds:
    base = _optimizer_bounds(config)
    lower = np.tile(np.asarray(base.lb, dtype=float), int(n_active))
    upper = np.tile(np.asarray(base.ub, dtype=float), int(n_active))
    return Bounds(lower, upper)


def _benchmark_regularization(config: Mapping[str, Any], x: np.ndarray) -> float:
    eta_lambda = float(config.get("regularization_lambda", 0.0))
    eta_alpha = float(config.get("regularization_alpha", 0.0))
    eta_omega = float(config.get("regularization_omega", 0.0))
    value = 0.0
    for idx in range(0, len(x), 3):
        value += eta_lambda * float(x[idx] ** 2)
        value += eta_alpha * float(x[idx + 1] ** 2)
        value += eta_omega * float(x[idx + 2] ** 2)
    return float(value)


def _max_active_drive_amplitude_rad_s(
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    corrections: Mapping[int, tuple[float, float, float]] | None,
    active_levels: Sequence[int],
) -> float:
    duration_s = float(config["duration_sqr_s"])
    tlist = _build_time_grid(duration_s, float(config["dt_s"]))
    sigma_fraction = float(config["sqr_sigma_fraction"])
    lam0 = sqr_lambda0_rad_s(duration_s)
    max_amp = 0.0
    for level in active_levels:
        theta_level = float(target.theta[int(level)])
        base_amp = _initial_drive_amplitude(theta_level, duration_s, sigma_fraction, tlist)
        d_lambda = 0.0
        if corrections is not None:
            d_lambda = float(corrections.get(int(level), (0.0, 0.0, 0.0))[0])
        max_amp = max(max_amp, abs(float(base_amp + d_lambda * lam0)))
    return float(max_amp)


def _max_active_detuning_rad_s(
    config: Mapping[str, Any],
    corrections: Mapping[int, tuple[float, float, float]] | None,
    active_levels: Sequence[int],
) -> float:
    max_detuning = 0.0
    for level in active_levels:
        correction = 0.0
        if corrections is not None:
            correction = float(corrections.get(int(level), (0.0, 0.0, 0.0))[2])
        detuning = abs(float(_conditional_detuning_rad_s(int(level), config) + correction))
        max_detuning = max(max_detuning, detuning)
    return float(max_detuning)


def _emit_guarded_progress_event(
    *,
    reporter: ProgressReporter,
    run_id: str,
    iteration: int,
    metrics: Mapping[str, Any],
    best_loss: float,
    best_iteration: int,
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    backend_label: str,
) -> None:
    active_levels = [int(level) for level in metrics.get("active_levels", [])]
    corrections = metrics.get("corrections")
    regularization = 0.0
    if corrections is not None:
        vector: list[float] = []
        for level in active_levels:
            vector.extend(corrections.get(int(level), (0.0, 0.0, 0.0)))
        regularization = _benchmark_regularization(config, np.asarray(vector, dtype=float))
    event = ProgressEvent(
        run_id=str(run_id),
        iteration=int(iteration),
        timestamp=float(time.time()),
        objective_total=float(metrics["loss_total"]),
        objective_terms={
            "infidelity": float(1.0 - float(metrics["logical_fidelity"])),
            "leakage": float(float(metrics.get("lambda_guard", 0.0)) * float(metrics["epsilon_guard"])),
            "time_reg": float(regularization),
            "grid_penalty": 0.0,
            "constraint_penalty": 0.0,
        },
        metrics={
            "fidelity_subspace": float(metrics["logical_fidelity"]),
            "leakage_avg": float(np.mean(metrics.get("guard_xy", [0.0])) if metrics.get("guard_xy") else 0.0),
            "leakage_worst": float(metrics["epsilon_guard"]),
        },
        best_so_far={
            "objective_total": float(best_loss),
            "iteration": int(best_iteration),
        },
        params_summary={
            "max_amp": _max_active_drive_amplitude_rad_s(target, config, corrections, active_levels),
            "max_detuning": _max_active_detuning_rad_s(config, corrections, active_levels),
            "times": {"SQR": float(config["duration_sqr_s"])},
        },
        backend=str(backend_label),
        solver_stats={
            "n_steps": int(max(1, len(_build_time_grid(float(config["duration_sqr_s"]), float(config["dt_s"]))))),
            "dt": float(config["dt_s"]),
            "solver_time_sec": 0.0,
        },
    )
    reporter.on_event(event.to_dict())


def extract_multitone_effective_qubit_unitary(
    manifold_n: int,
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    corrections: Mapping[int, tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    prepared = _build_multitone_simulation(target, config, corrections=corrections)
    return _project_manifold_unitary(prepared, manifold_n, config)


def evaluate_guarded_sqr_target(
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    corrections: Mapping[int, tuple[float, float, float]] | None = None,
    lambda_guard: float = 0.1,
    weight_mode: str = "uniform",
    poisson_alpha: complex | None = None,
) -> dict[str, Any]:
    logical_n = int(target.logical_n)
    total_levels = int(target.total_levels)
    weights = _logical_weights(logical_n, weight_mode=weight_mode, poisson_alpha=poisson_alpha)
    theta_cutoff = float(config.get("sqr_theta_cutoff", 1.0e-10))
    active_levels = _benchmark_active_levels(
        target,
        theta_cutoff,
        include_zero_theta_levels=bool(config.get("allow_zero_theta_corrections", True)),
    )
    rows: list[dict[str, Any]] = []
    logical_fidelities = np.zeros(logical_n, dtype=float)
    guard_xy: list[float] = []
    guard_z: list[float] = []
    prepared = _build_multitone_simulation(target, config, corrections=corrections)
    use_channel = bool(prepared["c_ops"])

    for n in range(total_levels):
        target_unitary = target_qubit_unitary(float(target.theta[n]), float(target.phi[n]))
        if use_channel:
            simulated, extra = _project_manifold_channel(prepared, n, config)
            fidelity = conditional_process_fidelity(target_unitary, simulated)
            ground_state = _devectorize_operator(simulated[:, 0], 2)
            x, y, z = _bloch_from_density_matrix(ground_state)
            achieved_theta = float("nan")
            achieved_phi = float("nan")
            achieved_axis_z = float("nan")
        else:
            simulated, extra = _project_manifold_unitary(prepared, n, config)
            fidelity = conditional_process_fidelity(target_unitary, simulated)
            x, y, z = _bloch_from_unitary_on_ground(simulated)
            achieved_theta, achieved_phi, achieved_axis_z = _unitary_rotation_parameters(simulated)
        row = {
            "n": int(n),
            "theta_target": float(target.theta[n]),
            "phi_target": float(target.phi[n]),
            "process_fidelity": float(np.clip(fidelity, 0.0, 1.0)),
            "bloch_x": float(x),
            "bloch_y": float(y),
            "bloch_z": float(z),
            "achieved_theta": float(achieved_theta),
            "achieved_phi": float(achieved_phi),
            "achieved_axis_z": float(achieved_axis_z),
            "simulation_mode": "channel" if use_channel else "unitary",
            **extra,
        }
        if n < logical_n:
            logical_fidelities[n] = row["process_fidelity"]
            row["is_guard"] = False
            row["weight"] = float(weights[n])
        else:
            epsilon_xy = float(math.sqrt(x * x + y * y))
            epsilon_z = float(1.0 - z)
            row["is_guard"] = True
            row["guard_xy"] = epsilon_xy
            row["guard_z"] = epsilon_z
            row["weight"] = 0.0
            guard_xy.append(epsilon_xy)
            guard_z.append(epsilon_z)
        rows.append(row)

    epsilon_guard = float(max(guard_xy)) if guard_xy else 0.0
    logical_fidelity = float(np.sum(weights * logical_fidelities))
    if corrections is None:
        reg = 0.0
    else:
        vector = []
        for level in active_levels:
            vector.extend(corrections.get(int(level), (0.0, 0.0, 0.0)))
        reg = _benchmark_regularization(config, np.asarray(vector, dtype=float))
    loss_total = float((1.0 - logical_fidelity) + float(lambda_guard) * epsilon_guard + reg)
    return {
        "target_id": target.target_id,
        "target_class": target.target_class,
        "logical_n": logical_n,
        "guard_levels": int(target.guard_levels),
        "active_levels": active_levels,
        "logical_weights": weights,
        "logical_fidelity": logical_fidelity,
        "epsilon_guard": epsilon_guard,
        "guard_xy": guard_xy,
        "guard_z": guard_z,
        "loss_total": loss_total,
        "simulation_mode": "channel" if use_channel else "unitary",
        "execution_path": "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.mesolve" if use_channel else "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator",
        "per_manifold": rows,
    }


def calibrate_guarded_sqr_target(
    target: RandomSQRTarget,
    config: Mapping[str, Any],
    lambda_guard: float = 0.1,
    weight_mode: str = "uniform",
    poisson_alpha: complex | None = None,
    fidelity_threshold: float = 0.99,
    guard_threshold: float = 1.0e-2,
    initial_vector: np.ndarray | None = None,
    reporter: ProgressReporter | None = None,
    progress_every: int = 1,
    run_id: str | None = None,
    backend_label: str = "pulse",
) -> GuardedBenchmarkResult:
    run_reporter = reporter or NullReporter()
    emit_every = max(1, int(progress_every))
    run_label = str(run_id or target.target_id)
    active_levels = _benchmark_active_levels(
        target,
        float(config.get("sqr_theta_cutoff", 1.0e-10)),
        include_zero_theta_levels=bool(config.get("allow_zero_theta_corrections", True)),
    )
    run_reporter.on_start(
        {
            "progress_schema_version": 1,
            "run_id": run_label,
            "backend": str(backend_label),
            "target_id": target.target_id,
            "duration_s": float(config["duration_sqr_s"]),
            "emit_every": emit_every,
            "timestamp": float(time.time()),
        }
    )
    if not active_levels:
        zero_result = SQRCalibrationResult(
            sqr_name=target.target_id,
            max_n=int(target.total_levels - 1),
            d_lambda=[0.0] * int(target.total_levels),
            d_alpha=[0.0] * int(target.total_levels),
            d_omega_rad_s=[0.0] * int(target.total_levels),
            theta_target=[float(value) for value in target.theta],
            phi_target=[float(value) for value in target.phi],
            initial_loss=[0.0] * int(target.total_levels),
            optimized_loss=[0.0] * int(target.total_levels),
            levels=[],
            metadata={"benchmark_mode": True, "config_snapshot": _relevant_config_snapshot(config)},
        )
        metrics = evaluate_guarded_sqr_target(target, config, corrections={}, lambda_guard=lambda_guard, weight_mode=weight_mode, poisson_alpha=poisson_alpha)
        metrics["lambda_guard"] = float(lambda_guard)
        metrics["corrections"] = {}
        _emit_guarded_progress_event(
            reporter=run_reporter,
            run_id=run_label,
            iteration=0,
            metrics=metrics,
            best_loss=float(metrics["loss_total"]),
            best_iteration=0,
            target=target,
            config=config,
            backend_label=backend_label,
        )
        run_reporter.on_end(
            {
                "progress_schema_version": 1,
                "run_id": run_label,
                "success": True,
                "message": "No active levels; identity benchmark.",
                "nit": 0,
                "best_objective_total": float(metrics["loss_total"]),
                "best_iteration": 0,
                "timestamp": float(time.time()),
            }
        )
        return GuardedBenchmarkResult(
            target_id=target.target_id,
            target_class=target.target_class,
            duration_s=float(config["duration_sqr_s"]),
            logical_n=int(target.logical_n),
            guard_levels=int(target.guard_levels),
            lambda_guard=float(lambda_guard),
            weight_mode=str(weight_mode),
            poisson_alpha=poisson_alpha,
            logical_fidelity=float(metrics["logical_fidelity"]),
            epsilon_guard=float(metrics["epsilon_guard"]),
            loss_total=float(metrics["loss_total"]),
            success=bool(metrics["logical_fidelity"] >= fidelity_threshold and metrics["epsilon_guard"] <= guard_threshold),
            converged=True,
            iterations=0,
            objective_evaluations=1,
            calibration=zero_result,
            per_manifold=metrics["per_manifold"],
            convergence_trace=[{"iteration": 0, "best_loss_total": float(metrics["loss_total"]), "best_logical_fidelity": float(metrics["logical_fidelity"]), "best_epsilon_guard": float(metrics["epsilon_guard"])}],
            metadata={
                "active_levels": [],
                "weight_mode": weight_mode,
                "simulation_mode": str(metrics.get("simulation_mode", "unitary")),
                "execution_path": str(metrics.get("execution_path", "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator")),
            },
        )

    bounds = _benchmark_bounds(config, len(active_levels))
    method_stage1 = str(config.get("optimizer_method_stage1", "Powell"))
    method_stage2 = str(config.get("optimizer_method_stage2", "L-BFGS-B"))
    x0 = (
        np.zeros(3 * len(active_levels), dtype=float)
        if initial_vector is None
        else np.asarray(initial_vector, dtype=float).reshape((3 * len(active_levels),))
    )
    trace: list[dict[str, float]] = []
    best_metrics: dict[str, Any] | None = None
    best_loss = float("inf")
    best_iteration = 0

    def evaluate_vector(x: np.ndarray) -> dict[str, Any]:
        corrections = _corrections_from_vector(active_levels, x)
        metrics = evaluate_guarded_sqr_target(
            target,
            config,
            corrections=corrections,
            lambda_guard=lambda_guard,
            weight_mode=weight_mode,
            poisson_alpha=poisson_alpha,
        )
        metrics["corrections"] = corrections
        metrics["lambda_guard"] = float(lambda_guard)
        return metrics

    def objective(x: np.ndarray) -> float:
        nonlocal best_metrics, best_loss, best_iteration
        metrics = evaluate_vector(np.asarray(x, dtype=float))
        current_loss = float(metrics["loss_total"])
        if current_loss <= best_loss:
            best_loss = current_loss
            best_metrics = metrics
            best_iteration = int(len(trace))
        trace.append(
            {
                "iteration": float(len(trace)),
                "best_loss_total": float(best_loss),
                "best_logical_fidelity": float(best_metrics["logical_fidelity"]) if best_metrics is not None else float("nan"),
                "best_epsilon_guard": float(best_metrics["epsilon_guard"]) if best_metrics is not None else float("nan"),
            }
        )
        if (len(trace) - 1) % emit_every == 0:
            _emit_guarded_progress_event(
                reporter=run_reporter,
                run_id=run_label,
                iteration=int(len(trace) - 1),
                metrics=metrics,
                best_loss=float(best_loss),
                best_iteration=int(best_iteration),
                target=target,
                config=config,
                backend_label=backend_label,
            )
        return current_loss

    stage1 = minimize(
        objective,
        x0=x0,
        method=method_stage1,
        bounds=bounds,
        options={"maxiter": int(config.get("optimizer_maxiter_stage1", 60)), "disp": False},
    )
    stage2 = minimize(
        objective,
        x0=np.asarray(stage1.x, dtype=float),
        method=method_stage2,
        bounds=bounds,
        options={"maxiter": int(config.get("optimizer_maxiter_stage2", 80))},
    )

    best_x = np.asarray(stage2.x if stage2.fun <= stage1.fun else stage1.x, dtype=float)
    final_metrics = evaluate_vector(best_x)
    if not trace or int(trace[-1]["iteration"]) != len(trace):
        _emit_guarded_progress_event(
            reporter=run_reporter,
            run_id=run_label,
            iteration=int(len(trace)),
            metrics=final_metrics,
            best_loss=float(min(best_loss, float(final_metrics["loss_total"]))),
            best_iteration=int(best_iteration if best_loss <= float(final_metrics["loss_total"]) else len(trace)),
            target=target,
            config=config,
            backend_label=backend_label,
        )
    final_corrections = final_metrics["corrections"]
    d_lambda, d_alpha, d_omega = _full_correction_arrays(active_levels, final_corrections, target.total_levels)
    initial_eval = evaluate_guarded_sqr_target(
        target,
        config,
        corrections={},
        lambda_guard=lambda_guard,
        weight_mode=weight_mode,
        poisson_alpha=poisson_alpha,
    )
    levels = []
    for n in range(target.total_levels):
        is_guard = n >= int(target.logical_n)
        optimized_row = final_metrics["per_manifold"][n]
        initial_row = initial_eval["per_manifold"][n]
        levels.append(
            SQRLevelCalibration(
                n=int(n),
                theta_target=float(target.theta[n]),
                phi_target=float(target.phi[n]),
                skipped=bool(n not in active_levels),
                initial_params=(0.0, 0.0, 0.0),
                optimized_params=(float(d_lambda[n]), float(d_alpha[n]), float(d_omega[n])),
                initial_loss=float(1.0 - initial_row["process_fidelity"]),
                optimized_loss=float(1.0 - optimized_row["process_fidelity"]),
                process_fidelity=float(optimized_row["process_fidelity"]),
                success_stage1=bool(stage1.success),
                success_stage2=bool(stage2.success),
                message_stage1=str(stage1.message),
                message_stage2=str(stage2.message),
            )
        )
        if is_guard:
            levels[-1].message_stage2 += " | guard"

    calibration = SQRCalibrationResult(
        sqr_name=target.target_id,
        max_n=int(target.total_levels - 1),
        d_lambda=d_lambda,
        d_alpha=d_alpha,
        d_omega_rad_s=d_omega,
        theta_target=[float(value) for value in target.theta],
        phi_target=[float(value) for value in target.phi],
        initial_loss=[float(1.0 - row["process_fidelity"]) for row in initial_eval["per_manifold"]],
        optimized_loss=[float(1.0 - row["process_fidelity"]) for row in final_metrics["per_manifold"]],
        levels=levels,
        metadata={
            "benchmark_mode": True,
            "amplitude_correction_convention": "amp = theta/(2T) + lambda0*d_lambda_norm",
            "config_snapshot": _relevant_config_snapshot(config),
            "logical_n": int(target.logical_n),
            "guard_levels": int(target.guard_levels),
            "lambda_guard": float(lambda_guard),
            "weight_mode": str(weight_mode),
            "poisson_alpha": None if poisson_alpha is None else [float(np.real(poisson_alpha)), float(np.imag(poisson_alpha))],
            "optimizer_method_stage1": method_stage1,
            "optimizer_method_stage2": method_stage2,
            "config_hash": _config_hash(config),
        },
    )

    nit_stage1 = int(getattr(stage1, "nit", 0) or 0)
    nit_stage2 = int(getattr(stage2, "nit", 0) or 0)
    objective_evaluations = int(getattr(stage1, "nfev", 0) or 0) + int(getattr(stage2, "nfev", 0) or 0)
    final_best_loss = float(min(best_loss, float(final_metrics["loss_total"])))
    if float(final_metrics["loss_total"]) <= best_loss:
        best_iteration = int(len(trace))
    run_reporter.on_end(
        {
            "progress_schema_version": 1,
            "run_id": run_label,
            "success": bool(stage1.success or stage2.success),
            "message": str(stage2.message if stage2.fun <= stage1.fun else stage1.message),
            "nit": int(nit_stage1 + nit_stage2),
            "best_objective_total": float(final_best_loss),
            "best_iteration": int(best_iteration),
            "timestamp": float(time.time()),
        }
    )
    return GuardedBenchmarkResult(
        target_id=target.target_id,
        target_class=target.target_class,
        duration_s=float(config["duration_sqr_s"]),
        logical_n=int(target.logical_n),
        guard_levels=int(target.guard_levels),
        lambda_guard=float(lambda_guard),
        weight_mode=str(weight_mode),
        poisson_alpha=poisson_alpha,
        logical_fidelity=float(final_metrics["logical_fidelity"]),
        epsilon_guard=float(final_metrics["epsilon_guard"]),
        loss_total=float(final_metrics["loss_total"]),
        success=bool(final_metrics["logical_fidelity"] >= fidelity_threshold and final_metrics["epsilon_guard"] <= guard_threshold),
        converged=bool(stage1.success or stage2.success),
        iterations=nit_stage1 + nit_stage2,
        objective_evaluations=objective_evaluations,
        calibration=calibration,
        per_manifold=final_metrics["per_manifold"],
        convergence_trace=trace,
        metadata={
            "active_levels": active_levels,
            "stage1_success": bool(stage1.success),
            "stage2_success": bool(stage2.success),
            "stage1_message": str(stage1.message),
            "stage2_message": str(stage2.message),
            "stage1_nit": nit_stage1,
            "stage2_nit": nit_stage2,
            "run_id": run_label,
            "simulation_mode": str(final_metrics.get("simulation_mode", "unitary")),
            "execution_path": str(final_metrics.get("execution_path", "cqed_sim.pulses.calibration.build_sqr_tone_specs -> cqed_sim.sim.runner.hamiltonian_time_slices -> qutip.propagator")),
        },
    )


def benchmark_random_sqr_targets_vs_duration(
    config: Mapping[str, Any],
    duration_list_s: Sequence[float],
    targets: Sequence[RandomSQRTarget],
    lambda_guard: float = 0.1,
    weight_mode: str = "uniform",
    poisson_alpha: complex | None = None,
    fidelity_threshold: float = 0.99,
    guard_threshold: float = 1.0e-2,
) -> list[GuardedBenchmarkResult]:
    results: list[GuardedBenchmarkResult] = []
    for duration_s in duration_list_s:
        run_config = dict(config)
        run_config["duration_sqr_s"] = float(duration_s)
        for target in targets:
            results.append(
                calibrate_guarded_sqr_target(
                    target=target,
                    config=run_config,
                    lambda_guard=lambda_guard,
                    weight_mode=weight_mode,
                    poisson_alpha=poisson_alpha,
                    fidelity_threshold=fidelity_threshold,
                    guard_threshold=guard_threshold,
                )
            )
    return results


def benchmark_results_table(results: Sequence[GuardedBenchmarkResult]) -> list[dict[str, Any]]:
    return [result.to_record() for result in results]


def summarize_duration_benchmark(results: Sequence[GuardedBenchmarkResult]) -> list[dict[str, Any]]:
    grouped: dict[float, list[GuardedBenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(float(result.duration_s), []).append(result)
    rows: list[dict[str, Any]] = []
    for duration_s in sorted(grouped):
        group = grouped[duration_s]
        fidelities = np.asarray([row.logical_fidelity for row in group], dtype=float)
        guard = np.asarray([row.epsilon_guard for row in group], dtype=float)
        success = np.asarray([1.0 if row.success else 0.0 for row in group], dtype=float)
        rows.append(
            {
                "T": float(duration_s),
                "n_trials": len(group),
                "f_median": float(np.median(fidelities)),
                "f_q25": float(np.quantile(fidelities, 0.25)),
                "f_q75": float(np.quantile(fidelities, 0.75)),
                "f_min": float(np.min(fidelities)),
                "f_max": float(np.max(fidelities)),
                "guard_median": float(np.median(guard)),
                "guard_q25": float(np.quantile(guard, 0.25)),
                "guard_q75": float(np.quantile(guard, 0.75)),
                "success_rate": float(np.mean(success)),
            }
        )
    return rows
