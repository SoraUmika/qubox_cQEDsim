from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import Bounds, minimize

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import manifold_transition_frequency
from cqed_sim.core.ideal_gates import qubit_rotation_xy, sqr_op
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import RotationGate
from cqed_sim.observables.trajectories import bloch_trajectory_from_states
from cqed_sim.pulses.calibration import build_sqr_tone_specs, sqr_lambda0_rad_s
from cqed_sim.pulses.envelopes import cosine_rise_envelope, normalized_gaussian
from cqed_sim.pulses.builders import build_rotation_pulse
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import CompiledSequence, SequenceCompiler
from cqed_sim.sim.extractors import conditioned_bloch_xyz
from cqed_sim.sim.runner import SimulationConfig, hamiltonian_time_slices, simulate_sequence


MODE_BASIC = "amp_phase"
MODE_EXTENDED = "amp_phase_detuning"
MODE_CHIRP = "amp_phase_detuning_ramp"


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


def polar_unitary(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.complex128))
    return normalize_unitary(u @ vh)


def process_fidelity(target: np.ndarray, simulated: np.ndarray) -> float:
    overlap = np.trace(np.asarray(target).conj().T @ np.asarray(simulated))
    return float(np.abs(overlap) ** 2 / 4.0)


def rotation_axis_parameters(unitary: np.ndarray) -> tuple[float, float, float, float, float]:
    u = normalize_unitary(np.asarray(unitary, dtype=np.complex128))
    trace = np.trace(u)
    cos_half = float(np.clip(np.real(trace / 2.0), -1.0, 1.0))
    theta = float(2.0 * np.arccos(cos_half))
    if theta < 1.0e-12:
        return 0.0, 1.0, 0.0, 0.0, 0.0
    sin_half = float(np.sin(theta / 2.0))
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
    return theta, nx, ny, nz, phi


@dataclass(frozen=True)
class SystemParams:
    n_max: int = 6
    omega_c_hz: float = 0.0
    omega_q_hz: float = 0.0
    qubit_alpha_hz: float = 0.0
    chi_nominal_hz: float = -2.84e6
    chi_easy_hz: float = -5.2e6
    chi_hard_hz: float = -1.0e6
    chi2_hz: float = 0.0
    chi3_hz: float = 0.0
    kerr_hz: float = 0.0
    use_rotating_frame: bool = True

    @property
    def n_cav(self) -> int:
        return int(self.n_max + 1)


@dataclass(frozen=True)
class PulseParams:
    duration_nominal_s: float = 1.0e-6
    duration_short_s: float = 3.5e-7
    duration_long_s: float = 1.7e-6
    sigma_fraction: float = 1.0 / 6.0
    envelope_kind: str = "gaussian"
    flat_top_rise_fraction: float = 0.12
    theta_cutoff: float = 1.0e-9
    include_zero_theta_tones: bool = True
    dt_eval_s: float = 2.0e-9
    dt_opt_s: float = 5.0e-9
    max_step_eval_s: float = 2.0e-9
    max_step_opt_s: float = 4.0e-9
    qutip_nsteps: int = 200000


@dataclass(frozen=True)
class ActiveSupportParams:
    mode: str = "contiguous"
    max_level_active: int = 1
    active_levels: tuple[int, ...] = ()
    active_weights: tuple[float, ...] = ()
    inference_state_label: str | None = None
    state_population_threshold: float = 1.0e-4
    infer_weights_from_state: bool = False
    inactive_weight: float = 0.02
    boundary_leakage_boost: float = 1.6
    include_projected_coherent: bool = True
    include_qubit_superposition: bool = True
    coherent_alpha: complex = complex(0.9, 0.2)


@dataclass(frozen=True)
class SupportObjectiveWeights:
    w_active_infidelity: float = 1.0
    w_active_theta: float = 0.20
    w_active_phase_axis: float = 0.35
    w_active_pre_z: float = 0.25
    w_active_post_z: float = 0.25
    w_active_state_mean: float = 0.55
    w_active_state_min: float = 0.45
    w_phase_superposition: float = 0.35
    w_leak_mean: float = 0.70
    w_leak_max: float = 0.55
    w_worst_block: float = 0.80
    active_block_fidelity_floor: float = 0.98
    active_state_fidelity_floor: float = 0.97


@dataclass(frozen=True)
class OptimizationParams:
    method_stage1: str = "Powell"
    method_stage2: str = "L-BFGS-B"
    maxiter_stage1_basic: int = 12
    maxiter_stage2_basic: int = 20
    maxiter_stage1_extended: int = 14
    maxiter_stage2_extended: int = 24
    maxiter_stage1_chirp: int = 14
    maxiter_stage2_chirp: int = 20
    amp_delta_bounds: tuple[float, float] = (-0.8, 0.8)
    allow_zero_theta_corrections: bool = True
    phase_delta_bounds: tuple[float, float] = (-np.pi, np.pi)
    detuning_hz_bounds: tuple[float, float] = (-2.4e6, 2.4e6)
    phase_ramp_hz_bounds: tuple[float, float] = (-1.8e6, 1.8e6)
    w_infid: float = 1.0
    w_phase: float = 0.25
    w_theta: float = 0.18
    w_residual_z: float = 0.22
    w_state: float = 0.35
    w_off_block: float = 0.18
    w_selectivity_mean: float = 0.45
    w_selectivity_max: float = 0.25
    objective_scope: str = "global"
    support_weights: SupportObjectiveWeights = field(default_factory=SupportObjectiveWeights)
    reg_amp: float = 1.0e-3
    reg_phase: float = 1.0e-3
    reg_detuning: float = 6.0e-4
    reg_phase_ramp: float = 4.0e-4


@dataclass(frozen=True)
class StudyParams:
    seed: int = 17
    theta_max_rad: float = float(0.92 * np.pi)
    coherent_alpha: complex = complex(1.1, 0.25)
    include_case_e: bool = True
    run_profiles: tuple[str, ...] = ("structured", "hard_random")
    output_dir: Path = Path("outputs/sqr_multitone_study")
    system: SystemParams = field(default_factory=SystemParams)
    pulse: PulseParams = field(default_factory=PulseParams)
    active_support: ActiveSupportParams = field(default_factory=ActiveSupportParams)
    optimization: OptimizationParams = field(default_factory=OptimizationParams)


@dataclass(frozen=True)
class TargetProfile:
    name: str
    mode: str
    theta: np.ndarray
    phi: np.ndarray
    seed: int

    @property
    def n_levels(self) -> int:
        return int(self.theta.size)


@dataclass(frozen=True)
class ToneControl:
    manifold: int
    omega_rad_s: float
    amp_rad_s: float
    phase_rad: float
    detuning_rad_s: float = 0.0
    phase_ramp_rad_s: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "manifold": int(self.manifold),
            "omega_rad_s": float(self.omega_rad_s),
            "amp_rad_s": float(self.amp_rad_s),
            "phase_rad": float(self.phase_rad),
            "detuning_rad_s": float(self.detuning_rad_s),
            "phase_ramp_rad_s": float(self.phase_ramp_rad_s),
        }


@dataclass
class CaseResult:
    case_id: str
    description: str
    controls: list[ToneControl]
    pulse: Pulse
    compiled: CompiledSequence
    unitary_qobj: qt.Qobj
    unitary: np.ndarray
    block_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    state_fidelities: dict[str, float]
    optimization_trace: list[dict[str, float]] = field(default_factory=list)
    optimization_status: dict[str, Any] | None = None


@dataclass
class ConventionAuditResult:
    verdict: str
    envelope_interpretation: str
    pure_i: dict[str, Any]
    pure_q: dict[str, Any]
    phase_sweep: list[dict[str, Any]]
    detuning_check: dict[str, Any]
    multitone_consistency: dict[str, Any]
    notes: list[str]


def build_model_and_frame(system: SystemParams, chi_hz: float) -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
    chi_higher = tuple(
        value
        for value in (
            hz_to_rad_s(float(system.chi2_hz)),
            hz_to_rad_s(float(system.chi3_hz)),
        )
        if abs(value) > 0.0
    )
    model = DispersiveTransmonCavityModel(
        omega_c=hz_to_rad_s(float(system.omega_c_hz)),
        omega_q=hz_to_rad_s(float(system.omega_q_hz)),
        alpha=hz_to_rad_s(float(system.qubit_alpha_hz)),
        chi=hz_to_rad_s(float(chi_hz)),
        chi_higher=chi_higher,
        kerr=hz_to_rad_s(float(system.kerr_hz)),
        n_cav=int(system.n_cav),
        n_tr=2,
    )
    frame = FrameSpec()
    if bool(system.use_rotating_frame):
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return model, frame


def _scale_to_theta_max(values: np.ndarray, theta_max: float, rng: np.random.Generator, max_scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs < 1.0e-12:
        return np.zeros_like(values)
    scale = float(rng.uniform(0.35, max_scale)) * float(theta_max) / max_abs
    return np.clip(scale * values, -theta_max, theta_max)


def build_target_profile(mode: str, n_levels: int, seed: int, theta_max: float) -> TargetProfile:
    rng = np.random.default_rng(int(seed))
    n = np.arange(int(n_levels), dtype=float)
    if mode == "structured":
        raw_theta = (
            0.92 * np.sin(2.0 * np.pi * (n + 0.2) / max(n_levels, 1))
            + 0.34 * np.sin(4.0 * np.pi * (n + 0.08) / max(n_levels, 1))
            + 0.12 * rng.normal(size=n_levels)
        )
        theta = _scale_to_theta_max(raw_theta, theta_max=float(theta_max), rng=rng, max_scale=0.82)
        phi = np.mod(
            0.2 * np.pi
            + 0.6 * np.pi * n / max(n_levels - 1, 1)
            + 0.22 * np.sin(2.0 * np.pi * n / max(n_levels, 1) + rng.normal()),
            2.0 * np.pi,
        )
        return TargetProfile(name=f"{mode}_seed{seed}", mode=mode, theta=theta, phi=phi, seed=int(seed))

    if mode == "hard_random":
        theta = rng.uniform(-theta_max, theta_max, size=n_levels)
        theta += 0.20 * theta_max * ((-1.0) ** np.arange(n_levels))
        theta = np.clip(theta + 0.10 * theta_max * rng.normal(size=n_levels), -theta_max, theta_max)
        phi = np.mod(
            rng.uniform(0.0, 2.0 * np.pi, size=n_levels)
            + 0.85 * ((-1.0) ** np.arange(n_levels)),
            2.0 * np.pi,
        )
        return TargetProfile(name=f"{mode}_seed{seed}", mode=mode, theta=theta, phi=phi, seed=int(seed))

    if mode == "fully_random":
        theta = rng.uniform(-theta_max, theta_max, size=n_levels)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_levels)
        return TargetProfile(name=f"{mode}_seed{seed}", mode=mode, theta=theta, phi=phi, seed=int(seed))

    if mode == "moderate_random":
        theta_span = float(0.62 * theta_max)
        theta = rng.uniform(-theta_span, theta_span, size=n_levels)
        theta = np.clip(theta + 0.08 * theta_span * rng.normal(size=n_levels), -theta_span, theta_span)
        phase_walk = np.cumsum(rng.normal(loc=0.0, scale=0.42, size=n_levels))
        phi = np.mod(0.45 * np.pi + phase_walk + 0.18 * rng.normal(size=n_levels), 2.0 * np.pi)
        return TargetProfile(name=f"{mode}_seed{seed}", mode=mode, theta=theta, phi=phi, seed=int(seed))

    raise ValueError(f"Unsupported mode '{mode}'.")


def build_target_blocks(profile: TargetProfile) -> dict[int, np.ndarray]:
    blocks: dict[int, np.ndarray] = {}
    for n in range(profile.n_levels):
        block = np.asarray(qubit_rotation_xy(float(profile.theta[n]), float(profile.phi[n])).full(), dtype=np.complex128)
        blocks[int(n)] = normalize_unitary(block)
    return blocks


def build_target_unitary(profile: TargetProfile) -> qt.Qobj:
    return sqr_op(profile.theta, profile.phi)


def extract_block_unitaries(unitary: np.ndarray, n_levels: int) -> tuple[dict[int, np.ndarray], float]:
    matrix = np.asarray(unitary, dtype=np.complex128)
    blocks: dict[int, np.ndarray] = {}
    mask = np.zeros_like(matrix, dtype=bool)
    for n in range(int(n_levels)):
        idx = slice(2 * n, 2 * n + 2)
        block = np.asarray(matrix[idx, idx], dtype=np.complex128)
        u, _, vh = np.linalg.svd(block)
        blocks[n] = u @ vh
        mask[idx, idx] = True
    off_block = matrix[~mask]
    off_block_norm = float(np.linalg.norm(off_block))
    return blocks, off_block_norm


def _block_rows_from_unitary(blocks: dict[int, np.ndarray], target_blocks: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in sorted(target_blocks):
        target = normalize_unitary(target_blocks[n])
        simulated_raw = np.asarray(blocks[n], dtype=np.complex128)
        simulated = normalize_unitary(simulated_raw)
        fidelity = process_fidelity(target, simulated)
        p_target = float(np.abs(target[1, 0]) ** 2)
        p_sim = float(np.abs(simulated[1, 0]) ** 2)
        theta_t, nx_t, ny_t, nz_t, phi_t = rotation_axis_parameters(target)
        theta_s, nx_s, ny_s, nz_s, phi_s = rotation_axis_parameters(simulated)
        error_u = normalize_unitary(target.conj().T @ simulated)
        theta_e, nx_e, ny_e, nz_e, phi_e = rotation_axis_parameters(error_u)
        phi_axis_error = float(wrap_pi(phi_s - phi_t))
        rows.append(
            {
                "n": int(n),
                "process_fidelity": float(np.clip(fidelity, 0.0, 1.0)),
                "process_infidelity": float(np.clip(1.0 - fidelity, 0.0, 1.0)),
                "population_target": p_target,
                "population_simulated": p_sim,
                "population_error": float(abs(p_sim - p_target)),
                "theta_target_rad": float(theta_t),
                "phi_target_rad": float(phi_t),
                "axis_target_x": float(nx_t),
                "axis_target_y": float(ny_t),
                "axis_target_z": float(nz_t),
                "theta_simulated_rad": float(theta_s),
                "phi_simulated_rad": float(phi_s),
                "phi_axis_error_rad": float(phi_axis_error),
                "axis_simulated_x": float(nx_s),
                "axis_simulated_y": float(ny_s),
                "axis_simulated_z": float(nz_s),
                "error_theta_rad": float(theta_e),
                "error_phi_rad": float(phi_e),
                "error_axis_x": float(nx_e),
                "error_axis_y": float(ny_e),
                "error_axis_z": float(nz_e),
                "residual_conditional_z_rad": float(theta_e * nz_e),
                "block_global_phase_rad": float(0.5 * np.angle(np.linalg.det(simulated_raw))),
            }
        )
    if not rows:
        return rows
    phase_ref = float(rows[0]["block_global_phase_rad"])
    for row in rows:
        row["relative_block_phase_rad"] = float(wrap_pi(row["block_global_phase_rad"] - phase_ref))
    return rows


def summarize_block_rows(rows: list[dict[str, Any]], off_block_norm: float) -> dict[str, Any]:
    if not rows:
        return {
            "n_levels": 0,
            "mean_process_fidelity": float("nan"),
            "min_process_fidelity": float("nan"),
            "mean_infidelity": float("nan"),
            "mean_population_error": float("nan"),
            "phase_rms_rad": float("nan"),
            "theta_rms_rad": float("nan"),
            "residual_z_rms_rad": float("nan"),
            "off_block_norm": float(off_block_norm),
            "population_phase_mismatch_example": None,
        }
    fidelity = np.asarray([row["process_fidelity"] for row in rows], dtype=float)
    infidelity = np.asarray([row["process_infidelity"] for row in rows], dtype=float)
    pop_err = np.asarray([row["population_error"] for row in rows], dtype=float)
    phase_axis = np.asarray([row["phi_axis_error_rad"] for row in rows], dtype=float)
    relative_phase = np.asarray([row["relative_block_phase_rad"] for row in rows], dtype=float)
    theta = np.asarray([row["error_theta_rad"] for row in rows], dtype=float)
    rz = np.asarray([row["residual_conditional_z_rad"] for row in rows], dtype=float)
    candidates = [row for row in rows if row["population_error"] < 0.03 and row["process_infidelity"] > 0.02]
    mismatch = None
    if candidates:
        pick = max(candidates, key=lambda row: row["process_infidelity"])
        mismatch = {
            "n": int(pick["n"]),
            "population_error": float(pick["population_error"]),
            "process_infidelity": float(pick["process_infidelity"]),
            "phi_axis_error_rad": float(pick["phi_axis_error_rad"]),
            "relative_block_phase_rad": float(pick["relative_block_phase_rad"]),
            "residual_conditional_z_rad": float(pick["residual_conditional_z_rad"]),
        }
    return {
        "n_levels": int(len(rows)),
        "mean_process_fidelity": float(np.mean(fidelity)),
        "min_process_fidelity": float(np.min(fidelity)),
        "mean_infidelity": float(np.mean(infidelity)),
        "mean_population_error": float(np.mean(pop_err)),
        "phase_rms_rad": float(np.sqrt(np.mean(phase_axis**2))),
        "phase_axis_rms_rad": float(np.sqrt(np.mean(phase_axis**2))),
        "relative_block_phase_rms_rad": float(np.sqrt(np.mean(relative_phase**2))),
        "theta_rms_rad": float(np.sqrt(np.mean(theta**2))),
        "residual_z_rms_rad": float(np.sqrt(np.mean(rz**2))),
        "max_abs_phase_axis_error_rad": float(np.max(np.abs(phase_axis))),
        "max_abs_relative_phase_rad": float(np.max(np.abs(relative_phase))),
        "off_block_norm": float(off_block_norm),
        "population_phase_mismatch_example": mismatch,
    }

def build_window(t_rel: np.ndarray, pulse: PulseParams) -> np.ndarray:
    if pulse.envelope_kind == "gaussian":
        return np.asarray(normalized_gaussian(t_rel, sigma_fraction=float(pulse.sigma_fraction)), dtype=np.complex128)
    if pulse.envelope_kind == "flat_top":
        window = np.asarray(cosine_rise_envelope(t_rel, rise_fraction=float(pulse.flat_top_rise_fraction)), dtype=np.complex128)
        area = float(np.trapz(np.real(window), t_rel))
        return window if abs(area) < 1.0e-12 else window / area
    raise ValueError(f"Unsupported envelope_kind '{pulse.envelope_kind}'.")


def multitone_envelope(t_rel: np.ndarray, duration_s: float, controls: list[ToneControl], pulse: PulseParams) -> np.ndarray:
    t_rel = np.asarray(t_rel, dtype=float)
    t = t_rel * float(duration_s)
    t_center = 0.5 * float(duration_s)
    env = build_window(t_rel, pulse)
    coeff = np.zeros_like(t, dtype=np.complex128)
    for tone in controls:
        phase_dyn = float(tone.phase_rad) + float(tone.phase_ramp_rad_s) * (t - t_center)
        omega = float(tone.omega_rad_s + tone.detuning_rad_s)
        coeff += float(tone.amp_rad_s) * np.exp(1j * phase_dyn) * np.exp(1j * omega * t)
    return env * coeff


def build_pulse_from_controls(controls: list[ToneControl], duration_s: float, pulse: PulseParams, label: str) -> Pulse:
    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return multitone_envelope(t_rel=t_rel, duration_s=float(duration_s), controls=controls, pulse=pulse)

    return Pulse(channel="qubit", t0=0.0, duration=float(duration_s), envelope=envelope, amp=1.0, phase=0.0, label=label)


def compile_single_pulse(pulse: Pulse, dt_s: float) -> CompiledSequence:
    return SequenceCompiler(dt=float(dt_s)).compile([pulse], t_end=float(pulse.t1 + dt_s))


def propagate_pulse_unitary(
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    compiled: CompiledSequence,
    max_step_s: float,
    qutip_nsteps: int,
) -> qt.Qobj:
    h = hamiltonian_time_slices(model, compiled, {"qubit": "qubit"}, frame=frame)
    options = {"atol": 1.0e-8, "rtol": 1.0e-7, "nsteps": int(qutip_nsteps)}
    if float(max_step_s) > 0.0:
        options["max_step"] = float(max_step_s)
    propagators = qt.propagator(h, compiled.tlist, options=options, tlist=compiled.tlist)
    return propagators[-1] if isinstance(propagators, list) else propagators


def build_controls_from_target(
    profile: TargetProfile,
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    duration_s: float,
    theta_cutoff: float,
    include_all_levels: bool = True,
) -> list[ToneControl]:
    raw_specs = build_sqr_tone_specs(
        model=model,
        frame=frame,
        theta_values=profile.theta.tolist(),
        phi_values=profile.phi.tolist(),
        duration_s=float(duration_s),
        include_all_levels=bool(include_all_levels),
        tone_cutoff=float(theta_cutoff),
    )
    return [
        ToneControl(
            manifold=int(spec.manifold),
            omega_rad_s=float(spec.omega_rad_s),
            amp_rad_s=float(spec.amp_rad_s),
            phase_rad=float(spec.phase_rad),
            detuning_rad_s=0.0,
            phase_ramp_rad_s=0.0,
        )
        for spec in raw_specs
    ]


def build_reference_states(
    model: DispersiveTransmonCavityModel,
    n_max: int,
    coherent_alpha: complex,
) -> dict[str, qt.Qobj]:
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    states: dict[str, qt.Qobj] = {}
    states["g,0"] = qt.tensor(g, qt.basis(model.n_cav, 0))
    chosen_n = sorted(set([1, min(3, n_max), n_max]))
    for n in chosen_n:
        if 0 <= n <= n_max:
            states[f"g,{n}"] = qt.tensor(g, qt.basis(model.n_cav, int(n)))
    if n_max >= 1:
        psi = (qt.basis(model.n_cav, 0) + qt.basis(model.n_cav, 1)).unit()
        states["g x (|0>+|1>)/sqrt2"] = qt.tensor(g, psi)
    even_levels = [n for n in (0, 2, 4) if n <= n_max]
    if len(even_levels) >= 2:
        super_even = sum(qt.basis(model.n_cav, n) for n in even_levels).unit()
        states[f"g x ({' + '.join(f'|{n}>' for n in even_levels)})"] = qt.tensor(g, super_even)
    states[f"g x |alpha={coherent_alpha.real:+.2f}{coherent_alpha.imag:+.2f}j>"] = qt.tensor(
        g, qt.coherent(model.n_cav, coherent_alpha)
    )
    if n_max >= 2:
        cav = (qt.basis(model.n_cav, 0) + qt.basis(model.n_cav, 2)).unit()
    else:
        cav = (qt.basis(model.n_cav, 0) + qt.basis(model.n_cav, 1)).unit()
    states["(|g>+|e>)/sqrt2 x cavity_superposition"] = qt.tensor((g + e).unit(), cav)
    return states


def select_support_reference_state(
    reference_states: dict[str, qt.Qobj],
    support: ActiveSupportParams,
) -> qt.Qobj | None:
    if not reference_states:
        return None
    if support.inference_state_label is not None:
        label = str(support.inference_state_label)
        if label in reference_states:
            return reference_states[label]
    for label in ("g x (|0>+|1>)/sqrt2", "g,0", "(|g>+|e>)/sqrt2 x cavity_superposition"):
        if label in reference_states:
            return reference_states[label]
    return next(iter(reference_states.values()))


def cavity_level_populations(state: qt.Qobj, n_levels: int) -> np.ndarray:
    rho = state if state.isoper else state.proj()
    rho_cav = qt.ptrace(rho, 1)
    diag = np.real(np.diag(np.asarray(rho_cav.full(), dtype=np.complex128)))
    n = min(int(n_levels), int(diag.size))
    out = np.clip(np.asarray(diag[:n], dtype=float), 0.0, None)
    s = float(np.sum(out))
    return out if s <= 1.0e-15 else out / s


def resolve_active_support(
    n_levels: int,
    support: ActiveSupportParams,
    reference_state: qt.Qobj | None = None,
) -> tuple[list[int], dict[int, float], list[int]]:
    mode = str(support.mode).strip().lower()
    if mode == "contiguous":
        max_level = int(max(0, support.max_level_active))
        levels = [n for n in range(min(int(n_levels), max_level + 1))]
    elif mode == "explicit":
        levels = sorted({int(n) for n in support.active_levels if 0 <= int(n) < int(n_levels)})
    elif mode == "from_state":
        if reference_state is None:
            raise ValueError("ActiveSupportParams.mode='from_state' requires a reference_state.")
        pops = cavity_level_populations(reference_state, n_levels=int(n_levels))
        threshold = float(max(0.0, support.state_population_threshold))
        levels = [int(n) for n in range(int(pops.size)) if float(pops[n]) >= threshold]
        if not levels and pops.size > 0:
            levels = [int(np.argmax(pops))]
    else:
        raise ValueError(f"Unsupported active-support mode '{support.mode}'.")
    if not levels:
        raise ValueError("Active support resolved to empty set.")

    raw_weights = np.ones(len(levels), dtype=float)
    if support.active_weights:
        if len(support.active_weights) != len(levels):
            raise ValueError("active_weights length must match resolved active support size.")
        raw_weights = np.asarray(support.active_weights, dtype=float)
    elif reference_state is not None and (bool(support.infer_weights_from_state) or mode == "from_state"):
        pops = cavity_level_populations(reference_state, n_levels=int(n_levels))
        raw_weights = np.asarray([float(pops[int(n)]) for n in levels], dtype=float)
        raw_weights = np.clip(raw_weights, 1.0e-12, None)
    raw_weights = np.clip(raw_weights, 1.0e-12, None)
    norm = float(np.sum(raw_weights))
    weights = raw_weights / norm
    weight_by_level = {int(level): float(weights[idx]) for idx, level in enumerate(levels)}
    inactive = [n for n in range(int(n_levels)) if n not in set(levels)]
    return levels, weight_by_level, inactive


def build_active_support_ensemble(
    model: DispersiveTransmonCavityModel,
    active_levels: list[int],
    support: ActiveSupportParams,
    active_weights: dict[int, float] | None = None,
) -> dict[str, qt.Qobj]:
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    states: dict[str, qt.Qobj] = {}
    for n in active_levels:
        states[f"active_g,{n}"] = qt.tensor(g, qt.basis(model.n_cav, int(n)))

    pair_candidates: list[tuple[int, int]] = []
    if len(active_levels) >= 2:
        pair_candidates.append((active_levels[0], active_levels[1]))
        pair_candidates.append((active_levels[0], active_levels[-1]))
    if len(active_levels) >= 4:
        pair_candidates.append((active_levels[1], active_levels[2]))
    unique_pairs: list[tuple[int, int]] = []
    for n, m in pair_candidates:
        if n == m:
            continue
        pair = tuple(sorted((int(n), int(m))))
        if pair not in unique_pairs:
            unique_pairs.append(pair)
    for n, m in unique_pairs:
        cav = (qt.basis(model.n_cav, n) + qt.basis(model.n_cav, m)).unit()
        states[f"sup_g,({n}+{m})"] = qt.tensor(g, cav)

    if len(active_levels) >= 2:
        cav_uniform = sum(qt.basis(model.n_cav, int(n)) for n in active_levels).unit()
        states["sup_g,uniform_active"] = qt.tensor(g, cav_uniform)
        if bool(support.include_qubit_superposition):
            states["sup_ge,uniform_active"] = qt.tensor((g + e).unit(), cav_uniform)
        if active_weights is not None and active_weights:
            amps = [
                float(np.sqrt(max(active_weights.get(int(n), 0.0), 1.0e-12))) * qt.basis(model.n_cav, int(n))
                for n in active_levels
            ]
            cav_weighted = sum(amps).unit()
            states["sup_g,weighted_active"] = qt.tensor(g, cav_weighted)
            if bool(support.include_qubit_superposition):
                states["sup_ge,weighted_active"] = qt.tensor((g + e).unit(), cav_weighted)

    if bool(support.include_projected_coherent):
        cav = qt.coherent(model.n_cav, support.coherent_alpha)
        proj = sum(qt.basis(model.n_cav, int(n)) * qt.basis(model.n_cav, int(n)).dag() for n in active_levels)
        cav_proj = proj * cav
        if float(cav_proj.norm()) > 1.0e-10:
            states[
                f"coherent_active,{support.coherent_alpha.real:+.2f}{support.coherent_alpha.imag:+.2f}j"
            ] = qt.tensor(g, cav_proj.unit())

    return states


def _unitary_block(matrix: np.ndarray, n: int) -> np.ndarray:
    idx = slice(2 * int(n), 2 * int(n) + 2)
    return np.asarray(matrix[idx, idx], dtype=np.complex128)


def _z_unitary(angle: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * float(angle)), 0.0], [0.0, np.exp(0.5j * float(angle))]],
        dtype=np.complex128,
    )


def _rxy_matrix(theta: float, phi: float) -> np.ndarray:
    return np.asarray(qubit_rotation_xy(float(theta), float(phi)).full(), dtype=np.complex128)


def _z_rxy_z(alpha: float, theta: float, phi: float, beta: float) -> np.ndarray:
    return _z_unitary(alpha) @ _rxy_matrix(theta, phi) @ _z_unitary(beta)


def decompose_z_rxy_z(unitary: np.ndarray) -> dict[str, float]:
    target = normalize_unitary(np.asarray(unitary, dtype=np.complex128))
    bounds = Bounds(
        np.array([-np.pi, 0.0, -np.pi, -np.pi], dtype=float),
        np.array([np.pi, np.pi, np.pi, np.pi], dtype=float),
    )

    def objective(vec: np.ndarray) -> float:
        alpha, theta, phi, beta = map(float, vec)
        recon = _z_rxy_z(alpha, theta, phi, beta)
        return float(np.linalg.norm(target - recon) ** 2 + 1.0e-6 * (alpha * alpha + beta * beta))

    seeds = (
        np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, np.pi / 2.0, 0.0, 0.0], dtype=float),
        np.array([0.4, np.pi / 3.0, 0.2, -0.2], dtype=float),
    )
    best = None
    for seed in seeds:
        result = minimize(objective, x0=seed, method="L-BFGS-B", bounds=bounds)
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    alpha, theta, phi, beta = map(float, best.x)
    return {
        "alpha_rad": float(wrap_pi(alpha)),
        "theta_rad": float(theta),
        "phi_rad": float(wrap_pi(phi)),
        "beta_rad": float(wrap_pi(beta)),
        "fit_error": float(best.fun),
    }


def _active_phase_profile(state: qt.Qobj, active_levels: list[int], branch: str = "ground") -> dict[int, float]:
    qubit_idx = 0 if branch == "ground" else 1
    amplitudes: dict[int, complex] = {}
    for n in active_levels:
        n_cav = int(state.dims[0][1])
        ket = qt.tensor(qt.basis(2, qubit_idx), qt.basis(n_cav, int(n)))
        value = ket.dag() * state
        amplitudes[int(n)] = complex(value.full()[0, 0]) if isinstance(value, qt.Qobj) else complex(value)
    ref_level = None
    for n in active_levels:
        if abs(amplitudes[int(n)]) > 1.0e-10:
            ref_level = int(n)
            break
    if ref_level is None:
        return {}
    ref = amplitudes[ref_level]
    out: dict[int, float] = {}
    for n in active_levels:
        amp = amplitudes[int(n)]
        if abs(amp) <= 1.0e-10:
            continue
        out[int(n)] = float(np.angle(amp / ref))
    return out


def _phase_proxy_error_for_superposition(
    ideal_state: qt.Qobj,
    simulated_state: qt.Qobj,
    active_levels: list[int],
) -> float:
    phase_ideal = _active_phase_profile(ideal_state, active_levels=active_levels, branch="ground")
    phase_sim = _active_phase_profile(simulated_state, active_levels=active_levels, branch="ground")
    common = [n for n in active_levels if n in phase_ideal and n in phase_sim]
    if len(common) < 2:
        return float("nan")
    err = np.asarray([wrap_pi(phase_sim[n] - phase_ideal[n]) for n in common], dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def support_state_leakage_metrics(
    states: dict[str, qt.Qobj],
    active_levels: list[int],
    n_cav: int,
) -> dict[str, Any]:
    if not states:
        return {"mean": float("nan"), "max": float("nan"), "by_state": {}}
    projector = sum(qt.basis(n_cav, int(n)) * qt.basis(n_cav, int(n)).dag() for n in active_levels)
    p_active = qt.tensor( qt.qeye(2),projector)
    by_state: dict[str, float] = {}
    values: list[float] = []
    for label, state in states.items():
        rho = state if state.isoper else state.proj()
        p_in = float(np.real((rho * p_active).tr()))
        leakage = float(max(0.0, 1.0 - p_in))
        by_state[label] = leakage
        values.append(leakage)
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "by_state": by_state,
    }


def apply_unitary_to_states(unitary: qt.Qobj, states: dict[str, qt.Qobj]) -> dict[str, qt.Qobj]:
    return {label: unitary * state for label, state in states.items()}


def state_fidelity_vs_ideal(ideal: dict[str, qt.Qobj], simulated: dict[str, qt.Qobj]) -> dict[str, float]:
    fidelities: dict[str, float] = {}
    for label, ideal_state in ideal.items():
        sim_state = simulated[label]
        fidelities[label] = float(abs(ideal_state.overlap(sim_state)) ** 2)
    return fidelities


def conditioned_bloch_snapshot(state: qt.Qobj, n_levels: int, threshold: float = 1.0e-8) -> list[dict[str, Any]]:
    rows = []
    for n in range(int(n_levels)):
        bx, by, bz, prob, valid = conditioned_bloch_xyz(state, n=n, fallback="nan")
        rows.append(
            {
                "n": int(n),
                "probability": float(prob),
                "valid": bool(valid and prob >= threshold),
                "x": float(bx),
                "y": float(by),
                "z": float(bz),
            }
        )
    return rows


def relative_phase_vector(state: qt.Qobj, n_levels: int, branch: str = "ground", threshold: float = 1.0e-10) -> list[dict[str, Any]]:
    rho = state if state.isoper else state.proj()
    n_cav = int(rho.dims[0][1])
    n_levels = min(int(n_levels), n_cav)
    ref = qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0))

    def scalar(value: Any) -> complex:
        if isinstance(value, qt.Qobj):
            return complex(value.full()[0, 0])
        return complex(value)

    p_ref = float(np.real(scalar(ref.dag() * rho * ref)))
    rows = []
    qubit = 0 if branch == "ground" else 1
    for n in range(n_levels):
        ket = qt.tensor( qt.basis(2, qubit),qt.basis(n_cav, n))
        p_target = float(np.real(scalar(ket.dag() * rho * ket)))
        coherence = scalar(ket.dag() * rho * ref)
        valid = bool(p_ref >= threshold and p_target >= threshold and abs(coherence) >= threshold)
        rows.append(
            {
                "n": int(n),
                "probability_target": p_target,
                "coherence_abs": float(abs(coherence)),
                "phase_rad": float(np.angle(coherence)) if valid else float("nan"),
                "valid": valid,
            }
        )
    return rows


def trajectory_relative_phase(
    states: list[qt.Qobj],
    n_values: list[int],
    branch: str = "ground",
    threshold: float = 1.0e-10,
) -> dict[int, np.ndarray]:
    out = {int(n): np.full(len(states), np.nan, dtype=float) for n in n_values}
    for idx, state in enumerate(states):
        phase_rows = relative_phase_vector(state, n_levels=max(n_values) + 1, branch=branch, threshold=threshold)
        by_n = {row["n"]: row for row in phase_rows}
        for n in n_values:
            row = by_n[int(n)]
            if row["valid"]:
                out[int(n)][idx] = float(row["phase_rad"])
    for n in n_values:
        valid = np.isfinite(out[int(n)])
        if np.count_nonzero(valid) >= 2:
            out[int(n)][valid] = np.unwrap(out[int(n)][valid])
    return out


def build_case(
    case_id: str,
    description: str,
    controls: list[ToneControl],
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    profile: TargetProfile,
    pulse_params: PulseParams,
    duration_s: float,
    dt_s: float,
    max_step_s: float,
    reference_states: dict[str, qt.Qobj],
    ideal_state_outputs: dict[str, qt.Qobj],
) -> CaseResult:
    pulse = build_pulse_from_controls(controls, duration_s=float(duration_s), pulse=pulse_params, label=case_id)
    compiled = compile_single_pulse(pulse, dt_s=float(dt_s))
    unitary_qobj = propagate_pulse_unitary(
        model=model,
        frame=frame,
        compiled=compiled,
        max_step_s=float(max_step_s),
        qutip_nsteps=int(pulse_params.qutip_nsteps),
    )
    unitary = np.asarray(unitary_qobj.full(), dtype=np.complex128)
    blocks, off_block_norm = extract_block_unitaries(unitary, n_levels=profile.n_levels)
    rows = _block_rows_from_unitary(blocks, build_target_blocks(profile))
    summary = summarize_block_rows(rows, off_block_norm=off_block_norm)
    selectivity = neighbor_selectivity_proxy(
        controls=controls,
        model=model,
        frame=frame,
        pulse=pulse_params,
        duration_s=float(duration_s),
        n_levels=int(profile.n_levels),
    )
    summary["neighbor_overlap_proxy_mean"] = float(selectivity["mean_neighbor_overlap"])
    summary["neighbor_overlap_proxy_max"] = float(selectivity["max_neighbor_overlap"])
    simulated_state_outputs = apply_unitary_to_states(unitary_qobj, reference_states)
    state_fidelity = state_fidelity_vs_ideal(ideal_state_outputs, simulated_state_outputs)
    summary["state_fidelity_mean"] = float(np.mean(list(state_fidelity.values())))
    summary["state_fidelity_min"] = float(np.min(list(state_fidelity.values())))
    return CaseResult(
        case_id=case_id,
        description=description,
        controls=controls,
        pulse=pulse,
        compiled=compiled,
        unitary_qobj=unitary_qobj,
        unitary=unitary,
        block_rows=rows,
        summary=summary,
        state_fidelities=state_fidelity,
    )


def active_controls(
    controls: list[ToneControl],
    theta_cutoff: float = 1.0e-12,
    include_zero_amp: bool = False,
) -> list[int]:
    if bool(include_zero_amp):
        return list(range(len(controls)))
    return [idx for idx, tone in enumerate(controls) if abs(float(tone.amp_rad_s)) > float(theta_cutoff)]


def parameter_layout(mode: str, active_indices: list[int], opt: OptimizationParams) -> tuple[np.ndarray, Bounds]:
    x0 = np.zeros(
        len(active_indices) * (2 if mode == MODE_BASIC else (3 if mode == MODE_EXTENDED else 4)),
        dtype=float,
    )
    lb: list[float] = []
    ub: list[float] = []
    for _ in active_indices:
        lb.append(float(opt.amp_delta_bounds[0]))
        ub.append(float(opt.amp_delta_bounds[1]))
        lb.append(float(opt.phase_delta_bounds[0]))
        ub.append(float(opt.phase_delta_bounds[1]))
        if mode in (MODE_EXTENDED, MODE_CHIRP):
            lb.append(float(hz_to_rad_s(opt.detuning_hz_bounds[0])))
            ub.append(float(hz_to_rad_s(opt.detuning_hz_bounds[1])))
        if mode == MODE_CHIRP:
            lb.append(float(hz_to_rad_s(opt.phase_ramp_hz_bounds[0])))
            ub.append(float(hz_to_rad_s(opt.phase_ramp_hz_bounds[1])))
    return x0, Bounds(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))


def controls_from_vector(
    base: list[ToneControl],
    active_indices: list[int],
    mode: str,
    vector: np.ndarray,
    duration_s: float,
) -> list[ToneControl]:
    out: list[ToneControl] = []
    cursor = 0
    lambda0 = sqr_lambda0_rad_s(duration_s)
    for idx, tone in enumerate(base):
        if idx not in active_indices:
            out.append(tone)
            continue
        amp_delta = float(vector[cursor])
        phase_delta = float(vector[cursor + 1])
        cursor += 2
        detuning = 0.0
        ramp = 0.0
        if mode in (MODE_EXTENDED, MODE_CHIRP):
            detuning = float(vector[cursor])
            cursor += 1
        if mode == MODE_CHIRP:
            ramp = float(vector[cursor])
            cursor += 1
        out.append(
            ToneControl(
                manifold=int(tone.manifold),
                omega_rad_s=float(tone.omega_rad_s),
                # Lab-aligned additive correction in normalized lambda0 units:
                # amp = base_amp + lambda0 * delta_lambda_norm.
                amp_rad_s=float(tone.amp_rad_s + lambda0 * amp_delta),
                phase_rad=float(tone.phase_rad + phase_delta),
                detuning_rad_s=detuning,
                phase_ramp_rad_s=ramp,
            )
        )
    return out


def regularization_cost(mode: str, vector: np.ndarray, opt: OptimizationParams) -> float:
    vec = np.asarray(vector, dtype=float)
    cursor = 0
    value = 0.0
    width = 2 if mode == MODE_BASIC else (3 if mode == MODE_EXTENDED else 4)
    for _ in range(int(len(vec) / width)):
        amp_delta = float(vec[cursor])
        phase_delta = float(vec[cursor + 1])
        cursor += 2
        value += float(opt.reg_amp * (amp_delta**2))
        value += float(opt.reg_phase * (phase_delta**2))
        if mode in (MODE_EXTENDED, MODE_CHIRP):
            detuning = float(vec[cursor])
            cursor += 1
            value += float(opt.reg_detuning * (rad_s_to_hz(detuning) / 1.0e6) ** 2)
        if mode == MODE_CHIRP:
            ramp = float(vec[cursor])
            cursor += 1
            value += float(opt.reg_phase_ramp * (rad_s_to_hz(ramp) / 1.0e6) ** 2)
    return float(value)


def sqr_manifold_waveform_frequency_rad_s(
    model: DispersiveTransmonCavityModel,
    manifold_n: int,
    frame: FrameSpec,
) -> float:
    return -float(manifold_transition_frequency(model, int(manifold_n), frame=frame))


def _spectral_overlap_proxy(delta_rad_s: float, pulse: PulseParams, duration_s: float) -> float:
    delta = float(delta_rad_s)
    duration = float(max(duration_s, 1.0e-12))
    if pulse.envelope_kind == "gaussian":
        sigma_t = float(max(pulse.sigma_fraction * duration, 1.0e-12))
        return float(np.exp(-0.5 * (delta * sigma_t) ** 2))
    if pulse.envelope_kind == "flat_top":
        rise = float(max(min(pulse.flat_top_rise_fraction, 0.49), 1.0e-4) * duration)
        flat = float(max(duration - 2.0 * rise, 1.0e-12))
        sinc = float(abs(np.sinc(delta * flat / (2.0 * np.pi))))
        taper = float(np.exp(-0.5 * (delta * max(rise, 1.0e-12) / 2.0) ** 2))
        return float(sinc * taper)
    return 0.0


def neighbor_selectivity_proxy(
    controls: list[ToneControl],
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    pulse: PulseParams,
    duration_s: float,
    n_levels: int,
) -> dict[str, float]:
    if not controls:
        return {"mean_neighbor_overlap": 0.0, "max_neighbor_overlap": 0.0}
    amp_scale = float(max(max(abs(float(t.amp_rad_s)), 1.0e-15) for t in controls))
    frequency_by_n = {
        int(n): sqr_manifold_waveform_frequency_rad_s(model=model, manifold_n=int(n), frame=frame)
        for n in range(int(n_levels))
    }
    values: list[float] = []
    max_value = 0.0
    for tone in controls:
        omega_eff = float(tone.omega_rad_s + tone.detuning_rad_s)
        amp_weight = float(abs(tone.amp_rad_s) / amp_scale)
        for neighbor in (int(tone.manifold) - 1, int(tone.manifold) + 1):
            if not (0 <= neighbor < int(n_levels)):
                continue
            delta = float(omega_eff - frequency_by_n[neighbor])
            overlap = _spectral_overlap_proxy(delta_rad_s=delta, pulse=pulse, duration_s=float(duration_s))
            value = float((amp_weight * overlap) ** 2)
            values.append(value)
            if value > max_value:
                max_value = value
    if not values:
        return {"mean_neighbor_overlap": 0.0, "max_neighbor_overlap": 0.0}
    return {
        "mean_neighbor_overlap": float(np.mean(values)),
        "max_neighbor_overlap": float(max_value),
    }


def active_support_leakage_proxy(
    controls: list[ToneControl],
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    pulse: PulseParams,
    duration_s: float,
    active_levels: list[int],
    active_weights: dict[int, float],
    boundary_boost: float,
    n_levels: int,
) -> dict[str, Any]:
    if not controls or not active_levels:
        return {
            "weighted_mean_leakage": 0.0,
            "max_leakage": 0.0,
            "worst_channel": None,
        }
    active_set = set(int(n) for n in active_levels)
    edge_levels = {min(active_set), max(active_set)}
    amp_scale = float(max(max(abs(float(t.amp_rad_s)), 1.0e-15) for t in controls))
    frequency_by_n = {
        int(n): sqr_manifold_waveform_frequency_rad_s(model=model, manifold_n=int(n), frame=frame)
        for n in range(int(n_levels))
    }
    values: list[float] = []
    worst = {"value": -1.0, "from": None, "to": None}
    for tone in controls:
        k = int(tone.manifold)
        if k not in active_set:
            continue
        omega_eff = float(tone.omega_rad_s + tone.detuning_rad_s)
        amp_weight = float(abs(tone.amp_rad_s) / amp_scale)
        support_weight = float(active_weights.get(k, 0.0))
        boundary = float(boundary_boost) if k in edge_levels else 1.0
        for n in (k - 1, k + 1):
            if not (0 <= n < int(n_levels)):
                continue
            if n in active_set:
                continue
            delta = float(omega_eff - frequency_by_n[int(n)])
            overlap = _spectral_overlap_proxy(delta_rad_s=delta, pulse=pulse, duration_s=float(duration_s))
            value = float((support_weight * boundary * amp_weight * overlap) ** 2)
            values.append(value)
            if value > float(worst["value"]):
                worst = {"value": float(value), "from": int(k), "to": int(n)}
    if not values:
        return {"weighted_mean_leakage": 0.0, "max_leakage": 0.0, "worst_channel": None}
    return {
        "weighted_mean_leakage": float(np.mean(values)),
        "max_leakage": float(np.max(values)),
        "worst_channel": {
            "from_manifold": worst["from"],
            "to_manifold": worst["to"],
            "value": float(worst["value"]),
        },
    }


def support_metrics_for_case(
    case: CaseResult,
    profile: TargetProfile,
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    pulse: PulseParams,
    duration_s: float,
    support: ActiveSupportParams,
    support_states: dict[str, qt.Qobj],
    support_ideal_outputs: dict[str, qt.Qobj],
    support_reference_state: qt.Qobj | None = None,
) -> dict[str, Any]:
    active_levels, active_weights, inactive_levels = resolve_active_support(
        profile.n_levels,
        support=support,
        reference_state=support_reference_state,
    )
    rows_by_n = {int(row["n"]): row for row in case.block_rows}
    w = np.asarray([float(active_weights[int(n)]) for n in active_levels], dtype=float)
    w = w / float(np.sum(w))

    active_fidelity = np.asarray([rows_by_n[int(n)]["process_fidelity"] for n in active_levels], dtype=float)
    theta_err = np.asarray(
        [rows_by_n[int(n)]["theta_simulated_rad"] - rows_by_n[int(n)]["theta_target_rad"] for n in active_levels],
        dtype=float,
    )
    phase_axis_err = np.asarray([rows_by_n[int(n)]["phi_axis_error_rad"] for n in active_levels], dtype=float)

    unitary = np.asarray(case.unitary, dtype=np.complex128)
    pre_z = np.zeros(len(active_levels), dtype=float)
    post_z = np.zeros(len(active_levels), dtype=float)
    active_rows: list[dict[str, Any]] = []
    for idx, n in enumerate(active_levels):
        simulated = normalize_unitary(_unitary_block(unitary, int(n)))
        target = normalize_unitary(_rxy_matrix(float(profile.theta[int(n)]), float(profile.phi[int(n)])))
        mismatch = normalize_unitary(target.conj().T @ simulated)
        decomp = decompose_z_rxy_z(mismatch)
        pre_z[idx] = float(decomp["alpha_rad"])
        post_z[idx] = float(decomp["beta_rad"])
        active_rows.append(
            {
                "n": int(n),
                "weight": float(active_weights[int(n)]),
                "process_fidelity": float(rows_by_n[int(n)]["process_fidelity"]),
                "theta_error_rad": float(theta_err[idx]),
                "phase_axis_error_rad": float(phase_axis_err[idx]),
                "residual_pre_z_rad": float(pre_z[idx]),
                "residual_post_z_rad": float(post_z[idx]),
            }
        )

    inactive_infidelity = np.asarray(
        [1.0 - float(rows_by_n[int(n)]["process_fidelity"]) for n in inactive_levels if int(n) in rows_by_n],
        dtype=float,
    )
    inactive_mean_infidelity = float(np.mean(inactive_infidelity)) if inactive_infidelity.size else 0.0

    support_outputs = apply_unitary_to_states(case.unitary_qobj, support_states)
    state_fid_map = state_fidelity_vs_ideal(support_ideal_outputs, support_outputs)
    state_values = np.asarray(list(state_fid_map.values()), dtype=float) if state_fid_map else np.asarray([], dtype=float)
    phase_super = []
    for label, ideal_state in support_ideal_outputs.items():
        if not label.startswith("sup_"):
            continue
        phase_err = _phase_proxy_error_for_superposition(
            ideal_state=ideal_state,
            simulated_state=support_outputs[label],
            active_levels=active_levels,
        )
        if np.isfinite(phase_err):
            phase_super.append(float(phase_err))
    phase_super = np.asarray(phase_super, dtype=float)
    phase_super_rms = float(np.sqrt(np.mean(phase_super**2))) if phase_super.size else 0.0
    state_leak = support_state_leakage_metrics(
        states=support_outputs,
        active_levels=active_levels,
        n_cav=int(model.n_cav),
    )

    leakage = active_support_leakage_proxy(
        controls=case.controls,
        model=model,
        frame=frame,
        pulse=pulse,
        duration_s=float(duration_s),
        active_levels=active_levels,
        active_weights=active_weights,
        boundary_boost=float(support.boundary_leakage_boost),
        n_levels=int(profile.n_levels),
    )

    return {
        "active_levels": [int(n) for n in active_levels],
        "active_weights": {str(int(n)): float(active_weights[int(n)]) for n in active_levels},
        "active_weighted_mean_process_fidelity": float(np.sum(w * active_fidelity)),
        "active_weighted_mean_infidelity": float(np.sum(w * (1.0 - active_fidelity))),
        "active_min_process_fidelity": float(np.min(active_fidelity)),
        "active_theta_rms_rad": float(np.sqrt(np.sum(w * (theta_err**2)))),
        "active_phase_axis_rms_rad": float(np.sqrt(np.sum(w * (phase_axis_err**2)))),
        "active_residual_pre_z_rms_rad": float(np.sqrt(np.sum(w * (pre_z**2)))),
        "active_residual_post_z_rms_rad": float(np.sqrt(np.sum(w * (post_z**2)))),
        "inactive_mean_infidelity": float(inactive_mean_infidelity),
        "support_state_fidelity_mean": float(np.mean(state_values)) if state_values.size else float("nan"),
        "support_state_fidelity_min": float(np.min(state_values)) if state_values.size else float("nan"),
        "support_state_fidelities": {k: float(v) for k, v in state_fid_map.items()},
        "support_phase_superposition_rms_rad": float(phase_super_rms),
        "support_state_leakage_mean": float(state_leak["mean"]),
        "support_state_leakage_max": float(state_leak["max"]),
        "support_state_leakage_by_state": {k: float(v) for k, v in state_leak["by_state"].items()},
        "support_weighted_leakage_mean": float(leakage["weighted_mean_leakage"]),
        "support_weighted_leakage_max": float(leakage["max_leakage"]),
        "support_worst_leakage_channel": leakage["worst_channel"],
        "active_block_rows": active_rows,
    }


def support_objective_loss_terms(
    support_metrics: dict[str, Any],
    support_weights: SupportObjectiveWeights,
    support: ActiveSupportParams,
    reg: float,
) -> dict[str, float]:
    # Do-not-care principle: inactive manifolds are spectators and are not target-unitary
    # constraints. They only enter the loss through weak inactive weighting and explicit
    # leakage penalties from active support.
    block_floor_penalty = max(
        0.0,
        float(support_weights.active_block_fidelity_floor) - float(support_metrics["active_min_process_fidelity"]),
    )
    state_floor_penalty = max(
        0.0,
        float(support_weights.active_state_fidelity_floor) - float(support_metrics["support_state_fidelity_min"]),
    )
    leak_mean_combined = float(
        0.70 * float(support_metrics["support_state_leakage_mean"])
        + 0.30 * float(support_metrics["support_weighted_leakage_mean"])
    )
    leak_max_combined = float(
        0.70 * float(support_metrics["support_state_leakage_max"])
        + 0.30 * float(support_metrics["support_weighted_leakage_max"])
    )
    return {
        "active_infidelity": float(support_weights.w_active_infidelity * float(support_metrics["active_weighted_mean_infidelity"])),
        "active_theta": float(support_weights.w_active_theta * float(support_metrics["active_theta_rms_rad"] ** 2)),
        "active_phase_axis": float(support_weights.w_active_phase_axis * float(support_metrics["active_phase_axis_rms_rad"] ** 2)),
        "active_pre_z": float(support_weights.w_active_pre_z * float(support_metrics["active_residual_pre_z_rms_rad"] ** 2)),
        "active_post_z": float(support_weights.w_active_post_z * float(support_metrics["active_residual_post_z_rms_rad"] ** 2)),
        "active_state_mean": float(
            support_weights.w_active_state_mean * float(1.0 - support_metrics["support_state_fidelity_mean"])
        ),
        "active_state_min": float(support_weights.w_active_state_min * float(state_floor_penalty**2)),
        "phase_superposition": float(
            support_weights.w_phase_superposition * float(support_metrics["support_phase_superposition_rms_rad"] ** 2)
        ),
        "leakage_mean": float(support_weights.w_leak_mean * leak_mean_combined),
        "leakage_max": float(support_weights.w_leak_max * leak_max_combined),
        "worst_block": float(support_weights.w_worst_block * float(block_floor_penalty**2)),
        "inactive_infidelity": float(float(support.inactive_weight) * float(support_metrics["inactive_mean_infidelity"])),
        "regularization": float(reg),
    }


def optimization_loss_terms(
    summary: dict[str, Any],
    opt: OptimizationParams,
    reg: float,
) -> dict[str, float]:
    return {
        "infidelity": float(opt.w_infid * float(summary["mean_infidelity"])),
        "phase_axis": float(opt.w_phase * float(summary["phase_rms_rad"] ** 2)),
        "theta": float(opt.w_theta * float(summary["theta_rms_rad"] ** 2)),
        "residual_z": float(opt.w_residual_z * float(summary["residual_z_rms_rad"] ** 2)),
        "state": float(opt.w_state * float(1.0 - summary["state_fidelity_mean"])),
        "off_block": float(opt.w_off_block * float(summary["off_block_norm"] ** 2)),
        "selectivity_mean": float(opt.w_selectivity_mean * float(summary.get("neighbor_overlap_proxy_mean", 0.0))),
        "selectivity_max": float(opt.w_selectivity_max * float(summary.get("neighbor_overlap_proxy_max", 0.0))),
        "regularization": float(reg),
    }


def optimization_loss(summary: dict[str, Any], opt: OptimizationParams, reg: float) -> float:
    terms = optimization_loss_terms(summary=summary, opt=opt, reg=reg)
    return float(sum(float(value) for value in terms.values()))


def optimize_case(
    mode: str,
    base_controls: list[ToneControl],
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    profile: TargetProfile,
    pulse_params: PulseParams,
    opt_params: OptimizationParams,
    duration_s: float,
    reference_states: dict[str, qt.Qobj],
    ideal_state_outputs: dict[str, qt.Qobj],
    support_config: ActiveSupportParams | None = None,
) -> CaseResult:
    active = active_controls(
        base_controls,
        theta_cutoff=float(pulse_params.theta_cutoff),
        include_zero_amp=bool(opt_params.allow_zero_theta_corrections),
    )
    x0, bounds = parameter_layout(mode=mode, active_indices=active, opt=opt_params)
    history: list[dict[str, float]] = []
    best = {"loss": float("inf"), "vector": x0.copy(), "summary": None}
    objective_scope = str(opt_params.objective_scope).strip().lower()

    support_states: dict[str, qt.Qobj] | None = None
    support_ideal_outputs: dict[str, qt.Qobj] | None = None
    support_active_levels: list[int] | None = None
    support_weight_map: dict[int, float] | None = None
    support_reference_state: qt.Qobj | None = None
    support_cfg = support_config
    if support_cfg is not None:
        support_reference_state = select_support_reference_state(reference_states, support=support_cfg)
        support_active_levels, support_weight_map, _ = resolve_active_support(
            profile.n_levels,
            support=support_cfg,
            reference_state=support_reference_state,
        )
        support_states = build_active_support_ensemble(
            model=model,
            active_levels=support_active_levels,
            support=support_cfg,
            active_weights=support_weight_map,
        )
        support_ideal_outputs = apply_unitary_to_states(build_target_unitary(profile), support_states)

    def evaluate(vector: np.ndarray, case_id: str) -> tuple[float, dict[str, Any]]:
        controls = controls_from_vector(
            base_controls,
            active_indices=active,
            mode=mode,
            vector=vector,
            duration_s=float(duration_s),
        )
        case = build_case(
            case_id=case_id,
            description=f"Optimization objective ({mode})",
            controls=controls,
            model=model,
            frame=frame,
            profile=profile,
            pulse_params=pulse_params,
            duration_s=float(duration_s),
            dt_s=float(pulse_params.dt_opt_s),
            max_step_s=float(pulse_params.max_step_opt_s),
            reference_states=reference_states,
            ideal_state_outputs=ideal_state_outputs,
        )
        reg = regularization_cost(mode=mode, vector=vector, opt=opt_params)
        support_metrics = None
        if support_cfg is not None and support_states is not None and support_ideal_outputs is not None:
            support_metrics = support_metrics_for_case(
                case=case,
                profile=profile,
                model=model,
                frame=frame,
                pulse=pulse_params,
                duration_s=float(duration_s),
                support=support_cfg,
                support_states=support_states,
                support_ideal_outputs=support_ideal_outputs,
                support_reference_state=support_reference_state,
            )
            case.summary["support_metrics"] = support_metrics
        if objective_scope == "support_aware":
            if support_metrics is None:
                raise ValueError("Support-aware objective requires support_config.")
            terms = support_objective_loss_terms(
                support_metrics=support_metrics,
                support_weights=opt_params.support_weights,
                support=support_cfg,
                reg=reg,
            )
        else:
            terms = optimization_loss_terms(case.summary, opt=opt_params, reg=reg)
        loss = float(sum(float(value) for value in terms.values()))
        summary = dict(case.summary)
        summary["objective_scope"] = objective_scope
        summary["loss_terms"] = {name: float(value) for name, value in terms.items()}
        summary["regularization"] = float(terms["regularization"])
        summary["loss_total"] = float(loss)
        if support_metrics is not None:
            summary["support_metrics"] = support_metrics
        return loss, summary

    def objective(vector: np.ndarray) -> float:
        loss, summary = evaluate(vector, case_id=f"opt_{mode}")
        terms = summary["loss_terms"]
        history.append(
            {
                "eval": float(len(history)),
                "loss_total": float(loss),
                "mean_infidelity": float(summary["mean_infidelity"]),
                "phase_rms_rad": float(summary["phase_rms_rad"]),
                "theta_rms_rad": float(summary["theta_rms_rad"]),
                "residual_z_rms_rad": float(summary["residual_z_rms_rad"]),
                "state_fidelity_mean": float(summary["state_fidelity_mean"]),
                "off_block_norm": float(summary["off_block_norm"]),
                "neighbor_overlap_proxy_mean": float(summary.get("neighbor_overlap_proxy_mean", float("nan"))),
                "neighbor_overlap_proxy_max": float(summary.get("neighbor_overlap_proxy_max", float("nan"))),
                "regularization": float(summary["regularization"]),
            }
        )
        if objective_scope == "support_aware":
            history[-1]["loss_active_infidelity"] = float(terms["active_infidelity"])
            history[-1]["loss_active_theta"] = float(terms["active_theta"])
            history[-1]["loss_active_phase_axis"] = float(terms["active_phase_axis"])
            history[-1]["loss_active_pre_z"] = float(terms["active_pre_z"])
            history[-1]["loss_active_post_z"] = float(terms["active_post_z"])
            history[-1]["loss_active_state_mean"] = float(terms["active_state_mean"])
            history[-1]["loss_active_state_min"] = float(terms["active_state_min"])
            history[-1]["loss_phase_superposition"] = float(terms["phase_superposition"])
            history[-1]["loss_leakage_mean"] = float(terms["leakage_mean"])
            history[-1]["loss_leakage_max"] = float(terms["leakage_max"])
            history[-1]["loss_worst_block"] = float(terms["worst_block"])
            history[-1]["loss_inactive_infidelity"] = float(terms["inactive_infidelity"])
        else:
            history[-1]["loss_infidelity"] = float(terms["infidelity"])
            history[-1]["loss_phase_axis"] = float(terms["phase_axis"])
            history[-1]["loss_theta"] = float(terms["theta"])
            history[-1]["loss_residual_z"] = float(terms["residual_z"])
            history[-1]["loss_state"] = float(terms["state"])
            history[-1]["loss_off_block"] = float(terms["off_block"])
            history[-1]["loss_selectivity_mean"] = float(terms["selectivity_mean"])
            history[-1]["loss_selectivity_max"] = float(terms["selectivity_max"])
        if objective_scope == "support_aware" and "support_metrics" in summary:
            s = summary["support_metrics"]
            history[-1]["active_weighted_mean_infidelity"] = float(s["active_weighted_mean_infidelity"])
            history[-1]["active_min_process_fidelity"] = float(s["active_min_process_fidelity"])
            history[-1]["support_state_fidelity_mean"] = float(s["support_state_fidelity_mean"])
            history[-1]["support_state_fidelity_min"] = float(s["support_state_fidelity_min"])
            history[-1]["support_phase_superposition_rms_rad"] = float(s["support_phase_superposition_rms_rad"])
            history[-1]["support_weighted_leakage_mean"] = float(s["support_weighted_leakage_mean"])
            history[-1]["support_weighted_leakage_max"] = float(s["support_weighted_leakage_max"])
        if loss < best["loss"]:
            best["loss"] = float(loss)
            best["vector"] = np.asarray(vector, dtype=float).copy()
            best["summary"] = summary
        return float(loss)

    if mode == MODE_BASIC:
        maxiter_stage1 = int(opt_params.maxiter_stage1_basic)
        maxiter_stage2 = int(opt_params.maxiter_stage2_basic)
    elif mode == MODE_EXTENDED:
        maxiter_stage1 = int(opt_params.maxiter_stage1_extended)
        maxiter_stage2 = int(opt_params.maxiter_stage2_extended)
    else:
        maxiter_stage1 = int(opt_params.maxiter_stage1_chirp)
        maxiter_stage2 = int(opt_params.maxiter_stage2_chirp)

    stage1 = minimize(
        objective,
        x0=x0,
        method=str(opt_params.method_stage1),
        bounds=bounds,
        options={"maxiter": maxiter_stage1, "disp": False},
    )
    stage2 = minimize(
        objective,
        x0=np.asarray(stage1.x, dtype=float),
        method=str(opt_params.method_stage2),
        bounds=bounds,
        options={"maxiter": maxiter_stage2},
    )
    candidate_vectors = [
        np.asarray(stage1.x, dtype=float),
        np.asarray(stage2.x, dtype=float),
        np.asarray(best["vector"], dtype=float),
    ]
    scores = [evaluate(vector, case_id=f"opt_{mode}_candidate")[0] for vector in candidate_vectors]
    final_vector = candidate_vectors[int(np.argmin(scores))]
    controls_opt = controls_from_vector(
        base_controls,
        active_indices=active,
        mode=mode,
        vector=final_vector,
        duration_s=float(duration_s),
    )
    case = build_case(
        case_id={"amp_phase": "C", "amp_phase_detuning": "D", "amp_phase_detuning_ramp": "E"}[mode],
        description=f"Optimized physical pulse ({mode})",
        controls=controls_opt,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=pulse_params,
        duration_s=float(duration_s),
        dt_s=float(pulse_params.dt_eval_s),
        max_step_s=float(pulse_params.max_step_eval_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal_state_outputs,
    )
    case.summary["objective_scope"] = objective_scope
    if support_cfg is not None and support_states is not None and support_ideal_outputs is not None:
        support_metrics = support_metrics_for_case(
            case=case,
            profile=profile,
            model=model,
            frame=frame,
            pulse=pulse_params,
            duration_s=float(duration_s),
            support=support_cfg,
            support_states=support_states,
            support_ideal_outputs=support_ideal_outputs,
            support_reference_state=support_reference_state,
        )
        case.summary["support_metrics"] = support_metrics
        case.summary["active_support_levels"] = [int(n) for n in support_active_levels or []]
        case.summary["active_support_weights"] = {str(int(k)): float(v) for k, v in (support_weight_map or {}).items()}
    final_reg = regularization_cost(mode=mode, vector=final_vector, opt=opt_params)
    if objective_scope == "support_aware":
        if "support_metrics" not in case.summary:
            raise ValueError("Support-aware objective requires support metrics in final case summary.")
        final_terms = support_objective_loss_terms(
            support_metrics=case.summary["support_metrics"],
            support_weights=opt_params.support_weights,
            support=support_cfg,
            reg=final_reg,
        )
    else:
        final_terms = optimization_loss_terms(case.summary, opt=opt_params, reg=final_reg)
    case.summary["loss_terms"] = {name: float(value) for name, value in final_terms.items()}
    case.summary["loss_total"] = float(sum(float(v) for v in final_terms.values()))
    case.optimization_trace = history
    case.optimization_status = {
        "mode": mode,
        "objective_scope": objective_scope,
        "active_tones": [int(base_controls[idx].manifold) for idx in active],
        "stage1": {
            "method": str(opt_params.method_stage1),
            "success": bool(stage1.success),
            "message": str(stage1.message),
            "fun": float(stage1.fun),
            "nit": int(getattr(stage1, "nit", 0)),
            "nfev": int(getattr(stage1, "nfev", 0)),
        },
        "stage2": {
            "method": str(opt_params.method_stage2),
            "success": bool(stage2.success),
            "message": str(stage2.message),
            "fun": float(stage2.fun),
            "nit": int(getattr(stage2, "nit", 0)),
            "nfev": int(getattr(stage2, "nfev", 0)),
        },
        "best_loss": float(min(scores)),
        "n_objective_calls": int(len(history)),
        "loss_terms_final": {name: float(value) for name, value in final_terms.items()},
    }
    return case

def plot_waveform(case: CaseResult, output_path: Path) -> None:
    signal = np.asarray(case.compiled.channels["qubit"].distorted, dtype=np.complex128)
    t_us = np.asarray(case.compiled.tlist, dtype=float) * 1.0e6
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.8), sharex=True)
    axes[0].plot(t_us, signal.real, color="tab:blue", linewidth=1.8)
    axes[0].set_ylabel("I(t) [rad/s]")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t_us, signal.imag, color="tab:orange", linewidth=1.8)
    axes[1].set_ylabel("Q(t) [rad/s]")
    axes[1].grid(alpha=0.25)
    axes[2].plot(t_us, np.abs(signal), color="tab:green", linewidth=1.8)
    axes[2].set_ylabel("|epsilon(t)|")
    axes[2].set_xlabel("Time [us]")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"Case {case.case_id}: time-domain multitone drive")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_spectrum(case: CaseResult, output_path: Path) -> None:
    signal = np.asarray(case.compiled.channels["qubit"].distorted, dtype=np.complex128)
    dt = float(case.compiled.dt)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(signal.size, d=dt))
    spec = np.fft.fftshift(np.fft.fft(signal))
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(freq_hz / 1.0e6, 20.0 * np.log10(np.abs(spec) / max(np.max(np.abs(spec)), 1.0e-18) + 1.0e-15), color="tab:purple")
    for tone in case.controls:
        marker = (tone.omega_rad_s + tone.detuning_rad_s) / (2.0 * np.pi) / 1.0e6
        ax.axvline(marker, color="0.55", linestyle="--", linewidth=0.8)
        ax.text(marker, -7.5, f"n={tone.manifold}", rotation=90, va="bottom", ha="right", fontsize=7)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Normalized spectrum [dB]")
    ax.set_title(f"Case {case.case_id}: multitone spectrum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_block_metric_comparison(
    profile: TargetProfile,
    case_results: dict[str, CaseResult],
    output_path: Path,
) -> None:
    n_values = np.arange(profile.n_levels, dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    color_map = {"B": "tab:red", "C": "tab:blue", "D": "tab:green", "E": "tab:purple"}
    for case_id, case in case_results.items():
        rows = {int(row["n"]): row for row in case.block_rows}
        color = color_map.get(case_id, "tab:gray")
        fidelity = np.asarray([rows[n]["process_fidelity"] for n in n_values], dtype=float)
        phase_err = np.asarray([rows[n]["phi_axis_error_rad"] for n in n_values], dtype=float)
        theta_err = np.asarray([rows[n]["error_theta_rad"] for n in n_values], dtype=float)
        rz = np.asarray([rows[n]["residual_conditional_z_rad"] for n in n_values], dtype=float)
        axes[0, 0].plot(n_values, fidelity, "o-", color=color, label=f"Case {case_id}")
        axes[0, 1].plot(n_values, phase_err, "o-", color=color, label=f"Case {case_id}")
        axes[1, 0].plot(n_values, theta_err, "o-", color=color, label=f"Case {case_id}")
        axes[1, 1].plot(n_values, rz, "o-", color=color, label=f"Case {case_id}")
    axes[0, 0].set_ylabel("Process fidelity")
    axes[0, 1].set_ylabel("Axis-phase error [rad]")
    axes[1, 0].set_ylabel("Unitary theta error [rad]")
    axes[1, 1].set_ylabel("Residual conditional Z [rad]")
    axes[1, 0].set_xlabel("Fock n")
    axes[1, 1].set_xlabel("Fock n")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    axes[0, 0].set_title("Blockwise fidelity")
    axes[0, 1].set_title("Phase-sensitive block error")
    axes[1, 0].set_title("Rotation-angle mismatch")
    axes[1, 1].set_title("Residual Z-like error")
    axes[0, 0].legend(loc="lower left", ncol=2, fontsize=8)
    fig.suptitle(f"{profile.name}: blockwise metrics by Fock manifold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_state_fidelity_comparison(
    profile: TargetProfile,
    case_results: dict[str, CaseResult],
    output_path: Path,
) -> None:
    labels = list(next(iter(case_results.values())).state_fidelities.keys())
    x = np.arange(len(labels), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    case_ids = list(case_results.keys())
    for idx, case_id in enumerate(case_ids):
        fid = np.asarray([case_results[case_id].state_fidelities[label] for label in labels], dtype=float)
        offset = (idx - (len(case_ids) - 1) / 2.0) * width
        ax.bar(x + offset, fid, width=width, label=f"Case {case_id}")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("State fidelity to ideal")
    ax.set_title(f"{profile.name}: final-state fidelity across representative inputs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_trajectory_comparison(
    profile: TargetProfile,
    cases: dict[str, CaseResult],
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    pulse_params: PulseParams,
    duration_s: float,
    initial_state: qt.Qobj,
    conditioned_levels: list[int],
    output_prefix: Path,
) -> dict[str, Any]:
    tlist = np.asarray(next(iter(cases.values())).compiled.tlist, dtype=float)
    ideal_states: list[qt.Qobj] = []
    for t in tlist:
        frac = float(np.clip(t / max(duration_s, 1.0e-18), 0.0, 1.0))
        u = sqr_op(profile.theta * frac, profile.phi)
        ideal_states.append(u * initial_state)
    ideal_traj = bloch_trajectory_from_states(ideal_states, conditioned_n_levels=conditioned_levels, probability_threshold=1.0e-8)
    traj_by_case: dict[str, dict[str, Any]] = {"A": ideal_traj}
    phase_by_case: dict[str, dict[int, np.ndarray]] = {
        "A": trajectory_relative_phase(ideal_states, n_values=conditioned_levels, branch="ground")
    }
    for case_id, case in cases.items():
        result = simulate_sequence(
            model,
            case.compiled,
            initial_state,
            {"qubit": "qubit"},
            config=SimulationConfig(frame=frame, max_step=float(pulse_params.max_step_eval_s), store_states=True),
        )
        if result.states is None:
            raise RuntimeError("Trajectory simulation expected stored states.")
        traj_by_case[case_id] = bloch_trajectory_from_states(
            result.states,
            conditioned_n_levels=conditioned_levels,
            probability_threshold=1.0e-8,
        )
        phase_by_case[case_id] = trajectory_relative_phase(result.states, n_values=conditioned_levels, branch="ground")

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 7.4), sharex=True)
    color_map = {"A": "black", "B": "tab:red", "C": "tab:blue", "D": "tab:green", "E": "tab:purple"}
    for case_id, traj in traj_by_case.items():
        color = color_map.get(case_id, "tab:gray")
        style = "-" if case_id == "A" else "--"
        axes[0].plot(tlist * 1.0e6, traj["x"], linestyle=style, color=color, linewidth=1.6, label=f"Case {case_id}")
        axes[1].plot(tlist * 1.0e6, traj["y"], linestyle=style, color=color, linewidth=1.6)
        axes[2].plot(tlist * 1.0e6, traj["z"], linestyle=style, color=color, linewidth=1.6)
    axes[0].set_ylabel("<sigma_x>")
    axes[1].set_ylabel("<sigma_y>")
    axes[2].set_ylabel("<sigma_z>")
    axes[2].set_xlabel("Time [us]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_ylim(-1.05, 1.05)
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"{profile.name}: qubit Bloch trajectory (unconditioned)")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_bloch.png"), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(len(conditioned_levels), 1, figsize=(10.8, 2.6 * len(conditioned_levels)), sharex=True)
    if len(conditioned_levels) == 1:
        axes = [axes]
    for ax, n in zip(axes, conditioned_levels, strict=False):
        for case_id, phase in phase_by_case.items():
            color = color_map.get(case_id, "tab:gray")
            style = "-" if case_id == "A" else "--"
            trace = phase[int(n)]
            ax.plot(tlist * 1.0e6, trace, linestyle=style, color=color, linewidth=1.6, label=f"Case {case_id}")
        ax.set_ylabel(f"arg(<g,{n}|rho|g,0>) [rad]")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [us]")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"{profile.name}: relative ground-manifold phase vs time")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_phase.png"), dpi=170)
    plt.close(fig)

    final_phase_error: dict[str, dict[str, float]] = {}
    for case_id, phase in phase_by_case.items():
        if case_id == "A":
            continue
        phase_err: dict[str, float] = {}
        for n in conditioned_levels:
            sim = phase[int(n)][-1]
            ref = phase_by_case["A"][int(n)][-1]
            phase_err[f"n{int(n)}"] = float(wrap_pi(sim - ref)) if np.isfinite(sim) and np.isfinite(ref) else float("nan")
        final_phase_error[case_id] = phase_err

    return {
        "trajectory_conditioned_levels": [int(n) for n in conditioned_levels],
        "final_relative_phase_error_rad": final_phase_error,
    }


def compute_single_tone_crosstalk(
    profile: TargetProfile,
    controls: list[ToneControl],
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    pulse_params: PulseParams,
    duration_s: float,
) -> dict[str, Any]:
    n_levels = profile.n_levels
    matrix = np.zeros((n_levels, n_levels), dtype=float)
    active_by_manifold = {tone.manifold: tone for tone in controls}
    for k in range(n_levels):
        tone = active_by_manifold.get(k)
        if tone is None:
            continue
        single = [ToneControl(manifold=tone.manifold, omega_rad_s=tone.omega_rad_s, amp_rad_s=tone.amp_rad_s, phase_rad=tone.phase_rad)]
        pulse = build_pulse_from_controls(single, duration_s=duration_s, pulse=pulse_params, label=f"single_n{k}")
        compiled = compile_single_pulse(pulse, dt_s=float(pulse_params.dt_eval_s))
        u = propagate_pulse_unitary(
            model=model,
            frame=frame,
            compiled=compiled,
            max_step_s=float(pulse_params.max_step_eval_s),
            qutip_nsteps=int(pulse_params.qutip_nsteps),
        )
        blocks, _ = extract_block_unitaries(np.asarray(u.full(), dtype=np.complex128), n_levels=n_levels)
        for n in range(n_levels):
            theta, *_ = rotation_axis_parameters(blocks[n])
            matrix[n, k] = float(theta)
    leakage = np.zeros_like(matrix)
    for k in range(n_levels):
        target = abs(matrix[k, k]) if abs(matrix[k, k]) > 1.0e-12 else 1.0
        leakage[:, k] = np.abs(matrix[:, k]) / target
        leakage[k, k] = 0.0
    near_neighbor = []
    for k in range(n_levels):
        for n in (k - 1, k + 1):
            if 0 <= n < n_levels:
                near_neighbor.append(float(leakage[n, k]))
    return {
        "theta_matrix_rad": matrix.tolist(),
        "relative_leakage_matrix": leakage.tolist(),
        "mean_neighbor_leakage": float(np.mean(near_neighbor)) if near_neighbor else 0.0,
        "max_neighbor_leakage": float(np.max(near_neighbor)) if near_neighbor else 0.0,
    }


def plot_crosstalk_heatmap(crosstalk: dict[str, Any], output_path: Path, title: str) -> None:
    data = np.asarray(crosstalk["relative_leakage_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.7, 4.8))
    im = ax.imshow(data, origin="lower", cmap="magma", aspect="auto")
    ax.set_xlabel("Driven tone manifold k")
    ax.set_ylabel("Affected manifold n")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|theta_n from tone k| / |theta_k|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_optimization_history(case: CaseResult, output_path: Path) -> None:
    if not case.optimization_trace:
        return
    x = np.arange(len(case.optimization_trace), dtype=int)
    loss = np.asarray([row["loss_total"] for row in case.optimization_trace], dtype=float)
    infid = np.asarray([row["mean_infidelity"] for row in case.optimization_trace], dtype=float)
    phase = np.asarray([row["phase_rms_rad"] for row in case.optimization_trace], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].plot(x, loss, color="tab:red", linewidth=1.8)
    ax[0].set_yscale("log")
    ax[0].set_title("Optimization loss")
    ax[0].set_xlabel("Objective call")
    ax[0].set_ylabel("Loss")
    ax[0].grid(alpha=0.25)
    ax[1].plot(x, infid, color="tab:blue", linewidth=1.7, label="Mean infidelity")
    ax[1].plot(x, phase, color="tab:green", linewidth=1.7, label="Phase RMS [rad]")
    ax[1].set_title("Primary error terms")
    ax[1].set_xlabel("Objective call")
    ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_convention_audit(output_dir: Path) -> ConventionAuditResult:
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    frame = FrameSpec()
    config = {"duration_rotation_s": 1.0, "rotation_sigma_fraction": 0.18}

    def n0_block(unitary_qobj: qt.Qobj) -> np.ndarray:
        full = np.asarray(unitary_qobj.full(), dtype=np.complex128)
        return full[:2, :2]

    def run_rotation(theta: float, phi: float, carrier: float = 0.0) -> tuple[np.ndarray, qt.Qobj]:
        gate = RotationGate(index=0, name="rot", theta=float(theta), phi=float(phi))
        pulses, drive_ops, _ = build_rotation_pulse(gate, config)
        base = pulses[0]
        pulse = Pulse(
            channel=base.channel,
            t0=base.t0,
            duration=base.duration,
            envelope=base.envelope,
            carrier=float(carrier),
            phase=base.phase,
            amp=base.amp,
            drag=base.drag,
            label=base.label,
        )
        compiled = SequenceCompiler(dt=0.002).compile([pulse], t_end=pulse.t1 + 0.002)
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state( 0,0),
            drive_ops,
            config=SimulationConfig(frame=frame, max_step=0.001),
        )
        rho_q = qt.ptrace(result.final_state, 0)
        bloch = np.array(
            [
                float(np.real((rho_q * qt.sigmax()).tr())),
                float(np.real((rho_q * qt.sigmay()).tr())),
                float(np.real((rho_q * qt.sigmaz()).tr())),
            ],
            dtype=float,
        )
        u = propagate_pulse_unitary(model, frame, compiled, max_step_s=0.001, qutip_nsteps=120000)
        return bloch, u

    bloch_i, unitary_i = run_rotation(theta=np.pi / 2.0, phi=0.0)
    bloch_q, unitary_q = run_rotation(theta=np.pi / 2.0, phi=np.pi / 2.0)

    phase_rows = []
    for phi in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0):
        _, u = run_rotation(theta=np.pi / 3.0, phi=float(phi))
        _, nx, ny, nz, _ = rotation_axis_parameters(n0_block(u))
        phase_rows.append({"phi_rad": float(phi), "axis_x": float(nx), "axis_y": float(ny), "axis_z": float(nz)})

    _, u_det_p = run_rotation(theta=np.pi / 2.0, phi=0.0, carrier=2.0 * np.pi * 0.25)
    _, u_det_m = run_rotation(theta=np.pi / 2.0, phi=0.0, carrier=-2.0 * np.pi * 0.25)
    err_p = normalize_unitary(n0_block(unitary_i).conj().T @ n0_block(u_det_p))
    err_m = normalize_unitary(n0_block(unitary_i).conj().T @ n0_block(u_det_m))
    theta_ep, _, _, nz_ep, _ = rotation_axis_parameters(err_p)
    theta_em, _, _, nz_em, _ = rotation_axis_parameters(err_m)

    tone = ToneControl(manifold=0, omega_rad_s=0.0, amp_rad_s=np.pi / 4.0, phase_rad=np.pi / 2.0)
    pulse = build_pulse_from_controls([tone], duration_s=1.0, pulse=PulseParams(), label="multitone_single")
    compiled = SequenceCompiler(dt=0.002).compile([pulse], t_end=1.002)
    unitary_mt = propagate_pulse_unitary(model, frame, compiled, max_step_s=0.001, qutip_nsteps=120000)
    overlap = np.trace(
        normalize_unitary(n0_block(unitary_q)).conj().T
        @ normalize_unitary(n0_block(unitary_mt))
    )
    mt_process_fid = float(np.abs(overlap) ** 2 / 4.0)

    sign_q = np.sign(bloch_q[0]) if abs(bloch_q[0]) > 1.0e-6 else 0.0
    if sign_q > 0:
        envelope = "epsilon(t) = 0.5 * (I + i Q)"
        verdict = "Gaussian convention matches required R_xy(theta,phi) with I->+x and Q->+y."
    elif sign_q < 0:
        envelope = "epsilon(t) = 0.5 * (I - i Q)"
        verdict = "Gaussian convention shows Q sign flip relative to required experimental convention."
    else:
        envelope = "Could not determine epsilon sign from Q test."
        verdict = "Convention audit inconclusive for Q sign."

    notes = [
        "Pulse._sample_analytic builds complex envelope with exp(+i*(carrier*t + phase)).",
        "multitone_gaussian_envelope uses exp(+i*(omega*t + phase)); SQR tone builder maps omega_waveform=-manifold_transition_frequency(...).",
        "hamiltonian_time_slices couples coeff to bdag and conj(coeff) to b on qubit channel.",
        "With |g>=+z and |e>=-z, numerical two-level tests determine the effective IQ sign unambiguously.",
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    phis = np.asarray([row["phi_rad"] for row in phase_rows], dtype=float)
    ax.plot(phis, [row["axis_x"] for row in phase_rows], "o-", label="axis x")
    ax.plot(phis, [row["axis_y"] for row in phase_rows], "s-", label="axis y")
    ax.plot(phis, [row["axis_z"] for row in phase_rows], "^-", label="axis z")
    ax.set_xlabel("Pulse phase phi [rad]")
    ax.set_ylabel("Extracted rotation-axis component")
    ax.set_title("Phase sweep: extracted axis from Gaussian pulse")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "phase0_axis_sweep.png", dpi=170)
    plt.close(fig)

    return ConventionAuditResult(
        verdict=verdict,
        envelope_interpretation=envelope,
        pure_i={"bloch_from_g": bloch_i.tolist()},
        pure_q={"bloch_from_g": bloch_q.tolist()},
        phase_sweep=phase_rows,
        detuning_check={
            "carrier_plus_error_theta_rad": float(theta_ep),
            "carrier_plus_error_axis_z": float(nz_ep),
            "carrier_minus_error_theta_rad": float(theta_em),
            "carrier_minus_error_axis_z": float(nz_em),
            "sign_flip_observed": bool(np.sign(nz_ep) == -np.sign(nz_em) and np.sign(nz_ep) != 0.0),
        },
        multitone_consistency={
            "single_tone_multitone_vs_gaussian_process_fidelity": float(mt_process_fid),
        },
        notes=notes,
    )

def serialize_case(case: CaseResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "description": case.description,
        "controls": [tone.as_dict() for tone in case.controls],
        "summary": case.summary,
        "block_rows": case.block_rows,
        "state_fidelities": case.state_fidelities,
        "optimization_status": case.optimization_status,
        "optimization_trace": case.optimization_trace,
    }


def profile_study(
    profile: TargetProfile,
    params: StudyParams,
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_u = build_target_unitary(profile)
    reference_states = build_reference_states(model, n_max=params.system.n_max, coherent_alpha=params.coherent_alpha)
    ideal_state_outputs = apply_unitary_to_states(target_u, reference_states)
    controls_naive = build_controls_from_target(
        profile=profile,
        model=model,
        frame=frame,
        duration_s=float(params.pulse.duration_nominal_s),
        theta_cutoff=float(params.pulse.theta_cutoff),
        include_all_levels=bool(params.pulse.include_zero_theta_tones),
    )

    case_b = build_case(
        case_id="B",
        description="Physical multi-tone pulse, naive initial guess",
        controls=controls_naive,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=params.pulse,
        duration_s=float(params.pulse.duration_nominal_s),
        dt_s=float(params.pulse.dt_eval_s),
        max_step_s=float(params.pulse.max_step_eval_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal_state_outputs,
    )
    case_c = optimize_case(
        mode=MODE_BASIC,
        base_controls=controls_naive,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=params.pulse,
        opt_params=params.optimization,
        duration_s=float(params.pulse.duration_nominal_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal_state_outputs,
    )
    case_d = optimize_case(
        mode=MODE_EXTENDED,
        base_controls=controls_naive,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=params.pulse,
        opt_params=params.optimization,
        duration_s=float(params.pulse.duration_nominal_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal_state_outputs,
    )

    case_results = {"B": case_b, "C": case_c, "D": case_d}
    if params.include_case_e:
        case_e = optimize_case(
            mode=MODE_CHIRP,
            base_controls=controls_naive,
            model=model,
            frame=frame,
            profile=profile,
            pulse_params=params.pulse,
            opt_params=params.optimization,
            duration_s=float(params.pulse.duration_nominal_s),
            reference_states=reference_states,
            ideal_state_outputs=ideal_state_outputs,
        )
        case_results["E"] = case_e

    plot_waveform(case_b, out_dir / "waveform_case_B.png")
    plot_spectrum(case_b, out_dir / "spectrum_case_B.png")
    plot_waveform(case_d, out_dir / "waveform_case_D.png")
    plot_spectrum(case_d, out_dir / "spectrum_case_D.png")
    if "E" in case_results:
        plot_waveform(case_results["E"], out_dir / "waveform_case_E.png")
        plot_spectrum(case_results["E"], out_dir / "spectrum_case_E.png")

    plot_block_metric_comparison(profile, case_results, out_dir / "block_metrics_cases_BCDE.png")
    plot_state_fidelity_comparison(profile, case_results, out_dir / "state_fidelity_cases_BCDE.png")
    for case_id in ("C", "D", "E"):
        if case_id in case_results:
            plot_optimization_history(case_results[case_id], out_dir / f"optimization_history_case_{case_id}.png")

    conditioned = [n for n in (0, 2, 4) if n < profile.n_levels]
    phase_state_key = next(
        (
            key
            for key in reference_states
            if key.startswith("g x (") and "|0>" in key and ("|2>" in key or "|4>" in key)
        ),
        None,
    )
    if phase_state_key is None and "g x (|0>+|1>)/sqrt2" in reference_states:
        phase_state_key = "g x (|0>+|1>)/sqrt2"
    traj_ref_state = reference_states[phase_state_key] if phase_state_key is not None else reference_states["g,0"]
    trajectory_summary = plot_trajectory_comparison(
        profile=profile,
        cases=case_results,
        model=model,
        frame=frame,
        pulse_params=params.pulse,
        duration_s=float(params.pulse.duration_nominal_s),
        initial_state=traj_ref_state,
        conditioned_levels=conditioned,
        output_prefix=out_dir / "trajectory",
    )

    crosstalk_naive = compute_single_tone_crosstalk(
        profile=profile,
        controls=case_b.controls,
        model=model,
        frame=frame,
        pulse_params=params.pulse,
        duration_s=float(params.pulse.duration_nominal_s),
    )
    crosstalk_opt = compute_single_tone_crosstalk(
        profile=profile,
        controls=case_d.controls,
        model=model,
        frame=frame,
        pulse_params=params.pulse,
        duration_s=float(params.pulse.duration_nominal_s),
    )
    plot_crosstalk_heatmap(crosstalk_naive, out_dir / "crosstalk_naive.png", "Case B cross-talk")
    plot_crosstalk_heatmap(crosstalk_opt, out_dir / "crosstalk_opt_case_D.png", "Case D cross-talk")

    ideal_snapshots: dict[str, Any] = {}
    for label, state in ideal_state_outputs.items():
        ideal_snapshots[label] = {
            "conditioned_bloch": conditioned_bloch_snapshot(state, n_levels=profile.n_levels),
            "relative_phase_ground": relative_phase_vector(state, n_levels=profile.n_levels, branch="ground"),
            "relative_phase_excited": relative_phase_vector(state, n_levels=profile.n_levels, branch="excited"),
        }

    return {
        "profile": {
            "name": profile.name,
            "mode": profile.mode,
            "seed": profile.seed,
            "theta": profile.theta.tolist(),
            "phi": profile.phi.tolist(),
        },
        "cases": {case_id: serialize_case(case) for case_id, case in case_results.items()},
        "ideal_reference": {
            "case_id": "A",
            "description": "Ideal operator only",
            "state_snapshots": ideal_snapshots,
        },
        "trajectory_summary": trajectory_summary,
        "crosstalk": {
            "naive_case_B": crosstalk_naive,
            "optimized_case_D": crosstalk_opt,
        },
    }


def scan_duration_dependence(
    profile: TargetProfile,
    params: StudyParams,
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
) -> dict[str, Any]:
    durations = {
        "short": float(params.pulse.duration_short_s),
        "nominal": float(params.pulse.duration_nominal_s),
        "long": float(params.pulse.duration_long_s),
    }
    reference_states = build_reference_states(model, n_max=params.system.n_max, coherent_alpha=params.coherent_alpha)
    ideal = apply_unitary_to_states(build_target_unitary(profile), reference_states)
    rows = []
    for label, duration in durations.items():
        controls = build_controls_from_target(
            profile=profile,
            model=model,
            frame=frame,
            duration_s=duration,
            theta_cutoff=float(params.pulse.theta_cutoff),
            include_all_levels=bool(params.pulse.include_zero_theta_tones),
        )
        case = build_case(
            case_id=f"B_{label}",
            description=f"Naive physical pulse ({label} duration)",
            controls=controls,
            model=model,
            frame=frame,
            profile=profile,
            pulse_params=params.pulse,
            duration_s=duration,
            dt_s=float(params.pulse.dt_eval_s),
            max_step_s=float(params.pulse.max_step_eval_s),
            reference_states=reference_states,
            ideal_state_outputs=ideal,
        )
        rows.append({"duration_label": label, "duration_s": duration, **case.summary})
    return {"rows": rows}


def scan_chi_dependence(
    profile: TargetProfile,
    params: StudyParams,
) -> dict[str, Any]:
    rows = []
    for tag, chi_hz in (
        ("easy", float(params.system.chi_easy_hz)),
        ("nominal", float(params.system.chi_nominal_hz)),
        ("hard", float(params.system.chi_hard_hz)),
    ):
        model, frame = build_model_and_frame(params.system, chi_hz=chi_hz)
        reference_states = build_reference_states(model, n_max=params.system.n_max, coherent_alpha=params.coherent_alpha)
        ideal = apply_unitary_to_states(build_target_unitary(profile), reference_states)
        controls = build_controls_from_target(
            profile=profile,
            model=model,
            frame=frame,
            duration_s=float(params.pulse.duration_nominal_s),
            theta_cutoff=float(params.pulse.theta_cutoff),
            include_all_levels=bool(params.pulse.include_zero_theta_tones),
        )
        case = build_case(
            case_id=f"B_chi_{tag}",
            description=f"Naive physical pulse (chi regime={tag})",
            controls=controls,
            model=model,
            frame=frame,
            profile=profile,
            pulse_params=params.pulse,
            duration_s=float(params.pulse.duration_nominal_s),
            dt_s=float(params.pulse.dt_eval_s),
            max_step_s=float(params.pulse.max_step_eval_s),
            reference_states=reference_states,
            ideal_state_outputs=ideal,
        )
        rows.append(
            {
                "regime": tag,
                "chi_hz": float(chi_hz),
                "chi_MHz": float(chi_hz / 1.0e6),
                **case.summary,
            }
        )
    return {"rows": rows}


def plot_scan_summary(duration_scan: dict[str, Any], chi_scan: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    d_rows = duration_scan["rows"]
    x = np.arange(len(d_rows), dtype=int)
    axes[0].plot(x, [row["mean_infidelity"] for row in d_rows], "o-", color="tab:red", label="Mean infidelity")
    axes[0].plot(x, [row["phase_rms_rad"] for row in d_rows], "s-", color="tab:blue", label="Phase RMS [rad]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([row["duration_label"] for row in d_rows])
    axes[0].set_title("Pulse duration dependence")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    c_rows = chi_scan["rows"]
    x2 = np.arange(len(c_rows), dtype=int)
    axes[1].plot(x2, [row["mean_infidelity"] for row in c_rows], "o-", color="tab:red", label="Mean infidelity")
    axes[1].plot(x2, [row["phase_rms_rad"] for row in c_rows], "s-", color="tab:blue", label="Phase RMS [rad]")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([row["regime"] for row in c_rows])
    axes[1].set_title("Dispersive-strength dependence")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

def write_report(
    output_path: Path,
    params: StudyParams,
    convention: ConventionAuditResult,
    profile_results: dict[str, dict[str, Any]],
    duration_scan: dict[str, Any],
    chi_scan: dict[str, Any],
    convention_post: ConventionAuditResult,
) -> None:
    lines = []
    lines.append("# SQR Multi-tone Study Report")
    lines.append("")
    lines.append("## Phase 0: Gaussian Convention Audit")
    lines.append(f"- Verdict: {convention.verdict}")
    lines.append(f"- Envelope interpretation: `{convention.envelope_interpretation}`")
    lines.append(f"- Pure I x90 from |g>: Bloch={convention.pure_i['bloch_from_g']}")
    lines.append(f"- Pure Q y90 from |g>: Bloch={convention.pure_q['bloch_from_g']}")
    lines.append(
        "- Phase sweep axes (phi, nx, ny, nz): "
        + ", ".join(
            f"({row['phi_rad']:.3f}, {row['axis_x']:.3f}, {row['axis_y']:.3f}, {row['axis_z']:.3f})"
            for row in convention.phase_sweep
        )
    )
    lines.append(
        "- Detuning sign check: "
        + f"+delta nz={convention.detuning_check['carrier_plus_error_axis_z']:.3f}, "
        + f"-delta nz={convention.detuning_check['carrier_minus_error_axis_z']:.3f}, "
        + f"flip={convention.detuning_check['sign_flip_observed']}"
    )
    lines.append(
        "- Multi-tone vs Gaussian convention consistency: "
        + f"process fidelity={convention.multitone_consistency['single_tone_multitone_vs_gaussian_process_fidelity']:.6f}"
    )
    lines.append("")

    lines.append("## Main SQR Cases")
    for profile_name, payload in profile_results.items():
        lines.append(f"### Profile: {profile_name}")
        lines.append(f"- Target mode: {payload['profile']['mode']}")
        lines.append(f"- Seed: {payload['profile']['seed']}")
        for case_id in ("B", "C", "D", "E"):
            if case_id not in payload["cases"]:
                continue
            summary = payload["cases"][case_id]["summary"]
            lines.append(
                f"- Case {case_id}: "
                + f"mean fidelity={summary['mean_process_fidelity']:.6f}, "
                + f"mean infidelity={summary['mean_infidelity']:.6f}, "
                + f"phase RMS={summary['phase_rms_rad']:.4f} rad, "
                + f"residual Z RMS={summary['residual_z_rms_rad']:.4f} rad, "
                + f"state fidelity mean={summary['state_fidelity_mean']:.6f}"
            )
            mismatch = summary.get("population_phase_mismatch_example")
            if mismatch is not None:
                lines.append(
                    f"  population-vs-phase mismatch example (n={mismatch['n']}): "
                    + f"population error={mismatch['population_error']:.4f}, "
                    + f"process infidelity={mismatch['process_infidelity']:.4f}, "
                    + f"relative block phase={mismatch['relative_block_phase_rad']:.4f} rad"
                )
        crosstalk_b = payload["crosstalk"]["naive_case_B"]
        crosstalk_d = payload["crosstalk"]["optimized_case_D"]
        lines.append(
            "- Neighbor cross-talk (mean |theta_{n+/-1}|/|theta_n|): "
            + f"Case B={crosstalk_b['mean_neighbor_leakage']:.4f}, "
            + f"Case D={crosstalk_d['mean_neighbor_leakage']:.4f}"
        )
        traj = payload["trajectory_summary"]["final_relative_phase_error_rad"]
        for case_id, phase_rows in traj.items():
            lines.append(f"- Final relative phase error Case {case_id}: {phase_rows}")
        lines.append("")

    lines.append("## Duration and Dispersive-Strength Scans")
    lines.append("- Duration scan (Case B naive):")
    for row in duration_scan["rows"]:
        lines.append(
            f"  - {row['duration_label']}: T={row['duration_s'] * 1e6:.3f} us, "
            + f"mean infidelity={row['mean_infidelity']:.6f}, phase RMS={row['phase_rms_rad']:.4f}"
        )
    lines.append("- Chi scan (Case B naive):")
    for row in chi_scan["rows"]:
        lines.append(
            f"  - {row['regime']}: chi={row['chi_MHz']:.3f} MHz, "
            + f"mean infidelity={row['mean_infidelity']:.6f}, phase RMS={row['phase_rms_rad']:.4f}"
        )
    lines.append("")

    lines.append("## Convention Regression (Case H)")
    lines.append(f"- Re-run verdict: {convention_post.verdict}")
    lines.append(f"- Re-run envelope interpretation: `{convention_post.envelope_interpretation}`")
    lines.append("")
    lines.append("## Parameter Snapshot")
    lines.append("```json")
    lines.append(json.dumps(asdict(params), indent=2, default=str))
    lines.append("```")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_study(params: StudyParams) -> dict[str, Any]:
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convention = run_convention_audit(output_dir=output_dir)
    model_nom, frame_nom = build_model_and_frame(params.system, chi_hz=float(params.system.chi_nominal_hz))

    profile_results: dict[str, dict[str, Any]] = {}
    for idx, mode in enumerate(params.run_profiles):
        profile = build_target_profile(
            mode=str(mode),
            n_levels=int(params.system.n_cav),
            seed=int(params.seed + 19 * idx),
            theta_max=float(params.theta_max_rad),
        )
        profile_out_dir = output_dir / profile.name
        profile_payload = profile_study(
            profile=profile,
            params=params,
            model=model_nom,
            frame=frame_nom,
            out_dir=profile_out_dir,
        )
        profile_results[profile.name] = profile_payload
        (profile_out_dir / "summary.json").write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")

    baseline_mode = "structured" if "structured" in params.run_profiles else params.run_profiles[0]
    baseline_profile = build_target_profile(
        mode=baseline_mode,
        n_levels=int(params.system.n_cav),
        seed=int(params.seed + 101),
        theta_max=float(params.theta_max_rad),
    )
    duration_scan = scan_duration_dependence(
        profile=baseline_profile,
        params=params,
        model=model_nom,
        frame=frame_nom,
    )
    chi_scan = scan_chi_dependence(
        profile=baseline_profile,
        params=params,
    )
    plot_scan_summary(duration_scan, chi_scan, output_dir / "duration_chi_scan.png")

    convention_post = run_convention_audit(output_dir=output_dir)
    report_path = output_dir / "report.md"
    write_report(
        output_path=report_path,
        params=params,
        convention=convention,
        profile_results=profile_results,
        duration_scan=duration_scan,
        chi_scan=chi_scan,
        convention_post=convention_post,
    )

    summary = {
        "output_dir": str(output_dir),
        "convention_audit": asdict(convention),
        "profiles": profile_results,
        "duration_scan": duration_scan,
        "chi_scan": chi_scan,
        "convention_regression": asdict(convention_post),
        "report_path": str(report_path),
        "audit_source_refs": {
            "pulse_sampler": inspect.getsource(Pulse._sample_analytic).splitlines()[0].strip(),
            "drive_hamiltonian_builder": inspect.getsource(hamiltonian_time_slices).splitlines()[0].strip(),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gaussian convention audit + multi-tone SQR study.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sqr_multitone_study"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for SQR targets.")
    parser.add_argument("--n-max", type=int, default=6, help="Maximum Fock manifold index.")
    parser.add_argument(
        "--profiles",
        type=str,
        default="structured,hard_random",
        help="Comma-separated target profile modes (structured, hard_random, fully_random).",
    )
    parser.add_argument("--skip-case-e", action="store_true", help="Skip chirped compensation optimization case.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = StudyParams(
        seed=int(args.seed),
        include_case_e=not bool(args.skip_case_e),
        run_profiles=tuple(x.strip() for x in str(args.profiles).split(",") if x.strip()),
        output_dir=Path(args.output_dir),
        system=SystemParams(n_max=int(args.n_max)),
    )
    summary = run_study(params)
    print(json.dumps(
        {
            "output_dir": summary["output_dir"],
            "report_path": summary["report_path"],
            "convention_verdict": summary["convention_audit"]["verdict"],
            "profiles": list(summary["profiles"].keys()),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
