from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import qutip as qt
from scipy.optimize import Bounds, minimize

from cqed_sim.calibration.conditioned_multitone import (
    ConditionedMultitoneCorrections,
    ConditionedMultitoneRunConfig,
    ConditionedMultitoneWaveform,
    ConditionedOptimizationConfig,
    ConditionedQubitTargets,
    ConditionedSectorMetrics,
    bloch_angles_from_density_matrix,
    build_conditioned_multitone_tones,
    build_conditioned_multitone_waveform,
    compile_conditioned_multitone_waveform,
    qubit_density_matrix_from_angles,
)
from cqed_sim.core.conventions import qubit_cavity_block_indices, qubit_cavity_index
from cqed_sim.core.ideal_gates import logical_block_phase_op, qubit_rotation_xy
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.sequence.scheduler import CompiledSequence
from cqed_sim.sim.extractors import bloch_xyz_from_qubit_state, conditioned_qubit_state, truncate_to_qubit_subspace
from cqed_sim.sim.runner import hamiltonian_time_slices
from cqed_sim.unitary_synthesis.metrics import LogicalBlockPhaseDiagnostics, logical_block_phase_diagnostics


def _wrap_pi(value: float | np.ndarray) -> float | np.ndarray:
    wrapped = (np.asarray(value, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(wrapped) == 0:
        return float(wrapped)
    return wrapped


def _normalize_weights(weights: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("weights must not be empty.")
    if np.any(arr < 0.0):
        raise ValueError("weights must be non-negative.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    arr = arr / total
    return tuple(float(value) for value in arr)


def _ensure_targets(
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
) -> ConditionedQubitTargets:
    if isinstance(targets, ConditionedQubitTargets):
        return targets
    return ConditionedQubitTargets.from_spec(targets)


def _resolve_logical_levels(
    targets: ConditionedQubitTargets,
    logical_levels: Sequence[int] | None,
) -> tuple[int, ...]:
    if logical_levels is None:
        levels = tuple(range(targets.n_levels))
    else:
        levels = tuple(int(level) for level in logical_levels)
    if not levels:
        raise ValueError("logical_levels must contain at least one cavity level.")
    if len(set(levels)) != len(levels):
        raise ValueError("logical_levels must not contain duplicates.")
    for level in levels:
        if level < 0 or level >= targets.n_levels:
            raise ValueError(
                f"logical level {level} is out of range for the provided target specification of length {targets.n_levels}."
            )
    return levels


def _logical_weights(targets: ConditionedQubitTargets, logical_levels: Sequence[int]) -> tuple[float, ...]:
    selected = [float(targets.weights[int(level)]) for level in logical_levels]
    if not any(weight > 0.0 for weight in selected):
        selected = [1.0] * len(selected)
    return _normalize_weights(selected)


def _logical_indices(model: DispersiveTransmonCavityModel, logical_levels: Sequence[int]) -> tuple[int, ...]:
    indices: list[int] = []
    for level in logical_levels:
        indices.extend(int(index) for index in qubit_cavity_block_indices(int(model.n_cav), int(level)))
    return tuple(indices)


def _full_dimension(model: DispersiveTransmonCavityModel) -> int:
    return int(model.n_tr) * int(model.n_cav)


def _solver_options(run_config: ConditionedMultitoneRunConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "atol": 1.0e-8,
        "rtol": 1.0e-7,
        "nsteps": 100000,
    }
    if run_config.max_step_s is not None:
        options["max_step"] = float(run_config.max_step_s)
    return options


def _classify_error(theta_error: float, phi_error: float, bloch_radius: float) -> str:
    if np.isnan(theta_error) and np.isnan(phi_error):
        return "undefined"
    if float(bloch_radius) < 0.98:
        return "mixed-or-decohered"
    theta_abs = float(abs(theta_error)) if np.isfinite(theta_error) else 0.0
    phi_abs = float(abs(phi_error)) if np.isfinite(phi_error) else 0.0
    if theta_abs < 0.03 and phi_abs < 0.03:
        return "small"
    if theta_abs > 2.0 * phi_abs and theta_abs > 0.05:
        return "amplitude-like"
    if phi_abs > 2.0 * theta_abs and phi_abs > 0.05:
        return "phase-or-detuning-like"
    return "mixed-or-crosstalk-like"


def _state_qobj(
    state: qt.Qobj | np.ndarray,
    *,
    full_dim: int,
    logical_dim: int,
    logical_indices: Sequence[int],
    subsystem_dims: Sequence[int],
) -> qt.Qobj:
    if isinstance(state, qt.Qobj):
        arr = np.asarray(state.full(), dtype=np.complex128)
    else:
        arr = np.asarray(state, dtype=np.complex128)
    logical_idx = np.asarray(tuple(int(index) for index in logical_indices), dtype=int)
    if arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape):
        vec = arr.reshape(-1)
        if vec.size == int(logical_dim):
            embedded = np.zeros(int(full_dim), dtype=np.complex128)
            embedded[logical_idx] = vec
            vec = embedded
        elif vec.size != int(full_dim):
            raise ValueError(f"State vector has length {vec.size}, expected {logical_dim} or {full_dim}.")
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return qt.Qobj(vec.reshape((-1, 1)), dims=[list(subsystem_dims), [1] * len(subsystem_dims)])
    if arr.ndim != 2:
        raise ValueError("States must be vectors or density matrices.")
    matrix = arr
    if matrix.shape == (int(logical_dim), int(logical_dim)):
        embedded = np.zeros((int(full_dim), int(full_dim)), dtype=np.complex128)
        embedded[np.ix_(logical_idx, logical_idx)] = matrix
        matrix = embedded
    elif matrix.shape != (int(full_dim), int(full_dim)):
        raise ValueError(
            f"State operator has shape {matrix.shape}, expected {(logical_dim, logical_dim)} or {(full_dim, full_dim)}."
        )
    return qt.Qobj(matrix, dims=[list(subsystem_dims), list(subsystem_dims)])


def _state_fidelity(actual: qt.Qobj, target: qt.Qobj) -> float:
    if not actual.isoper and not target.isoper:
        actual_vec = np.asarray(actual.full(), dtype=np.complex128).reshape(-1)
        target_vec = np.asarray(target.full(), dtype=np.complex128).reshape(-1)
        return float(np.clip(abs(np.vdot(target_vec, actual_vec)) ** 2, 0.0, 1.0))
    rho_actual = actual if actual.isoper else actual.proj()
    rho_target = target if target.isoper else target.proj()
    return float(np.clip(qt.fidelity(rho_actual, rho_target) ** 2, 0.0, 1.0))


def _restricted_process_fidelity(target_operator: np.ndarray, actual_operator: np.ndarray) -> float:
    target = np.asarray(target_operator, dtype=np.complex128)
    actual = np.asarray(actual_operator, dtype=np.complex128)
    if target.shape != actual.shape or target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("restricted process fidelity requires same-shape square matrices.")
    dim = float(target.shape[0])
    overlap = np.trace(target.conj().T @ actual)
    return float(np.clip(abs(overlap) ** 2 / (dim * dim), 0.0, 1.0))


def _restricted_fro_error(target_operator: np.ndarray, actual_operator: np.ndarray) -> float:
    target = np.asarray(target_operator, dtype=np.complex128)
    actual = np.asarray(actual_operator, dtype=np.complex128)
    denom = float(np.linalg.norm(target, ord="fro"))
    if denom <= 0.0:
        return float(np.linalg.norm(actual - target, ord="fro"))
    return float(np.linalg.norm(actual - target, ord="fro") / denom)


def _logical_block_slices(logical_levels: Sequence[int]) -> tuple[slice, ...]:
    return tuple(slice(2 * index, 2 * index + 2) for index, _ in enumerate(logical_levels))


def _logical_block_phase_matrix(
    model: DispersiveTransmonCavityModel,
    logical_block_phase: "LogicalBlockPhaseCorrection",
) -> np.ndarray:
    if not logical_block_phase.logical_levels:
        return np.eye(_full_dimension(model), dtype=np.complex128)
    return np.asarray(
        logical_block_phase_op(
            logical_block_phase.phases_rad,
            fock_levels=logical_block_phase.logical_levels,
            cavity_dim=int(model.n_cav),
            qubit_dim=int(model.n_tr),
        ).full(),
        dtype=np.complex128,
    )


def _apply_logical_block_phase(
    full_operator: np.ndarray,
    model: DispersiveTransmonCavityModel,
    logical_block_phase: "LogicalBlockPhaseCorrection",
) -> np.ndarray:
    full = np.asarray(full_operator, dtype=np.complex128)
    if not logical_block_phase.logical_levels:
        return full
    return _logical_block_phase_matrix(model, logical_block_phase) @ full


@dataclass(frozen=True)
class TargetedSubspaceTransferSet:
    input_states: tuple[qt.Qobj | np.ndarray, ...]
    target_states: tuple[qt.Qobj | np.ndarray, ...]
    weights: tuple[float, ...] = ()
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.input_states) == 0:
            raise ValueError("TargetedSubspaceTransferSet requires at least one state pair.")
        if len(self.input_states) != len(self.target_states):
            raise ValueError("input_states and target_states must have the same length.")
        if self.labels and len(self.labels) != len(self.input_states):
            raise ValueError("labels must match the number of state pairs.")
        weights = (
            _normalize_weights(self.weights)
            if self.weights
            else tuple(float(1.0 / len(self.input_states)) for _ in range(len(self.input_states)))
        )
        labels = self.labels or tuple(f"state_{index}" for index in range(len(self.input_states)))
        object.__setattr__(self, "weights", tuple(float(weight) for weight in weights))
        object.__setattr__(self, "labels", tuple(str(label) for label in labels))


@dataclass(frozen=True)
class TargetedSubspaceObjectiveWeights:
    qubit_weight: float = 1.0
    subspace_weight: float = 1.0
    preservation_weight: float = 1.0
    leakage_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("qubit_weight", "subspace_weight", "preservation_weight", "leakage_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")

    def as_dict(self) -> dict[str, float]:
        return {
            "qubit_weight": float(self.qubit_weight),
            "subspace_weight": float(self.subspace_weight),
            "preservation_weight": float(self.preservation_weight),
            "leakage_weight": float(self.leakage_weight),
        }


@dataclass(frozen=True)
class LogicalBlockPhaseCorrection:
    logical_levels: tuple[int, ...] = ()
    phases_rad: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        levels = tuple(int(level) for level in self.logical_levels)
        phases = tuple(float(value) for value in self.phases_rad)
        if len(levels) != len(phases):
            raise ValueError("logical_levels and phases_rad must have the same length.")
        if len(set(levels)) != len(levels):
            raise ValueError("logical_levels must not contain duplicates.")
        object.__setattr__(self, "logical_levels", levels)
        object.__setattr__(self, "phases_rad", phases)

    @classmethod
    def zeros(cls, logical_levels: Sequence[int]) -> "LogicalBlockPhaseCorrection":
        levels = tuple(int(level) for level in logical_levels)
        return cls(logical_levels=levels, phases_rad=tuple(0.0 for _ in levels))

    def phases_for_levels(self, logical_levels: Sequence[int]) -> tuple[float, ...]:
        mapping = {int(level): float(phase) for level, phase in zip(self.logical_levels, self.phases_rad, strict=True)}
        return tuple(float(mapping.get(int(level), 0.0)) for level in logical_levels)

    def reindexed(self, logical_levels: Sequence[int]) -> "LogicalBlockPhaseCorrection":
        levels = tuple(int(level) for level in logical_levels)
        return LogicalBlockPhaseCorrection(logical_levels=levels, phases_rad=self.phases_for_levels(levels))

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_levels": [int(level) for level in self.logical_levels],
            "phases_rad": [float(value) for value in self.phases_rad],
        }


@dataclass(frozen=True)
class TargetedSubspaceOptimizationConfig:
    conditioned: ConditionedOptimizationConfig = field(default_factory=ConditionedOptimizationConfig)
    include_block_phase: bool = False
    block_phase_levels: tuple[int, ...] = ()
    block_phase_bounds_rad: tuple[float, float] = (-np.pi, np.pi)
    regularization_block_phase: float = 0.0
    block_phase_reference_level: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_phase_levels", tuple(int(level) for level in self.block_phase_levels))
        if len(set(self.block_phase_levels)) != len(self.block_phase_levels):
            raise ValueError("block_phase_levels must not contain duplicates.")
        lower, upper = (float(self.block_phase_bounds_rad[0]), float(self.block_phase_bounds_rad[1]))
        if lower > upper:
            raise ValueError("block_phase_bounds_rad must satisfy lower <= upper.")
        if float(self.regularization_block_phase) < 0.0:
            raise ValueError("regularization_block_phase must be non-negative.")
        if self.block_phase_reference_level is not None:
            object.__setattr__(self, "block_phase_reference_level", int(self.block_phase_reference_level))


@dataclass(frozen=True)
class TargetedSubspaceBasisMetric:
    input_label: str
    input_block: int
    input_qubit_level: int
    same_block_population: float
    other_target_population: float
    leakage_population: float
    dominant_block: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_label": str(self.input_label),
            "input_block": int(self.input_block),
            "input_qubit_level": int(self.input_qubit_level),
            "same_block_population": float(self.same_block_population),
            "other_target_population": float(self.other_target_population),
            "leakage_population": float(self.leakage_population),
            "dominant_block": int(self.dominant_block),
        }


@dataclass(frozen=True)
class TargetedSubspaceTransferMetric:
    label: str
    weight: float
    fidelity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "weight": float(self.weight),
            "fidelity": float(self.fidelity),
        }


@dataclass
class TargetedSubspaceValidationResult:
    targets: ConditionedQubitTargets
    logical_levels: tuple[int, ...]
    logical_weights: tuple[float, ...]
    corrections: ConditionedMultitoneCorrections
    logical_block_phase: LogicalBlockPhaseCorrection
    best_fit_logical_block_phase: LogicalBlockPhaseCorrection
    objective_weights: TargetedSubspaceObjectiveWeights
    restricted_operator: np.ndarray = field(repr=False)
    block_population_matrix: np.ndarray = field(repr=False)
    conditioned_sector_metrics: tuple[ConditionedSectorMetrics, ...] = ()
    basis_metrics: tuple[TargetedSubspaceBasisMetric, ...] = ()
    transfer_metrics: tuple[TargetedSubspaceTransferMetric, ...] = ()
    block_phase_diagnostics: LogicalBlockPhaseDiagnostics | None = None
    restricted_process_fidelity: float = float("nan")
    uncorrected_restricted_process_fidelity: float = float("nan")
    best_fit_restricted_process_fidelity: float = float("nan")
    restricted_fro_error: float = float("nan")
    restricted_unitarity_error: float = float("nan")
    state_transfer_fidelity_mean: float = float("nan")
    state_transfer_fidelity_min: float = float("nan")
    same_block_population_mean: float = 0.0
    same_block_population_min: float = 0.0
    other_target_population_mean: float = 0.0
    other_target_population_max: float = 0.0
    leakage_outside_target_mean: float = 0.0
    leakage_outside_target_max: float = 0.0
    qubit_loss: float = 0.0
    subspace_loss: float = 0.0
    preservation_loss: float = 0.0
    leakage_loss: float = 0.0
    weighted_loss: float = 0.0
    target_operator: np.ndarray | None = field(default=None, repr=False)
    full_operator: np.ndarray | None = field(default=None, repr=False)
    waveform: ConditionedMultitoneWaveform | None = field(default=None, repr=False)
    compiled: CompiledSequence | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def basis_rows(self) -> list[dict[str, Any]]:
        return [metric.as_dict() for metric in self.basis_metrics]

    def transfer_rows(self) -> list[dict[str, Any]]:
        return [metric.as_dict() for metric in self.transfer_metrics]

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_levels": [int(level) for level in self.logical_levels],
            "logical_weights": [float(weight) for weight in self.logical_weights],
            "logical_block_phase": self.logical_block_phase.as_dict(),
            "best_fit_logical_block_phase": self.best_fit_logical_block_phase.as_dict(),
            "restricted_process_fidelity": float(self.restricted_process_fidelity),
            "uncorrected_restricted_process_fidelity": float(self.uncorrected_restricted_process_fidelity),
            "best_fit_restricted_process_fidelity": float(self.best_fit_restricted_process_fidelity),
            "restricted_fro_error": float(self.restricted_fro_error),
            "restricted_unitarity_error": float(self.restricted_unitarity_error),
            "state_transfer_fidelity_mean": float(self.state_transfer_fidelity_mean),
            "state_transfer_fidelity_min": float(self.state_transfer_fidelity_min),
            "same_block_population_mean": float(self.same_block_population_mean),
            "same_block_population_min": float(self.same_block_population_min),
            "other_target_population_mean": float(self.other_target_population_mean),
            "other_target_population_max": float(self.other_target_population_max),
            "leakage_outside_target_mean": float(self.leakage_outside_target_mean),
            "leakage_outside_target_max": float(self.leakage_outside_target_max),
            "qubit_loss": float(self.qubit_loss),
            "subspace_loss": float(self.subspace_loss),
            "preservation_loss": float(self.preservation_loss),
            "leakage_loss": float(self.leakage_loss),
            "weighted_loss": float(self.weighted_loss),
            "objective_weights": self.objective_weights.as_dict(),
            "block_population_matrix": np.asarray(self.block_population_matrix, dtype=float).tolist(),
            "conditioned_sector_metrics": [metric.as_dict() for metric in self.conditioned_sector_metrics],
            "basis_metrics": self.basis_rows(),
            "transfer_metrics": self.transfer_rows(),
            "block_phase_diagnostics": (
                None if self.block_phase_diagnostics is None else self.block_phase_diagnostics.as_dict()
            ),
            "metadata": dict(self.metadata),
        }


@dataclass
class TargetedSubspaceOptimizationResult:
    initial_result: TargetedSubspaceValidationResult
    optimized_result: TargetedSubspaceValidationResult
    optimized_corrections: ConditionedMultitoneCorrections
    optimized_logical_block_phase: LogicalBlockPhaseCorrection
    active_levels: tuple[int, ...]
    parameters: tuple[str, ...]
    block_phase_levels: tuple[int, ...] = ()
    history: list[dict[str, float]] = field(default_factory=list)
    success_stage1: bool = False
    success_stage2: bool = False
    message_stage1: str = ""
    message_stage2: str = ""

    def improvement_summary(self) -> dict[str, Any]:
        return {
            "initial_weighted_loss": float(self.initial_result.weighted_loss),
            "optimized_weighted_loss": float(self.optimized_result.weighted_loss),
            "initial_restricted_process_fidelity": float(self.initial_result.restricted_process_fidelity),
            "optimized_restricted_process_fidelity": float(self.optimized_result.restricted_process_fidelity),
            "initial_block_phase_rms_rad": float(
                float("nan")
                if self.initial_result.block_phase_diagnostics is None
                else self.initial_result.block_phase_diagnostics.rms_block_phase_error_rad
            ),
            "optimized_block_phase_rms_rad": float(
                float("nan")
                if self.optimized_result.block_phase_diagnostics is None
                else self.optimized_result.block_phase_diagnostics.rms_block_phase_error_rad
            ),
            "initial_same_block_population_mean": float(self.initial_result.same_block_population_mean),
            "optimized_same_block_population_mean": float(self.optimized_result.same_block_population_mean),
            "initial_leakage_outside_target_mean": float(self.initial_result.leakage_outside_target_mean),
            "optimized_leakage_outside_target_mean": float(self.optimized_result.leakage_outside_target_mean),
            "active_levels": [int(level) for level in self.active_levels],
            "block_phase_levels": [int(level) for level in self.block_phase_levels],
            "parameters": [str(name) for name in self.parameters],
        }


def build_block_rotation_target_operator(
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    *,
    logical_levels: Sequence[int] | None = None,
) -> np.ndarray:
    target_obj = _ensure_targets(targets)
    resolved_levels = _resolve_logical_levels(target_obj, logical_levels)
    dim = 2 * len(resolved_levels)
    operator = np.zeros((dim, dim), dtype=np.complex128)
    for block_index, level in enumerate(resolved_levels):
        operator[2 * block_index : 2 * block_index + 2, 2 * block_index : 2 * block_index + 2] = np.asarray(
            qubit_rotation_xy(float(target_obj.theta[int(level)]), float(target_obj.phi[int(level)])).full(),
            dtype=np.complex128,
        )
    return operator


def build_spanning_state_transfer_set(
    target_operator: np.ndarray,
    *,
    include_pairwise_superpositions: bool = True,
) -> TargetedSubspaceTransferSet:
    operator = np.asarray(target_operator, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("target_operator must be a square matrix.")
    dim = int(operator.shape[0])
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    labels: list[str] = []
    eye = np.eye(dim, dtype=np.complex128)
    for basis_index in range(dim):
        psi = eye[:, basis_index]
        inputs.append(psi)
        outputs.append(operator @ psi)
        labels.append(f"basis_{basis_index}")
    if include_pairwise_superpositions:
        for left in range(dim):
            for right in range(left + 1, dim):
                plus = (eye[:, left] + eye[:, right]) / np.sqrt(2.0)
                phase = (eye[:, left] + 1.0j * eye[:, right]) / np.sqrt(2.0)
                inputs.extend((plus, phase))
                outputs.extend((operator @ plus, operator @ phase))
                labels.extend((f"plus_{left}_{right}", f"phase_{left}_{right}"))
    return TargetedSubspaceTransferSet(
        input_states=tuple(inputs),
        target_states=tuple(outputs),
        labels=tuple(labels),
    )


def _conditioned_metric(
    targets: ConditionedQubitTargets,
    *,
    level: int,
    weight: float,
    rho_q: qt.Qobj,
    sector_population: float,
) -> ConditionedSectorMetrics:
    target_bloch_x = float(np.sin(float(targets.theta[level])) * np.cos(float(targets.phi[level])))
    target_bloch_y = float(np.sin(float(targets.theta[level])) * np.sin(float(targets.phi[level])))
    target_bloch_z = float(np.cos(float(targets.theta[level])))
    x, y, z = bloch_xyz_from_qubit_state(rho_q)
    theta_sim, phi_sim, bloch_radius = bloch_angles_from_density_matrix(rho_q)
    theta_target = float(targets.theta[level])
    phi_target = float(np.mod(targets.phi[level], 2.0 * np.pi))
    theta_error = float("nan") if np.isnan(theta_sim) else _wrap_pi(theta_sim - theta_target)
    phi_error = float("nan") if np.isnan(phi_sim) else _wrap_pi(phi_sim - phi_target)
    target_dm = qubit_density_matrix_from_angles(theta_target, phi_target)
    # When rho_q comes from an n_tr > 2 model, truncate to {|g>,|e>} for
    # the fidelity comparison (target_dm is always 2x2).
    rho_for_fidelity = rho_q
    if rho_q.shape[0] > 2:
        rho_for_fidelity, _ = truncate_to_qubit_subspace(rho_q)
    fidelity = float(np.clip(np.real((target_dm * rho_for_fidelity).tr()), 0.0, 1.0))
    purity = float(np.real((rho_q * rho_q).tr()))
    bloch_distance = float(
        np.sqrt(
            (x - target_bloch_x) ** 2
            + (y - target_bloch_y) ** 2
            + (z - target_bloch_z) ** 2
        )
    )
    return ConditionedSectorMetrics(
        n=int(level),
        weight=float(weight),
        fidelity=float(fidelity),
        target_theta_rad=float(theta_target),
        target_phi_rad=float(phi_target),
        target_bloch_x=float(target_bloch_x),
        target_bloch_y=float(target_bloch_y),
        target_bloch_z=float(target_bloch_z),
        simulated_bloch_x=float(x),
        simulated_bloch_y=float(y),
        simulated_bloch_z=float(z),
        bloch_radius=float(bloch_radius),
        purity=float(purity),
        theta_simulated_rad=float(theta_sim),
        phi_simulated_rad=float(phi_sim),
        theta_error_rad=float(theta_error),
        phi_error_rad=float(phi_error),
        bloch_distance=float(bloch_distance),
        sector_population=float(sector_population),
        dominant_error=_classify_error(theta_error, phi_error, bloch_radius),
    )


def _conditioned_metrics_from_operator(
    actual_full_operator: np.ndarray,
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets,
    logical_levels: Sequence[int],
    logical_weights: Sequence[float],
) -> tuple[ConditionedSectorMetrics, ...]:
    full = np.asarray(actual_full_operator, dtype=np.complex128)
    metrics: list[ConditionedSectorMetrics] = []
    for weight, level in zip(logical_weights, logical_levels, strict=True):
        initial = model.basis_state(0, int(level))
        final_vector = full @ np.asarray(initial.full(), dtype=np.complex128).reshape(-1)
        final_state = qt.Qobj(final_vector.reshape((-1, 1)), dims=initial.dims)
        rho_q, population, valid = conditioned_qubit_state(final_state, n=int(level), fallback="zero")
        if not valid:
            n_q = int(model.n_tr)
            rho_q = qt.Qobj(np.zeros((n_q, n_q), dtype=np.complex128), dims=[[n_q], [n_q]])
        metrics.append(
            _conditioned_metric(
                targets,
                level=int(level),
                weight=float(weight),
                rho_q=rho_q,
                sector_population=float(population),
            )
        )
    return tuple(metrics)


def _basis_population_metrics(
    actual_full_operator: np.ndarray,
    model: DispersiveTransmonCavityModel,
    logical_levels: Sequence[int],
) -> tuple[
    tuple[TargetedSubspaceBasisMetric, ...],
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    full = np.asarray(actual_full_operator, dtype=np.complex128)
    logical_blocks = [np.asarray(qubit_cavity_block_indices(int(model.n_cav), int(level)), dtype=int) for level in logical_levels]
    basis_rows: list[TargetedSubspaceBasisMetric] = []
    block_matrix = np.zeros((len(logical_blocks), len(logical_blocks)), dtype=float)
    same_values: list[float] = []
    other_values: list[float] = []
    leak_values: list[float] = []
    for input_block_index, level in enumerate(logical_levels):
        per_qubit_rows: list[np.ndarray] = []
        for qubit_level in (0, 1):
            input_index = qubit_cavity_index(int(model.n_cav), qubit_level, int(level))
            column = full[:, int(input_index)]
            per_block = np.asarray(
                [float(np.sum(np.abs(column[indices]) ** 2)) for indices in logical_blocks],
                dtype=float,
            )
            same_block = float(per_block[input_block_index])
            targeted_total = float(np.sum(per_block))
            other_target = float(max(targeted_total - same_block, 0.0))
            leakage = float(max(1.0 - targeted_total, 0.0))
            dominant_block = int(logical_levels[int(np.argmax(per_block))]) if per_block.size else int(level)
            basis_rows.append(
                TargetedSubspaceBasisMetric(
                    input_label=f"|{'g' if qubit_level == 0 else 'e'},{int(level)}>",
                    input_block=int(level),
                    input_qubit_level=int(qubit_level),
                    same_block_population=float(same_block),
                    other_target_population=float(other_target),
                    leakage_population=float(leakage),
                    dominant_block=dominant_block,
                )
            )
            per_qubit_rows.append(per_block)
            same_values.append(same_block)
            other_values.append(other_target)
            leak_values.append(leakage)
        block_matrix[:, input_block_index] = np.mean(np.asarray(per_qubit_rows, dtype=float), axis=0)
    return (
        tuple(basis_rows),
        block_matrix,
        float(np.mean(same_values)),
        float(np.min(same_values)),
        float(np.mean(other_values)),
        float(np.max(other_values)),
        float(np.mean(leak_values)),
        float(np.max(leak_values)),
    )


def _transfer_metrics(
    actual_full_operator: np.ndarray,
    model: DispersiveTransmonCavityModel,
    logical_levels: Sequence[int],
    transfer_set: TargetedSubspaceTransferSet | None,
) -> tuple[tuple[TargetedSubspaceTransferMetric, ...], float, float]:
    if transfer_set is None:
        return tuple(), float("nan"), float("nan")
    logical_indices = _logical_indices(model, logical_levels)
    full_dim = _full_dimension(model)
    logical_dim = len(logical_indices)
    subsystem_dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims", (int(model.n_tr), int(model.n_cav))))
    full = np.asarray(actual_full_operator, dtype=np.complex128)
    rows: list[TargetedSubspaceTransferMetric] = []
    fidelities: list[float] = []
    for label, weight, input_state, target_state in zip(
        transfer_set.labels,
        transfer_set.weights,
        transfer_set.input_states,
        transfer_set.target_states,
        strict=True,
    ):
        input_qobj = _state_qobj(
            input_state,
            full_dim=full_dim,
            logical_dim=logical_dim,
            logical_indices=logical_indices,
            subsystem_dims=subsystem_dims,
        )
        target_qobj = _state_qobj(
            target_state,
            full_dim=full_dim,
            logical_dim=logical_dim,
            logical_indices=logical_indices,
            subsystem_dims=subsystem_dims,
        )
        if input_qobj.isoper:
            actual_qobj = qt.Qobj(full @ np.asarray(input_qobj.full(), dtype=np.complex128) @ full.conj().T, dims=input_qobj.dims)
        else:
            actual_qobj = qt.Qobj(full @ np.asarray(input_qobj.full(), dtype=np.complex128), dims=input_qobj.dims)
        fidelity = _state_fidelity(actual_qobj, target_qobj)
        rows.append(TargetedSubspaceTransferMetric(label=str(label), weight=float(weight), fidelity=float(fidelity)))
        fidelities.append(float(fidelity))
    weights = np.asarray(transfer_set.weights, dtype=float)
    fidelity_arr = np.asarray(fidelities, dtype=float)
    return tuple(rows), float(np.sum(weights * fidelity_arr)), float(np.min(fidelity_arr))


def analyze_targeted_subspace_operator(
    actual_full_operator: np.ndarray,
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    *,
    logical_levels: Sequence[int] | None = None,
    corrections: ConditionedMultitoneCorrections | None = None,
    logical_block_phase: LogicalBlockPhaseCorrection | None = None,
    objective_weights: TargetedSubspaceObjectiveWeights | None = None,
    target_operator: np.ndarray | None = None,
    transfer_set: TargetedSubspaceTransferSet | None = None,
    waveform: ConditionedMultitoneWaveform | None = None,
    compiled: CompiledSequence | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TargetedSubspaceValidationResult:
    target_obj = _ensure_targets(targets)
    resolved_levels = _resolve_logical_levels(target_obj, logical_levels)
    logical_weights = _logical_weights(target_obj, resolved_levels)
    weights = TargetedSubspaceObjectiveWeights() if objective_weights is None else objective_weights
    correction_obj = (
        ConditionedMultitoneCorrections.zeros(target_obj.n_levels)
        if corrections is None
        else corrections.padded(target_obj.n_levels)
    )
    full = np.asarray(actual_full_operator, dtype=np.complex128)
    expected_shape = (_full_dimension(model), _full_dimension(model))
    if full.shape != expected_shape:
        raise ValueError(f"actual_full_operator has shape {full.shape}, expected {expected_shape}.")
    logical_indices = _logical_indices(model, resolved_levels)
    resolved_target_operator = (
        build_block_rotation_target_operator(target_obj, logical_levels=resolved_levels)
        if target_operator is None
        else np.asarray(target_operator, dtype=np.complex128)
    )
    raw_restricted = np.asarray(full[np.ix_(logical_indices, logical_indices)], dtype=np.complex128)
    resolved_logical_block_phase = (
        LogicalBlockPhaseCorrection.zeros(resolved_levels)
        if logical_block_phase is None
        else logical_block_phase.reindexed(resolved_levels)
    )
    corrected_full = _apply_logical_block_phase(full, model, resolved_logical_block_phase)
    restricted = np.asarray(corrected_full[np.ix_(logical_indices, logical_indices)], dtype=np.complex128)
    if resolved_target_operator.shape != raw_restricted.shape:
        raise ValueError(
            f"target_operator has shape {resolved_target_operator.shape}, expected {raw_restricted.shape} for the selected logical levels."
        )
    block_phase_diagnostics = logical_block_phase_diagnostics(
        raw_restricted,
        resolved_target_operator,
        block_slices=_logical_block_slices(resolved_levels),
        applied_correction_phases=resolved_logical_block_phase.phases_for_levels(resolved_levels),
    )
    best_fit_logical_block_phase = LogicalBlockPhaseCorrection(
        logical_levels=tuple(int(level) for level in resolved_levels),
        phases_rad=tuple(float(value) for value in block_phase_diagnostics.best_fit_correction_phases_rad),
    )
    best_fit_full = _apply_logical_block_phase(full, model, best_fit_logical_block_phase)
    best_fit_restricted = np.asarray(best_fit_full[np.ix_(logical_indices, logical_indices)], dtype=np.complex128)
    conditioned_metrics = _conditioned_metrics_from_operator(corrected_full, model, target_obj, resolved_levels, logical_weights)
    basis_metrics, block_matrix, same_mean, same_min, other_mean, other_max, leak_mean, leak_max = _basis_population_metrics(
        corrected_full,
        model,
        resolved_levels,
    )
    transfer_metrics, transfer_mean, transfer_min = _transfer_metrics(corrected_full, model, resolved_levels, transfer_set)
    qubit_loss = float(np.sum(np.asarray(logical_weights, dtype=float) * (1.0 - np.asarray([row.fidelity for row in conditioned_metrics], dtype=float))))
    restricted_fidelity = _restricted_process_fidelity(resolved_target_operator, restricted)
    uncorrected_restricted_fidelity = _restricted_process_fidelity(resolved_target_operator, raw_restricted)
    best_fit_restricted_fidelity = _restricted_process_fidelity(resolved_target_operator, best_fit_restricted)
    restricted_fro_error = _restricted_fro_error(resolved_target_operator, restricted)
    restricted_unitarity_error = float(np.linalg.norm(restricted.conj().T @ restricted - np.eye(restricted.shape[0], dtype=np.complex128), ord="fro"))
    subspace_loss = float(1.0 - restricted_fidelity)
    if transfer_set is not None and not np.isfinite(subspace_loss):
        subspace_loss = float(1.0 - transfer_mean)
    preservation_loss = float(other_mean)
    leakage_loss = float(leak_mean)
    weighted_loss = float(
        float(weights.qubit_weight) * qubit_loss
        + float(weights.subspace_weight) * subspace_loss
        + float(weights.preservation_weight) * preservation_loss
        + float(weights.leakage_weight) * leakage_loss
    )
    metadata_payload = dict(metadata or {})
    metadata_payload.update(
        {
            "logical_indices": [int(index) for index in logical_indices],
            "subspace_metric_source": "restricted_operator",
            "has_transfer_set": bool(transfer_set is not None),
            "logical_block_phase_applied": resolved_logical_block_phase.as_dict(),
            "logical_block_phase_best_fit": best_fit_logical_block_phase.as_dict(),
        }
    )
    return TargetedSubspaceValidationResult(
        targets=target_obj,
        logical_levels=tuple(int(level) for level in resolved_levels),
        logical_weights=tuple(float(weight) for weight in logical_weights),
        corrections=correction_obj,
        logical_block_phase=resolved_logical_block_phase,
        best_fit_logical_block_phase=best_fit_logical_block_phase,
        objective_weights=weights,
        restricted_operator=restricted,
        target_operator=resolved_target_operator,
        block_population_matrix=block_matrix,
        conditioned_sector_metrics=conditioned_metrics,
        basis_metrics=basis_metrics,
        transfer_metrics=transfer_metrics,
        block_phase_diagnostics=block_phase_diagnostics,
        restricted_process_fidelity=float(restricted_fidelity),
        uncorrected_restricted_process_fidelity=float(uncorrected_restricted_fidelity),
        best_fit_restricted_process_fidelity=float(best_fit_restricted_fidelity),
        restricted_fro_error=float(restricted_fro_error),
        restricted_unitarity_error=float(restricted_unitarity_error),
        state_transfer_fidelity_mean=float(transfer_mean),
        state_transfer_fidelity_min=float(transfer_min),
        same_block_population_mean=float(same_mean),
        same_block_population_min=float(same_min),
        other_target_population_mean=float(other_mean),
        other_target_population_max=float(other_max),
        leakage_outside_target_mean=float(leak_mean),
        leakage_outside_target_max=float(leak_max),
        qubit_loss=float(qubit_loss),
        subspace_loss=float(subspace_loss),
        preservation_loss=float(preservation_loss),
        leakage_loss=float(leakage_loss),
        weighted_loss=float(weighted_loss),
        full_operator=corrected_full,
        waveform=waveform,
        compiled=compiled,
        metadata=metadata_payload,
    )


def _final_full_unitary(
    model: DispersiveTransmonCavityModel,
    waveform: ConditionedMultitoneWaveform,
    run_config: ConditionedMultitoneRunConfig,
    *,
    compiled: CompiledSequence,
) -> np.ndarray:
    hamiltonian = hamiltonian_time_slices(model, compiled, waveform.drive_ops, frame=run_config.frame)
    propagators = qt.propagator(
        hamiltonian,
        compiled.tlist,
        options=_solver_options(run_config),
        tlist=compiled.tlist,
    )
    final = propagators[-1] if isinstance(propagators, list) else propagators
    return np.asarray(final.full(), dtype=np.complex128)


def evaluate_targeted_subspace_multitone(
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    waveform: ConditionedMultitoneWaveform,
    run_config: ConditionedMultitoneRunConfig,
    *,
    corrections: ConditionedMultitoneCorrections | None = None,
    logical_block_phase: LogicalBlockPhaseCorrection | None = None,
    logical_levels: Sequence[int] | None = None,
    objective_weights: TargetedSubspaceObjectiveWeights | None = None,
    target_operator: np.ndarray | None = None,
    transfer_set: TargetedSubspaceTransferSet | None = None,
) -> TargetedSubspaceValidationResult:
    compiled = compile_conditioned_multitone_waveform(waveform, run_config)
    full_operator = _final_full_unitary(model, waveform, run_config, compiled=compiled)
    return analyze_targeted_subspace_operator(
        full_operator,
        model,
        targets,
        logical_levels=logical_levels,
        corrections=corrections,
        logical_block_phase=logical_block_phase,
        objective_weights=objective_weights,
        target_operator=target_operator,
        transfer_set=transfer_set,
        waveform=waveform,
        compiled=compiled,
        metadata={
            "t_s": np.asarray(compiled.tlist, dtype=float).tolist(),
            "tone_specs": waveform.tone_rows(),
            "waveform_metadata": dict(waveform.metadata),
        },
    )


def run_targeted_subspace_multitone_validation(
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    run_config: ConditionedMultitoneRunConfig,
    *,
    corrections: ConditionedMultitoneCorrections | None = None,
    logical_block_phase: LogicalBlockPhaseCorrection | None = None,
    logical_levels: Sequence[int] | None = None,
    objective_weights: TargetedSubspaceObjectiveWeights | None = None,
    target_operator: np.ndarray | None = None,
    transfer_set: TargetedSubspaceTransferSet | None = None,
    channel: str = "qubit",
    drive_target: str = "qubit",
    label: str | None = None,
) -> TargetedSubspaceValidationResult:
    target_obj = _ensure_targets(targets)
    correction_obj = (
        ConditionedMultitoneCorrections.zeros(target_obj.n_levels)
        if corrections is None
        else corrections.padded(target_obj.n_levels)
    )
    tones = build_conditioned_multitone_tones(model, target_obj, run_config, corrections=correction_obj)
    waveform = build_conditioned_multitone_waveform(
        tones,
        run_config,
        channel=channel,
        drive_target=drive_target,
        label=label,
    )
    return evaluate_targeted_subspace_multitone(
        model,
        target_obj,
        waveform,
        run_config,
        corrections=correction_obj,
        logical_block_phase=logical_block_phase,
        logical_levels=logical_levels,
        objective_weights=objective_weights,
        target_operator=target_operator,
        transfer_set=transfer_set,
    )


def _active_levels(
    logical_levels: Sequence[int],
    logical_weights: Sequence[float],
    optimization_config: ConditionedOptimizationConfig,
) -> tuple[int, ...]:
    if optimization_config.active_levels:
        return tuple(int(level) for level in optimization_config.active_levels)
    return tuple(
        int(level)
        for level, weight in zip(logical_levels, logical_weights, strict=True)
        if float(weight) > 0.0
    )


def _coerce_targeted_optimization_config(
    optimization_config: ConditionedOptimizationConfig | TargetedSubspaceOptimizationConfig | None,
) -> TargetedSubspaceOptimizationConfig:
    if optimization_config is None:
        return TargetedSubspaceOptimizationConfig()
    if isinstance(optimization_config, TargetedSubspaceOptimizationConfig):
        return optimization_config
    return TargetedSubspaceOptimizationConfig(conditioned=optimization_config)


def _validate_block_phase_levels(
    logical_levels: Sequence[int],
    block_phase_levels: Sequence[int],
) -> tuple[int, ...]:
    resolved = tuple(int(level) for level in block_phase_levels)
    logical_set = {int(level) for level in logical_levels}
    for level in resolved:
        if level not in logical_set:
            raise ValueError(f"block_phase level {level} is not part of the selected logical levels {tuple(logical_levels)}.")
    return resolved


def _block_phase_parameter_levels(
    logical_levels: Sequence[int],
    logical_weights: Sequence[float],
    optimization_config: TargetedSubspaceOptimizationConfig,
) -> tuple[tuple[int, ...], int | None]:
    if not optimization_config.include_block_phase:
        return tuple(), None
    if optimization_config.block_phase_levels:
        active = _validate_block_phase_levels(logical_levels, optimization_config.block_phase_levels)
    else:
        active = tuple(
            int(level)
            for level, weight in zip(logical_levels, logical_weights, strict=True)
            if float(weight) > 0.0
        )
    if not active:
        return tuple(), None
    reference = active[0] if optimization_config.block_phase_reference_level is None else int(optimization_config.block_phase_reference_level)
    if reference not in active:
        raise ValueError("block_phase_reference_level must be one of the active logical block-phase levels.")
    return tuple(level for level in active if level != reference), int(reference)


def _gauge_fix_logical_block_phase(
    logical_block_phase: LogicalBlockPhaseCorrection,
    reference_level: int | None,
) -> LogicalBlockPhaseCorrection:
    if reference_level is None or reference_level not in set(logical_block_phase.logical_levels):
        return logical_block_phase
    levels = tuple(int(level) for level in logical_block_phase.logical_levels)
    phases = np.asarray(logical_block_phase.phases_for_levels(levels), dtype=float)
    reference_index = levels.index(int(reference_level))
    phases = _wrap_pi(phases - float(phases[reference_index]))
    return LogicalBlockPhaseCorrection(logical_levels=levels, phases_rad=tuple(float(value) for value in phases))


def _vector_from_logical_block_phase(
    logical_block_phase: LogicalBlockPhaseCorrection,
    parameter_levels: Sequence[int],
) -> np.ndarray:
    if not parameter_levels:
        return np.zeros(0, dtype=float)
    phase_map = {
        int(level): float(phase)
        for level, phase in zip(logical_block_phase.logical_levels, logical_block_phase.phases_rad, strict=True)
    }
    return np.asarray([float(phase_map.get(int(level), 0.0)) for level in parameter_levels], dtype=float)


def _logical_block_phase_from_vector(
    base: LogicalBlockPhaseCorrection,
    vector: np.ndarray,
    parameter_levels: Sequence[int],
) -> LogicalBlockPhaseCorrection:
    phase_map = {
        int(level): float(phase)
        for level, phase in zip(base.logical_levels, base.phases_rad, strict=True)
    }
    data = np.asarray(vector, dtype=float).reshape(-1)
    if data.size != len(parameter_levels):
        raise ValueError(f"Expected block-phase vector of length {len(parameter_levels)}, received {data.size}.")
    for level, value in zip(parameter_levels, data, strict=True):
        phase_map[int(level)] = float(value)
    return LogicalBlockPhaseCorrection(
        logical_levels=tuple(int(level) for level in base.logical_levels),
        phases_rad=tuple(float(phase_map[int(level)]) for level in base.logical_levels),
    )


def _block_phase_bounds(
    optimization_config: TargetedSubspaceOptimizationConfig,
    parameter_levels: Sequence[int],
) -> Bounds:
    lower = np.full(len(parameter_levels), float(optimization_config.block_phase_bounds_rad[0]), dtype=float)
    upper = np.full(len(parameter_levels), float(optimization_config.block_phase_bounds_rad[1]), dtype=float)
    return Bounds(lower, upper)


def _block_phase_regularization_cost(
    logical_block_phase: LogicalBlockPhaseCorrection,
    parameter_levels: Sequence[int],
    optimization_config: TargetedSubspaceOptimizationConfig,
) -> float:
    if not parameter_levels:
        return 0.0
    phase_map = {
        int(level): float(phase)
        for level, phase in zip(logical_block_phase.logical_levels, logical_block_phase.phases_rad, strict=True)
    }
    value = sum(float(phase_map[int(level)] ** 2) for level in parameter_levels)
    return float(optimization_config.regularization_block_phase) * float(value)


def _vector_from_corrections(
    corrections: ConditionedMultitoneCorrections,
    active_levels: Sequence[int],
    parameters: Sequence[str],
) -> np.ndarray:
    corr = corrections.padded(max(int(max(active_levels, default=-1)) + 1, len(corrections.d_lambda), 1))
    vector: list[float] = []
    for level in active_levels:
        for name in parameters:
            if name == "d_lambda":
                vector.append(float(corr.d_lambda[int(level)]))
            elif name == "d_alpha":
                vector.append(float(corr.d_alpha[int(level)]))
            elif name == "d_omega":
                vector.append(float(corr.d_omega_rad_s[int(level)]))
    return np.asarray(vector, dtype=float)


def _corrections_from_vector(
    base: ConditionedMultitoneCorrections,
    vector: np.ndarray,
    n_levels: int,
    active_levels: Sequence[int],
    parameters: Sequence[str],
) -> ConditionedMultitoneCorrections:
    corr = base.padded(n_levels)
    d_lambda = np.asarray(corr.d_lambda, dtype=float)
    d_alpha = np.asarray(corr.d_alpha, dtype=float)
    d_omega = np.asarray(corr.d_omega_rad_s, dtype=float)
    data = np.asarray(vector, dtype=float).reshape(-1)
    expected = len(active_levels) * len(parameters)
    if data.size != expected:
        raise ValueError(f"Expected optimization vector of length {expected}, received {data.size}.")
    offset = 0
    for level in active_levels:
        for name in parameters:
            if name == "d_lambda":
                d_lambda[int(level)] = float(data[offset])
            elif name == "d_alpha":
                d_alpha[int(level)] = float(data[offset])
            elif name == "d_omega":
                d_omega[int(level)] = float(data[offset])
            offset += 1
    return ConditionedMultitoneCorrections(
        d_lambda=tuple(float(value) for value in d_lambda),
        d_alpha=tuple(float(value) for value in d_alpha),
        d_omega_rad_s=tuple(float(value) for value in d_omega),
    )


def _optimization_bounds(
    optimization_config: ConditionedOptimizationConfig,
    active_levels: Sequence[int],
) -> Bounds:
    lower: list[float] = []
    upper: list[float] = []
    for _level in active_levels:
        for name in optimization_config.parameters:
            if name == "d_lambda":
                lower.append(float(optimization_config.d_lambda_bounds[0]))
                upper.append(float(optimization_config.d_lambda_bounds[1]))
            elif name == "d_alpha":
                lower.append(float(optimization_config.d_alpha_bounds[0]))
                upper.append(float(optimization_config.d_alpha_bounds[1]))
            elif name == "d_omega":
                lower.append(float(2.0 * np.pi * optimization_config.d_omega_hz_bounds[0]))
                upper.append(float(2.0 * np.pi * optimization_config.d_omega_hz_bounds[1]))
    return Bounds(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _regularization_cost(
    corrections: ConditionedMultitoneCorrections,
    active_levels: Sequence[int],
    parameters: Sequence[str],
    optimization_config: ConditionedOptimizationConfig,
) -> float:
    value = 0.0
    for level in active_levels:
        d_lambda, d_alpha, d_omega = corrections.correction_for_n(int(level))
        for name in parameters:
            if name == "d_lambda":
                value += float(optimization_config.regularization_lambda) * float(d_lambda**2)
            elif name == "d_alpha":
                value += float(optimization_config.regularization_alpha) * float(d_alpha**2)
            elif name == "d_omega":
                value += float(optimization_config.regularization_omega) * float(d_omega**2)
    return float(value)


def optimize_targeted_subspace_multitone(
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    run_config: ConditionedMultitoneRunConfig,
    *,
    logical_levels: Sequence[int] | None = None,
    initial_corrections: ConditionedMultitoneCorrections | None = None,
    initial_logical_block_phase: LogicalBlockPhaseCorrection | None = None,
    optimization_config: ConditionedOptimizationConfig | TargetedSubspaceOptimizationConfig | None = None,
    objective_weights: TargetedSubspaceObjectiveWeights | None = None,
    target_operator: np.ndarray | None = None,
    transfer_set: TargetedSubspaceTransferSet | None = None,
    channel: str = "qubit",
    drive_target: str = "qubit",
    label: str | None = None,
) -> TargetedSubspaceOptimizationResult:
    target_obj = _ensure_targets(targets)
    resolved_levels = _resolve_logical_levels(target_obj, logical_levels)
    logical_weights = _logical_weights(target_obj, resolved_levels)
    opt_cfg = _coerce_targeted_optimization_config(optimization_config)
    base_corr = (
        ConditionedMultitoneCorrections.zeros(target_obj.n_levels)
        if initial_corrections is None
        else initial_corrections.padded(target_obj.n_levels)
    )
    block_phase_parameter_levels, block_phase_reference_level = _block_phase_parameter_levels(
        resolved_levels,
        logical_weights,
        opt_cfg,
    )
    base_block_phase = _gauge_fix_logical_block_phase(
        LogicalBlockPhaseCorrection.zeros(resolved_levels)
        if initial_logical_block_phase is None
        else initial_logical_block_phase.reindexed(resolved_levels),
        block_phase_reference_level,
    )
    active_levels = _active_levels(resolved_levels, logical_weights, opt_cfg.conditioned)
    correction_x0 = _vector_from_corrections(base_corr, active_levels, opt_cfg.conditioned.parameters)
    block_phase_x0 = _vector_from_logical_block_phase(base_block_phase, block_phase_parameter_levels)
    x0 = np.concatenate([correction_x0, block_phase_x0])
    corr_bounds = _optimization_bounds(opt_cfg.conditioned, active_levels)
    block_bounds = _block_phase_bounds(opt_cfg, block_phase_parameter_levels)
    bounds = Bounds(
        np.concatenate([np.asarray(corr_bounds.lb, dtype=float), np.asarray(block_bounds.lb, dtype=float)]),
        np.concatenate([np.asarray(corr_bounds.ub, dtype=float), np.asarray(block_bounds.ub, dtype=float)]),
    )
    history: list[dict[str, float]] = []

    initial_result = run_targeted_subspace_multitone_validation(
        model,
        target_obj,
        run_config,
        corrections=base_corr,
        logical_block_phase=base_block_phase,
        logical_levels=resolved_levels,
        objective_weights=objective_weights,
        target_operator=target_operator,
        transfer_set=transfer_set,
        channel=channel,
        drive_target=drive_target,
        label=label,
    )

    if x0.size == 0:
        return TargetedSubspaceOptimizationResult(
            initial_result=initial_result,
            optimized_result=initial_result,
            optimized_corrections=base_corr,
            optimized_logical_block_phase=base_block_phase,
            active_levels=tuple(int(level) for level in active_levels),
            block_phase_levels=tuple(int(level) for level in block_phase_parameter_levels),
            parameters=tuple(),
            history=history,
            success_stage1=True,
            success_stage2=True,
            message_stage1="No active optimization variables.",
            message_stage2="No active optimization variables.",
        )

    n_corr = int(correction_x0.size)

    def objective(vector: np.ndarray) -> float:
        vector_arr = np.asarray(vector, dtype=float).reshape(-1)
        corrections = _corrections_from_vector(
            base_corr,
            vector_arr[:n_corr],
            target_obj.n_levels,
            active_levels,
            opt_cfg.conditioned.parameters,
        )
        logical_block_phase = _logical_block_phase_from_vector(
            base_block_phase,
            vector_arr[n_corr:],
            block_phase_parameter_levels,
        )
        validation = run_targeted_subspace_multitone_validation(
            model,
            target_obj,
            run_config,
            corrections=corrections,
            logical_block_phase=logical_block_phase,
            logical_levels=resolved_levels,
            objective_weights=objective_weights,
            target_operator=target_operator,
            transfer_set=transfer_set,
            channel=channel,
            drive_target=drive_target,
            label=label,
        )
        regularization = _regularization_cost(corrections, active_levels, opt_cfg.conditioned.parameters, opt_cfg.conditioned)
        regularization += _block_phase_regularization_cost(logical_block_phase, block_phase_parameter_levels, opt_cfg)
        objective_value = float(validation.weighted_loss + regularization)
        history.append(
            {
                "evaluation": float(len(history)),
                "weighted_loss": float(validation.weighted_loss),
                "regularization": float(regularization),
                "objective": float(objective_value),
                "restricted_process_fidelity": float(validation.restricted_process_fidelity),
                "logical_block_phase_rms_rad": float(
                    float("nan")
                    if validation.block_phase_diagnostics is None
                    else validation.block_phase_diagnostics.rms_block_phase_error_rad
                ),
                "same_block_population_mean": float(validation.same_block_population_mean),
                "leakage_outside_target_mean": float(validation.leakage_outside_target_mean),
            }
        )
        return objective_value

    stage1 = minimize(
        objective,
        x0=x0,
        method=str(opt_cfg.conditioned.method_stage1),
        bounds=bounds,
        options={"maxiter": int(opt_cfg.conditioned.maxiter_stage1), "disp": False},
    )

    stage2 = None
    if opt_cfg.conditioned.method_stage2:
        stage2 = minimize(
            objective,
            x0=np.asarray(stage1.x, dtype=float),
            method=str(opt_cfg.conditioned.method_stage2),
            bounds=bounds,
            options={"maxiter": int(opt_cfg.conditioned.maxiter_stage2)},
        )

    candidates = [np.asarray(stage1.x, dtype=float)]
    candidate_scores = [float(stage1.fun)]
    if stage2 is not None:
        candidates.append(np.asarray(stage2.x, dtype=float))
        candidate_scores.append(float(stage2.fun))
    best_vector = candidates[int(np.argmin(candidate_scores))]
    optimized_corrections = _corrections_from_vector(
        base_corr,
        best_vector[:n_corr],
        target_obj.n_levels,
        active_levels,
        opt_cfg.conditioned.parameters,
    )
    optimized_logical_block_phase = _logical_block_phase_from_vector(
        base_block_phase,
        best_vector[n_corr:],
        block_phase_parameter_levels,
    )
    optimized_result = run_targeted_subspace_multitone_validation(
        model,
        target_obj,
        run_config,
        corrections=optimized_corrections,
        logical_block_phase=optimized_logical_block_phase,
        logical_levels=resolved_levels,
        objective_weights=objective_weights,
        target_operator=target_operator,
        transfer_set=transfer_set,
        channel=channel,
        drive_target=drive_target,
        label=label,
    )
    return TargetedSubspaceOptimizationResult(
        initial_result=initial_result,
        optimized_result=optimized_result,
        optimized_corrections=optimized_corrections,
        optimized_logical_block_phase=optimized_logical_block_phase,
        active_levels=tuple(int(level) for level in active_levels),
        block_phase_levels=tuple(int(level) for level in block_phase_parameter_levels),
        parameters=tuple(str(name) for name in opt_cfg.conditioned.parameters)
        + (("block_phase",) if block_phase_parameter_levels else tuple()),
        history=history,
        success_stage1=bool(stage1.success),
        success_stage2=False if stage2 is None else bool(stage2.success),
        message_stage1=str(stage1.message),
        message_stage2="" if stage2 is None else str(stage2.message),
    )


__all__ = [
    "TargetedSubspaceTransferSet",
    "TargetedSubspaceObjectiveWeights",
    "LogicalBlockPhaseCorrection",
    "TargetedSubspaceOptimizationConfig",
    "TargetedSubspaceBasisMetric",
    "TargetedSubspaceTransferMetric",
    "TargetedSubspaceValidationResult",
    "TargetedSubspaceOptimizationResult",
    "build_block_rotation_target_operator",
    "build_spanning_state_transfer_set",
    "analyze_targeted_subspace_operator",
    "evaluate_targeted_subspace_multitone",
    "run_targeted_subspace_multitone_validation",
    "optimize_targeted_subspace_multitone",
]