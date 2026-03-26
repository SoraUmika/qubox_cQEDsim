from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import multiprocessing as mp
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from cqed_sim.unitary_synthesis.metrics import subspace_unitary_fidelity

from .hardware import apply_control_pipeline, selected_control_indices, selected_iq_pairs
from .initial_guesses import random_control_schedule, zero_control_schedule
from .objectives import StateTransferObjective, UnitaryObjective
from .parameterizations import ControlSchedule
from .penalties import (
    AmplitudePenalty,
    BoundPenalty,
    BoundaryConditionPenalty,
    IQRadiusPenalty,
    LeakagePenalty,
    SlewRatePenalty,
)
from .propagators import backward_target_history, build_propagation_data, propagate_state_history
from .result import GrapeIterationRecord, GrapeResult
from .utils import dense_projector
from .utils import finite_bound_scale


@dataclass(frozen=True)
class GrapeConfig:
    optimizer_method: str = "L-BFGS-B"
    maxiter: int = 200
    ftol: float = 1.0e-9
    gtol: float = 1.0e-6
    initial_guess: str = "random"
    random_scale: float = 0.15
    seed: int | None = None
    history_every: int = 1
    apply_hardware_in_forward_model: bool = True
    report_command_reference: bool = True
    scipy_options: dict[str, Any] = field(default_factory=dict)
    engine: str = "numpy"
    jax_device: str | None = None
    show_progress: bool = False

    def __post_init__(self) -> None:
        if str(self.initial_guess).lower() not in {"random", "zeros"}:
            raise ValueError("GrapeConfig.initial_guess must be 'random' or 'zeros'.")
        if int(self.maxiter) <= 0:
            raise ValueError("GrapeConfig.maxiter must be positive.")
        if float(self.ftol) < 0.0 or float(self.gtol) < 0.0:
            raise ValueError("GrapeConfig tolerances must be non-negative.")
        if int(self.history_every) <= 0:
            raise ValueError("GrapeConfig.history_every must be positive.")
        if str(self.engine).lower() not in {"numpy", "jax"}:
            raise ValueError("GrapeConfig.engine must be 'numpy' or 'jax'.")


@dataclass(frozen=True)
class _PreparedObjective:
    kind: str
    name: str
    weight: float
    initial_states: np.ndarray
    target_states: np.ndarray
    state_weights: np.ndarray
    labels: tuple[str, ...]
    raw_objective: StateTransferObjective | UnitaryObjective


@dataclass(frozen=True)
class _PreparedLeakagePenalty:
    penalty: LeakagePenalty
    projector: np.ndarray


@dataclass
class _ScheduleEvaluation:
    objective: float
    gradient: np.ndarray
    metrics: dict[str, Any]
    system_metrics: tuple[dict[str, Any], ...]
    nominal_final_unitary: np.ndarray | None
    resolved_waveforms: Any


def _prepare_objective(problem, objective: StateTransferObjective | UnitaryObjective) -> _PreparedObjective:
    if isinstance(objective, StateTransferObjective):
        initial_states, target_states, weights, labels = objective.resolved_pairs(full_dim=problem.full_dim)
        return _PreparedObjective(
            kind="state_transfer",
            name=str(objective.name),
            weight=float(objective.weight),
            initial_states=initial_states,
            target_states=target_states,
            state_weights=weights,
            labels=labels,
            raw_objective=objective,
        )
    if isinstance(objective, UnitaryObjective):
        initial_states, target_states, weights, labels = objective.resolved_pairs(full_dim=problem.full_dim)
        return _PreparedObjective(
            kind="unitary",
            name=str(objective.name),
            weight=float(objective.weight),
            initial_states=initial_states,
            target_states=target_states,
            state_weights=weights,
            labels=labels,
            raw_objective=objective,
        )
    raise TypeError(f"Unsupported objective type '{type(objective).__name__}'.")


def _prepare_leakage_penalties(problem) -> tuple[_PreparedLeakagePenalty, ...]:
    prepared: list[_PreparedLeakagePenalty] = []
    for penalty in problem.penalties:
        if isinstance(penalty, LeakagePenalty):
            if penalty.subspace.full_dim != problem.full_dim:
                raise ValueError(
                    f"Leakage penalty subspace has full_dim={penalty.subspace.full_dim}, expected {problem.full_dim}."
                )
            prepared.append(_PreparedLeakagePenalty(penalty=penalty, projector=dense_projector(penalty.subspace)))
    return tuple(prepared)


def _unitary_block_slices(objective: UnitaryObjective):
    if objective.phase_blocks:
        return tuple(tuple(int(index) for index in block) for block in objective.phase_blocks)
    if objective.gauge() == "block" and objective.subspace is not None and objective.subspace.kind == "qubit_cavity_block":
        return tuple(objective.subspace.per_fock_blocks())
    return None


def _unitary_target_matrix(problem, objective: UnitaryObjective) -> np.ndarray:
    target = np.asarray(objective.target_operator, dtype=np.complex128)
    if objective.subspace is None:
        return target
    if target.shape == (objective.subspace.dim, objective.subspace.dim):
        return target
    if target.shape == (problem.full_dim, problem.full_dim):
        return objective.subspace.restrict_operator(target)
    raise ValueError(
        f"Unitary target '{objective.name}' has shape {target.shape}, which does not match the full space or supplied subspace."
    )


def _evaluate_state_objective(prepared: _PreparedObjective, propagation, control_operators: tuple[np.ndarray, ...]):
    forward = propagate_state_history(propagation, prepared.initial_states)
    backward = backward_target_history(propagation, prepared.target_states)
    final_states = forward[:, -1, :]
    overlaps = np.sum(np.conj(prepared.target_states) * final_states, axis=1)
    fidelities = np.abs(overlaps) ** 2
    cost = 1.0 - float(np.sum(prepared.state_weights * fidelities))
    gradient = np.zeros((len(control_operators), len(propagation.slice_unitaries)), dtype=float)

    for step_index in range(len(propagation.slice_unitaries)):
        psi_j = forward[:, step_index, :]
        chi_j1 = backward[:, step_index + 1, :]
        for control_index, operator in enumerate(control_operators):
            d_unitary = propagation.slice_derivative(step_index, control_index, operator)
            dpsi = psi_j @ d_unitary.T
            amplitudes = np.sum(np.conj(chi_j1) * dpsi, axis=1)
            gradient[control_index, step_index] += -2.0 * float(
                np.real(np.sum(prepared.state_weights * np.conj(overlaps) * amplitudes))
            )

    metrics = {
        "fidelity_weighted": float(np.sum(prepared.state_weights * fidelities)),
        "fidelity_mean": float(np.mean(fidelities)),
        "fidelity_min": float(np.min(fidelities)),
        "fidelity_max": float(np.max(fidelities)),
        "infidelity": float(cost),
    }
    return cost, gradient, metrics, forward


def _evaluate_leakage_records(
    forward: np.ndarray,
    propagation,
    control_operators: tuple[np.ndarray, ...],
    projector: np.ndarray,
):
    final_states = forward[:, -1, :]
    kept_states = final_states @ projector.T
    keep_probabilities = np.real(np.sum(np.conj(final_states) * kept_states, axis=1))
    leakages = np.clip(1.0 - keep_probabilities, 0.0, 1.0)

    backward = np.zeros_like(forward)
    backward[:, -1, :] = kept_states
    for step_index in range(len(propagation.slice_unitaries) - 1, -1, -1):
        backward[:, step_index, :] = backward[:, step_index + 1, :] @ propagation.slice_unitaries[step_index].conj()

    pair_gradients = np.zeros((forward.shape[0], len(control_operators), len(propagation.slice_unitaries)), dtype=float)
    for step_index in range(len(propagation.slice_unitaries)):
        psi_j = forward[:, step_index, :]
        lambda_j1 = backward[:, step_index + 1, :]
        for control_index, operator in enumerate(control_operators):
            d_unitary = propagation.slice_derivative(step_index, control_index, operator)
            dpsi = psi_j @ d_unitary.T
            amplitudes = np.sum(np.conj(lambda_j1) * dpsi, axis=1)
            pair_gradients[:, control_index, step_index] = -2.0 * np.real(amplitudes)

    return leakages, pair_gradients


def _penalty_domain_values(schedule: ControlSchedule, applied, penalty) -> tuple[np.ndarray, Any, str]:
    domain = str(getattr(penalty, "apply_to", "command")).lower()
    if domain in {"parameter", "parameters", "schedule"}:
        return np.asarray(schedule.values, dtype=float), (lambda grad: np.asarray(grad, dtype=float)), "parameter"
    if domain == "command":
        return np.asarray(applied.resolved.command_values, dtype=float), applied.pullback_command, "command"
    if domain == "physical":
        return np.asarray(applied.resolved.physical_values, dtype=float), applied.pullback_physical, "physical"
    raise ValueError(f"Unsupported penalty domain '{domain}'.")


def _embed_selected_gradient(shape: tuple[int, int], selected: tuple[int, ...], local_gradient: np.ndarray) -> np.ndarray:
    gradient = np.zeros(shape, dtype=float)
    if selected:
        gradient[np.asarray(selected, dtype=int), :] = np.asarray(local_gradient, dtype=float)
    return gradient


def _evaluate_control_penalties(problem, schedule: ControlSchedule, applied) -> tuple[float, np.ndarray, dict[str, Any]]:
    total = 0.0
    gradient = np.zeros_like(schedule.values, dtype=float)
    metrics: dict[str, Any] = {
        "amplitude_penalty": 0.0,
        "slew_penalty": 0.0,
        "bound_penalty": 0.0,
        "boundary_penalty": 0.0,
        "iq_radius_penalty": 0.0,
    }

    for penalty in problem.penalties:
        if isinstance(penalty, LeakagePenalty):
            continue

        values, pullback, _domain = _penalty_domain_values(schedule, applied, penalty)

        if isinstance(penalty, AmplitudePenalty):
            centered = np.asarray(values, dtype=float) - float(penalty.reference)
            raw = float(np.mean(np.square(centered))) if centered.size else 0.0
            contribution = float(penalty.weight) * raw
            total += contribution
            if centered.size:
                gradient += pullback(float(penalty.weight) * (2.0 / centered.size) * centered)
            metrics["amplitude_penalty"] += contribution
            continue

        if isinstance(penalty, SlewRatePenalty):
            if values.shape[1] < 2:
                continue
            diffs = np.diff(values, axis=1)
            raw = float(np.mean(np.square(diffs))) if diffs.size else 0.0
            contribution = float(penalty.weight) * raw
            total += contribution
            metrics["slew_penalty"] += contribution
            norm = float(diffs.size)
            local_gradient = np.zeros_like(values, dtype=float)
            scale = 2.0 / max(norm, 1.0)
            local_gradient[:, 0] += scale * (values[:, 0] - values[:, 1])
            local_gradient[:, -1] += scale * (values[:, -1] - values[:, -2])
            if values.shape[1] > 2:
                local_gradient[:, 1:-1] += scale * (2.0 * values[:, 1:-1] - values[:, :-2] - values[:, 2:])
            gradient += pullback(float(penalty.weight) * local_gradient)
            continue

        if isinstance(penalty, BoundPenalty):
            selected = selected_control_indices(
                problem.control_terms,
                control_names=tuple(penalty.control_names),
                export_channels=tuple(penalty.export_channels),
            )
            if not selected:
                continue
            subset = np.asarray(values[np.asarray(selected, dtype=int), :], dtype=float)
            below = np.clip(float(penalty.lower_bound) - subset, 0.0, None)
            above = np.clip(subset - float(penalty.upper_bound), 0.0, None)
            raw = float(np.mean(np.square(below) + np.square(above))) if subset.size else 0.0
            contribution = float(penalty.weight) * raw
            total += contribution
            metrics["bound_penalty"] += contribution
            local_gradient = np.zeros_like(subset, dtype=float)
            if subset.size:
                local_gradient += (-2.0 / subset.size) * below
                local_gradient += (2.0 / subset.size) * above
                gradient += pullback(
                    float(penalty.weight)
                    * _embed_selected_gradient(values.shape, selected, local_gradient)
                )
            continue

        if isinstance(penalty, BoundaryConditionPenalty):
            selected = selected_control_indices(
                problem.control_terms,
                control_names=tuple(penalty.control_names),
                export_channels=tuple(penalty.export_channels),
            )
            if not selected:
                continue
            subset = np.asarray(values[np.asarray(selected, dtype=int), :], dtype=float)
            local_gradient = np.zeros_like(subset, dtype=float)
            raw = 0.0
            if bool(penalty.apply_start):
                span = min(int(penalty.ramp_slices), subset.shape[1])
                start_values = subset[:, :span]
                raw += float(np.mean(np.square(start_values))) if start_values.size else 0.0
                if start_values.size:
                    local_gradient[:, :span] += (2.0 / start_values.size) * start_values
            if bool(penalty.apply_end):
                span = min(int(penalty.ramp_slices), subset.shape[1])
                end_values = subset[:, -span:]
                raw += float(np.mean(np.square(end_values))) if end_values.size else 0.0
                if end_values.size:
                    local_gradient[:, -span:] += (2.0 / end_values.size) * end_values
            contribution = float(penalty.weight) * float(raw)
            total += contribution
            metrics["boundary_penalty"] += contribution
            gradient += pullback(float(penalty.weight) * _embed_selected_gradient(values.shape, selected, local_gradient))
            continue

        if isinstance(penalty, IQRadiusPenalty):
            pairs = selected_iq_pairs(
                problem.control_terms,
                control_names=tuple(penalty.control_names),
                export_channels=tuple(penalty.export_channels),
            )
            if not pairs:
                continue
            local_gradient = np.zeros_like(values, dtype=float)
            raw_terms: list[np.ndarray] = []
            amplitude_max = float(penalty.amplitude_max)
            for _channel, i_index, q_index in pairs:
                i_values = np.asarray(values[i_index, :], dtype=float)
                q_values = np.asarray(values[q_index, :], dtype=float)
                radius = np.sqrt(np.square(i_values) + np.square(q_values))
                violation = np.clip(radius - amplitude_max, 0.0, None)
                raw_terms.append(np.square(violation))
                active = radius > 1.0e-18
                scale = np.zeros_like(radius, dtype=float)
                scale[active] = (2.0 * violation[active]) / radius[active]
                local_gradient[i_index, :] += scale * i_values
                local_gradient[q_index, :] += scale * q_values
            if raw_terms:
                stacked = np.concatenate([term.reshape(-1) for term in raw_terms])
                raw = float(np.mean(stacked)) if stacked.size else 0.0
                contribution = float(penalty.weight) * raw
                total += contribution
                metrics["iq_radius_penalty"] += contribution
                norm = float(sum(term.size for term in raw_terms))
                gradient += pullback(float(penalty.weight) * (local_gradient / max(norm, 1.0)))
            continue

    metrics["control_penalty_total"] = float(total)
    return float(total), gradient, metrics


def _aggregate_systems(system_objectives, system_gradients, systems, mode: str) -> tuple[float, np.ndarray, dict[str, Any]]:
    if mode == "worst":
        index = int(np.argmax(system_objectives))
        return (
            float(system_objectives[index]),
            np.asarray(system_gradients[index], dtype=float),
            {"aggregate_mode": "worst", "active_system": str(systems[index].label)},
        )
    weights = np.asarray([float(system.weight) for system in systems], dtype=float)
    weights = weights / np.sum(weights)
    objective = float(np.sum(weights * np.asarray(system_objectives, dtype=float)))
    gradient = np.zeros_like(system_gradients[0], dtype=float)
    for weight, grad in zip(weights, system_gradients, strict=True):
        gradient += float(weight) * np.asarray(grad, dtype=float)
    return objective, gradient, {"aggregate_mode": "mean", "system_weights": [float(weight) for weight in weights]}


def _evaluate_system_jax(
    system,
    physical_values: np.ndarray,
    step_durations_s: np.ndarray,
    prepared_objectives,
    prepared_leakage_penalties,
    problem,
    *,
    jax_evaluator=None,
    device: str | None = None,
):
    """Evaluate one ensemble system using JAX autodiff.

    Returns ``(cost, gradient_physical, system_metrics, U_final, evaluator)``
    where *gradient_physical* is w.r.t. the physical control values and
    *evaluator* is the (possibly newly-built) JIT-compiled evaluator to
    be cached for subsequent calls.
    """
    from .propagators_jax import build_jax_evaluator

    if jax_evaluator is None:
        pair_counts = tuple(int(p.initial_states.shape[0]) for p in prepared_objectives)
        leak_mets = tuple(str(pp.penalty.metric) for pp in prepared_leakage_penalties)
        jax_evaluator = build_jax_evaluator(
            system.drift_hamiltonian,
            system.control_operators,
            step_durations_s,
            int(physical_values.shape[1]),
            objective_pair_counts=pair_counts,
            leakage_metrics=leak_mets,
            device=device,
        )

    # Concatenate probe-state data across objectives
    all_init = np.concatenate([p.initial_states for p in prepared_objectives])
    all_tgt = np.concatenate([p.target_states for p in prepared_objectives])
    all_sw = np.concatenate([p.state_weights for p in prepared_objectives])
    obj_wt = np.array([float(p.weight) for p in prepared_objectives], dtype=float)

    # Leakage inputs
    if prepared_leakage_penalties:
        projectors = [pp.projector for pp in prepared_leakage_penalties]
        leak_wt = np.array([float(pp.penalty.weight) for pp in prepared_leakage_penalties], dtype=float)
        leak_pair_wt = [
            np.concatenate([p.state_weights * float(p.weight) for p in prepared_objectives])
            for _ in prepared_leakage_penalties
        ]
    else:
        projectors = None
        leak_wt = None
        leak_pair_wt = None

    cost, gradient_physical, U_final, fidelities = jax_evaluator(
        physical_values, all_init, all_tgt, all_sw, obj_wt,
        projectors=projectors,
        leak_penalty_weights=leak_wt,
        leak_pair_weights=leak_pair_wt,
    )

    # Build per-objective metric reports
    objective_reports: list[dict[str, Any]] = []
    offset = 0
    for prepared in prepared_objectives:
        n = int(prepared.initial_states.shape[0])
        obj_fids = fidelities[offset:offset + n]
        offset += n

        report: dict[str, Any] = {
            "name": str(prepared.name),
            "kind": str(prepared.kind),
            "weight": float(prepared.weight),
            "weighted_cost": float(prepared.weight) * float(1.0 - np.sum(prepared.state_weights * obj_fids)),
            "fidelity_weighted": float(np.sum(prepared.state_weights * obj_fids)),
            "fidelity_mean": float(np.mean(obj_fids)),
            "fidelity_min": float(np.min(obj_fids)),
            "fidelity_max": float(np.max(obj_fids)),
            "infidelity": float(1.0 - np.sum(prepared.state_weights * obj_fids)),
        }

        if prepared.kind == "unitary" and isinstance(prepared.raw_objective, UnitaryObjective):
            uo = prepared.raw_objective
            target_matrix = _unitary_target_matrix(problem, uo)
            if uo.subspace is None:
                actual_matrix = U_final
            else:
                actual_matrix = uo.subspace.restrict_operator(U_final)
            report["exact_unitary_fidelity"] = float(
                subspace_unitary_fidelity(
                    actual_matrix,
                    target_matrix,
                    gauge=uo.gauge(),
                    block_slices=_unitary_block_slices(uo),
                )
            )

        objective_reports.append(report)

    # Leakage metrics (for reporting only; gradient already in cost)
    leakage_report: dict[str, float] = {}
    if prepared_leakage_penalties:
        all_final = all_init @ U_final.T
        combined_weights = np.concatenate(
            [p.state_weights * float(p.weight) for p in prepared_objectives]
        )
        for j, pp in enumerate(prepared_leakage_penalties):
            kept = all_final @ pp.projector.T
            keep_probs = np.real(np.sum(np.conj(all_final) * kept, axis=1))
            leakages = np.clip(1.0 - keep_probs, 0.0, 1.0)
            if pp.penalty.metric == "worst":
                raw = float(np.max(leakages))
            else:
                norm = float(np.sum(combined_weights))
                raw = float(np.sum(combined_weights * leakages) / max(norm, 1e-18))
            contribution = float(pp.penalty.weight) * raw
            leakage_report[f"leakage_penalty_{j}"] = contribution
            leakage_report[f"leakage_raw_{j}"] = raw

    system_metrics: dict[str, Any] = {
        "label": str(system.label),
        "weight": float(system.weight),
        "objective": float(cost),
        "objectives": objective_reports,
        **leakage_report,
    }

    return float(cost), gradient_physical, system_metrics, U_final, jax_evaluator


def _evaluate_schedule(
    problem,
    schedule: ControlSchedule,
    prepared_objectives,
    prepared_leakage_penalties,
    *,
    apply_hardware: bool = True,
    engine: str = "numpy",
    jax_device: str | None = None,
    _jax_cache: dict | None = None,
) -> _ScheduleEvaluation:
    applied = apply_control_pipeline(problem, schedule, apply_hardware=apply_hardware)
    resolved = applied.resolved
    physical_values = np.asarray(resolved.physical_values, dtype=float)
    system_objectives: list[float] = []
    system_gradients: list[np.ndarray] = []
    system_metrics: list[dict[str, Any]] = []
    nominal_final_unitary: np.ndarray | None = None

    for system_index, system in enumerate(problem.systems):
        # ----- JAX engine: single autodiff call per system -----
        if engine == "jax":
            cached_eval = _jax_cache.get(system_index) if _jax_cache is not None else None
            step_durations_s = np.asarray(problem.time_grid.step_durations_s, dtype=float)
            obj_total, grad_phys, sys_metrics, final_U, evaluator = _evaluate_system_jax(
                system, physical_values, step_durations_s,
                prepared_objectives, prepared_leakage_penalties, problem,
                jax_evaluator=cached_eval, device=jax_device,
            )
            if _jax_cache is not None:
                _jax_cache[system_index] = evaluator
            gradient_total = applied.pullback_physical(grad_phys)
            system_objectives.append(obj_total)
            system_gradients.append(gradient_total)
            system_metrics.append(sys_metrics)
            if system_index == 0:
                nominal_final_unitary = final_U
            continue

        # ----- NumPy engine (existing path) -----
        propagation = build_propagation_data(
            drift_hamiltonian=system.drift_hamiltonian,
            control_operators=system.control_operators,
            control_values=physical_values,
            step_durations_s=np.asarray(problem.time_grid.step_durations_s, dtype=float),
        )
        if system_index == 0:
            nominal_final_unitary = propagation.final_unitary

        objective_total = 0.0
        gradient_total = np.zeros_like(schedule.values, dtype=float)
        objective_reports: list[dict[str, Any]] = []
        leakage_records: dict[int, list[tuple[float, float, np.ndarray]]] = {index: [] for index in range(len(prepared_leakage_penalties))}

        for prepared in prepared_objectives:
            cost, gradient_physical, metrics, forward = _evaluate_state_objective(prepared, propagation, system.control_operators)
            weighted_cost = float(prepared.weight) * float(cost)
            objective_total += weighted_cost
            gradient_total += float(prepared.weight) * applied.pullback_physical(gradient_physical)

            report = {
                "name": str(prepared.name),
                "kind": str(prepared.kind),
                "weight": float(prepared.weight),
                "weighted_cost": float(weighted_cost),
                **metrics,
            }

            if prepared.kind == "unitary":
                unitary_objective = prepared.raw_objective
                assert isinstance(unitary_objective, UnitaryObjective)
                target_matrix = _unitary_target_matrix(problem, unitary_objective)
                if unitary_objective.subspace is None:
                    actual_matrix = propagation.final_unitary
                else:
                    actual_matrix = unitary_objective.subspace.restrict_operator(propagation.final_unitary)
                report["exact_unitary_fidelity"] = float(
                    subspace_unitary_fidelity(
                        actual_matrix,
                        target_matrix,
                        gauge=unitary_objective.gauge(),
                        block_slices=_unitary_block_slices(unitary_objective),
                    )
                )

            objective_reports.append(report)

            for penalty_index, prepared_penalty in enumerate(prepared_leakage_penalties):
                leakages, pair_gradients = _evaluate_leakage_records(
                    forward,
                    propagation,
                    system.control_operators,
                    prepared_penalty.projector,
                )
                for pair_index, leakage in enumerate(leakages):
                    pair_weight = float(prepared.weight) * float(prepared.state_weights[pair_index])
                    leakage_records[penalty_index].append(
                        (pair_weight, float(leakage), np.asarray(pair_gradients[pair_index], dtype=float))
                    )

        leakage_report: dict[str, float] = {}
        for penalty_index, prepared_penalty in enumerate(prepared_leakage_penalties):
            records = leakage_records[penalty_index]
            if not records:
                raw = 0.0
                gradient = np.zeros_like(schedule.values, dtype=float)
            else:
                weights = np.asarray([row[0] for row in records], dtype=float)
                leakages = np.asarray([row[1] for row in records], dtype=float)
                gradients = np.asarray([row[2] for row in records], dtype=float)
                if prepared_penalty.penalty.metric == "worst":
                    active = int(np.argmax(leakages))
                    raw = float(leakages[active])
                    gradient = np.asarray(gradients[active], dtype=float)
                else:
                    norm = float(np.sum(weights))
                    raw = float(np.sum(weights * leakages) / max(norm, 1.0e-18))
                    gradient = np.sum(weights[:, None, None] * gradients, axis=0) / max(norm, 1.0e-18)
            contribution = float(prepared_penalty.penalty.weight) * float(raw)
            objective_total += contribution
            gradient_total += float(prepared_penalty.penalty.weight) * applied.pullback_physical(gradient)
            leakage_report[f"leakage_penalty_{penalty_index}"] = float(contribution)
            leakage_report[f"leakage_raw_{penalty_index}"] = float(raw)

        system_objectives.append(float(objective_total))
        system_gradients.append(np.asarray(gradient_total, dtype=float))
        system_metrics.append(
            {
                "label": str(system.label),
                "weight": float(system.weight),
                "objective": float(objective_total),
                "objectives": objective_reports,
                **leakage_report,
            }
        )

    aggregate_objective, aggregate_gradient, aggregate_report = _aggregate_systems(
        system_objectives,
        system_gradients,
        problem.systems,
        problem.ensemble_aggregate,
    )
    control_penalty, control_penalty_grad, control_penalty_metrics = _evaluate_control_penalties(problem, schedule, applied)
    total_objective = float(aggregate_objective + control_penalty)
    total_gradient = np.asarray(aggregate_gradient + control_penalty_grad, dtype=float)

    nominal_fidelity = float("nan")
    if system_metrics and system_metrics[0]["objectives"]:
        first_objective = system_metrics[0]["objectives"][0]
        nominal_fidelity = float(first_objective.get("exact_unitary_fidelity", first_objective.get("fidelity_weighted", np.nan)))

    metrics = {
        "objective_total": float(total_objective),
        "objective_system": float(aggregate_objective),
        "objective_control_penalty": float(control_penalty),
        "gradient_norm": float(np.linalg.norm(total_gradient.reshape(-1))),
        "nominal_fidelity": float(nominal_fidelity),
        **aggregate_report,
        **control_penalty_metrics,
        **resolved.parameterization_metrics,
        **resolved.hardware_metrics,
    }
    if apply_hardware:
        metrics["nominal_physical_fidelity"] = float(nominal_fidelity)
    else:
        metrics["nominal_command_fidelity"] = float(nominal_fidelity)
    return _ScheduleEvaluation(
        objective=float(total_objective),
        gradient=total_gradient,
        metrics=metrics,
        system_metrics=tuple(system_metrics),
        nominal_final_unitary=nominal_final_unitary,
        resolved_waveforms=resolved,
    )


class GrapeSolver:
    def __init__(self, config: GrapeConfig | None = None) -> None:
        self.config = GrapeConfig() if config is None else config

    def _initial_schedule(self, problem, initial_schedule: ControlSchedule | np.ndarray | None) -> ControlSchedule:
        if initial_schedule is None:
            if str(self.config.initial_guess).lower() == "zeros":
                return zero_control_schedule(problem)
            return random_control_schedule(problem, seed=self.config.seed, scale=float(self.config.random_scale))
        if isinstance(initial_schedule, ControlSchedule):
            return initial_schedule.clipped()
        return ControlSchedule(problem.parameterization, problem.parameterization.clip(np.asarray(initial_schedule, dtype=float)))

    def solve(self, problem, *, initial_schedule: ControlSchedule | np.ndarray | None = None) -> GrapeResult:
        prepared_objectives = tuple(_prepare_objective(problem, objective) for objective in problem.objectives)
        prepared_leakage_penalties = _prepare_leakage_penalties(problem)
        schedule0 = self._initial_schedule(problem, initial_schedule)

        scale_matrix = np.asarray(
            [
                [
                    finite_bound_scale(term.amplitude_bounds[0], term.amplitude_bounds[1], fallback=1.0)
                    for _ in range(problem.n_slices)
                ]
                for term in problem.control_terms
            ],
            dtype=float,
        )
        scale_vector = scale_matrix.reshape(-1)
        scaled_bounds = tuple(
            (float(lower) / max(scale, 1.0e-18), float(upper) / max(scale, 1.0e-18))
            for (lower, upper), scale in zip(problem.parameterization.bounds(), scale_vector, strict=True)
        )

        last_vector: np.ndarray | None = None
        last_evaluation: _ScheduleEvaluation | None = None
        history: list[GrapeIterationRecord] = []
        start_time = time.perf_counter()
        evaluation_counter = 0
        _engine = str(self.config.engine).lower()
        _jax_device = self.config.jax_device
        _jax_cache: dict[int, Any] = {}

        def evaluate(vector: np.ndarray) -> _ScheduleEvaluation:
            nonlocal last_vector, last_evaluation, evaluation_counter
            if last_vector is not None and np.array_equal(vector, last_vector):
                assert last_evaluation is not None
                return last_evaluation

            evaluation_counter += 1
            physical_vector = np.asarray(vector, dtype=float).reshape(-1) * scale_vector
            schedule = ControlSchedule.from_flattened(problem.parameterization, physical_vector).clipped()
            evaluation = _evaluate_schedule(
                problem,
                schedule,
                prepared_objectives,
                prepared_leakage_penalties,
                apply_hardware=bool(self.config.apply_hardware_in_forward_model),
                engine=_engine,
                jax_device=_jax_device,
                _jax_cache=_jax_cache,
            )
            last_vector = np.array(vector, copy=True)
            last_evaluation = evaluation

            if evaluation_counter == 1 or evaluation_counter % int(self.config.history_every) == 0:
                history.append(
                    GrapeIterationRecord(
                        evaluation=evaluation_counter,
                        objective=float(evaluation.objective),
                        gradient_norm=float(np.linalg.norm(evaluation.gradient.reshape(-1))),
                        elapsed_s=float(time.perf_counter() - start_time),
                        metrics=dict(evaluation.metrics),
                    )
                )
            return evaluation

        def objective_function(vector: np.ndarray) -> float:
            return float(evaluate(vector).objective)

        def gradient_function(vector: np.ndarray) -> np.ndarray:
            return evaluate(vector).gradient.reshape(-1) * scale_vector

        options = dict(self.config.scipy_options)
        options.setdefault("maxiter", int(self.config.maxiter))
        if str(self.config.optimizer_method).upper() == "L-BFGS-B":
            options.setdefault("ftol", float(self.config.ftol))
            options.setdefault("gtol", float(self.config.gtol))

        _pbar = None
        _callback = None
        if bool(self.config.show_progress):
            try:
                from tqdm.auto import tqdm as _TqdmCls
                _pbar = _TqdmCls(
                    total=int(self.config.maxiter),
                    desc="GRAPE",
                    unit="iter",
                    dynamic_ncols=True,
                    leave=True,
                )

                def _callback(*_args):
                    _m = last_evaluation.metrics if last_evaluation is not None else {}
                    _pbar.set_postfix(
                        fidelity=f"{float(_m.get('nominal_fidelity', float('nan'))):.4f}",
                        cost=f"{float(_m.get('objective_total', float('nan'))):.4g}",
                        refresh=False,
                    )
                    _pbar.update(1)
            except Exception:
                pass

        optimizer_result = minimize(
            objective_function,
            schedule0.flattened() / scale_vector,
            method=str(self.config.optimizer_method),
            jac=gradient_function,
            bounds=scaled_bounds,
            options=options,
            callback=_callback,
        )

        if _pbar is not None:
            _pbar.close()

        final_schedule = ControlSchedule.from_flattened(problem.parameterization, optimizer_result.x * scale_vector).clipped()
        final_evaluation = _evaluate_schedule(
            problem,
            final_schedule,
            prepared_objectives,
            prepared_leakage_penalties,
            apply_hardware=bool(self.config.apply_hardware_in_forward_model),
            engine=_engine,
            jax_device=_jax_device,
            _jax_cache=_jax_cache,
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
                    engine=_engine,
                    jax_device=_jax_device,
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
                    engine=_engine,
                    jax_device=_jax_device,
                )
                final_metrics["nominal_command_fidelity"] = float(final_evaluation.metrics.get("nominal_fidelity", np.nan))
                final_metrics["objective_command_reference"] = float(final_evaluation.objective)
                final_metrics["nominal_physical_fidelity"] = float(physical_reference.metrics.get("nominal_fidelity", np.nan))
                final_metrics["objective_physical_reference"] = float(physical_reference.objective)
        final_metrics["hardware_forward_model_applied"] = bool(self.config.apply_hardware_in_forward_model and problem.hardware_model is not None)
        if not history or history[-1].evaluation != evaluation_counter:
            history.append(
                GrapeIterationRecord(
                    evaluation=max(evaluation_counter, 1),
                    objective=float(final_evaluation.objective),
                    gradient_norm=float(np.linalg.norm(final_evaluation.gradient.reshape(-1))),
                    elapsed_s=float(time.perf_counter() - start_time),
                    metrics=dict(final_evaluation.metrics),
                )
            )

        return GrapeResult(
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
                "nit": int(getattr(optimizer_result, "nit", 0)),
                "nfev": int(getattr(optimizer_result, "nfev", 0)),
                "njev": int(getattr(optimizer_result, "njev", 0) or 0),
                "status": int(getattr(optimizer_result, "status", 0)),
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
        )


def solve_grape(problem, *, config: GrapeConfig | None = None, initial_schedule: ControlSchedule | np.ndarray | None = None) -> GrapeResult:
    return GrapeSolver(config=config).solve(problem, initial_schedule=initial_schedule)


# ---------------------------------------------------------------------------
# Multi-start support
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrapeMultistartConfig:
    """Configuration for a multi-start GRAPE run.

    Runs *n_restarts* independent GRAPE optimizations, each starting from a
    different random seed, and returns all results sorted by objective value
    (best first).  Set *max_workers > 1* to execute restarts in parallel.

    Args:
        n_restarts: Number of independent random restarts to run.
        max_workers: Number of parallel workers.  ``1`` means serial execution.
        mp_context: Parallelism strategy.

            - ``"thread"`` (default): Uses :class:`~concurrent.futures.ThreadPoolExecutor`.
              Zero startup overhead.  Works well because NumPy/SciPy linear-algebra
              calls (``expm``, BLAS) release the GIL, and the JAX engine runs entirely
              in XLA (also GIL-free).  Recommended for most workloads on all platforms.
            - ``"loky"``: Uses the ``loky`` reusable process pool (from ``joblib``).
              Full process isolation with persistent workers; near-zero per-task
              startup overhead.  Requires ``loky`` to be installed.
            - ``"spawn"`` / ``"fork"``: Standard :mod:`multiprocessing` contexts via
              :class:`~concurrent.futures.ProcessPoolExecutor`.  ``"spawn"`` is safe
              on all platforms but carries ~4-5 s per-worker startup overhead on
              Windows.  ``"fork"`` is faster but only available on Unix.

        return_all: If ``True``, return all restart results sorted best-first.
            If ``False`` (default), return only the single best result inside
            a list of length 1.
    """

    n_restarts: int = 4
    max_workers: int = 1
    mp_context: str = "thread"
    return_all: bool = True
    show_progress: bool = False

    def __post_init__(self) -> None:
        if int(self.n_restarts) < 1:
            raise ValueError("GrapeMultistartConfig.n_restarts must be at least 1.")
        if int(self.max_workers) < 1:
            raise ValueError("GrapeMultistartConfig.max_workers must be at least 1.")
        allowed_contexts = {"thread", "loky", "spawn", "fork", "forkserver"}
        if str(self.mp_context).lower() not in allowed_contexts:
            raise ValueError(
                f"GrapeMultistartConfig.mp_context must be one of {sorted(allowed_contexts)}."
            )


# Global state for process-based parallel workers (module-level for pickle).
_PARALLEL_GRAPE_PROBLEM: Any = None
_PARALLEL_GRAPE_CONFIG: GrapeConfig | None = None


def _init_parallel_grape(problem: Any, config: GrapeConfig) -> None:
    global _PARALLEL_GRAPE_PROBLEM, _PARALLEL_GRAPE_CONFIG
    _PARALLEL_GRAPE_PROBLEM = problem
    _PARALLEL_GRAPE_CONFIG = config


def _make_restart_config(base: GrapeConfig, seed: int) -> GrapeConfig:
    return GrapeConfig(
        optimizer_method=base.optimizer_method,
        maxiter=base.maxiter,
        ftol=base.ftol,
        gtol=base.gtol,
        initial_guess="random",
        random_scale=base.random_scale,
        seed=int(seed),
        history_every=base.history_every,
        apply_hardware_in_forward_model=base.apply_hardware_in_forward_model,
        report_command_reference=base.report_command_reference,
        scipy_options=dict(base.scipy_options),
        engine=base.engine,
        jax_device=base.jax_device,
    )


def _run_parallel_grape_restart(seed: int) -> GrapeResult:
    if _PARALLEL_GRAPE_PROBLEM is None or _PARALLEL_GRAPE_CONFIG is None:
        raise RuntimeError("Parallel GRAPE worker was not initialized.")
    cfg = _make_restart_config(_PARALLEL_GRAPE_CONFIG, seed)
    return GrapeSolver(config=cfg).solve(_PARALLEL_GRAPE_PROBLEM)


def _run_thread_grape_restart(problem: Any, base_config: GrapeConfig, seed: int) -> GrapeResult:
    """Worker for thread-based parallelism (no global state needed)."""
    cfg = _make_restart_config(base_config, seed)
    return GrapeSolver(config=cfg).solve(problem)


def solve_grape_multistart(
    problem,
    *,
    config: GrapeConfig | None = None,
    multistart_config: GrapeMultistartConfig | None = None,
) -> list[GrapeResult]:
    """Run GRAPE from multiple random starting points.

    Returns all restart results sorted by objective value (lowest first, i.e.
    best result first) when *multistart_config.return_all* is ``True``, or a
    single-element list containing only the best result otherwise.

    **Parallelism strategies** (``multistart_config.mp_context``):

    - ``"thread"`` (default): Thread-based parallelism via
      :class:`~concurrent.futures.ThreadPoolExecutor`.  Zero startup overhead.
      Works well because NumPy/SciPy linear-algebra calls release the GIL.
      With the ``"jax"`` engine, XLA computation is fully GIL-free for
      near-ideal multi-start scaling.
    - ``"loky"``: Persistent reusable process pool (requires ``loky``).
      Near-zero per-restart startup overhead while maintaining full process
      isolation.
    - ``"spawn"`` / ``"fork"``: Standard :mod:`multiprocessing` contexts.
      ``"spawn"`` adds ~4-5 s startup per worker on Windows.

    Args:
        problem: A :class:`~cqed_sim.optimal_control.ControlProblem` instance.
        config: Base :class:`GrapeConfig`.  The ``seed`` field is overridden
            per restart.  Defaults to :class:`GrapeConfig()`.
        multistart_config: Multi-start settings.  Defaults to
            :class:`GrapeMultistartConfig()`.

    Returns:
        List of :class:`GrapeResult` instances, sorted best-first.
    """
    cfg = config or GrapeConfig()
    ms_cfg = multistart_config or GrapeMultistartConfig()
    base_seed = int(cfg.seed) if cfg.seed is not None else 0
    seeds = [base_seed + restart_index for restart_index in range(int(ms_cfg.n_restarts))]
    context = str(ms_cfg.mp_context).lower()

    def _wrap_progress(iterable, total):
        """Optionally wrap *iterable* with a tqdm progress bar."""
        if not bool(ms_cfg.show_progress):
            return iterable
        try:
            from tqdm.auto import tqdm as _tqdm
            return _tqdm(iterable, total=total, desc="GRAPE restarts", unit="restart", dynamic_ncols=True)
        except Exception:
            return iterable

    if int(ms_cfg.max_workers) <= 1 or int(ms_cfg.n_restarts) <= 1:
        # Serial execution
        results: list[GrapeResult] = []
        for seed in _wrap_progress(seeds, total=len(seeds)):
            restart_cfg = _make_restart_config(cfg, seed)
            results.append(GrapeSolver(config=restart_cfg).solve(problem))

    elif context == "thread":
        # Thread-based: zero startup overhead, GIL released by NumPy/SciPy/XLA
        with ThreadPoolExecutor(max_workers=int(ms_cfg.max_workers)) as executor:
            futures = [
                executor.submit(_run_thread_grape_restart, problem, cfg, seed)
                for seed in seeds
            ]
            results = [f.result() for f in _wrap_progress(futures, total=len(futures))]

    elif context == "loky":
        try:
            import loky
        except ImportError as exc:
            raise ImportError(
                "The 'loky' package is required for mp_context='loky'. "
                "Install with: pip install loky"
            ) from exc
        executor = loky.get_reusable_executor(max_workers=int(ms_cfg.max_workers))
        futures = [
            executor.submit(_run_thread_grape_restart, problem, cfg, seed)
            for seed in seeds
        ]
        results = [f.result() for f in _wrap_progress(futures, total=len(futures))]

    else:
        # Process-based (spawn / fork / forkserver)
        ctx = mp.get_context(context)
        with ProcessPoolExecutor(
            max_workers=int(ms_cfg.max_workers),
            mp_context=ctx,
            initializer=_init_parallel_grape,
            initargs=(problem, cfg),
        ) as executor:
            results = list(_wrap_progress(executor.map(_run_parallel_grape_restart, seeds), total=len(seeds)))

    results.sort(key=lambda r: float(r.objective_value))
    if not bool(ms_cfg.return_all):
        return results[:1]
    return results


__all__ = [
    "GrapeConfig",
    "GrapeMultistartConfig",
    "GrapeSolver",
    "solve_grape",
    "solve_grape_multistart",
]