from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from cqed_sim.unitary_synthesis.metrics import subspace_unitary_fidelity

from .initial_guesses import random_control_schedule, zero_control_schedule
from .objectives import StateTransferObjective, UnitaryObjective
from .parameterizations import ControlSchedule
from .penalties import AmplitudePenalty, LeakagePenalty, SlewRatePenalty
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
    scipy_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.initial_guess).lower() not in {"random", "zeros"}:
            raise ValueError("GrapeConfig.initial_guess must be 'random' or 'zeros'.")
        if int(self.maxiter) <= 0:
            raise ValueError("GrapeConfig.maxiter must be positive.")
        if float(self.ftol) < 0.0 or float(self.gtol) < 0.0:
            raise ValueError("GrapeConfig tolerances must be non-negative.")
        if int(self.history_every) <= 0:
            raise ValueError("GrapeConfig.history_every must be positive.")


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


def _evaluate_control_penalties(problem, values: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
    total = 0.0
    gradient = np.zeros_like(values, dtype=float)
    metrics = {
        "amplitude_penalty": 0.0,
        "slew_penalty": 0.0,
    }

    for penalty in problem.penalties:
        if isinstance(penalty, AmplitudePenalty):
            centered = np.asarray(values, dtype=float) - float(penalty.reference)
            raw = float(np.mean(np.square(centered))) if centered.size else 0.0
            contribution = float(penalty.weight) * raw
            total += contribution
            if centered.size:
                gradient += float(penalty.weight) * (2.0 / centered.size) * centered
            metrics["amplitude_penalty"] += contribution
        elif isinstance(penalty, SlewRatePenalty):
            if values.shape[1] < 2:
                continue
            diffs = np.diff(values, axis=1)
            raw = float(np.mean(np.square(diffs))) if diffs.size else 0.0
            contribution = float(penalty.weight) * raw
            total += contribution
            metrics["slew_penalty"] += contribution
            norm = float(diffs.size)
            scale = 2.0 * float(penalty.weight) / max(norm, 1.0)
            gradient[:, 0] += scale * (values[:, 0] - values[:, 1])
            gradient[:, -1] += scale * (values[:, -1] - values[:, -2])
            if values.shape[1] > 2:
                gradient[:, 1:-1] += scale * (2.0 * values[:, 1:-1] - values[:, :-2] - values[:, 2:])

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


def _evaluate_schedule(problem, schedule: ControlSchedule, prepared_objectives, prepared_leakage_penalties) -> _ScheduleEvaluation:
    values = np.asarray(schedule.values, dtype=float)
    system_objectives: list[float] = []
    system_gradients: list[np.ndarray] = []
    system_metrics: list[dict[str, Any]] = []
    nominal_final_unitary: np.ndarray | None = None

    for system_index, system in enumerate(problem.systems):
        propagation = build_propagation_data(
            drift_hamiltonian=system.drift_hamiltonian,
            control_operators=system.control_operators,
            control_values=values,
            step_durations_s=np.asarray(problem.time_grid.step_durations_s, dtype=float),
        )
        if system_index == 0:
            nominal_final_unitary = propagation.final_unitary

        objective_total = 0.0
        gradient_total = np.zeros_like(values, dtype=float)
        objective_reports: list[dict[str, Any]] = []
        leakage_records: dict[int, list[tuple[float, float, np.ndarray]]] = {index: [] for index in range(len(prepared_leakage_penalties))}

        for prepared in prepared_objectives:
            cost, gradient, metrics, forward = _evaluate_state_objective(prepared, propagation, system.control_operators)
            weighted_cost = float(prepared.weight) * float(cost)
            objective_total += weighted_cost
            gradient_total += float(prepared.weight) * gradient

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
                gradient = np.zeros_like(values, dtype=float)
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
            gradient_total += float(prepared_penalty.penalty.weight) * gradient
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
    control_penalty, control_penalty_grad, control_penalty_metrics = _evaluate_control_penalties(problem, values)
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
    }
    return _ScheduleEvaluation(
        objective=float(total_objective),
        gradient=total_gradient,
        metrics=metrics,
        system_metrics=tuple(system_metrics),
        nominal_final_unitary=nominal_final_unitary,
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

        def evaluate(vector: np.ndarray) -> _ScheduleEvaluation:
            nonlocal last_vector, last_evaluation, evaluation_counter
            if last_vector is not None and np.array_equal(vector, last_vector):
                assert last_evaluation is not None
                return last_evaluation

            evaluation_counter += 1
            physical_vector = np.asarray(vector, dtype=float).reshape(-1) * scale_vector
            schedule = ControlSchedule.from_flattened(problem.parameterization, physical_vector).clipped()
            evaluation = _evaluate_schedule(problem, schedule, prepared_objectives, prepared_leakage_penalties)
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

        optimizer_result = minimize(
            objective_function,
            schedule0.flattened() / scale_vector,
            method=str(self.config.optimizer_method),
            jac=gradient_function,
            bounds=scaled_bounds,
            options=options,
        )

        final_schedule = ControlSchedule.from_flattened(problem.parameterization, optimizer_result.x * scale_vector).clipped()
        final_evaluation = _evaluate_schedule(problem, final_schedule, prepared_objectives, prepared_leakage_penalties)
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
            metrics=dict(final_evaluation.metrics),
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
        )


def solve_grape(problem, *, config: GrapeConfig | None = None, initial_schedule: ControlSchedule | np.ndarray | None = None) -> GrapeResult:
    return GrapeSolver(config=config).solve(problem, initial_schedule=initial_schedule)


__all__ = ["GrapeConfig", "GrapeSolver", "solve_grape"]