from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Sequence

import numpy as np

from .grape import GrapeConfig, GrapeResult, solve_grape
from .parameterizations import ControlSchedule, PiecewiseConstantParameterization, PiecewiseConstantTimeGrid
from .problems import ControlProblem
from .result import ControlResult
from .structured import StructuredControlConfig, StructuredPulseParameterization, solve_structured_control


def _clone_parameterization_with_time_grid(parameterization: Any, time_grid: PiecewiseConstantTimeGrid) -> Any:
    if not is_dataclass(parameterization):
        raise TypeError(
            "Gate-time workflows require a dataclass-based parameterization so the time grid can be cloned safely."
        )
    return replace(parameterization, time_grid=time_grid)


def _clone_problem(
    problem: ControlProblem,
    parameterization: Any,
    *,
    metadata_updates: dict[str, Any] | None = None,
) -> ControlProblem:
    metadata = dict(problem.metadata)
    if metadata_updates:
        metadata.update(dict(metadata_updates))
    return ControlProblem(
        parameterization=parameterization,
        systems=tuple(problem.systems),
        objectives=tuple(problem.objectives),
        penalties=tuple(problem.penalties),
        ensemble_aggregate=str(problem.ensemble_aggregate),
        hardware_model=problem.hardware_model,
        metadata=metadata,
    )


def _adapt_initial_schedule(
    parameterization: Any,
    initial_schedule: ControlSchedule | np.ndarray | None,
    *,
    allow_shape_mismatch: bool = False,
) -> ControlSchedule | None:
    if initial_schedule is None:
        return None
    values = (
        np.asarray(initial_schedule.values, dtype=float)
        if isinstance(initial_schedule, ControlSchedule)
        else np.asarray(initial_schedule, dtype=float)
    )
    try:
        clipped = parameterization.clip(values)
    except Exception as exc:
        if allow_shape_mismatch:
            return None
        raise ValueError(
            "Initial schedule is incompatible with the target parameterization shape or bounds."
        ) from exc
    return ControlSchedule(parameterization, clipped)


def _normalize_durations(durations_s: Sequence[float]) -> tuple[float, ...]:
    durations = tuple(float(value) for value in durations_s)
    if not durations:
        raise ValueError("Gate-time optimization requires at least one candidate duration.")
    if any(duration <= 0.0 for duration in durations):
        raise ValueError("All candidate durations must be positive.")
    return durations


def _nominal_fidelity(result: ControlResult) -> float:
    metrics = dict(getattr(result, "metrics", {}))
    for key in ("nominal_physical_fidelity", "nominal_command_fidelity", "nominal_fidelity"):
        value = metrics.get(key)
        if value is None:
            continue
        fidelity = float(value)
        if np.isfinite(fidelity):
            return fidelity
    return float("nan")


@dataclass(frozen=True)
class GateTimeOptimizationConfig:
    warm_start_strategy: str = "previous_best"
    max_workers: int = 1

    def __post_init__(self) -> None:
        allowed = {"none", "previous_best"}
        if str(self.warm_start_strategy).lower() not in allowed:
            raise ValueError(f"warm_start_strategy must be one of {sorted(allowed)}.")
        if int(self.max_workers) < 1:
            raise ValueError("GateTimeOptimizationConfig.max_workers must be at least 1.")


@dataclass(frozen=True)
class GateTimeCandidate:
    duration_s: float
    step_durations_s: tuple[float, ...]
    result: ControlResult


@dataclass(frozen=True)
class GateTimeOptimizationResult:
    solver_name: str
    candidates: tuple[GateTimeCandidate, ...]
    best_index: int
    search_strategy: str = "duration_sweep"
    warm_start_strategy: str = "previous_best"
    max_workers: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("GateTimeOptimizationResult requires at least one candidate.")
        if int(self.best_index) < 0 or int(self.best_index) >= len(self.candidates):
            raise ValueError("GateTimeOptimizationResult.best_index is out of range.")

    @property
    def best(self) -> GateTimeCandidate:
        return self.candidates[int(self.best_index)]

    @property
    def best_result(self) -> ControlResult:
        return self.best.result

    @property
    def best_duration_s(self) -> float:
        return float(self.best.duration_s)


@dataclass(frozen=True)
class StructuredToGrapeResult:
    structured_problem: ControlProblem
    grape_problem: ControlProblem
    structured_result: ControlResult
    warm_start_schedule: ControlSchedule
    grape_result: GrapeResult
    metrics: dict[str, Any] = field(default_factory=dict)


def build_grape_refinement_problem(
    problem: ControlProblem,
    *,
    time_grid: PiecewiseConstantTimeGrid | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> ControlProblem:
    refinement_grid = problem.time_grid if time_grid is None else time_grid
    parameterization = PiecewiseConstantParameterization(
        time_grid=refinement_grid,
        control_terms=tuple(problem.control_terms),
    )
    default_updates = {"grape_refinement_source_parameterization": type(problem.parameterization).__name__}
    if metadata_updates:
        default_updates.update(dict(metadata_updates))
    return _clone_problem(problem, parameterization, metadata_updates=default_updates)


def _run_gate_time_candidate(
    problem: ControlProblem,
    duration_s: float,
    *,
    solver_name: str,
    solver_config: Any,
    initial_schedule: ControlSchedule | np.ndarray | None,
) -> GateTimeCandidate:
    scaled_time_grid = problem.time_grid.scaled_to_duration(duration_s)
    parameterization = _clone_parameterization_with_time_grid(problem.parameterization, scaled_time_grid)
    scaled_problem = _clone_problem(
        problem,
        parameterization,
        metadata_updates={
            "gate_time_duration_s": float(duration_s),
            "gate_time_solver": str(solver_name),
        },
    )
    candidate_initial_schedule = _adapt_initial_schedule(
        scaled_problem.parameterization,
        initial_schedule,
        allow_shape_mismatch=True,
    )
    if str(solver_name) == "grape":
        result = solve_grape(
            scaled_problem,
            config=solver_config,
            initial_schedule=candidate_initial_schedule,
        )
    elif str(solver_name) == "structured-control":
        result = solve_structured_control(
            scaled_problem,
            config=solver_config,
            initial_schedule=candidate_initial_schedule,
        )
    else:
        raise ValueError(f"Unsupported gate-time solver '{solver_name}'.")
    return GateTimeCandidate(
        duration_s=float(duration_s),
        step_durations_s=tuple(float(value) for value in scaled_time_grid.step_durations_s),
        result=result,
    )


def _optimize_gate_time(
    problem: ControlProblem,
    durations_s: Sequence[float],
    *,
    solver_name: str,
    solver_config: Any,
    gate_time_config: GateTimeOptimizationConfig | None,
    initial_schedule: ControlSchedule | np.ndarray | None,
) -> GateTimeOptimizationResult:
    duration_values = _normalize_durations(durations_s)
    config = GateTimeOptimizationConfig() if gate_time_config is None else gate_time_config
    warm_start_strategy = str(config.warm_start_strategy).lower()
    max_workers = int(config.max_workers)

    if max_workers > 1 and warm_start_strategy != "none":
        raise ValueError(
            "Parallel gate-time optimization requires warm_start_strategy='none' because candidates run independently."
        )

    candidates: list[GateTimeCandidate] = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_gate_time_candidate,
                    problem,
                    duration,
                    solver_name=solver_name,
                    solver_config=solver_config,
                    initial_schedule=initial_schedule,
                )
                for duration in duration_values
            ]
            for future in futures:
                candidates.append(future.result())
    else:
        current_initial_schedule = initial_schedule
        best_candidate: GateTimeCandidate | None = None
        for duration in duration_values:
            candidate = _run_gate_time_candidate(
                problem,
                duration,
                solver_name=solver_name,
                solver_config=solver_config,
                initial_schedule=current_initial_schedule,
            )
            candidates.append(candidate)
            if best_candidate is None or float(candidate.result.objective_value) < float(best_candidate.result.objective_value):
                best_candidate = candidate
            if warm_start_strategy == "previous_best" and best_candidate is not None:
                current_initial_schedule = best_candidate.result.schedule

    objectives = np.asarray([float(candidate.result.objective_value) for candidate in candidates], dtype=float)
    best_index = int(np.argmin(objectives))
    fidelities = np.asarray([_nominal_fidelity(candidate.result) for candidate in candidates], dtype=float)

    return GateTimeOptimizationResult(
        solver_name=str(solver_name),
        candidates=tuple(candidates),
        best_index=best_index,
        search_strategy="duration_sweep",
        warm_start_strategy=str(config.warm_start_strategy),
        max_workers=max_workers,
        metrics={
            "candidate_count": int(len(candidates)),
            "objective_min": float(np.min(objectives)),
            "objective_max": float(np.max(objectives)),
            "objective_span": float(np.max(objectives) - np.min(objectives)),
            "best_duration_s": float(candidates[best_index].duration_s),
            "best_nominal_fidelity": float(fidelities[best_index]),
            "searched_durations_s": [float(duration) for duration in duration_values],
        },
    )


def optimize_gate_time_with_grape(
    problem: ControlProblem,
    *,
    durations_s: Sequence[float],
    config: GrapeConfig | None = None,
    gate_time_config: GateTimeOptimizationConfig | None = None,
    initial_schedule: ControlSchedule | np.ndarray | None = None,
) -> GateTimeOptimizationResult:
    solver_config = GrapeConfig() if config is None else config
    return _optimize_gate_time(
        problem,
        durations_s,
        solver_name="grape",
        solver_config=solver_config,
        gate_time_config=gate_time_config,
        initial_schedule=initial_schedule,
    )


def optimize_gate_time_with_structured_control(
    problem: ControlProblem,
    *,
    durations_s: Sequence[float],
    config: StructuredControlConfig | None = None,
    gate_time_config: GateTimeOptimizationConfig | None = None,
    initial_schedule: ControlSchedule | np.ndarray | None = None,
) -> GateTimeOptimizationResult:
    solver_config = StructuredControlConfig() if config is None else config
    return _optimize_gate_time(
        problem,
        durations_s,
        solver_name="structured-control",
        solver_config=solver_config,
        gate_time_config=gate_time_config,
        initial_schedule=initial_schedule,
    )


def solve_structured_then_grape(
    problem: ControlProblem,
    *,
    structured_config: StructuredControlConfig | None = None,
    grape_problem: ControlProblem | None = None,
    grape_config: GrapeConfig | None = None,
    structured_initial_schedule: ControlSchedule | np.ndarray | None = None,
    grape_initial_schedule: ControlSchedule | np.ndarray | None = None,
) -> StructuredToGrapeResult:
    if not isinstance(problem.parameterization, StructuredPulseParameterization):
        raise TypeError(
            "solve_structured_then_grape requires a ControlProblem with StructuredPulseParameterization."
        )

    structured_result = solve_structured_control(
        problem,
        config=structured_config,
        initial_schedule=structured_initial_schedule,
    )
    refinement_problem = build_grape_refinement_problem(problem) if grape_problem is None else grape_problem
    warm_start_schedule = _adapt_initial_schedule(
        refinement_problem.parameterization,
        structured_result.command_values,
    )
    assert warm_start_schedule is not None
    refinement_initial_schedule = (
        warm_start_schedule
        if grape_initial_schedule is None
        else _adapt_initial_schedule(refinement_problem.parameterization, grape_initial_schedule)
    )
    assert refinement_initial_schedule is not None
    grape_result = solve_grape(
        refinement_problem,
        config=grape_config,
        initial_schedule=refinement_initial_schedule,
    )

    structured_fidelity = _nominal_fidelity(structured_result)
    refined_fidelity = _nominal_fidelity(grape_result)
    return StructuredToGrapeResult(
        structured_problem=problem,
        grape_problem=refinement_problem,
        structured_result=structured_result,
        warm_start_schedule=warm_start_schedule,
        grape_result=grape_result,
        metrics={
            "structured_objective": float(structured_result.objective_value),
            "grape_objective": float(grape_result.objective_value),
            "objective_improvement": float(structured_result.objective_value - grape_result.objective_value),
            "structured_nominal_fidelity": float(structured_fidelity),
            "grape_nominal_fidelity": float(refined_fidelity),
            "nominal_fidelity_gain": float(refined_fidelity - structured_fidelity),
        },
    )


__all__ = [
    "GateTimeOptimizationConfig",
    "GateTimeCandidate",
    "GateTimeOptimizationResult",
    "StructuredToGrapeResult",
    "build_grape_refinement_problem",
    "optimize_gate_time_with_grape",
    "optimize_gate_time_with_structured_control",
    "solve_structured_then_grape",
]