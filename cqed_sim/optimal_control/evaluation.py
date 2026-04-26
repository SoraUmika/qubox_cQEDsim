from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import qutip as qt

from cqed_sim.core import FrameSpec
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import NoiseSpec, SimulationConfig, prepare_simulation

from .hardware import resolve_control_schedule
from .objectives import StateTransferObjective, UnitaryObjective
from .parameterizations import ControlSchedule
from .utils import dense_projector, json_ready

if TYPE_CHECKING:
    from .problems import ControlProblem


@dataclass(frozen=True)
class ControlEvaluationCase:
    model: Any
    label: str = "nominal"
    frame: FrameSpec = field(default_factory=FrameSpec)
    noise: NoiseSpec | None = None
    weight: float = 1.0
    compiler_dt_s: float | None = None
    max_step_s: float | None = None
    nsteps: int | None = None
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    simulation_config: SimulationConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.weight) <= 0.0:
            raise ValueError("ControlEvaluationCase.weight must be positive.")
        if self.compiler_dt_s is not None and float(self.compiler_dt_s) <= 0.0:
            raise ValueError("ControlEvaluationCase.compiler_dt_s must be positive when provided.")
        if self.max_step_s is not None and float(self.max_step_s) <= 0.0:
            raise ValueError("ControlEvaluationCase.max_step_s must be positive when provided.")
        if self.nsteps is not None and int(self.nsteps) <= 0:
            raise ValueError("ControlEvaluationCase.nsteps must be positive when provided.")
        object.__setattr__(self, "solver_options", dict(self.solver_options))


def _simulation_config_for_case(case: ControlEvaluationCase) -> SimulationConfig:
    if case.simulation_config is not None:
        return case.simulation_config
    return SimulationConfig(
        frame=case.frame,
        max_step=case.max_step_s,
        nsteps=case.nsteps,
        solver_options=case.solver_options,
    )


@dataclass(frozen=True)
class ControlObjectiveEvaluation:
    name: str
    kind: str
    weight: float
    pair_labels: tuple[str, ...]
    fidelities: tuple[float, ...]
    fidelity_weighted: float
    fidelity_mean: float
    fidelity_min: float
    fidelity_max: float
    leakage_values: tuple[float, ...] = ()
    leakage_weighted: float | None = None
    leakage_mean: float | None = None
    leakage_max: float | None = None


@dataclass(frozen=True)
class ControlMemberEvaluation:
    label: str
    weight: float
    objective_reports: tuple[ControlObjectiveEvaluation, ...]
    aggregate_fidelity: float
    aggregate_infidelity: float
    aggregate_leakage: float | None = None
    compiler_dt_s: float | None = None
    max_step_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlEvaluationResult:
    aggregate_mode: str
    member_reports: tuple[ControlMemberEvaluation, ...]
    metrics: dict[str, Any]
    pulse_metadata: dict[str, Any]
    compiler_dt_s: float
    duration_s: float
    waveform_mode: str = "command"
    parameterization_metrics: dict[str, Any] = field(default_factory=dict)
    hardware_metrics: dict[str, Any] = field(default_factory=dict)
    hardware_reports: tuple[dict[str, Any], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "aggregate_mode": str(self.aggregate_mode),
            "metrics": dict(self.metrics),
            "pulse_metadata": dict(self.pulse_metadata),
            "compiler_dt_s": float(self.compiler_dt_s),
            "duration_s": float(self.duration_s),
            "waveform_mode": str(self.waveform_mode),
            "parameterization_metrics": dict(self.parameterization_metrics),
            "hardware_metrics": dict(self.hardware_metrics),
            "hardware_reports": list(self.hardware_reports),
            "member_reports": [
                {
                    "label": report.label,
                    "weight": float(report.weight),
                    "aggregate_fidelity": float(report.aggregate_fidelity),
                    "aggregate_infidelity": float(report.aggregate_infidelity),
                    "aggregate_leakage": None if report.aggregate_leakage is None else float(report.aggregate_leakage),
                    "compiler_dt_s": None if report.compiler_dt_s is None else float(report.compiler_dt_s),
                    "max_step_s": None if report.max_step_s is None else float(report.max_step_s),
                    "metadata": dict(report.metadata),
                    "objectives": [
                        {
                            "name": objective.name,
                            "kind": objective.kind,
                            "weight": float(objective.weight),
                            "pair_labels": list(objective.pair_labels),
                            "fidelities": [float(value) for value in objective.fidelities],
                            "fidelity_weighted": float(objective.fidelity_weighted),
                            "fidelity_mean": float(objective.fidelity_mean),
                            "fidelity_min": float(objective.fidelity_min),
                            "fidelity_max": float(objective.fidelity_max),
                            "leakage_values": [float(value) for value in objective.leakage_values],
                            "leakage_weighted": None if objective.leakage_weighted is None else float(objective.leakage_weighted),
                            "leakage_mean": None if objective.leakage_mean is None else float(objective.leakage_mean),
                            "leakage_max": None if objective.leakage_max is None else float(objective.leakage_max),
                        }
                        for objective in report.objective_reports
                    ],
                }
                for report in self.member_reports
            ],
        }
        return json_ready(payload)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")
        return output_path


def _default_compiler_dt(step_durations_s: np.ndarray) -> float:
    min_step = float(np.min(np.asarray(step_durations_s, dtype=float)))
    return float(min(1.0e-9, max(min_step / 8.0, 1.0e-10)))


def _qobj_dims(model: Any, full_dim: int) -> list[list[int]]:
    dims = tuple(int(value) for value in getattr(model, "subsystem_dims", (full_dim,)))
    if int(np.prod(np.asarray(dims, dtype=int))) != int(full_dim):
        return [[int(full_dim)], [1]]
    return [list(dims), [1] * len(dims)]


def _ket_from_vector(vector: np.ndarray, dims: list[list[int]]) -> qt.Qobj:
    data = np.asarray(vector, dtype=np.complex128).reshape(-1, 1)
    return qt.Qobj(data, dims=dims)


def _projector_qobj(projector: np.ndarray, dims: list[list[int]]) -> qt.Qobj:
    return qt.Qobj(np.asarray(projector, dtype=np.complex128), dims=[dims[0], dims[0]])


def _density_matrix(state: qt.Qobj) -> qt.Qobj:
    return state if state.isoper else qt.ket2dm(state)


def _state_fidelity(final_state: qt.Qobj, target_state: qt.Qobj) -> float:
    return float(qt.metrics.fidelity(final_state, target_state))


def _state_leakage(final_state: qt.Qobj, projector: qt.Qobj) -> float:
    rho = _density_matrix(final_state)
    retained = float(np.real((projector * rho).tr()))
    retained = float(np.clip(retained, 0.0, 1.0))
    return float(1.0 - retained)


def _resolved_objective_pairs(problem: "ControlProblem", objective: StateTransferObjective | UnitaryObjective):
    if isinstance(objective, StateTransferObjective):
        initial_states, target_states, weights, labels = objective.resolved_pairs(full_dim=problem.full_dim)
        return "state_transfer", initial_states, target_states, weights, labels, None
    if isinstance(objective, UnitaryObjective):
        initial_states, target_states, weights, labels = objective.resolved_pairs(full_dim=problem.full_dim)
        projector = None if objective.subspace is None else dense_projector(objective.subspace)
        return "unitary", initial_states, target_states, weights, labels, projector
    raise TypeError(f"Unsupported objective type '{type(objective).__name__}'.")


def _evaluate_case(
    problem: "ControlProblem",
    case: ControlEvaluationCase,
    compiled,
    drive_ops: dict[str, Any],
) -> ControlMemberEvaluation:
    if int(np.prod(np.asarray(getattr(case.model, "subsystem_dims", (problem.full_dim,)), dtype=int))) != int(problem.full_dim):
        raise ValueError(
            f"Evaluation case '{case.label}' has model dimension incompatible with ControlProblem.full_dim={problem.full_dim}."
        )

    config = _simulation_config_for_case(case)
    session = prepare_simulation(case.model, compiled, drive_ops, config=config, noise=case.noise, e_ops={})
    dims = _qobj_dims(case.model, problem.full_dim)
    objective_reports: list[ControlObjectiveEvaluation] = []
    objective_weights: list[float] = []
    objective_fidelities: list[float] = []
    objective_leakages: list[float] = []

    for objective in problem.objectives:
        kind, initial_states, target_states, state_weights, labels, projector = _resolved_objective_pairs(problem, objective)
        initial_qobjs = [_ket_from_vector(vector, dims) for vector in initial_states]
        target_qobjs = [_ket_from_vector(vector, dims) for vector in target_states]
        final_states = [session.run(initial_state).final_state for initial_state in initial_qobjs]

        fidelities = np.asarray(
            [_state_fidelity(final_state, target_state) for final_state, target_state in zip(final_states, target_qobjs, strict=True)],
            dtype=float,
        )
        weighted_fidelity = float(np.sum(np.asarray(state_weights, dtype=float) * fidelities))

        leakage_values: tuple[float, ...] = ()
        leakage_weighted: float | None = None
        leakage_mean: float | None = None
        leakage_max: float | None = None
        if projector is not None:
            projector_qobj = _projector_qobj(projector, dims)
            leakages = np.asarray([_state_leakage(final_state, projector_qobj) for final_state in final_states], dtype=float)
            leakage_values = tuple(float(value) for value in leakages)
            leakage_weighted = float(np.sum(np.asarray(state_weights, dtype=float) * leakages))
            leakage_mean = float(np.mean(leakages))
            leakage_max = float(np.max(leakages))
            objective_leakages.append(float(leakage_weighted))

        objective_reports.append(
            ControlObjectiveEvaluation(
                name=str(getattr(objective, "name", kind)),
                kind=str(kind),
                weight=float(getattr(objective, "weight", 1.0)),
                pair_labels=tuple(str(label) for label in labels),
                fidelities=tuple(float(value) for value in fidelities),
                fidelity_weighted=float(weighted_fidelity),
                fidelity_mean=float(np.mean(fidelities)),
                fidelity_min=float(np.min(fidelities)),
                fidelity_max=float(np.max(fidelities)),
                leakage_values=leakage_values,
                leakage_weighted=leakage_weighted,
                leakage_mean=leakage_mean,
                leakage_max=leakage_max,
            )
        )
        objective_weights.append(float(getattr(objective, "weight", 1.0)))
        objective_fidelities.append(float(weighted_fidelity))

    objective_weight_array = np.asarray(objective_weights, dtype=float)
    objective_weight_array = objective_weight_array / np.sum(objective_weight_array)
    aggregate_fidelity = float(np.sum(objective_weight_array * np.asarray(objective_fidelities, dtype=float)))
    aggregate_infidelity = float(1.0 - aggregate_fidelity)
    aggregate_leakage = None
    if objective_leakages:
        aggregate_leakage = float(np.mean(np.asarray(objective_leakages, dtype=float)))

    return ControlMemberEvaluation(
        label=str(case.label),
        weight=float(case.weight),
        objective_reports=tuple(objective_reports),
        aggregate_fidelity=float(aggregate_fidelity),
        aggregate_infidelity=float(aggregate_infidelity),
        aggregate_leakage=aggregate_leakage,
        compiler_dt_s=None if case.compiler_dt_s is None else float(case.compiler_dt_s),
        max_step_s=None if case.max_step_s is None else float(case.max_step_s),
        metadata={
            "model_type": type(case.model).__name__,
            "noise": None
            if case.noise is None
            else {
                key: value
                for key, value in vars(case.noise).items()
                if value is not None and value != 0.0
            },
            **dict(case.metadata),
        },
    )


def evaluate_control_with_simulator(
    problem: "ControlProblem",
    schedule: ControlSchedule | np.ndarray,
    *,
    cases: Sequence[ControlEvaluationCase] = (),
    model: Any | None = None,
    frame: FrameSpec | None = None,
    noise: NoiseSpec | None = None,
    compiler_dt_s: float | None = None,
    max_step_s: float | None = None,
    nsteps: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
    simulation_config: SimulationConfig | None = None,
    aggregate_mode: str | None = None,
    waveform_mode: str = "problem_default",
) -> ControlEvaluationResult:
    if cases and model is not None:
        raise ValueError("Pass either explicit cases or a single model/frame/noise specification, not both.")
    if isinstance(schedule, ControlSchedule):
        control_schedule = schedule.clipped()
    else:
        control_schedule = ControlSchedule(problem.parameterization, problem.parameterization.clip(np.asarray(schedule, dtype=float)))

    if not cases:
        if model is None:
            raise ValueError("evaluate_control_with_simulator requires either cases=... or model=... .")
        cases = (
            ControlEvaluationCase(
                model=model,
                label="nominal",
                frame=FrameSpec() if frame is None else frame,
                noise=noise,
                compiler_dt_s=compiler_dt_s,
                max_step_s=max_step_s,
                nsteps=nsteps,
                solver_options=dict(solver_options or {}),
                simulation_config=simulation_config,
            ),
        )

    resolved = resolve_control_schedule(problem, control_schedule, apply_hardware=True)
    resolved_step_durations_s = np.diff(np.asarray(resolved.time_boundaries_s, dtype=float))
    resolved_duration_s = float(np.sum(resolved_step_durations_s))
    resolved_waveform_mode = str(waveform_mode).lower()
    if resolved_waveform_mode == "problem_default":
        resolved_waveform_mode = "physical" if problem.hardware_model is not None else "command"
    if resolved_waveform_mode == "command":
        waveform_values = np.asarray(resolved.command_values, dtype=float)
    elif resolved_waveform_mode == "physical":
        waveform_values = np.asarray(resolved.physical_values, dtype=float)
    else:
        raise ValueError("waveform_mode must be 'problem_default', 'command', or 'physical'.")

    pulses, drive_ops, pulse_meta = control_schedule.to_pulses(waveform_values=waveform_values)
    pulse_meta = {
        **dict(pulse_meta),
        "waveform_mode": str(resolved_waveform_mode),
        "parameterization_metrics": dict(resolved.parameterization_metrics),
        "hardware_metrics": dict(resolved.hardware_metrics),
    }
    default_dt = _default_compiler_dt(resolved_step_durations_s) if compiler_dt_s is None else float(compiler_dt_s)
    compiled_cache: dict[float, Any] = {}
    member_reports: list[ControlMemberEvaluation] = []

    for case in cases:
        case_dt = float(default_dt if case.compiler_dt_s is None else case.compiler_dt_s)
        compiled = compiled_cache.get(case_dt)
        if compiled is None:
            compiled = SequenceCompiler(dt=case_dt).compile(pulses, t_end=resolved_duration_s)
            compiled_cache[case_dt] = compiled
        resolved_case = case
        updates: dict[str, Any] = {}
        if case.compiler_dt_s is None:
            updates["compiler_dt_s"] = case_dt
        if case.simulation_config is None:
            if simulation_config is not None:
                updates["simulation_config"] = simulation_config
            if case.max_step_s is None and max_step_s is not None:
                updates["max_step_s"] = max_step_s
            if case.nsteps is None and nsteps is not None:
                updates["nsteps"] = nsteps
            if not case.solver_options and solver_options:
                updates["solver_options"] = dict(solver_options)
        if updates:
            resolved_case = replace(case, **updates)
        member_reports.append(_evaluate_case(problem, resolved_case, compiled, drive_ops))

    mode = str(problem.ensemble_aggregate if aggregate_mode is None else aggregate_mode)
    if mode not in {"mean", "worst"}:
        raise ValueError("aggregate_mode must be 'mean' or 'worst'.")

    if mode == "worst":
        active_index = int(np.argmin(np.asarray([report.aggregate_fidelity for report in member_reports], dtype=float)))
        active_report = member_reports[active_index]
        metrics = {
            "aggregate_mode": "worst",
            "aggregate_fidelity": float(active_report.aggregate_fidelity),
            "aggregate_infidelity": float(active_report.aggregate_infidelity),
            "aggregate_leakage": None if active_report.aggregate_leakage is None else float(active_report.aggregate_leakage),
            "active_case": str(active_report.label),
        }
    else:
        weights = np.asarray([float(report.weight) for report in member_reports], dtype=float)
        weights = weights / np.sum(weights)
        fidelities = np.asarray([float(report.aggregate_fidelity) for report in member_reports], dtype=float)
        infidelities = np.asarray([float(report.aggregate_infidelity) for report in member_reports], dtype=float)
        leakage_pairs = [
            (float(weight), float(report.aggregate_leakage))
            for weight, report in zip(weights, member_reports, strict=True)
            if report.aggregate_leakage is not None
        ]
        metrics = {
            "aggregate_mode": "mean",
            "aggregate_fidelity": float(np.sum(weights * fidelities)),
            "aggregate_infidelity": float(np.sum(weights * infidelities)),
            "aggregate_leakage": None
            if not leakage_pairs
            else float(
                np.sum(np.asarray([pair[0] for pair in leakage_pairs], dtype=float) * np.asarray([pair[1] for pair in leakage_pairs], dtype=float))
                / np.sum(np.asarray([pair[0] for pair in leakage_pairs], dtype=float))
            ),
            "case_weights": [float(weight) for weight in weights],
        }

    return ControlEvaluationResult(
        aggregate_mode=mode,
        member_reports=tuple(member_reports),
        metrics=metrics,
        pulse_metadata=dict(pulse_meta),
        compiler_dt_s=float(default_dt),
        duration_s=float(resolved_duration_s),
        waveform_mode=str(resolved_waveform_mode),
        parameterization_metrics=dict(resolved.parameterization_metrics),
        hardware_metrics=dict(resolved.hardware_metrics),
        hardware_reports=tuple({"name": report.name, "metrics": dict(report.metrics)} for report in resolved.hardware_reports),
    )


__all__ = [
    "ControlEvaluationCase",
    "ControlObjectiveEvaluation",
    "ControlMemberEvaluation",
    "ControlEvaluationResult",
    "evaluate_control_with_simulator",
]
