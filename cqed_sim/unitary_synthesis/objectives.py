from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


_METRIC_ALIASES = {
    "state_transfer": "state_average_fidelity",
    "state_average_fidelity": "state_average_fidelity",
    "state_min_fidelity": "state_min_fidelity",
    "weighted_state_infidelity": "weighted_state_infidelity",
    "observable_mismatch": "weighted_observable_error",
    "weighted_observable_error": "weighted_observable_error",
    "trajectory": "trajectory_task_loss",
    "trajectory_task_loss": "trajectory_task_loss",
    "unitary_trace_overlap": "subspace_unitary_overlap",
    "unitary_trace_fidelity": "subspace_unitary_process_fidelity",
    "unitary_process_fidelity": "subspace_unitary_process_fidelity",
    "unitary_coherent": "subspace_unitary_process_fidelity",
    "unitary_subspace": "subspace_unitary_process_fidelity",
    "subspace_unitary_overlap": "subspace_unitary_overlap",
    "subspace_unitary_process_fidelity": "subspace_unitary_process_fidelity",
    "isometry": "isometry_coherent_fidelity",
    "isometry_coherent": "isometry_coherent_fidelity",
    "isometry_coherent_fidelity": "isometry_coherent_fidelity",
    "isometry_basis": "isometry_basis_fidelity",
    "isometry_basis_fidelity": "isometry_basis_fidelity",
    "isometry_retention": "isometry_retention",
    "isometry_output_leakage": "isometry_output_leakage",
    "logical_leakage": "logical_leakage_worst",
    "logical_leakage_average": "logical_leakage_average",
    "logical_leakage_worst": "logical_leakage_worst",
    "path_leakage": "path_leakage_worst",
    "path_leakage_average": "path_leakage_average",
    "path_leakage_worst": "path_leakage_worst",
    "edge_population": "edge_population_worst",
    "edge_population_average": "edge_population_average",
    "edge_population_worst": "edge_population_worst",
    "channel_overlap": "channel_overlap",
    "channel_process_fidelity": "channel_process_fidelity",
    "channel_choi_fidelity": "channel_process_fidelity",
    "channel_avg_gate_fidelity": "channel_average_gate_fidelity",
    "channel_average_gate_fidelity": "channel_average_gate_fidelity",
    "channel_entanglement_fidelity": "channel_entanglement_fidelity",
    "channel_choi_error": "channel_choi_error",
}


_METRIC_SENSE = {
    "state_average_fidelity": "maximize",
    "state_min_fidelity": "maximize",
    "weighted_state_infidelity": "minimize",
    "weighted_observable_error": "minimize",
    "trajectory_task_loss": "minimize",
    "subspace_unitary_overlap": "maximize",
    "subspace_unitary_process_fidelity": "maximize",
    "isometry_coherent_fidelity": "maximize",
    "isometry_basis_fidelity": "maximize",
    "isometry_retention": "maximize",
    "isometry_output_leakage": "minimize",
    "logical_leakage_average": "minimize",
    "logical_leakage_worst": "minimize",
    "path_leakage_average": "minimize",
    "path_leakage_worst": "minimize",
    "edge_population_average": "minimize",
    "edge_population_worst": "minimize",
    "channel_overlap": "maximize",
    "channel_process_fidelity": "maximize",
    "channel_average_gate_fidelity": "maximize",
    "channel_entanglement_fidelity": "maximize",
    "channel_choi_error": "minimize",
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    weight: float = 1.0
    sense: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.weight) <= 0.0:
            raise ValueError("MetricSpec.weight must be positive.")
        if self.sense is not None and str(self.sense) not in {"maximize", "minimize"}:
            raise ValueError("MetricSpec.sense must be 'maximize' or 'minimize' when provided.")
        object.__setattr__(self, "name", normalize_metric_name(self.name))
        object.__setattr__(self, "options", dict(self.options))

    def resolved_sense(self) -> str:
        return str(self.sense or metric_sense(self.name))

    def to_record(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "weight": float(self.weight),
            "sense": self.resolved_sense(),
            "options": dict(self.options),
        }


def normalize_metric_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key not in _METRIC_ALIASES:
        raise ValueError(f"Unsupported metric '{name}'.")
    return str(_METRIC_ALIASES[key])


def metric_sense(name: str) -> str:
    canonical = normalize_metric_name(name)
    return str(_METRIC_SENSE[canonical])


def metric_cost(name: str, value: float, *, sense: str | None = None) -> float:
    resolved_sense = str(sense or metric_sense(name))
    if resolved_sense == "maximize":
        return float(1.0 - value)
    return float(value)


def default_metric_specs(*, target_type: str, open_system: bool = False) -> tuple[MetricSpec, ...]:
    normalized = str(target_type)
    if normalized == "unitary":
        return (MetricSpec("state_average_fidelity"),) if open_system else (MetricSpec("subspace_unitary_process_fidelity"),)
    if normalized == "state_mapping":
        return (MetricSpec("state_average_fidelity"),)
    if normalized == "reduced_state_mapping":
        return (MetricSpec("state_average_fidelity"),)
    if normalized == "isometry":
        return (MetricSpec("channel_process_fidelity"),) if open_system else (MetricSpec("isometry_coherent_fidelity"),)
    if normalized == "channel":
        return (MetricSpec("channel_process_fidelity"),)
    if normalized == "observable":
        return (MetricSpec("weighted_observable_error"),)
    if normalized == "trajectory":
        return (MetricSpec("trajectory_task_loss"),)
    raise ValueError(f"Unsupported target_type '{target_type}'.")


def objective_preset_defaults(*, objective: str, target_type: str, open_system: bool = False) -> tuple[tuple[MetricSpec, ...], dict[str, float]]:
    normalized = str(objective).strip().lower().replace("-", "_")
    if normalized in {"unitary_coherent", "unitary_subspace", "isometry_coherent", "state_transfer", "channel_process_fidelity", "channel_avg_gate_fidelity"}:
        metric_name = {
            "unitary_coherent": "subspace_unitary_process_fidelity",
            "unitary_subspace": "subspace_unitary_process_fidelity",
            "isometry_coherent": "isometry_coherent_fidelity" if not open_system else "channel_process_fidelity",
            "state_transfer": "state_average_fidelity",
            "channel_process_fidelity": "channel_process_fidelity",
            "channel_avg_gate_fidelity": "channel_average_gate_fidelity",
        }[normalized]
        return (MetricSpec(metric_name),), {}
    if normalized == "robust_gate":
        return default_metric_specs(target_type=target_type, open_system=open_system), {"robustness_weight": 1.0}
    if normalized == "hardware_friendly_gate":
        return default_metric_specs(target_type=target_type, open_system=open_system), {
            "duration_weight": 0.1,
            "smoothness_weight": 0.1,
            "hardware_penalty_weight": 1.0,
        }
    return (MetricSpec(normalized),), {}


def resolve_metric_specs(
    *,
    target_type: str,
    open_system: bool = False,
    metric: str | MetricSpec | None = None,
    metrics: Sequence[str | MetricSpec] | None = None,
    objective: str | None = None,
) -> tuple[tuple[MetricSpec, ...], dict[str, float]]:
    if metrics is not None and metric is not None:
        raise ValueError("Use either metric or metrics, not both.")
    if metrics is not None:
        return tuple(spec if isinstance(spec, MetricSpec) else MetricSpec(str(spec)) for spec in metrics), {}
    if metric is not None:
        if isinstance(metric, MetricSpec):
            return (metric,), {}
        return (MetricSpec(str(metric)),), {}
    if objective is not None:
        return objective_preset_defaults(objective=objective, target_type=target_type, open_system=open_system)
    return default_metric_specs(target_type=target_type, open_system=open_system), {}


__all__ = [
    "MetricSpec",
    "default_metric_specs",
    "metric_cost",
    "metric_sense",
    "normalize_metric_name",
    "objective_preset_defaults",
    "resolve_metric_specs",
]