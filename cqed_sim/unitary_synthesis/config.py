from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .subspace import Subspace


@dataclass(frozen=True)
class LeakagePenalty:
    weight: float = 0.0
    allowed_subspace: Subspace | Sequence[int] | None = None
    metric: str = "worst"
    checkpoint_weight: float = 0.0
    checkpoints: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.metric not in {"worst", "average"}:
            raise ValueError("LeakagePenalty.metric must be 'worst' or 'average'.")
        if float(self.weight) < 0.0 or float(self.checkpoint_weight) < 0.0:
            raise ValueError("LeakagePenalty weights must be non-negative.")
        object.__setattr__(self, "checkpoints", tuple(int(idx) for idx in self.checkpoints))

    def resolve_subspace(self, default: Subspace | None) -> Subspace | None:
        if self.allowed_subspace is None:
            return default
        if isinstance(self.allowed_subspace, Subspace):
            return self.allowed_subspace
        if default is None:
            raise ValueError("A default subspace is required when allowed_subspace is provided as indices.")
        return Subspace.custom(
            full_dim=default.full_dim,
            indices=tuple(int(idx) for idx in self.allowed_subspace),
        )

    def to_record(self) -> dict[str, Any]:
        subspace = self.allowed_subspace
        if isinstance(subspace, Subspace):
            subspace_payload: Any = {
                "kind": subspace.kind,
                "full_dim": int(subspace.full_dim),
                "indices": list(subspace.indices),
                "labels": list(subspace.labels),
            }
        elif subspace is None:
            subspace_payload = None
        else:
            subspace_payload = [int(idx) for idx in subspace]
        return {
            "weight": float(self.weight),
            "metric": str(self.metric),
            "checkpoint_weight": float(self.checkpoint_weight),
            "checkpoints": [int(idx) for idx in self.checkpoints],
            "allowed_subspace": subspace_payload,
        }


@dataclass(frozen=True)
class SynthesisConstraints:
    max_amplitude: float | None = None
    max_duration: float | None = None
    max_primitives: int | None = None
    allowed_primitive_counts: tuple[int, ...] = ()
    smoothness_penalty: bool = False
    smoothness_weight: float = 1.0
    max_bandwidth: float | None = None
    bandwidth_weight: float = 1.0
    forbidden_parameter_ranges: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    forbidden_range_weight: float = 1.0
    duration_mode: str = "penalty"

    def __post_init__(self) -> None:
        if self.duration_mode not in {"penalty", "hard"}:
            raise ValueError("SynthesisConstraints.duration_mode must be 'penalty' or 'hard'.")
        if self.max_amplitude is not None and self.max_amplitude <= 0.0:
            raise ValueError("max_amplitude must be positive when provided.")
        if self.max_duration is not None and self.max_duration <= 0.0:
            raise ValueError("max_duration must be positive when provided.")
        if self.max_primitives is not None and self.max_primitives <= 0:
            raise ValueError("max_primitives must be positive when provided.")
        if self.max_bandwidth is not None and self.max_bandwidth <= 0.0:
            raise ValueError("max_bandwidth must be positive when provided.")
        if self.smoothness_weight < 0.0 or self.bandwidth_weight < 0.0 or self.forbidden_range_weight < 0.0:
            raise ValueError("Constraint penalty weights must be non-negative.")
        normalized: dict[str, tuple[tuple[float, float], ...]] = {}
        for key, ranges in dict(self.forbidden_parameter_ranges).items():
            rows: list[tuple[float, float]] = []
            for item in ranges:
                lo, hi = float(item[0]), float(item[1])
                rows.append((min(lo, hi), max(lo, hi)))
            normalized[str(key)] = tuple(rows)
        object.__setattr__(self, "allowed_primitive_counts", tuple(int(v) for v in self.allowed_primitive_counts))
        object.__setattr__(self, "forbidden_parameter_ranges", normalized)

    def matches_parameter(self, pattern: str, candidate: str) -> bool:
        return fnmatch.fnmatch(candidate, pattern)

    def validate_sequence_length(self, primitive_count: int) -> None:
        count = int(primitive_count)
        if self.max_primitives is not None and count > int(self.max_primitives):
            raise ValueError(
                f"Sequence has {count} primitives but SynthesisConstraints.max_primitives={int(self.max_primitives)}."
            )
        if self.allowed_primitive_counts and count not in set(self.allowed_primitive_counts):
            allowed = ", ".join(str(v) for v in self.allowed_primitive_counts)
            raise ValueError(f"Sequence has {count} primitives but allowed_primitive_counts are {{{allowed}}}.")

    def to_record(self) -> dict[str, Any]:
        return {
            "max_amplitude": None if self.max_amplitude is None else float(self.max_amplitude),
            "max_duration": None if self.max_duration is None else float(self.max_duration),
            "max_primitives": None if self.max_primitives is None else int(self.max_primitives),
            "allowed_primitive_counts": [int(v) for v in self.allowed_primitive_counts],
            "smoothness_penalty": bool(self.smoothness_penalty),
            "smoothness_weight": float(self.smoothness_weight),
            "max_bandwidth": None if self.max_bandwidth is None else float(self.max_bandwidth),
            "bandwidth_weight": float(self.bandwidth_weight),
            "forbidden_parameter_ranges": {
                str(key): [[float(lo), float(hi)] for lo, hi in ranges]
                for key, ranges in self.forbidden_parameter_ranges.items()
            },
            "forbidden_range_weight": float(self.forbidden_range_weight),
            "duration_mode": str(self.duration_mode),
        }


@dataclass(frozen=True)
class MultiObjective:
    fidelity_weight: float = 1.0
    task_weight: float | None = None
    leakage_weight: float = 0.0
    duration_weight: float = 0.0
    gate_count_weight: float = 0.0
    pulse_power_weight: float = 0.0
    robustness_weight: float = 0.0
    smoothness_weight: float = 0.0
    hardware_penalty_weight: float = 1.0
    mode: str = "weighted_sum"

    def __post_init__(self) -> None:
        if self.mode not in {"weighted_sum"}:
            raise ValueError("MultiObjective.mode currently supports only 'weighted_sum'.")
        for name in (
            "fidelity_weight",
            "task_weight",
            "leakage_weight",
            "duration_weight",
            "gate_count_weight",
            "pulse_power_weight",
            "robustness_weight",
            "smoothness_weight",
            "hardware_penalty_weight",
        ):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = float(raw)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

    def to_record(self) -> dict[str, Any]:
        return {
            "fidelity_weight": float(self.fidelity_weight),
            "task_weight": None if self.task_weight is None else float(self.task_weight),
            "leakage_weight": float(self.leakage_weight),
            "duration_weight": float(self.duration_weight),
            "gate_count_weight": float(self.gate_count_weight),
            "pulse_power_weight": float(self.pulse_power_weight),
            "robustness_weight": float(self.robustness_weight),
            "smoothness_weight": float(self.smoothness_weight),
            "hardware_penalty_weight": float(self.hardware_penalty_weight),
            "mode": str(self.mode),
        }


@dataclass(frozen=True)
class ExecutionOptions:
    engine: str = "auto"
    fallback_engine: str = "legacy"
    device: str = "auto"
    use_fast_path: bool = True
    jit: bool = True
    vectorized_candidates: bool = True
    candidate_batch_size: int = 0
    cache_fast_path: bool = True

    def __post_init__(self) -> None:
        if self.engine not in {"auto", "legacy", "numpy", "jax"}:
            raise ValueError("ExecutionOptions.engine must be 'auto', 'legacy', 'numpy', or 'jax'.")
        if self.fallback_engine not in {"legacy", "numpy"}:
            raise ValueError("ExecutionOptions.fallback_engine must be 'legacy' or 'numpy'.")
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ValueError("ExecutionOptions.device must be 'auto', 'cpu', or 'gpu'.")
        if int(self.candidate_batch_size) < 0:
            raise ValueError("ExecutionOptions.candidate_batch_size must be non-negative.")

    def to_record(self) -> dict[str, Any]:
        return {
            "engine": str(self.engine),
            "fallback_engine": str(self.fallback_engine),
            "device": str(self.device),
            "use_fast_path": bool(self.use_fast_path),
            "jit": bool(self.jit),
            "vectorized_candidates": bool(self.vectorized_candidates),
            "candidate_batch_size": int(self.candidate_batch_size),
            "cache_fast_path": bool(self.cache_fast_path),
        }


@dataclass(frozen=True)
class Normal:
    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0.0:
            raise ValueError("Normal.std must be non-negative.")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(float(self.mean), float(self.std)))

    def nominal(self) -> float:
        return float(self.mean)

    def to_record(self) -> dict[str, Any]:
        return {"kind": "normal", "mean": float(self.mean), "std": float(self.std)}


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("Uniform requires low <= high.")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(float(self.low), float(self.high)))

    def nominal(self) -> float:
        return 0.5 * (float(self.low) + float(self.high))

    def to_record(self) -> dict[str, Any]:
        return {"kind": "uniform", "low": float(self.low), "high": float(self.high)}


DistributionSpec = Normal | Uniform | float | int


class ParameterDistribution:
    def __init__(
        self,
        *,
        sample_count: int = 4,
        include_nominal: bool = True,
        aggregate: str = "mean",
        **parameters: DistributionSpec,
    ) -> None:
        if aggregate not in {"mean", "worst"}:
            raise ValueError("ParameterDistribution.aggregate must be 'mean' or 'worst'.")
        if int(sample_count) <= 0:
            raise ValueError("ParameterDistribution.sample_count must be positive.")
        if not parameters:
            raise ValueError("ParameterDistribution requires at least one uncertain parameter.")
        self.sample_count = int(sample_count)
        self.include_nominal = bool(include_nominal)
        self.aggregate = str(aggregate)
        self.parameters = dict(parameters)

    def __iter__(self):
        return iter(self.parameters.items())

    @staticmethod
    def _sample_one(spec: DistributionSpec, rng: np.random.Generator) -> float:
        if hasattr(spec, "sample"):
            return float(spec.sample(rng))  # type: ignore[call-arg]
        return float(spec)

    @staticmethod
    def _nominal_one(spec: DistributionSpec) -> float:
        if hasattr(spec, "nominal"):
            return float(spec.nominal())  # type: ignore[call-arg]
        return float(spec)

    def sample_assignments(self, rng: np.random.Generator) -> list[dict[str, float]]:
        samples: list[dict[str, float]] = []
        if self.include_nominal:
            samples.append({name: self._nominal_one(spec) for name, spec in self.parameters.items()})
        for _ in range(self.sample_count):
            samples.append({name: self._sample_one(spec, rng) for name, spec in self.parameters.items()})
        return samples

    @staticmethod
    def _set_nested_attr(target: Any, path: str, value: float) -> None:
        parts = str(path).split(".")
        obj = target
        for name in parts[:-1]:
            obj = getattr(obj, name)
        setattr(obj, parts[-1], float(value))

    def apply(self, model: Any, assignment: Mapping[str, float]) -> Any:
        updated = copy.deepcopy(model)
        for name, value in assignment.items():
            self._set_nested_attr(updated, str(name), float(value))
        return updated

    def to_record(self) -> dict[str, Any]:
        def _record(spec: DistributionSpec) -> dict[str, Any]:
            if hasattr(spec, "to_record"):
                return dict(spec.to_record())  # type: ignore[call-arg]
            try:
                value = float(spec)
            except Exception:
                return {"kind": "custom", "repr": repr(spec)}
            return {"kind": "fixed", "value": value}

        return {
            "sample_count": int(self.sample_count),
            "include_nominal": bool(self.include_nominal),
            "aggregate": str(self.aggregate),
            "parameters": {str(name): _record(spec) for name, spec in self.parameters.items()},
        }
