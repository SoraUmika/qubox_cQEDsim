from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .utils import json_ready


@dataclass(frozen=True)
class MeasurementOutcome:
    eigenvalue: complex | None
    probability: float
    bond_state: np.ndarray
    outcome_index: int | None = None

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "eigenvalue": None if self.eigenvalue is None else complex(self.eigenvalue),
                "probability": float(self.probability),
                "outcome_index": None if self.outcome_index is None else int(self.outcome_index),
                "bond_state": np.asarray(self.bond_state, dtype=np.complex128),
            }
        )


@dataclass(frozen=True)
class BranchRecord:
    probability: float
    estimator_value: complex
    measurement_eigenvalues: tuple[complex, ...]
    accepted_probability: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "probability": float(self.probability),
            "accepted_probability": None if self.accepted_probability is None else float(self.accepted_probability),
            "estimator_value": complex(self.estimator_value),
            "measurement_eigenvalues": [complex(v) for v in self.measurement_eigenvalues],
        }


@dataclass
class CorrelatorEstimate:
    mean: complex
    variance: float
    stderr: float
    attempted_shots: int
    accepted_shots: int
    burn_in_steps: int
    total_steps: int
    schedule_record: dict[str, Any]
    samples: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "mean": complex(self.mean),
                "variance": float(self.variance),
                "stderr": float(self.stderr),
                "attempted_shots": int(self.attempted_shots),
                "accepted_shots": int(self.accepted_shots),
                "burn_in_steps": int(self.burn_in_steps),
                "total_steps": int(self.total_steps),
                "schedule": dict(self.schedule_record),
                "samples": None if self.samples is None else np.asarray(self.samples, dtype=np.complex128),
                "metadata": dict(self.metadata),
            }
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_record(), indent=2), encoding="utf-8")
        return path


@dataclass
class ExactCorrelatorResult:
    mean: complex
    variance: float
    branch_probability_sum: float
    accepted_probability_sum: float
    normalization_error: float
    burn_in_steps: int
    total_steps: int
    schedule_record: dict[str, Any]
    branches: list[BranchRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stderr(self) -> float:
        if self.accepted_probability_sum <= 0.0:
            return float("nan")
        return 0.0

    def to_record(self) -> dict[str, Any]:
        return {
            "mean": complex(self.mean),
            "variance": float(self.variance),
            "branch_probability_sum": float(self.branch_probability_sum),
            "accepted_probability_sum": float(self.accepted_probability_sum),
            "normalization_error": float(self.normalization_error),
            "burn_in_steps": int(self.burn_in_steps),
            "total_steps": int(self.total_steps),
            "schedule": dict(self.schedule_record),
            "branches": [branch.to_record() for branch in self.branches],
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_ready(self.to_record()), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class ChannelDiagnostics:
    physical_dim: int
    bond_dim: int
    kraus_count: int
    kraus_completeness_error: float
    trace_preservation_error: float
    right_canonical_error: float
    unitary_error: float | None
    hermiticity_preserving: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "physical_dim": int(self.physical_dim),
            "bond_dim": int(self.bond_dim),
            "kraus_count": int(self.kraus_count),
            "kraus_completeness_error": float(self.kraus_completeness_error),
            "trace_preservation_error": float(self.trace_preservation_error),
            "right_canonical_error": float(self.right_canonical_error),
            "unitary_error": None if self.unitary_error is None else float(self.unitary_error),
            "hermiticity_preserving": bool(self.hermiticity_preserving),
        }


@dataclass
class BurnInSummary:
    steps: int
    residuals: np.ndarray
    final_state: np.ndarray

    @property
    def max_residual(self) -> float:
        if self.residuals.size == 0:
            return 0.0
        return float(np.max(self.residuals))

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "steps": int(self.steps),
                "residuals": np.asarray(self.residuals, dtype=float),
                "max_residual": float(self.max_residual),
                "final_state": np.asarray(self.final_state, dtype=np.complex128),
            }
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_record(), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class EstimatorComparison:
    monte_carlo_mean: complex
    exact_mean: complex
    absolute_error: float
    monte_carlo_stderr: float

    @property
    def within_two_sigma(self) -> bool:
        if not np.isfinite(self.monte_carlo_stderr):
            return False
        return bool(self.absolute_error <= 2.0 * float(self.monte_carlo_stderr))

    def to_record(self) -> dict[str, Any]:
        return {
            "monte_carlo_mean": complex(self.monte_carlo_mean),
            "exact_mean": complex(self.exact_mean),
            "absolute_error": float(self.absolute_error),
            "monte_carlo_stderr": float(self.monte_carlo_stderr),
            "within_two_sigma": bool(self.within_two_sigma),
        }
