from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass
class ReadoutMetricSet:
    assignment_fidelity: float
    physical_qnd_fidelity: float
    measured_two_shot_qnd_fidelity: float | None = None
    p_0_to_1: float = 0.0
    p_1_to_0: float = 0.0
    leakage_probability: float = 0.0
    residual_resonator_photons: float = 0.0
    residual_filter_photons: float = 0.0
    pulse_energy: float = 0.0
    slew_penalty: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def objective(
        self,
        *,
        wA: float = 1.0,
        wQ: float = 1.0,
        wL: float = 1.0,
        wR: float = 1.0,
        wE: float = 0.0,
        wS: float = 0.0,
        wM: float = 0.0,
        mist_penalty: float = 0.0,
    ) -> float:
        return float(
            wA * (1.0 - self.assignment_fidelity)
            + wQ * (1.0 - self.physical_qnd_fidelity)
            + wL * self.leakage_probability
            + wR * (self.residual_resonator_photons + self.residual_filter_photons)
            + wE * self.pulse_energy
            + wS * self.slew_penalty
            + wM * float(mist_penalty)
        )


def assignment_fidelity(confusion: Sequence[Sequence[float]]) -> float:
    matrix = np.asarray(confusion, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("confusion must be a nonempty matrix.")
    dim = min(matrix.shape)
    return float(np.mean(np.diag(matrix[:dim, :dim])))


def physical_qnd_fidelity(transition_matrix: Sequence[Sequence[float]], computational_indices: Sequence[int] = (0, 1)) -> float:
    matrix = np.asarray(transition_matrix, dtype=float)
    indices = tuple(int(idx) for idx in computational_indices)
    if not indices:
        raise ValueError("computational_indices must be nonempty.")
    return float(np.mean([matrix[idx, col] for col, idx in enumerate(indices) if idx < matrix.shape[0] and col < matrix.shape[1]]))


def measured_two_shot_qnd_fidelity(two_shot_confusion: Sequence[Sequence[float]]) -> float:
    return assignment_fidelity(two_shot_confusion)


def transition_probability(transition_matrix: Sequence[Sequence[float]], target: int, prepared: int) -> float:
    matrix = np.asarray(transition_matrix, dtype=float)
    if int(target) >= matrix.shape[0] or int(prepared) >= matrix.shape[1]:
        return 0.0
    return float(matrix[int(target), int(prepared)])


def leakage_probability_from_transition(
    transition_matrix: Sequence[Sequence[float]],
    *,
    computational_rows: Sequence[int] = (0, 1),
) -> float:
    matrix = np.asarray(transition_matrix, dtype=float)
    rows = [int(row) for row in computational_rows if int(row) < matrix.shape[0]]
    if matrix.size == 0:
        return 0.0
    retained = np.sum(matrix[rows, :], axis=0) if rows else np.zeros(matrix.shape[1], dtype=float)
    return float(np.mean(np.maximum(0.0, 1.0 - retained)))


def residual_photons(value: float | Mapping[object, float] | Sequence[float]) -> float:
    if isinstance(value, Mapping):
        return float(max((float(v) for v in value.values()), default=0.0))
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(np.max(arr) if arr.size else 0.0)


def pulse_energy(samples: Sequence[complex], *, dt: float) -> float:
    values = np.asarray(samples, dtype=np.complex128)
    return float(np.sum(np.abs(values) ** 2) * float(dt))


def slew_penalty(
    samples: Sequence[complex],
    *,
    dt: float,
    max_slew: float | None = None,
) -> float:
    values = np.asarray(samples, dtype=np.complex128)
    if values.size <= 1:
        return 0.0
    slew = np.abs(np.diff(values)) / float(dt)
    if max_slew is None:
        return float(np.sum(slew * slew) * float(dt))
    excess = np.maximum(0.0, slew - float(max_slew))
    return float(np.sum(excess * excess) * float(dt))


def compute_readout_metrics(
    *,
    confusion: Sequence[Sequence[float]],
    transition_matrix: Sequence[Sequence[float]],
    pulse_samples: Sequence[complex],
    dt: float,
    residual_resonator: float | Mapping[object, float] | Sequence[float] = 0.0,
    residual_filter: float | Mapping[object, float] | Sequence[float] = 0.0,
    two_shot_confusion: Sequence[Sequence[float]] | None = None,
    leakage_probability: float | None = None,
    max_slew: float | None = None,
    extra: Mapping[str, float] | None = None,
) -> ReadoutMetricSet:
    transition = np.asarray(transition_matrix, dtype=float)
    leak = leakage_probability_from_transition(transition) if leakage_probability is None else float(leakage_probability)
    return ReadoutMetricSet(
        assignment_fidelity=assignment_fidelity(confusion),
        physical_qnd_fidelity=physical_qnd_fidelity(transition),
        measured_two_shot_qnd_fidelity=None if two_shot_confusion is None else measured_two_shot_qnd_fidelity(two_shot_confusion),
        p_0_to_1=transition_probability(transition, 1, 0),
        p_1_to_0=transition_probability(transition, 0, 1),
        leakage_probability=leak,
        residual_resonator_photons=residual_photons(residual_resonator),
        residual_filter_photons=residual_photons(residual_filter),
        pulse_energy=pulse_energy(pulse_samples, dt=float(dt)),
        slew_penalty=slew_penalty(pulse_samples, dt=float(dt), max_slew=max_slew),
        extra={str(k): float(v) for k, v in dict(extra or {}).items()},
    )


__all__ = [
    "ReadoutMetricSet",
    "assignment_fidelity",
    "compute_readout_metrics",
    "leakage_probability_from_transition",
    "measured_two_shot_qnd_fidelity",
    "physical_qnd_fidelity",
    "pulse_energy",
    "residual_photons",
    "slew_penalty",
    "transition_probability",
]
