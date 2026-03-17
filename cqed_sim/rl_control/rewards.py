from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import qutip as qt

from .metrics import parity_expectation, sparse_wigner_samples, state_fidelity


def _metric(metrics: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if metrics is None:
        return float(default)
    value = metrics.get(key, default)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return float(default)
    return float(value)


@dataclass(frozen=True)
class StateFidelityReward:
    weight: float = 1.0

    def evaluate(self, *, state: qt.Qobj | None, task: Any, model: Any, metrics: dict[str, Any] | None = None, **_: Any) -> tuple[float, dict[str, float]]:
        if state is None:
            value = _metric(metrics, "state_fidelity_mean")
        elif task.target_state_factory is not None:
            value = state_fidelity(state, task.build_target_state(model))
        else:
            value = _metric(metrics, "state_fidelity", 0.0)
        return float(self.weight * value), {"state_fidelity": float(value)}


@dataclass(frozen=True)
class ProcessFidelityReward:
    weight: float = 1.0

    def evaluate(self, *, metrics: dict[str, Any] | None = None, **_: Any) -> tuple[float, dict[str, float]]:
        value = _metric(metrics, "process_fidelity", _metric(metrics, "state_fidelity_mean", 0.0))
        return float(self.weight * value), {"process_fidelity": float(value)}


@dataclass(frozen=True)
class ParityRewardTerm:
    weight: float = 1.0
    displacements: tuple[complex, ...] = (0.0j,)

    def evaluate(self, *, state: qt.Qobj | None, task: Any, model: Any, **_: Any) -> tuple[float, dict[str, float]]:
        if state is None or task.target_state_factory is None:
            return 0.0, {"parity_match": 0.0}
        target_state = task.build_target_state(model)
        actual = np.asarray([parity_expectation(state, displacement=point) for point in self.displacements], dtype=float)
        target = np.asarray([parity_expectation(target_state, displacement=point) for point in self.displacements], dtype=float)
        mismatch = float(np.mean(np.abs(actual - target)))
        reward = float(self.weight * max(0.0, 1.0 - mismatch))
        return reward, {"parity_match": float(max(0.0, 1.0 - mismatch))}


@dataclass(frozen=True)
class WignerSampleRewardTerm:
    weight: float = 1.0
    points: tuple[complex, ...] = (0.0j, 0.5 + 0.0j, -0.5 + 0.0j)

    def evaluate(self, *, state: qt.Qobj | None, task: Any, model: Any, **_: Any) -> tuple[float, dict[str, float]]:
        if state is None or task.target_state_factory is None:
            return 0.0, {"wigner_match": 0.0}
        target_state = task.build_target_state(model)
        actual = sparse_wigner_samples(state, self.points)
        target = sparse_wigner_samples(target_state, self.points)
        mismatch = float(np.mean(np.abs(actual - target)))
        reward = float(self.weight * max(0.0, 1.0 - mismatch))
        return reward, {"wigner_match": float(max(0.0, 1.0 - mismatch))}


@dataclass(frozen=True)
class LeakagePenaltyTerm:
    weight: float = 0.25

    def evaluate(self, *, metrics: dict[str, Any] | None = None, **_: Any) -> tuple[float, dict[str, float]]:
        leakage = _metric(metrics, "leakage_average", 0.0)
        penalty = float(-self.weight * leakage)
        return penalty, {"leakage_penalty": float(penalty)}


@dataclass(frozen=True)
class AncillaReturnPenaltyTerm:
    weight: float = 0.25

    def evaluate(self, *, metrics: dict[str, Any] | None = None, **_: Any) -> tuple[float, dict[str, float]]:
        ancilla_return = _metric(metrics, "ancilla_return", _metric(metrics, "ancilla_return_mean", 1.0))
        penalty = float(-self.weight * max(0.0, 1.0 - ancilla_return))
        return penalty, {"ancilla_return_penalty": float(penalty)}


@dataclass(frozen=True)
class MeasurementAssignmentRewardTerm:
    weight: float = 0.4
    use_observed_probabilities: bool = True

    def evaluate(self, *, measurement: Any = None, task: Any = None, metrics: dict[str, Any] | None = None, **_: Any) -> tuple[float, dict[str, float]]:
        target_level = int(getattr(task, "target_ancilla_level", 0)) if task is not None else 0
        target_label = "g" if target_level == 0 else "e"
        if measurement is None:
            ancilla_return = _metric(metrics, "ancilla_return", _metric(metrics, "ancilla_return_mean", 1.0 if target_level == 0 else 0.0))
            value = ancilla_return if target_level == 0 else max(0.0, 1.0 - ancilla_return)
        else:
            probabilities = measurement.observed_probabilities if self.use_observed_probabilities else measurement.probabilities
            value = float(probabilities.get(target_label, 0.0))
        return float(self.weight * value), {"measurement_assignment": float(value)}


@dataclass(frozen=True)
class ControlCostPenaltyTerm:
    weight: float = 0.05
    duration_scale: float = 100.0e-9
    amplitude_scale: float = 2.0 * np.pi * 5.0e6
    measurement_cost: float = 0.1

    def evaluate(self, *, segment: Any = None, **_: Any) -> tuple[float, dict[str, float]]:
        if segment is None:
            return 0.0, {"control_cost": 0.0}
        duration = float(segment.metadata.get("duration", getattr(segment, "duration", 0.0)))
        max_abs_amp = float(segment.metadata.get("max_abs_amp", 0.0))
        measurement_requested = bool(getattr(segment, "measurement_requested", False))
        cost = duration / max(self.duration_scale, 1.0e-18)
        cost += max_abs_amp / max(self.amplitude_scale, 1.0e-18)
        if measurement_requested:
            cost += float(self.measurement_cost)
        penalty = float(-self.weight * cost)
        return penalty, {"control_cost": float(penalty)}


class CompositeReward:
    def __init__(self, terms: Iterable[Any]):
        self.terms = tuple(terms)

    def compute(self, **kwargs: Any) -> tuple[float, dict[str, float]]:
        total = 0.0
        breakdown: dict[str, float] = {}
        for index, term in enumerate(self.terms):
            value, details = term.evaluate(**kwargs)
            total += float(value)
            if details:
                for key, item in details.items():
                    breakdown[key] = float(item)
            else:
                breakdown[f"term_{index}"] = float(value)
        breakdown["total"] = float(total)
        return float(total), breakdown


def build_reward_model(mode: str, *, include_control_cost: bool = True) -> CompositeReward:
    mode_key = str(mode).strip().lower()
    if mode_key in {"state", "state_preparation"}:
        terms: list[Any] = [StateFidelityReward(weight=1.0), LeakagePenaltyTerm(weight=0.2), AncillaReturnPenaltyTerm(weight=0.2)]
    elif mode_key in {"gate", "unitary", "unitary_synthesis"}:
        terms = [ProcessFidelityReward(weight=1.0), LeakagePenaltyTerm(weight=0.25), AncillaReturnPenaltyTerm(weight=0.15)]
    elif mode_key == "cat":
        terms = [StateFidelityReward(weight=0.6), ParityRewardTerm(weight=0.25), WignerSampleRewardTerm(weight=0.2), LeakagePenaltyTerm(weight=0.2)]
    elif mode_key in {"measurement", "measurement_proxy", "proxy"}:
        terms = [
            MeasurementAssignmentRewardTerm(weight=0.55),
            ParityRewardTerm(weight=0.2),
            WignerSampleRewardTerm(weight=0.15),
            LeakagePenaltyTerm(weight=0.15),
            AncillaReturnPenaltyTerm(weight=0.1),
        ]
    else:
        raise ValueError(f"Unsupported reward mode '{mode}'.")
    if include_control_cost:
        terms.append(ControlCostPenaltyTerm())
    return CompositeReward(terms)


__all__ = [
    "StateFidelityReward",
    "ProcessFidelityReward",
    "ParityRewardTerm",
    "WignerSampleRewardTerm",
    "LeakagePenaltyTerm",
    "AncillaReturnPenaltyTerm",
    "MeasurementAssignmentRewardTerm",
    "ControlCostPenaltyTerm",
    "CompositeReward",
    "build_reward_model",
]