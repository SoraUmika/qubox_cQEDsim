from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .observables import PhysicalObservable, as_observable


@dataclass(frozen=True)
class ObservableInsertion:
    step: int
    observable: PhysicalObservable
    label: str | None = None

    def __post_init__(self) -> None:
        if int(self.step) <= 0:
            raise ValueError("ObservableInsertion.step must be positive.")
        object.__setattr__(self, "step", int(self.step))

    def to_record(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "label": self.label if self.label is not None else self.observable.label,
            "observable": self.observable.to_record(),
        }


def _coerce_insertion(value: ObservableInsertion | Mapping[str, Any]) -> ObservableInsertion:
    if isinstance(value, ObservableInsertion):
        return value
    if "step" not in value:
        raise ValueError("Observable schedule records must contain 'step'.")
    if "operator" in value:
        operator = value["operator"]
    elif "observable" in value:
        operator = value["observable"]
    else:
        raise ValueError("Observable schedule records must contain 'operator' or 'observable'.")
    return ObservableInsertion(
        step=int(value["step"]),
        observable=as_observable(operator, label=value.get("label")),
        label=value.get("label"),
    )


@dataclass(frozen=True)
class ObservableSchedule:
    insertions: Sequence[ObservableInsertion | Mapping[str, Any]]
    total_steps: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        normalized = tuple(sorted((_coerce_insertion(item) for item in self.insertions), key=lambda item: item.step))
        steps = [item.step for item in normalized]
        if len(set(steps)) != len(steps):
            raise ValueError("ObservableSchedule does not allow multiple insertions at the same step.")
        inferred_total = max(steps, default=0)
        total = inferred_total if self.total_steps is None else int(self.total_steps)
        if total < inferred_total:
            raise ValueError("ObservableSchedule.total_steps must be at least the largest measured step.")
        object.__setattr__(self, "insertions", normalized)
        object.__setattr__(self, "total_steps", total)

    @property
    def measured_steps(self) -> tuple[int, ...]:
        return tuple(item.step for item in self.insertions)

    def insertion_for_step(self, step: int) -> ObservableInsertion | None:
        for item in self.insertions:
            if item.step == int(step):
                return item
        return None

    def to_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total_steps": int(self.total_steps or 0),
            "insertions": [item.to_record() for item in self.insertions],
        }
