from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .results import CorrelatorEstimate, ExactCorrelatorResult
from .sampler import HolographicSampler
from .schedules import ObservableInsertion, ObservableSchedule


@dataclass(frozen=True)
class TimeSlice:
    """Future-facing time-sliced schedule block for holographic dynamics programs."""

    steps: int
    insertions: Sequence[ObservableInsertion | Mapping[str, Any]] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        if int(self.steps) <= 0:
            raise ValueError("TimeSlice.steps must be positive.")
        object.__setattr__(self, "steps", int(self.steps))
        object.__setattr__(self, "insertions", tuple(self.insertions))

    def to_schedule(self, *, offset: int = 0) -> ObservableSchedule:
        return ObservableSchedule(self.shifted_insertions(offset=offset), total_steps=int(offset) + int(self.steps), label=self.label)

    def shifted_insertions(self, *, offset: int = 0) -> list[dict[str, Any]]:
        shifted: list[dict[str, Any]] = []
        for insertion in self.insertions:
            if isinstance(insertion, ObservableInsertion):
                shifted.append(
                    {
                        "step": int(offset) + int(insertion.step),
                        "operator": insertion.observable.matrix,
                        "label": insertion.label,
                    }
                )
            else:
                shifted.append(
                    {
                        "step": int(offset) + int(insertion["step"]),
                        "operator": insertion["operator"],
                        "label": insertion.get("label"),
                    }
                )
        return shifted

    def to_record(self) -> dict[str, Any]:
        return self.to_schedule(offset=0).to_record()


@dataclass(frozen=True)
class HoloQUADSProgram:
    """Lightweight schedule composition scaffold for future holoQUADS work."""

    slices: Sequence[TimeSlice]
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("HoloQUADSProgram requires at least one TimeSlice.")
        object.__setattr__(self, "slices", tuple(self.slices))

    @property
    def total_steps(self) -> int:
        return int(sum(item.steps for item in self.slices))

    def combined_schedule(self) -> ObservableSchedule:
        offset = 0
        merged: list[dict[str, Any]] = []
        for item in self.slices:
            merged.extend(item.shifted_insertions(offset=offset))
            offset += item.steps
        return ObservableSchedule(merged, total_steps=self.total_steps, label=self.label)

    def evaluate(
        self,
        sampler: HolographicSampler,
        *,
        shots: int = 1_000,
        exact: bool = False,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> CorrelatorEstimate | ExactCorrelatorResult:
        schedule = self.combined_schedule()
        if exact:
            return sampler.enumerate_correlator(schedule)
        return sampler.sample_correlator(schedule, shots=shots, seed=seed, show_progress=show_progress)
