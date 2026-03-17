from __future__ import annotations

from dataclasses import dataclass

from cqed_sim.unitary_synthesis.subspace import Subspace


@dataclass(frozen=True)
class AmplitudePenalty:
    weight: float = 0.0
    reference: float = 0.0

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("AmplitudePenalty.weight must be non-negative.")


@dataclass(frozen=True)
class SlewRatePenalty:
    weight: float = 0.0

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("SlewRatePenalty.weight must be non-negative.")


@dataclass(frozen=True)
class LeakagePenalty:
    subspace: Subspace
    weight: float = 0.0
    metric: str = "average"

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("LeakagePenalty.weight must be non-negative.")
        if self.metric not in {"average", "worst"}:
            raise ValueError("LeakagePenalty.metric must be 'average' or 'worst'.")


__all__ = ["AmplitudePenalty", "SlewRatePenalty", "LeakagePenalty"]