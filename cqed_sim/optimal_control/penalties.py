from __future__ import annotations

from dataclasses import dataclass

from cqed_sim.unitary_synthesis.subspace import Subspace


@dataclass(frozen=True)
class AmplitudePenalty:
    weight: float = 0.0
    reference: float = 0.0
    apply_to: str = "command"

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("AmplitudePenalty.weight must be non-negative.")
        if str(self.apply_to).lower() not in {"parameter", "parameters", "schedule", "command", "physical"}:
            raise ValueError("AmplitudePenalty.apply_to must be 'parameter', 'command', or 'physical'.")


@dataclass(frozen=True)
class SlewRatePenalty:
    weight: float = 0.0
    apply_to: str = "command"

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("SlewRatePenalty.weight must be non-negative.")
        if str(self.apply_to).lower() not in {"parameter", "parameters", "schedule", "command", "physical"}:
            raise ValueError("SlewRatePenalty.apply_to must be 'parameter', 'command', or 'physical'.")


@dataclass(frozen=True)
class BoundPenalty:
    weight: float = 0.0
    lower_bound: float = -float("inf")
    upper_bound: float = float("inf")
    apply_to: str = "command"
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("BoundPenalty.weight must be non-negative.")
        if float(self.lower_bound) > float(self.upper_bound):
            raise ValueError("BoundPenalty requires lower_bound <= upper_bound.")
        if str(self.apply_to).lower() not in {"parameter", "parameters", "schedule", "command", "physical"}:
            raise ValueError("BoundPenalty.apply_to must be 'parameter', 'command', or 'physical'.")


@dataclass(frozen=True)
class BoundaryConditionPenalty:
    weight: float = 0.0
    ramp_slices: int = 1
    apply_start: bool = True
    apply_end: bool = True
    apply_to: str = "command"
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("BoundaryConditionPenalty.weight must be non-negative.")
        if int(self.ramp_slices) < 1:
            raise ValueError("BoundaryConditionPenalty.ramp_slices must be at least 1.")
        if not bool(self.apply_start) and not bool(self.apply_end):
            raise ValueError("BoundaryConditionPenalty must apply to the start, the end, or both.")
        if str(self.apply_to).lower() not in {"parameter", "parameters", "schedule", "command", "physical"}:
            raise ValueError("BoundaryConditionPenalty.apply_to must be 'parameter', 'command', or 'physical'.")


@dataclass(frozen=True)
class IQRadiusPenalty:
    amplitude_max: float
    weight: float = 0.0
    apply_to: str = "command"
    control_names: tuple[str, ...] = ()
    export_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if float(self.amplitude_max) <= 0.0:
            raise ValueError("IQRadiusPenalty.amplitude_max must be positive.")
        if float(self.weight) < 0.0:
            raise ValueError("IQRadiusPenalty.weight must be non-negative.")
        if str(self.apply_to).lower() not in {"parameter", "parameters", "schedule", "command", "physical"}:
            raise ValueError("IQRadiusPenalty.apply_to must be 'parameter', 'command', or 'physical'.")


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


__all__ = [
    "AmplitudePenalty",
    "SlewRatePenalty",
    "BoundPenalty",
    "BoundaryConditionPenalty",
    "IQRadiusPenalty",
    "LeakagePenalty",
]