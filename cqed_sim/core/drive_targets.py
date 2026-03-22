from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import qutip as qt


@dataclass(frozen=True)
class TransmonTransitionDriveSpec:
    """Drive a selected transmon transition |lower> <-> |upper>."""

    lower_level: int
    upper_level: int


@dataclass(frozen=True)
class SidebandDriveSpec:
    """Drive an effective red/blue sideband on a selected bosonic mode."""

    mode: str = "storage"
    lower_level: int = 0
    upper_level: int = 1
    sideband: str = "red"


DriveTarget = str | TransmonTransitionDriveSpec | SidebandDriveSpec


def resolve_drive_target_operators(
    model: Any,
    target: DriveTarget,
    couplings: Mapping[str, tuple[qt.Qobj, qt.Qobj]] | None = None,
) -> tuple[qt.Qobj, qt.Qobj]:
    """Resolve a user-facing drive target into raising/lowering operators."""
    if isinstance(target, str):
        resolved_couplings = dict(model.drive_coupling_operators()) if couplings is None else dict(couplings)
        if target not in resolved_couplings:
            raise ValueError(f"Unsupported target '{target}'.")
        return resolved_couplings[target]
    if isinstance(target, TransmonTransitionDriveSpec):
        if not hasattr(model, "transmon_transition_operators"):
            raise ValueError("Model does not support structured transmon transition targets.")
        return model.transmon_transition_operators(target.lower_level, target.upper_level)
    if isinstance(target, SidebandDriveSpec):
        if not hasattr(model, "sideband_drive_operators"):
            raise ValueError("Model does not support structured sideband targets.")
        return model.sideband_drive_operators(
            mode=target.mode,
            lower_level=target.lower_level,
            upper_level=target.upper_level,
            sideband=target.sideband,
        )
    raise TypeError(f"Unsupported drive target type '{type(target).__name__}'.")


__all__ = [
    "DriveTarget",
    "SidebandDriveSpec",
    "TransmonTransitionDriveSpec",
    "resolve_drive_target_operators",
]

