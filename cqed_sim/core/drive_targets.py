from __future__ import annotations

from dataclasses import dataclass


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

