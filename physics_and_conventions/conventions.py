from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Callable

import numpy as np


class UnitType(str, Enum):
    HZ = "Hz"
    RAD_PER_S = "rad/s"


class DetuningSign(str, Enum):
    SYSTEM_MINUS_DRIVE = "system_minus_drive"
    DRIVE_MINUS_SYSTEM = "drive_minus_system"


class TensorOrdering(str, Enum):
    QUBIT_STORAGE = "qubit⊗storage"
    QUBIT_STORAGE_READOUT = "qubit⊗storage⊗readout"


internal_units = UnitType.RAD_PER_S


def to_internal_units(frequency_hz: float) -> float:
    return float(2.0 * np.pi * frequency_hz)


def from_internal_units(frequency_rad_s: float) -> float:
    return float(frequency_rad_s / (2.0 * np.pi))


def validate_detuning(delta: float, sign: DetuningSign = DetuningSign.SYSTEM_MINUS_DRIVE) -> float:
    if sign not in (DetuningSign.SYSTEM_MINUS_DRIVE, DetuningSign.DRIVE_MINUS_SYSTEM):
        raise ValueError(f"Unsupported detuning sign '{sign}'.")
    return float(delta if sign == DetuningSign.SYSTEM_MINUS_DRIVE else -delta)


def enforce_conventions(
    *,
    units: UnitType = UnitType.RAD_PER_S,
    detuning_sign: DetuningSign = DetuningSign.SYSTEM_MINUS_DRIVE,
) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args, **kwargs):
            if units != UnitType.RAD_PER_S:
                raise ValueError("cqed_sim internal APIs expect angular frequencies in rad/s.")
            kwargs.setdefault("_validated_detuning_sign", detuning_sign)
            return function(*args, **kwargs)

        return wrapped

    return decorator


@contextmanager
def convention_scope(
    *,
    units: UnitType = UnitType.RAD_PER_S,
    detuning_sign: DetuningSign = DetuningSign.SYSTEM_MINUS_DRIVE,
):
    if units != UnitType.RAD_PER_S:
        raise ValueError("The runtime convention scope only supports rad/s internal units.")
    yield {"units": units, "detuning_sign": detuning_sign}


__all__ = [
    "UnitType",
    "DetuningSign",
    "TensorOrdering",
    "internal_units",
    "to_internal_units",
    "from_internal_units",
    "validate_detuning",
    "enforce_conventions",
    "convention_scope",
]
