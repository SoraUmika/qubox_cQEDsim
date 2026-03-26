"""Calibration map abstraction for :class:`~cqed_sim.control.ControlLine`.

A calibration map converts the device-level signal ``u_dev(t)`` to the
Hamiltonian coefficient ``c(t)`` used by the simulator:

    c(t) = calibration_map.apply(u_dev(t))

For a linear calibration this reduces to scalar multiplication.  For a
nonlinear calibration (e.g. measured power law, two-tone mixer saturation)
an arbitrary callable may be used.

Public API
----------
:class:`CalibrationMap`
    Abstract base class.

:class:`LinearCalibrationMap`
    Linear (scalar) calibration: ``c = gain * u_dev``.

:class:`CallableCalibrationMap`
    Wraps an arbitrary Python callable.  Does **not** support GRAPE Mode B
    (gradient-through-hardware) or JSON serialization.

:func:`calibration_map_from_dict`
    Reconstruct a :class:`CalibrationMap` from its serialized dict.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class CalibrationMap(ABC):
    """Abstract base for calibration maps.

    Converts device-level signal ``u_dev(t)`` to Hamiltonian coefficient
    ``c(t)``:

        c(t) = calibration_map.apply(u_dev(t))

    Subclasses must implement :meth:`apply`.  For GRAPE Mode B (gradient-
    through-hardware optimisation), subclasses should also override
    :meth:`as_hardware_map` to return a differentiable
    :class:`~cqed_sim.optimal_control.hardware.HardwareMap`.
    """

    @abstractmethod
    def apply(self, u_dev: np.ndarray) -> np.ndarray:
        """Apply calibration to a device waveform.

        Parameters
        ----------
        u_dev:
            Complex or real array of device-level signal samples (any shape).

        Returns
        -------
        np.ndarray
            Complex array of Hamiltonian coefficients (same shape as *u_dev*).
        """
        raise NotImplementedError

    def as_hardware_map(self, export_channels: tuple[str, ...] = ()) -> Any | None:
        """Return a :class:`~cqed_sim.optimal_control.hardware.HardwareMap`
        equivalent to this calibration, or ``None`` if not supported.

        The default returns ``None`` (calibration not representable as a
        differentiable hardware map).  :class:`LinearCalibrationMap` overrides
        this to return
        :class:`~cqed_sim.optimal_control.hardware.GainHardwareMap` when
        *gain* ≠ 1.
        """
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict.

        Raises
        ------
        TypeError
            If this map cannot be serialized (e.g.
            :class:`CallableCalibrationMap`).
        """
        raise TypeError(
            f"{type(self).__name__} does not support serialization.  "
            "Use LinearCalibrationMap for serializable workflows."
        )


@dataclass(frozen=True)
class LinearCalibrationMap(CalibrationMap):
    """Linear (scalar multiplication) calibration map.

    Applies:

        c(t) = gain * u_dev(t)

    This is the standard calibration for converting AWG output voltage to
    effective Rabi coupling strength in rad/s.

    Parameters
    ----------
    gain:
        Scalar multiplier.  May be negative (phase flip) or zero.
        Defaults to ``1.0`` (no scaling).
    """

    gain: float = 1.0

    def apply(self, u_dev: np.ndarray) -> np.ndarray:
        return np.asarray(u_dev, dtype=np.complex128) * float(self.gain)

    def as_hardware_map(self, export_channels: tuple[str, ...] = ()) -> Any | None:
        from cqed_sim.optimal_control.hardware import GainHardwareMap

        if abs(float(self.gain) - 1.0) > 1e-15:
            return GainHardwareMap(gain=float(self.gain), export_channels=export_channels)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "LinearCalibrationMap", "gain": float(self.gain)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearCalibrationMap":
        """Reconstruct from a serialized dict."""
        return cls(gain=float(data["gain"]))


class CallableCalibrationMap(CalibrationMap):
    """Calibration map backed by an arbitrary callable.

    Useful for nonlinear calibrations, power laws, or lookup-table interpolation.

    .. warning::
        ``CallableCalibrationMap`` does **not** support GRAPE Mode B
        (gradient-through-hardware) because no closed-form pullback is
        available.  It also cannot be serialized to JSON.  Use
        :class:`LinearCalibrationMap` when reproducibility and exact gradients
        are required.

    Parameters
    ----------
    fn:
        Python callable that accepts and returns a complex :class:`numpy.ndarray`.
        Signature: ``fn(u_dev: np.ndarray) -> np.ndarray``.
    label:
        Human-readable label for display and logging.  Defaults to
        ``"callable"``.
    """

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        label: str = "callable",
    ) -> None:
        self._fn = fn
        self._label = str(label)

    @property
    def fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """The wrapped callable."""
        return self._fn

    @property
    def label(self) -> str:
        """Human-readable label."""
        return self._label

    def apply(self, u_dev: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(u_dev), dtype=np.complex128)

    def to_dict(self) -> dict[str, Any]:
        raise TypeError(
            "CallableCalibrationMap cannot be serialized to JSON.  "
            "Replace with LinearCalibrationMap for serializable workflows."
        )

    def __repr__(self) -> str:
        return f"CallableCalibrationMap(label={self._label!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CallableCalibrationMap) and self._fn is other._fn

    def __hash__(self) -> int:
        return id(self._fn)


def calibration_map_from_dict(data: dict[str, Any]) -> CalibrationMap:
    """Reconstruct a :class:`CalibrationMap` from its serialized dict.

    Parameters
    ----------
    data:
        Output of :meth:`CalibrationMap.to_dict`.

    Returns
    -------
    CalibrationMap

    Raises
    ------
    ValueError
        If the ``"type"`` field is unknown.
    """
    type_name = str(data.get("type", ""))
    if type_name == "LinearCalibrationMap":
        return LinearCalibrationMap.from_dict(data)
    raise ValueError(
        f"Unknown CalibrationMap type '{type_name}'.  "
        "Only 'LinearCalibrationMap' is supported for deserialization."
    )


__all__ = [
    "CalibrationMap",
    "LinearCalibrationMap",
    "CallableCalibrationMap",
    "calibration_map_from_dict",
]
