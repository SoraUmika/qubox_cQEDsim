from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from cqed_sim.core.frame import FrameSpec

if TYPE_CHECKING:
    from cqed_sim.core.model import DispersiveTransmonCavityModel


def falling_factorial_scalar(n: int, order: int) -> float:
    out = 1.0
    for k in range(int(order)):
        out *= float(int(n) - k)
    return out


def manifold_transition_frequency(
    model: "DispersiveTransmonCavityModel",
    n: int,
    frame: FrameSpec | None = None,
) -> float:
    """Return the |g,n> <-> |e,n> transition frequency in the specified frame."""
    return transmon_transition_frequency(
        model,
        cavity_level=int(n),
        lower_level=0,
        upper_level=1,
        frame=frame,
    )


def transmon_transition_frequency(
    model,
    *,
    cavity_level: int = 0,
    storage_level: int = 0,
    readout_level: int = 0,
    mode_levels: dict[str, int] | tuple[int, ...] | list[int] | None = None,
    lower_level: int = 0,
    upper_level: int = 1,
    frame: FrameSpec | None = None,
) -> float:
    if hasattr(model, "transmon_transition_frequency"):
        method = model.transmon_transition_frequency
        parameters = inspect.signature(method).parameters
        kwargs = {"lower_level": int(lower_level), "upper_level": int(upper_level), "frame": frame}
        if "mode_levels" in parameters:
            kwargs["mode_levels"] = (
                mode_levels
                if mode_levels is not None
                else {"storage": int(storage_level), "readout": int(readout_level), "cavity": int(cavity_level)}
            )
        elif "cavity_level" in parameters:
            kwargs["cavity_level"] = int(cavity_level)
        else:
            kwargs["storage_level"] = int(storage_level)
            kwargs["readout_level"] = int(readout_level)
        return float(model.transmon_transition_frequency(**kwargs))
    raise TypeError("model does not expose a transmon_transition_frequency helper.")


def sideband_transition_frequency(
    model,
    *,
    cavity_level: int = 0,
    storage_level: int = 0,
    readout_level: int = 0,
    mode_levels: dict[str, int] | tuple[int, ...] | list[int] | None = None,
    lower_level: int = 0,
    upper_level: int = 1,
    mode: str = "storage",
    sideband: str = "red",
    frame: FrameSpec | None = None,
) -> float:
    if hasattr(model, "sideband_transition_frequency"):
        method = model.sideband_transition_frequency
        parameters = inspect.signature(method).parameters
        kwargs = {
            "lower_level": int(lower_level),
            "upper_level": int(upper_level),
            "sideband": str(sideband),
            "frame": frame,
        }
        if "mode" in parameters:
            kwargs["mode"] = str(mode)
        if "mode_levels" in parameters:
            kwargs["mode_levels"] = (
                mode_levels
                if mode_levels is not None
                else {"storage": int(storage_level), "readout": int(readout_level), "cavity": int(cavity_level)}
            )
        elif "cavity_level" in parameters:
            kwargs["cavity_level"] = int(cavity_level)
        else:
            kwargs["storage_level"] = int(storage_level)
            kwargs["readout_level"] = int(readout_level)
        return float(model.sideband_transition_frequency(**kwargs))
    raise TypeError("model does not expose a sideband_transition_frequency helper.")


def effective_sideband_rabi_frequency(coupling: float, detuning: float) -> float:
    return float((4.0 * float(coupling) ** 2 + float(detuning) ** 2) ** 0.5)


def carrier_for_transition_frequency(transition_frequency: float) -> float:
    """Map a rotating-frame transition frequency onto cqed_sim's Pulse.carrier."""
    return -float(transition_frequency)


def transition_frequency_from_carrier(carrier: float) -> float:
    """Return the rotating-frame transition frequency resonant with a Pulse.carrier."""
    return -float(carrier)


def drive_frequency_for_transition_frequency(transition_frequency: float, frame_frequency: float) -> float:
    """Return the positive physical drive frequency for a rotating-frame transition.

    The input ``transition_frequency`` is the transition frequency in the chosen
    rotating frame. The returned value is the corresponding positive lab-style
    tone frequency.
    """
    return float(frame_frequency) + float(transition_frequency)


def transition_frequency_from_drive_frequency(drive_frequency: float, frame_frequency: float) -> float:
    """Return the rotating-frame transition frequency addressed by a positive drive tone."""
    return float(drive_frequency) - float(frame_frequency)


def internal_carrier_from_drive_frequency(drive_frequency: float, frame_frequency: float) -> float:
    """Convert a positive physical drive frequency into cqed_sim's raw Pulse.carrier."""
    return carrier_for_transition_frequency(
        transition_frequency_from_drive_frequency(drive_frequency, frame_frequency)
    )


def drive_frequency_from_internal_carrier(carrier: float, frame_frequency: float) -> float:
    """Convert cqed_sim's raw Pulse.carrier into a positive physical drive frequency."""
    return drive_frequency_for_transition_frequency(
        transition_frequency_from_carrier(carrier),
        frame_frequency,
    )


__all__ = [
    "falling_factorial_scalar",
    "manifold_transition_frequency",
    "transmon_transition_frequency",
    "sideband_transition_frequency",
    "effective_sideband_rabi_frequency",
    "carrier_for_transition_frequency",
    "transition_frequency_from_carrier",
    "drive_frequency_for_transition_frequency",
    "transition_frequency_from_drive_frequency",
    "internal_carrier_from_drive_frequency",
    "drive_frequency_from_internal_carrier",
]
