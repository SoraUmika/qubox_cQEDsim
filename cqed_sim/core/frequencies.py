from __future__ import annotations

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
    lower_level: int = 0,
    upper_level: int = 1,
    frame: FrameSpec | None = None,
) -> float:
    if hasattr(model, "transmon_transition_frequency"):
        kwargs = {"lower_level": int(lower_level), "upper_level": int(upper_level), "frame": frame}
        dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims", ()))
        if len(dims) == 2:
            kwargs["cavity_level"] = int(cavity_level)
        elif len(dims) == 3:
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
    lower_level: int = 0,
    upper_level: int = 1,
    mode: str = "storage",
    sideband: str = "red",
    frame: FrameSpec | None = None,
) -> float:
    if hasattr(model, "sideband_transition_frequency"):
        kwargs = {
            "lower_level": int(lower_level),
            "upper_level": int(upper_level),
            "sideband": str(sideband),
            "frame": frame,
        }
        dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims", ()))
        if len(dims) == 2:
            kwargs["cavity_level"] = int(cavity_level)
        elif len(dims) == 3:
            kwargs["mode"] = str(mode)
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


__all__ = [
    "falling_factorial_scalar",
    "manifold_transition_frequency",
    "transmon_transition_frequency",
    "sideband_transition_frequency",
    "effective_sideband_rabi_frequency",
    "carrier_for_transition_frequency",
    "transition_frequency_from_carrier",
]
