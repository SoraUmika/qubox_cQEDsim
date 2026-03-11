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
    frame = frame or FrameSpec()
    base = float(model.omega_q - frame.omega_q_frame)
    out = base + float(int(n)) * float(model.chi)
    for i, coeff in enumerate(model.chi_higher, start=2):
        out += falling_factorial_scalar(int(n), i) * float(coeff)
    return float(out)


def carrier_for_transition_frequency(transition_frequency: float) -> float:
    """Map a rotating-frame transition frequency onto cqed_sim's Pulse.carrier."""
    return -float(transition_frequency)


def transition_frequency_from_carrier(carrier: float) -> float:
    """Return the rotating-frame transition frequency resonant with a Pulse.carrier."""
    return -float(carrier)


__all__ = [
    "falling_factorial_scalar",
    "manifold_transition_frequency",
    "carrier_for_transition_frequency",
    "transition_frequency_from_carrier",
]
