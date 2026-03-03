from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSpec:
    """Per-mode rotating frame frequencies in rad/s."""

    omega_c_frame: float = 0.0
    omega_q_frame: float = 0.0

