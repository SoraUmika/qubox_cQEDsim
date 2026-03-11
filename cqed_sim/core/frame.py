from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSpec:
    """Per-mode rotating-frame frequencies in rad/s."""

    omega_c_frame: float = 0.0
    omega_q_frame: float = 0.0
    omega_r_frame: float = 0.0

    @property
    def omega_s_frame(self) -> float:
        """Alias the legacy cavity-frame name to storage for three-mode code."""
        return self.omega_c_frame
