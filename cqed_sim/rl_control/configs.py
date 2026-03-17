from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cqed_sim.core import FrameSpec
from cqed_sim.measurement import QubitMeasurementSpec
from cqed_sim.pulses import HardwareConfig
from cqed_sim.sim import NoiseSpec


ModelRegime = Literal["reduced_dispersive", "full_pulse"]
TaskKind = Literal["state_preparation", "unitary_synthesis"]
RandomizationMode = Literal["train", "eval"]


@dataclass(frozen=True)
class ReducedDispersiveModelConfig:
    omega_c: float
    omega_q: float
    alpha: float
    chi: float
    kerr: float = 0.0
    chi_higher: tuple[float, ...] = ()
    kerr_higher: tuple[float, ...] = ()
    n_cav: int = 10
    n_tr: int = 3


@dataclass(frozen=True)
class FullPulseModelConfig:
    omega_c: float
    omega_q: float
    alpha: float
    exchange_g: float
    kerr: float = 0.0
    cross_kerr: float = 0.0
    n_cav: int = 10
    n_tr: int = 4


@dataclass
class HybridSystemConfig:
    regime: ModelRegime = "reduced_dispersive"
    reduced_model: ReducedDispersiveModelConfig | None = None
    full_model: FullPulseModelConfig | None = None
    frame: FrameSpec = FrameSpec()
    use_model_rotating_frame: bool = True
    noise: NoiseSpec | None = None
    hardware: dict[str, HardwareConfig] = field(default_factory=dict)
    crosstalk_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    dt: float = 4.0e-9
    max_step: float | None = None


@dataclass
class HybridEnvConfig:
    system: HybridSystemConfig
    task: Any
    action_space: Any
    observation_model: Any
    reward_model: Any
    randomizer: Any | None = None
    randomization_mode: RandomizationMode = "train"
    episode_horizon: int = 4
    measurement_spec: QubitMeasurementSpec | None = None
    collapse_on_measurement: bool = False
    auto_measurement: bool = False
    seed: int | None = None
    store_states_for_diagnostics: bool = True
    diagnostics_wigner_points: int = 41


__all__ = [
    "ModelRegime",
    "TaskKind",
    "RandomizationMode",
    "ReducedDispersiveModelConfig",
    "FullPulseModelConfig",
    "HybridSystemConfig",
    "HybridEnvConfig",
]