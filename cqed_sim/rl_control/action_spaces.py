from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


DEFAULT_DETUNING_BOUNDS = (-2.0 * np.pi * 10.0e6, 2.0 * np.pi * 10.0e6)
DEFAULT_DURATION_BOUNDS = (16.0e-9, 400.0e-9)


@dataclass(frozen=True)
class QubitGaussianAction:
    theta: float
    phi: float
    detuning: float
    duration: float
    drag: float = 0.0


@dataclass(frozen=True)
class CavityDisplacementAction:
    alpha: complex
    duration: float
    detuning: float = 0.0


@dataclass(frozen=True)
class SidebandAction:
    amplitude: float
    detuning: float
    duration: float
    lower_level: int = 0
    upper_level: int = 1
    mode: str = "storage"
    sideband: str = "red"
    phase: float = 0.0


@dataclass(frozen=True)
class WaitAction:
    duration: float


@dataclass(frozen=True)
class MeasurementAction:
    collapse: bool = False


@dataclass(frozen=True)
class ResetAction:
    ideal: bool = True


@dataclass(frozen=True)
class HybridBlockAction:
    qubit_theta: float = 0.0
    qubit_phi: float = 0.0
    qubit_detuning: float = 0.0
    qubit_duration: float = 0.0
    qubit_drag: float = 0.0
    cavity_alpha: complex = 0.0j
    cavity_detuning: float = 0.0
    cavity_duration: float = 0.0
    sideband_amplitude: float = 0.0
    sideband_detuning: float = 0.0
    sideband_duration: float = 0.0
    sideband_phase: float = 0.0
    wait_duration: float = 0.0
    measurement_requested: bool = False


@dataclass(frozen=True)
class WaveformAction:
    channel_samples: dict[str, np.ndarray]
    dt: float
    duration: float


@dataclass(frozen=True)
class PrimitiveAction:
    primitive: str
    duration: float = 0.0
    amplitude: float = 0.0
    phase: float = 0.0
    detuning: float = 0.0
    drag: float = 0.0
    alpha: complex = 0.0j
    lower_level: int = 0
    upper_level: int = 1
    mode: str = "storage"
    sideband: str = "red"
    collapse: bool = False


def _vector(action: Any, expected_size: int) -> np.ndarray:
    array = np.asarray(action, dtype=float).reshape(-1)
    if array.size != int(expected_size):
        raise ValueError(f"Expected action vector of size {expected_size}, got {array.size}.")
    return array


class ParametricPulseActionSpace:
    def __init__(
        self,
        *,
        family: str = "hybrid_block",
        theta_bounds: tuple[float, float] = (-2.0 * np.pi, 2.0 * np.pi),
        alpha_bounds: tuple[float, float] = (-2.0, 2.0),
        amplitude_bounds: tuple[float, float] = (0.0, 2.0 * np.pi * 6.0e6),
        detuning_bounds: tuple[float, float] = DEFAULT_DETUNING_BOUNDS,
        duration_bounds: tuple[float, float] = DEFAULT_DURATION_BOUNDS,
        sideband_levels: tuple[int, int] = (0, 1),
    ):
        self.family = str(family)
        self.theta_bounds = tuple(float(value) for value in theta_bounds)
        self.alpha_bounds = tuple(float(value) for value in alpha_bounds)
        self.amplitude_bounds = tuple(float(value) for value in amplitude_bounds)
        self.detuning_bounds = tuple(float(value) for value in detuning_bounds)
        self.duration_bounds = tuple(float(value) for value in duration_bounds)
        self.sideband_levels = (int(sideband_levels[0]), int(sideband_levels[1]))

        if self.family == "qubit_gaussian":
            self.names = ("theta", "phi", "detuning", "duration", "drag")
            self.low = np.asarray([
                self.theta_bounds[0],
                -np.pi,
                self.detuning_bounds[0],
                self.duration_bounds[0],
                -2.0,
            ], dtype=float)
            self.high = np.asarray([
                self.theta_bounds[1],
                np.pi,
                self.detuning_bounds[1],
                self.duration_bounds[1],
                2.0,
            ], dtype=float)
        elif self.family == "cavity_displacement":
            self.names = ("alpha_re", "alpha_im", "detuning", "duration")
            self.low = np.asarray([
                self.alpha_bounds[0],
                self.alpha_bounds[0],
                self.detuning_bounds[0],
                self.duration_bounds[0],
            ], dtype=float)
            self.high = np.asarray([
                self.alpha_bounds[1],
                self.alpha_bounds[1],
                self.detuning_bounds[1],
                self.duration_bounds[1],
            ], dtype=float)
        elif self.family == "sideband":
            self.names = ("amplitude", "detuning", "duration", "phase")
            self.low = np.asarray([
                self.amplitude_bounds[0],
                self.detuning_bounds[0],
                self.duration_bounds[0],
                -np.pi,
            ], dtype=float)
            self.high = np.asarray([
                self.amplitude_bounds[1],
                self.detuning_bounds[1],
                self.duration_bounds[1],
                np.pi,
            ], dtype=float)
        elif self.family == "hybrid_block":
            self.names = (
                "qubit_theta",
                "qubit_phi",
                "qubit_detuning",
                "qubit_duration",
                "qubit_drag",
                "cavity_re",
                "cavity_im",
                "cavity_detuning",
                "cavity_duration",
                "sideband_amplitude",
                "sideband_detuning",
                "sideband_duration",
                "sideband_phase",
                "wait_duration",
                "measurement_requested",
            )
            self.low = np.asarray([
                self.theta_bounds[0],
                -np.pi,
                self.detuning_bounds[0],
                0.0,
                -2.0,
                self.alpha_bounds[0],
                self.alpha_bounds[0],
                self.detuning_bounds[0],
                0.0,
                self.amplitude_bounds[0],
                self.detuning_bounds[0],
                0.0,
                -np.pi,
                0.0,
                0.0,
            ], dtype=float)
            self.high = np.asarray([
                self.theta_bounds[1],
                np.pi,
                self.detuning_bounds[1],
                self.duration_bounds[1],
                2.0,
                self.alpha_bounds[1],
                self.alpha_bounds[1],
                self.detuning_bounds[1],
                self.duration_bounds[1],
                self.amplitude_bounds[1],
                self.detuning_bounds[1],
                self.duration_bounds[1],
                np.pi,
                self.duration_bounds[1],
                1.0,
            ], dtype=float)
        else:
            raise ValueError(f"Unsupported parametric family '{self.family}'.")

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.names),)

    def clip(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.clip(_vector(action, len(self.names)), self.low, self.high)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.low, self.high)

    def parse(self, action: Any) -> Any:
        if isinstance(action, (QubitGaussianAction, CavityDisplacementAction, SidebandAction, HybridBlockAction)):
            return action
        clipped = self.clip(action)
        if self.family == "qubit_gaussian":
            return QubitGaussianAction(
                theta=float(clipped[0]),
                phi=float(clipped[1]),
                detuning=float(clipped[2]),
                duration=float(clipped[3]),
                drag=float(clipped[4]),
            )
        if self.family == "cavity_displacement":
            return CavityDisplacementAction(
                alpha=complex(float(clipped[0]), float(clipped[1])),
                detuning=float(clipped[2]),
                duration=float(clipped[3]),
            )
        if self.family == "sideband":
            return SidebandAction(
                amplitude=float(clipped[0]),
                detuning=float(clipped[1]),
                duration=float(clipped[2]),
                phase=float(clipped[3]),
                lower_level=self.sideband_levels[0],
                upper_level=self.sideband_levels[1],
            )
        return HybridBlockAction(
            qubit_theta=float(clipped[0]),
            qubit_phi=float(clipped[1]),
            qubit_detuning=float(clipped[2]),
            qubit_duration=float(clipped[3]),
            qubit_drag=float(clipped[4]),
            cavity_alpha=complex(float(clipped[5]), float(clipped[6])),
            cavity_detuning=float(clipped[7]),
            cavity_duration=float(clipped[8]),
            sideband_amplitude=float(clipped[9]),
            sideband_detuning=float(clipped[10]),
            sideband_duration=float(clipped[11]),
            sideband_phase=float(clipped[12]),
            wait_duration=float(clipped[13]),
            measurement_requested=bool(clipped[14] >= 0.5),
        )

    def flatten(self, action: Any) -> np.ndarray:
        if isinstance(action, QubitGaussianAction):
            return np.asarray([action.theta, action.phi, action.detuning, action.duration, action.drag], dtype=float)
        if isinstance(action, CavityDisplacementAction):
            return np.asarray([action.alpha.real, action.alpha.imag, action.detuning, action.duration], dtype=float)
        if isinstance(action, SidebandAction):
            return np.asarray([action.amplitude, action.detuning, action.duration, action.phase], dtype=float)
        if isinstance(action, HybridBlockAction):
            return np.asarray([
                action.qubit_theta,
                action.qubit_phi,
                action.qubit_detuning,
                action.qubit_duration,
                action.qubit_drag,
                action.cavity_alpha.real,
                action.cavity_alpha.imag,
                action.cavity_detuning,
                action.cavity_duration,
                action.sideband_amplitude,
                action.sideband_detuning,
                action.sideband_duration,
                action.sideband_phase,
                action.wait_duration,
                1.0 if action.measurement_requested else 0.0,
            ], dtype=float)
        return self.clip(action)


class PrimitiveActionSpace:
    def __init__(
        self,
        *,
        primitives: Sequence[str] = ("qubit_gaussian", "cavity_displacement", "sideband", "wait", "measure", "reset"),
        duration_bounds: tuple[float, float] = DEFAULT_DURATION_BOUNDS,
        theta_bounds: tuple[float, float] = (-2.0 * np.pi, 2.0 * np.pi),
        alpha_bounds: tuple[float, float] = (-2.0, 2.0),
        amplitude_bounds: tuple[float, float] = (0.0, 2.0 * np.pi * 6.0e6),
        detuning_bounds: tuple[float, float] = DEFAULT_DETUNING_BOUNDS,
        sideband_levels: tuple[int, int] = (0, 1),
    ):
        self.primitives = tuple(str(name) for name in primitives)
        self.duration_bounds = tuple(float(value) for value in duration_bounds)
        self.theta_bounds = tuple(float(value) for value in theta_bounds)
        self.alpha_bounds = tuple(float(value) for value in alpha_bounds)
        self.amplitude_bounds = tuple(float(value) for value in amplitude_bounds)
        self.detuning_bounds = tuple(float(value) for value in detuning_bounds)
        self.sideband_levels = (int(sideband_levels[0]), int(sideband_levels[1]))
        self.low = np.asarray([0.0, 0.0, -2.0 * np.pi, -2.0 * np.pi, self.detuning_bounds[0], -2.0, 0.0], dtype=float)
        self.high = np.asarray([
            float(len(self.primitives) - 1),
            self.duration_bounds[1],
            2.0 * np.pi,
            2.0 * np.pi,
            self.detuning_bounds[1],
            2.0,
            1.0,
        ], dtype=float)

    @property
    def shape(self) -> tuple[int, ...]:
        return (7,)

    def clip(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.clip(_vector(action, 7), self.low, self.high)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        action = rng.uniform(self.low, self.high)
        action[0] = float(rng.integers(0, len(self.primitives)))
        return action

    def parse(self, action: Any) -> Any:
        if isinstance(action, (PrimitiveAction, QubitGaussianAction, CavityDisplacementAction, SidebandAction, WaitAction, MeasurementAction, ResetAction)):
            return action
        if isinstance(action, dict):
            primitive = str(action.get("primitive", "wait"))
            return PrimitiveAction(
                primitive=primitive,
                duration=float(action.get("duration", 0.0)),
                amplitude=float(action.get("amplitude", 0.0)),
                phase=float(action.get("phase", 0.0)),
                detuning=float(action.get("detuning", 0.0)),
                drag=float(action.get("drag", 0.0)),
                alpha=complex(action.get("alpha", 0.0j)),
                lower_level=int(action.get("lower_level", self.sideband_levels[0])),
                upper_level=int(action.get("upper_level", self.sideband_levels[1])),
                mode=str(action.get("mode", "storage")),
                sideband=str(action.get("sideband", "red")),
                collapse=bool(action.get("collapse", False)),
            )
        vector = self.clip(action)
        primitive = self.primitives[int(np.round(vector[0]))]
        if primitive == "qubit_gaussian":
            return QubitGaussianAction(
                theta=float(np.clip(vector[2], self.theta_bounds[0], self.theta_bounds[1])),
                phi=float(np.clip(vector[3], -np.pi, np.pi)),
                detuning=float(vector[4]),
                duration=float(vector[1]),
                drag=float(vector[5]),
            )
        if primitive == "cavity_displacement":
            alpha_re = float(np.clip(vector[2], self.alpha_bounds[0], self.alpha_bounds[1]))
            alpha_im = float(np.clip(vector[3], self.alpha_bounds[0], self.alpha_bounds[1]))
            return CavityDisplacementAction(alpha=complex(alpha_re, alpha_im), detuning=float(vector[4]), duration=float(vector[1]))
        if primitive == "sideband":
            return SidebandAction(
                amplitude=float(np.clip(abs(vector[2]), self.amplitude_bounds[0], self.amplitude_bounds[1])),
                phase=float(np.clip(vector[3], -np.pi, np.pi)),
                detuning=float(vector[4]),
                duration=float(vector[1]),
                lower_level=self.sideband_levels[0],
                upper_level=self.sideband_levels[1],
            )
        if primitive == "measure":
            return MeasurementAction(collapse=bool(vector[6] >= 0.5))
        if primitive == "reset":
            return ResetAction(ideal=True)
        return WaitAction(duration=float(vector[1]))

    def flatten(self, action: Any) -> np.ndarray:
        if isinstance(action, PrimitiveAction):
            primitive_index = self.primitives.index(action.primitive)
            return np.asarray([
                float(primitive_index),
                action.duration,
                action.amplitude if action.primitive == "sideband" else action.alpha.real if action.primitive == "cavity_displacement" else 0.0,
                action.phase if action.primitive == "sideband" else action.alpha.imag if action.primitive == "cavity_displacement" else 0.0,
                action.detuning,
                action.drag,
                1.0 if action.collapse else 0.0,
            ], dtype=float)
        if isinstance(action, QubitGaussianAction):
            return np.asarray([
                float(self.primitives.index("qubit_gaussian")),
                action.duration,
                action.theta,
                action.phi,
                action.detuning,
                action.drag,
                0.0,
            ], dtype=float)
        if isinstance(action, CavityDisplacementAction):
            return np.asarray([
                float(self.primitives.index("cavity_displacement")),
                action.duration,
                action.alpha.real,
                action.alpha.imag,
                action.detuning,
                0.0,
                0.0,
            ], dtype=float)
        if isinstance(action, SidebandAction):
            return np.asarray([
                float(self.primitives.index("sideband")),
                action.duration,
                action.amplitude,
                action.phase,
                action.detuning,
                0.0,
                0.0,
            ], dtype=float)
        if isinstance(action, WaitAction):
            return np.asarray([float(self.primitives.index("wait")), action.duration, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        if isinstance(action, MeasurementAction):
            return np.asarray([float(self.primitives.index("measure")), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if action.collapse else 0.0], dtype=float)
        if isinstance(action, ResetAction):
            return np.asarray([float(self.primitives.index("reset")), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        return self.clip(action)


class WaveformActionSpace:
    def __init__(
        self,
        *,
        segments: int = 16,
        duration: float = 160.0e-9,
        channels: Sequence[str] = ("qubit", "storage"),
        amplitude_limit: float = 2.0 * np.pi * 6.0e6,
    ):
        self.segments = int(segments)
        self.duration = float(duration)
        self.channels = tuple(str(channel) for channel in channels)
        self.amplitude_limit = float(amplitude_limit)
        self.low = -self.amplitude_limit * np.ones(2 * len(self.channels) * self.segments, dtype=float)
        self.high = self.amplitude_limit * np.ones(2 * len(self.channels) * self.segments, dtype=float)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.low.size,)

    def clip(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.clip(_vector(action, self.low.size), self.low, self.high)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.low, self.high)

    def parse(self, action: Any) -> WaveformAction:
        if isinstance(action, WaveformAction):
            return action
        clipped = self.clip(action)
        block = clipped.reshape(len(self.channels), 2, self.segments)
        channel_samples = {
            channel: np.asarray(block[index, 0] + 1j * block[index, 1], dtype=np.complex128)
            for index, channel in enumerate(self.channels)
        }
        dt = float(self.duration / max(self.segments, 1))
        return WaveformAction(channel_samples=channel_samples, dt=dt, duration=self.duration)

    def flatten(self, action: Any) -> np.ndarray:
        if isinstance(action, WaveformAction):
            arrays: list[np.ndarray] = []
            for channel in self.channels:
                samples = np.asarray(action.channel_samples[channel], dtype=np.complex128)
                arrays.extend([samples.real, samples.imag])
            return np.concatenate(arrays)
        return self.clip(action)


__all__ = [
    "QubitGaussianAction",
    "CavityDisplacementAction",
    "SidebandAction",
    "WaitAction",
    "MeasurementAction",
    "ResetAction",
    "HybridBlockAction",
    "WaveformAction",
    "PrimitiveAction",
    "ParametricPulseActionSpace",
    "PrimitiveActionSpace",
    "WaveformActionSpace",
]