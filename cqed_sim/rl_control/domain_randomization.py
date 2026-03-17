from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np


class ParameterPrior(Protocol):
    def sample(self, rng: np.random.Generator) -> Any:
        ...


@dataclass(frozen=True)
class FixedPrior:
    value: Any

    def sample(self, rng: np.random.Generator) -> Any:
        del rng
        return self.value


@dataclass(frozen=True)
class UniformPrior:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(float(self.low), float(self.high)))


@dataclass(frozen=True)
class NormalPrior:
    mean: float
    sigma: float
    low: float | None = None
    high: float | None = None

    def sample(self, rng: np.random.Generator) -> float:
        value = float(rng.normal(float(self.mean), float(self.sigma)))
        if self.low is not None:
            value = max(float(self.low), value)
        if self.high is not None:
            value = min(float(self.high), value)
        return float(value)


@dataclass(frozen=True)
class ChoicePrior:
    values: tuple[Any, ...]
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("ChoicePrior.values must not be empty.")
        if self.probabilities is not None and len(self.probabilities) != len(self.values):
            raise ValueError("ChoicePrior.probabilities must match ChoicePrior.values.")

    def sample(self, rng: np.random.Generator) -> Any:
        if self.probabilities is None:
            return self.values[int(rng.integers(0, len(self.values)))]
        probabilities = np.asarray(self.probabilities, dtype=float)
        probabilities = probabilities / np.sum(probabilities)
        index = int(rng.choice(np.arange(len(self.values)), p=probabilities))
        return self.values[index]


@dataclass(frozen=True)
class RandomizationSample:
    model_overrides: dict[str, Any] = field(default_factory=dict)
    noise_overrides: dict[str, Any] = field(default_factory=dict)
    hardware_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    measurement_overrides: dict[str, Any] = field(default_factory=dict)
    drift_state: dict[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": dict(self.model_overrides),
            "noise": dict(self.noise_overrides),
            "hardware": {channel: dict(values) for channel, values in self.hardware_overrides.items()},
            "measurement": dict(self.measurement_overrides),
            "drift": dict(self.drift_state),
        }


def _sample_mapping(
    priors: dict[str, ParameterPrior] | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not priors:
        return {}
    return {key: prior.sample(rng) for key, prior in priors.items()}


def _sample_hardware_mapping(
    priors: dict[str, dict[str, ParameterPrior]] | None,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    if not priors:
        return {}
    return {
        channel: _sample_mapping(channel_priors, rng)
        for channel, channel_priors in priors.items()
    }


@dataclass
class DomainRandomizer:
    model_priors_train: dict[str, ParameterPrior] = field(default_factory=dict)
    noise_priors_train: dict[str, ParameterPrior] = field(default_factory=dict)
    hardware_priors_train: dict[str, dict[str, ParameterPrior]] = field(default_factory=dict)
    measurement_priors_train: dict[str, ParameterPrior] = field(default_factory=dict)
    drift_priors_train: dict[str, ParameterPrior] = field(default_factory=dict)
    model_priors_eval: dict[str, ParameterPrior] | None = None
    noise_priors_eval: dict[str, ParameterPrior] | None = None
    hardware_priors_eval: dict[str, dict[str, ParameterPrior]] | None = None
    measurement_priors_eval: dict[str, ParameterPrior] | None = None
    drift_priors_eval: dict[str, ParameterPrior] | None = None

    def _resolved_priors(
        self,
        mode: str,
    ) -> tuple[
        dict[str, ParameterPrior],
        dict[str, ParameterPrior],
        dict[str, dict[str, ParameterPrior]],
        dict[str, ParameterPrior],
        dict[str, ParameterPrior],
    ]:
        if str(mode) == "eval":
            return (
                self.model_priors_eval if self.model_priors_eval is not None else self.model_priors_train,
                self.noise_priors_eval if self.noise_priors_eval is not None else self.noise_priors_train,
                self.hardware_priors_eval if self.hardware_priors_eval is not None else self.hardware_priors_train,
                self.measurement_priors_eval if self.measurement_priors_eval is not None else self.measurement_priors_train,
                self.drift_priors_eval if self.drift_priors_eval is not None else self.drift_priors_train,
            )
        return (
            self.model_priors_train,
            self.noise_priors_train,
            self.hardware_priors_train,
            self.measurement_priors_train,
            self.drift_priors_train,
        )

    def sample(self, *, seed: int | None = None, mode: str = "train") -> RandomizationSample:
        rng = np.random.default_rng(seed)
        model_priors, noise_priors, hardware_priors, measurement_priors, drift_priors = self._resolved_priors(mode)
        return RandomizationSample(
            model_overrides=_sample_mapping(model_priors, rng),
            noise_overrides=_sample_mapping(noise_priors, rng),
            hardware_overrides=_sample_hardware_mapping(hardware_priors, rng),
            measurement_overrides=_sample_mapping(measurement_priors, rng),
            drift_state=_sample_mapping(drift_priors, rng),
        )

    def sweep(self, seeds: Sequence[int], *, mode: str = "train") -> list[RandomizationSample]:
        return [self.sample(seed=int(seed), mode=mode) for seed in seeds]


__all__ = [
    "ParameterPrior",
    "FixedPrior",
    "UniformPrior",
    "NormalPrior",
    "ChoicePrior",
    "RandomizationSample",
    "DomainRandomizer",
]