from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cqed_sim.rl_control.domain_randomization import DomainRandomizer, ParameterPrior


@dataclass(frozen=True)
class CalibrationEvidence:
    model_posteriors: dict[str, ParameterPrior] = field(default_factory=dict)
    noise_posteriors: dict[str, ParameterPrior] = field(default_factory=dict)
    hardware_posteriors: dict[str, dict[str, ParameterPrior]] = field(default_factory=dict)
    measurement_posteriors: dict[str, ParameterPrior] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def randomizer_from_calibration(evidence: CalibrationEvidence) -> DomainRandomizer:
    return DomainRandomizer(
        model_priors_train=dict(evidence.model_posteriors),
        noise_priors_train=dict(evidence.noise_posteriors),
        hardware_priors_train={channel: dict(values) for channel, values in evidence.hardware_posteriors.items()},
        measurement_priors_train=dict(evidence.measurement_posteriors),
        model_priors_eval=dict(evidence.model_posteriors),
        noise_priors_eval=dict(evidence.noise_posteriors),
        hardware_priors_eval={channel: dict(values) for channel, values in evidence.hardware_posteriors.items()},
        measurement_priors_eval=dict(evidence.measurement_posteriors),
    )


__all__ = ["CalibrationEvidence", "randomizer_from_calibration"]