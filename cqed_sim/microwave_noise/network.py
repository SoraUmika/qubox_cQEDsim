from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core import loss_db_to_power_transmission
from .thermal_photons import effective_temperature, n_bose


LossStageKind = Literal[
    "commercial_attenuator",
    "eccosorb",
    "lossy_coax",
    "cavity_attenuator",
    "unknown",
]

FilterStageKind = Literal["lowpass", "bandpass", "reflective", "purcell", "unknown"]


def _validate_nonnegative(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative.")


def _in_passband(frequency_hz: float, center_hz: float | None, width_hz: float | None) -> bool:
    if center_hz is None or width_hz is None:
        return True
    _validate_nonnegative("bandwidth_hz", width_hz)
    return abs(float(frequency_hz) - float(center_hz)) <= 0.5 * float(width_hz)


@dataclass(frozen=True)
class PassiveLossStage:
    """Passive dissipative loss stage modeled as a thermal beam splitter."""

    name: str
    attenuation_dB: float
    physical_temperature_K: float
    center_frequency_hz: float | None = None
    bandwidth_hz: float | None = None
    kind: LossStageKind = "unknown"

    def __post_init__(self) -> None:
        _validate_nonnegative("attenuation_dB", self.attenuation_dB)
        _validate_nonnegative("physical_temperature_K", self.physical_temperature_K)
        if self.center_frequency_hz is not None:
            _validate_nonnegative("center_frequency_hz", self.center_frequency_hz)
        if self.bandwidth_hz is not None:
            _validate_nonnegative("bandwidth_hz", self.bandwidth_hz)

    def transmission_eta(self, frequency_hz: float) -> float:
        """Return power transmission at ``frequency_hz``."""

        _validate_nonnegative("frequency_hz", frequency_hz)
        if not _in_passband(frequency_hz, self.center_frequency_hz, self.bandwidth_hz):
            return 1.0
        return float(loss_db_to_power_transmission(float(self.attenuation_dB)))

    def propagate_nbar(self, frequency_hz: float, nbar_in: float) -> float:
        """Propagate occupation through this cold or warm dissipative element."""

        _validate_nonnegative("nbar_in", nbar_in)
        eta = self.transmission_eta(frequency_hz)
        return float(eta * float(nbar_in) + (1.0 - eta) * n_bose(frequency_hz, self.physical_temperature_K))


@dataclass(frozen=True)
class LosslessFilterStage:
    """Frequency-selective lossless filter.

    In-band, the normally ordered occupation is passed through unchanged. Out
    of band, the transmitted occupation is rejected, but no cold thermal bath is
    added because the filter is modeled as lossless/reflective.
    """

    name: str
    passband_center_hz: float
    passband_width_hz: float
    stopband_rejection_dB: float
    kind: FilterStageKind = "unknown"

    def __post_init__(self) -> None:
        _validate_nonnegative("passband_center_hz", self.passband_center_hz)
        _validate_nonnegative("passband_width_hz", self.passband_width_hz)
        _validate_nonnegative("stopband_rejection_dB", self.stopband_rejection_dB)

    def transmission_eta(self, frequency_hz: float) -> float:
        """Return power transmission at ``frequency_hz``."""

        _validate_nonnegative("frequency_hz", frequency_hz)
        if _in_passband(frequency_hz, self.passband_center_hz, self.passband_width_hz):
            return 1.0
        return float(loss_db_to_power_transmission(float(self.stopband_rejection_dB)))

    def propagate_nbar(self, frequency_hz: float, nbar_in: float) -> float:
        """Transmit occupation without adding a thermal bath."""

        _validate_nonnegative("nbar_in", nbar_in)
        return float(self.transmission_eta(frequency_hz) * float(nbar_in))


@dataclass(frozen=True)
class MicrowaveNoiseChain:
    """A cascade of passive loss stages and lossless filters."""

    stages: list[PassiveLossStage | LosslessFilterStage]
    input_temperature_K: float

    def __post_init__(self) -> None:
        _validate_nonnegative("input_temperature_K", self.input_temperature_K)

    def propagate_nbar(self, frequency_hz: float) -> float:
        """Return final normally ordered occupation at ``frequency_hz``."""

        nbar = n_bose(frequency_hz, self.input_temperature_K)
        for stage in self.stages:
            nbar = stage.propagate_nbar(frequency_hz, nbar)
        return float(nbar)

    def equivalent_output_temperature(self, frequency_hz: float) -> float:
        """Return the Bose-equivalent output temperature."""

        return effective_temperature(frequency_hz, self.propagate_nbar(frequency_hz))

    def stage_by_stage_table(self, frequency_hz: float) -> list[dict[str, float | str]]:
        """Return a compact diagnostic table for the cascade."""

        rows: list[dict[str, float | str]] = []
        nbar = n_bose(frequency_hz, self.input_temperature_K)
        rows.append(
            {
                "name": "input",
                "kind": "source",
                "eta": 1.0,
                "n_in": float(nbar),
                "n_out": float(nbar),
                "temperature_K": float(self.input_temperature_K),
            }
        )
        for stage in self.stages:
            n_in = float(nbar)
            nbar = stage.propagate_nbar(frequency_hz, nbar)
            row = {
                "name": stage.name,
                "kind": stage.kind,
                "eta": float(stage.transmission_eta(frequency_hz)),
                "n_in": n_in,
                "n_out": float(nbar),
            }
            if isinstance(stage, PassiveLossStage):
                row["temperature_K"] = float(stage.physical_temperature_K)
            rows.append(row)
        return rows


__all__ = [
    "FilterStageKind",
    "LossStageKind",
    "LosslessFilterStage",
    "MicrowaveNoiseChain",
    "PassiveLossStage",
]
