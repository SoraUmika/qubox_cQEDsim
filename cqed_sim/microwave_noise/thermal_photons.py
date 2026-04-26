from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .core import (
    bose_occupation,
    occupation_to_effective_temperature,
    resonator_lindblad_rates,
    resonator_thermal_occupation,
)


BathKind = Literal[
    "cold_internal",
    "hot_line",
    "attenuator",
    "filter",
    "cavity_attenuator",
    "added_noise",
    "unknown",
]


def _validate_nonnegative_scalar(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative.")


def n_bose(frequency_hz: float, temperature_K: float) -> float:
    """Return the Bose-Einstein photon occupation for frequency in Hz."""

    _validate_nonnegative_scalar("frequency_hz", frequency_hz)
    _validate_nonnegative_scalar("temperature_K", temperature_K)
    return float(bose_occupation(float(frequency_hz), float(temperature_K)))


def n_bose_angular(omega_rad_s: float, temperature_K: float) -> float:
    """Return the Bose-Einstein photon occupation for angular frequency in rad/s."""

    _validate_nonnegative_scalar("omega_rad_s", omega_rad_s)
    return n_bose(float(omega_rad_s) / (2.0 * np.pi), float(temperature_K))


def effective_temperature(frequency_hz: float, nbar: float) -> float:
    """Invert the Bose-Einstein occupation at ``frequency_hz``."""

    _validate_nonnegative_scalar("frequency_hz", frequency_hz)
    _validate_nonnegative_scalar("nbar", nbar)
    return float(occupation_to_effective_temperature(float(frequency_hz), float(nbar)))


def thermal_lindblad_rates(kappa_rad_s: float, nbar: float) -> tuple[float, float]:
    """Return ``(kappa_down, kappa_up)`` for a bosonic thermal bath."""

    return resonator_lindblad_rates(float(kappa_rad_s), float(nbar))


@dataclass(frozen=True)
class BathSpec:
    """One thermal or calibrated bath coupled to a bosonic mode."""

    name: str
    kappa_rad_s: float
    temperature_K: float | None = None
    nbar: float | None = None
    kind: BathKind = "unknown"

    def __post_init__(self) -> None:
        _validate_nonnegative_scalar("kappa_rad_s", self.kappa_rad_s)
        if self.temperature_K is not None:
            _validate_nonnegative_scalar("temperature_K", self.temperature_K)
        if self.nbar is not None:
            _validate_nonnegative_scalar("nbar", self.nbar)
        if self.temperature_K is not None and self.nbar is not None:
            raise ValueError("BathSpec accepts either temperature_K or nbar, not both.")
        if self.kappa_rad_s > 0.0 and self.temperature_K is None and self.nbar is None:
            raise ValueError("A nonzero-coupled bath must specify temperature_K or nbar.")

    def resolved_nbar(self, omega_rad_s: float) -> float:
        """Return the bath occupation at angular frequency ``omega_rad_s``."""

        _validate_nonnegative_scalar("omega_rad_s", omega_rad_s)
        if self.nbar is not None:
            return float(self.nbar)
        if self.temperature_K is not None:
            return n_bose_angular(float(omega_rad_s), float(self.temperature_K))
        return 0.0


@dataclass(frozen=True)
class ModeBathModel:
    """A resonator mode coupled to one or more thermal baths."""

    mode_name: str
    omega_rad_s: float
    baths: list[BathSpec]

    def __post_init__(self) -> None:
        _validate_nonnegative_scalar("omega_rad_s", self.omega_rad_s)
        if not self.baths:
            raise ValueError("ModeBathModel requires at least one BathSpec.")

    def total_kappa(self) -> float:
        """Return the total mode linewidth from all baths."""

        total = float(sum(bath.kappa_rad_s for bath in self.baths))
        if total <= 0.0:
            raise ValueError("total kappa must be positive.")
        return total

    def effective_nbar(self) -> float:
        """Return the linewidth-weighted effective mode occupation."""

        kappas = [bath.kappa_rad_s for bath in self.baths]
        nbars = [bath.resolved_nbar(self.omega_rad_s) for bath in self.baths]
        return resonator_thermal_occupation(kappas, nbars)

    def effective_temperature(self) -> float:
        """Return the Bose-equivalent effective mode temperature in kelvin."""

        return effective_temperature(self.omega_rad_s / (2.0 * np.pi), self.effective_nbar())

    def thermal_lindblad_rates(self) -> tuple[float, float]:
        """Return ``(downward_rate, upward_rate)`` for the effective bath."""

        return thermal_lindblad_rates(self.total_kappa(), self.effective_nbar())


__all__ = [
    "BathKind",
    "BathSpec",
    "ModeBathModel",
    "effective_temperature",
    "n_bose",
    "n_bose_angular",
    "thermal_lindblad_rates",
]
