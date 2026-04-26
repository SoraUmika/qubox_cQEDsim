from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .core import thermal_photon_dephasing


def _validate_rate(name: str, value: float, *, positive: bool = False) -> None:
    value = float(value)
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if not positive and value < 0.0:
        raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class ThermalPhotonDephasingContribution:
    """Per-mode contribution to thermal-photon-induced qubit dephasing."""

    mode_name: str
    nbar: float
    kappa_rad_s: float
    chi_rad_s: float
    gamma_phi_rad_s: float


def gamma_phi_thermal(
    nbar: float,
    kappa_rad_s: float,
    chi_rad_s: float,
    *,
    exact: bool = True,
    approximation: str | None = None,
) -> float:
    """Thermal-photon-induced pure dephasing rate in angular-rate units.

    The default is the Zhang/Clerk-Utami expression already used by
    :func:`thermal_photon_dephasing`; ``approximation`` may be ``"weak"`` or
    ``"strong_low_occupation"``.
    """

    _validate_rate("nbar", nbar)
    _validate_rate("kappa_rad_s", kappa_rad_s, positive=True)
    return thermal_photon_dephasing(
        float(kappa_rad_s),
        float(chi_rad_s),
        float(nbar),
        exact=bool(exact),
        approximation=approximation,
    )


def gamma_phi_lorentzian_interpolation(nbar: float, kappa_rad_s: float, chi_rad_s: float) -> float:
    """Return ``nbar*kappa*chi**2/(kappa**2+chi**2)``.

    This is a compact interpolation used in some cavity-thermal-photon estimates.
    It is exposed explicitly so callers do not confuse it with the more general
    canonical expression in :func:`gamma_phi_thermal`.
    """

    _validate_rate("nbar", nbar)
    _validate_rate("kappa_rad_s", kappa_rad_s, positive=True)
    denominator = float(kappa_rad_s) ** 2 + float(chi_rad_s) ** 2
    if denominator == 0.0:
        return 0.0
    return float(float(nbar) * float(kappa_rad_s) * float(chi_rad_s) ** 2 / denominator)


def gamma_phi_strong_dispersive_N(nbar: float, kappa_rad_s: float, N: int) -> float:
    """Photon-number-conditioned escape/dephasing rate in the strong-dispersive limit."""

    _validate_rate("nbar", nbar)
    _validate_rate("kappa_rad_s", kappa_rad_s)
    if int(N) != N or int(N) < 0:
        raise ValueError("N must be a nonnegative integer.")
    number = int(N)
    # Strong-dispersive photon escape rate used in Sears et al., PRB 2012.
    return float(float(kappa_rad_s) * ((float(nbar) + 1.0) * number + float(nbar) * (number + 1)))


def Tphi_from_gamma(gamma_phi_rad_s: float) -> float:
    """Return ``T_phi`` in seconds from a dephasing rate."""

    _validate_rate("gamma_phi_rad_s", gamma_phi_rad_s)
    if float(gamma_phi_rad_s) == 0.0:
        return float("inf")
    return 1.0 / float(gamma_phi_rad_s)


def T2_from_T1_Tphi(T1_s: float, Tphi_s: float) -> float:
    """Use ``1/T2 = 1/(2*T1) + 1/Tphi``."""

    _validate_rate("T1_s", T1_s, positive=True)
    _validate_rate("Tphi_s", Tphi_s, positive=True)
    return float(1.0 / (1.0 / (2.0 * float(T1_s)) + 1.0 / float(Tphi_s)))


def Tphi_from_T1_T2(T1_s: float, T2_s: float) -> float:
    """Extract pure-dephasing time from ``T1`` and ``T2``."""

    _validate_rate("T1_s", T1_s, positive=True)
    _validate_rate("T2_s", T2_s, positive=True)
    inv_tphi = 1.0 / float(T2_s) - 1.0 / (2.0 * float(T1_s))
    if inv_tphi <= 0.0:
        return float("inf")
    return float(1.0 / inv_tphi)


def gamma_phi_multimode(
    nbar_values: Sequence[float],
    kappa_rad_s_values: Sequence[float],
    chi_rad_s_values: Sequence[float],
    *,
    mode_names: Sequence[str] | None = None,
    exact: bool = True,
    approximation: str | None = None,
) -> tuple[float, tuple[ThermalPhotonDephasingContribution, ...]]:
    """Return additive multimode dephasing and per-mode contributions."""

    if not (len(nbar_values) == len(kappa_rad_s_values) == len(chi_rad_s_values)):
        raise ValueError("nbar, kappa, and chi sequences must have the same length.")
    if mode_names is not None and len(mode_names) != len(nbar_values):
        raise ValueError("mode_names must match the length of the input sequences.")

    names = list(mode_names) if mode_names is not None else [f"mode_{index}" for index in range(len(nbar_values))]
    contributions = []
    for name, nbar, kappa, chi in zip(names, nbar_values, kappa_rad_s_values, chi_rad_s_values):
        gamma = gamma_phi_thermal(nbar, kappa, chi, exact=exact, approximation=approximation)
        contributions.append(
            ThermalPhotonDephasingContribution(
                mode_name=str(name),
                nbar=float(nbar),
                kappa_rad_s=float(kappa),
                chi_rad_s=float(chi),
                gamma_phi_rad_s=float(gamma),
            )
        )
    total = float(np.sum([item.gamma_phi_rad_s for item in contributions]))
    return total, tuple(contributions)


__all__ = [
    "ThermalPhotonDephasingContribution",
    "T2_from_T1_Tphi",
    "Tphi_from_T1_T2",
    "Tphi_from_gamma",
    "gamma_phi_lorentzian_interpolation",
    "gamma_phi_multimode",
    "gamma_phi_strong_dispersive_N",
    "gamma_phi_thermal",
]
