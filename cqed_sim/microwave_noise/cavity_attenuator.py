from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .photon_dephasing import T2_from_T1_Tphi, Tphi_from_gamma, gamma_phi_thermal
from .thermal_photons import effective_temperature, n_bose_angular


def _validate_nonnegative(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative.")


def _resolve_bath_nbar(
    omega_rad_s: float,
    kappa_rad_s: float,
    temperature_K: float | None,
    nbar: float | None,
    label: str,
) -> float:
    if temperature_K is not None and nbar is not None:
        raise ValueError(f"{label} bath accepts either temperature_K or nbar, not both.")
    if temperature_K is not None:
        _validate_nonnegative(f"{label}_temperature_K", temperature_K)
    if nbar is not None:
        _validate_nonnegative(f"{label}_nbar", nbar)
    if kappa_rad_s > 0.0 and temperature_K is None and nbar is None:
        raise ValueError(f"{label} bath must specify temperature_K or nbar when kappa is nonzero.")
    if nbar is not None:
        return float(nbar)
    if temperature_K is not None:
        return n_bose_angular(omega_rad_s, temperature_K)
    return 0.0


@dataclass(frozen=True)
class EffectiveCavityAttenuator:
    """Effective readout mode coupled to a cold internal bath and hot external line."""

    omega_ro_rad_s: float
    kappa_internal_rad_s: float
    kappa_external_rad_s: float
    internal_temperature_K: float | None = None
    internal_nbar: float | None = None
    external_temperature_K: float | None = None
    external_nbar: float | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative("omega_ro_rad_s", self.omega_ro_rad_s)
        _validate_nonnegative("kappa_internal_rad_s", self.kappa_internal_rad_s)
        _validate_nonnegative("kappa_external_rad_s", self.kappa_external_rad_s)
        _resolve_bath_nbar(
            self.omega_ro_rad_s,
            self.kappa_internal_rad_s,
            self.internal_temperature_K,
            self.internal_nbar,
            "internal",
        )
        _resolve_bath_nbar(
            self.omega_ro_rad_s,
            self.kappa_external_rad_s,
            self.external_temperature_K,
            self.external_nbar,
            "external",
        )

    def effective_kappa(self) -> float:
        """Return total readout linewidth."""

        total = float(self.kappa_internal_rad_s + self.kappa_external_rad_s)
        if total <= 0.0:
            raise ValueError("effective kappa must be positive.")
        return total

    def effective_nbar(self) -> float:
        """Return the weighted readout-mode thermal occupation."""

        n_internal = _resolve_bath_nbar(
            self.omega_ro_rad_s,
            self.kappa_internal_rad_s,
            self.internal_temperature_K,
            self.internal_nbar,
            "internal",
        )
        n_external = _resolve_bath_nbar(
            self.omega_ro_rad_s,
            self.kappa_external_rad_s,
            self.external_temperature_K,
            self.external_nbar,
            "external",
        )
        total = self.effective_kappa()
        # Effective cold-bath weighting used for Wang et al., PRApplied 2019.
        return float((self.kappa_internal_rad_s * n_internal + self.kappa_external_rad_s * n_external) / total)

    def effective_Teff(self) -> float:
        """Return the Bose-equivalent readout mode temperature."""

        return effective_temperature(self.omega_ro_rad_s / (2.0 * np.pi), self.effective_nbar())

    def gamma_phi(self, chi_rad_s: float) -> float:
        """Return residual-photon dephasing from the effective readout bath."""

        return gamma_phi_thermal(self.effective_nbar(), self.effective_kappa(), chi_rad_s)


@dataclass(frozen=True)
class TwoModeCavityAttenuatorModel:
    """Linear two-mode hybridization model for a readout cavity and lossy attenuator."""

    omega_readout_rad_s: float
    omega_attenuator_rad_s: float
    coupling_J_rad_s: float
    kappa_readout_internal_rad_s: float = 0.0
    kappa_attenuator_internal_rad_s: float = 0.0
    kappa_external_rad_s: float = 0.0
    chi_bare_readout_rad_s: float = 0.0

    def __post_init__(self) -> None:
        _validate_nonnegative("omega_readout_rad_s", self.omega_readout_rad_s)
        _validate_nonnegative("omega_attenuator_rad_s", self.omega_attenuator_rad_s)
        _validate_nonnegative("kappa_readout_internal_rad_s", self.kappa_readout_internal_rad_s)
        _validate_nonnegative("kappa_attenuator_internal_rad_s", self.kappa_attenuator_internal_rad_s)
        _validate_nonnegative("kappa_external_rad_s", self.kappa_external_rad_s)

    def _eigensystem(self) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.array(
            [
                [float(self.omega_readout_rad_s), float(self.coupling_J_rad_s)],
                [float(self.coupling_J_rad_s), float(self.omega_attenuator_rad_s)],
            ],
            dtype=float,
        )
        frequencies, eigenvectors = np.linalg.eigh(matrix)
        return frequencies, eigenvectors

    def hybridized_modes(self) -> dict[str, np.ndarray]:
        """Return hybridized frequencies and eigenvectors.

        Eigenvector rows correspond to bare readout and attenuator amplitudes;
        columns correspond to the returned hybridized modes.
        """

        frequencies, eigenvectors = self._eigensystem()
        return {"frequencies_rad_s": frequencies, "eigenvectors": eigenvectors}

    def participation_ratios(self) -> dict[str, np.ndarray]:
        """Return bare-mode participation ratios for each hybridized mode."""

        _, eigenvectors = self._eigensystem()
        readout = np.abs(eigenvectors[0, :]) ** 2
        attenuator = np.abs(eigenvectors[1, :]) ** 2
        return {"readout": readout, "attenuator": attenuator}

    def effective_chi_per_mode(self) -> np.ndarray:
        """Return dispersive shifts inherited from readout-mode participation."""

        return self.participation_ratios()["readout"] * float(self.chi_bare_readout_rad_s)

    def effective_kappa_per_mode(self) -> np.ndarray:
        """Return linewidths inherited from readout and attenuator participation."""

        participation = self.participation_ratios()
        return (
            participation["readout"] * float(self.kappa_readout_internal_rad_s)
            + participation["attenuator"]
            * float(self.kappa_attenuator_internal_rad_s + self.kappa_external_rad_s)
        )

    def effective_nbar_per_mode(
        self,
        *,
        readout_internal_nbar: float = 0.0,
        attenuator_internal_nbar: float = 0.0,
        external_nbar: float = 0.0,
    ) -> np.ndarray:
        """Return effective occupations from weighted bath participation."""

        _validate_nonnegative("readout_internal_nbar", readout_internal_nbar)
        _validate_nonnegative("attenuator_internal_nbar", attenuator_internal_nbar)
        _validate_nonnegative("external_nbar", external_nbar)
        participation = self.participation_ratios()
        k_readout = participation["readout"] * float(self.kappa_readout_internal_rad_s)
        k_att_int = participation["attenuator"] * float(self.kappa_attenuator_internal_rad_s)
        k_ext = participation["attenuator"] * float(self.kappa_external_rad_s)
        total = k_readout + k_att_int + k_ext
        numerator = (
            k_readout * float(readout_internal_nbar)
            + k_att_int * float(attenuator_internal_nbar)
            + k_ext * float(external_nbar)
        )
        return np.divide(numerator, total, out=np.zeros_like(numerator, dtype=float), where=total > 0.0)


def required_internal_to_external_ratio(
    target_nbar: float,
    external_nbar: float,
    *,
    internal_nbar: float = 0.0,
) -> float:
    """Return the minimum ``kappa_internal/kappa_external`` to reach ``target_nbar``."""

    _validate_nonnegative("target_nbar", target_nbar)
    _validate_nonnegative("external_nbar", external_nbar)
    _validate_nonnegative("internal_nbar", internal_nbar)
    if target_nbar <= internal_nbar:
        return float("inf")
    if external_nbar <= target_nbar:
        return 0.0
    return float((external_nbar - target_nbar) / (target_nbar - internal_nbar))


def sweep_cavity_attenuator_design(
    ratios: np.ndarray,
    *,
    omega_ro_rad_s: float,
    kappa_external_rad_s: float,
    external_nbar: float,
    chi_rad_s: float,
    T1_s: float,
    internal_nbar: float = 0.0,
) -> dict[str, np.ndarray]:
    """Sweep cold-bath coupling ratios for design studies."""

    ratio_values = np.asarray(ratios, dtype=float)
    if np.any(ratio_values < 0.0):
        raise ValueError("ratios must be nonnegative.")
    nbar_values = []
    gamma_values = []
    tphi_values = []
    t2_over_2t1 = []
    for ratio in ratio_values:
        model = EffectiveCavityAttenuator(
            omega_ro_rad_s=float(omega_ro_rad_s),
            kappa_internal_rad_s=float(ratio) * float(kappa_external_rad_s),
            kappa_external_rad_s=float(kappa_external_rad_s),
            internal_nbar=float(internal_nbar),
            external_nbar=float(external_nbar),
        )
        nbar_eff = model.effective_nbar()
        gamma = model.gamma_phi(float(chi_rad_s))
        tphi = Tphi_from_gamma(gamma)
        t2 = T2_from_T1_Tphi(float(T1_s), tphi)
        nbar_values.append(nbar_eff)
        gamma_values.append(gamma)
        tphi_values.append(tphi)
        t2_over_2t1.append(t2 / (2.0 * float(T1_s)))
    return {
        "ratio": ratio_values,
        "nbar_eff": np.asarray(nbar_values, dtype=float),
        "gamma_phi_rad_s": np.asarray(gamma_values, dtype=float),
        "Tphi_s": np.asarray(tphi_values, dtype=float),
        "T2_over_2T1": np.asarray(t2_over_2t1, dtype=float),
    }


__all__ = [
    "EffectiveCavityAttenuator",
    "TwoModeCavityAttenuatorModel",
    "required_internal_to_external_ratio",
    "sweep_cavity_attenuator_design",
]
