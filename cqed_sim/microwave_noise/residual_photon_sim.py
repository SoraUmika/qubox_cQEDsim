from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.core import FrameSpec
from cqed_sim.core.readout_model import DispersiveReadoutTransmonStorageModel
from cqed_sim.sim.noise import NoiseSpec, collapse_operators

from .photon_dephasing import (
    T2_from_T1_Tphi,
    Tphi_from_T1_T2,
    Tphi_from_gamma,
    gamma_phi_strong_dispersive_N,
    gamma_phi_thermal,
)


@dataclass
class RamseyResult:
    """Deterministic Ramsey-style residual-photon dephasing result."""

    times_s: np.ndarray
    signal: np.ndarray
    fitted_T2star_s: float
    fitted_frequency_rad_s: float
    gamma_phi_rad_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NoiseInducedDephasingResult:
    """Synthetic or fitted added-noise dephasing extraction result."""

    nadd_values: np.ndarray
    T1_values_s: np.ndarray
    T2e_values_s: np.ndarray
    Tphi_values_s: np.ndarray
    fitted_nth: float
    fitted_slope_rad_s: float
    confidence_interval: tuple[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _thermal_dm(dim: int, nbar: float) -> qt.Qobj:
    if nbar <= 0.0:
        return qt.fock_dm(int(dim), 0)
    return qt.thermal_dm(int(dim), float(nbar))


def _density_product_for_model(model: Any, noise: NoiseSpec) -> qt.Qobj:
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
    if len(dims) < 2:
        raise ValueError("Residual-photon Ramsey helpers require at least one bosonic mode.")
    if dims[0] < 2:
        raise ValueError("The leading transmon/qubit subsystem must have at least two levels.")
    plus = (qt.basis(dims[0], 0) + qt.basis(dims[0], 1)).unit()
    factors: list[qt.Qobj] = [plus * plus.dag()]
    if len(dims) == 2:
        factors.append(_thermal_dm(dims[1], float(noise.nth)))
    elif len(dims) == 3:
        nth_storage = noise.nth_storage if noise.nth_storage is not None else noise.nth
        nth_readout = noise.nth_readout if noise.nth_readout is not None else noise.nth
        factors.append(_thermal_dm(dims[1], float(nth_storage or 0.0)))
        factors.append(_thermal_dm(dims[2], float(nth_readout or 0.0)))
    else:
        raise ValueError("Only two- and three-mode cQED tensor layouts are supported.")
    return qt.tensor(*factors)


def _qubit_coherence_signal(states: list[qt.Qobj]) -> tuple[np.ndarray, np.ndarray]:
    coherences = []
    for state in states:
        rho_q = qt.ptrace(state, 0)
        coherences.append(complex(rho_q[0, 1]))
    coherence = np.asarray(coherences, dtype=complex)
    return 2.0 * np.real(coherence), coherence


def _fit_coherence(times_s: np.ndarray, coherence: np.ndarray, *, gamma1_rad_s: float = 0.0) -> tuple[float, float, float]:
    times = np.asarray(times_s, dtype=float)
    magnitude = np.abs(np.asarray(coherence, dtype=complex))
    valid = (times > times[0]) & (magnitude > 1.0e-12)
    if np.count_nonzero(valid) >= 2:
        slope = np.polyfit(times[valid] - times[0], np.log(magnitude[valid] / max(magnitude[0], 1.0e-300)), 1)[0]
        decay_rate = max(0.0, float(-slope))
    else:
        decay_rate = 0.0
    phase = np.unwrap(np.angle(coherence))
    if len(times) >= 2:
        fitted_frequency = float(np.polyfit(times - times[0], phase - phase[0], 1)[0])
    else:
        fitted_frequency = 0.0
    gamma_phi = max(0.0, decay_rate - 0.5 * float(gamma1_rad_s))
    t2star = float("inf") if decay_rate == 0.0 else 1.0 / decay_rate
    return t2star, fitted_frequency, gamma_phi


def simulate_ramsey_with_residual_photons(
    model: Any,
    times_s: np.ndarray,
    noise: NoiseSpec,
    *,
    frame: FrameSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> RamseyResult:
    """Run deterministic Lindblad Ramsey free evolution from a thermal bosonic state."""

    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times_s must be a one-dimensional array with at least two entries.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("times_s must be sorted.")
    rho0 = _density_product_for_model(model, noise)
    hamiltonian = model.static_hamiltonian(frame=frame or FrameSpec())
    result = qt.mesolve(
        hamiltonian,
        rho0,
        times,
        c_ops=collapse_operators(model, noise),
        e_ops=[],
        options={"store_states": True},
    )
    signal, coherence = _qubit_coherence_signal(list(result.states))
    t2star, frequency, gamma_phi = _fit_coherence(times, coherence, gamma1_rad_s=noise.gamma1)
    return RamseyResult(
        times_s=times,
        signal=signal,
        fitted_T2star_s=t2star,
        fitted_frequency_rad_s=frequency,
        gamma_phi_rad_s=gamma_phi,
        metadata={"sequence": "ramsey", **(metadata or {})},
    )


def simulate_hahn_echo_with_residual_photons(
    model: Any,
    times_s: np.ndarray,
    noise: NoiseSpec,
    *,
    frame: FrameSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> RamseyResult:
    """Run a small deterministic Hahn-echo replay with an ideal qubit pi pulse."""

    times = np.asarray(times_s, dtype=float)
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
    if dims[0] < 2:
        raise ValueError("The leading transmon/qubit subsystem must have at least two levels.")
    qubit_pi = np.eye(dims[0], dtype=complex)
    qubit_pi[0, 0] = 0.0
    qubit_pi[1, 1] = 0.0
    qubit_pi[0, 1] = 1.0
    qubit_pi[1, 0] = 1.0
    sigma_x = qt.tensor(qt.Qobj(qubit_pi), *(qt.qeye(dim) for dim in dims[1:]))
    hamiltonian = model.static_hamiltonian(frame=frame or FrameSpec())
    c_ops = collapse_operators(model, noise)
    rho0 = _density_product_for_model(model, noise)
    states: list[qt.Qobj] = []
    for tau in times:
        if tau < 0.0:
            raise ValueError("times_s must be nonnegative.")
        if tau == 0.0:
            states.append(rho0)
            continue
        first = qt.mesolve(hamiltonian, rho0, [0.0, 0.5 * tau], c_ops=c_ops, e_ops=[], options={"store_states": True})
        after_pi = sigma_x * first.states[-1] * sigma_x.dag()
        second = qt.mesolve(
            hamiltonian,
            after_pi,
            [0.5 * tau, tau],
            c_ops=c_ops,
            e_ops=[],
            options={"store_states": True},
        )
        states.append(second.states[-1])
    signal, coherence = _qubit_coherence_signal(states)
    t2star, frequency, gamma_phi = _fit_coherence(times, coherence, gamma1_rad_s=noise.gamma1)
    return RamseyResult(
        times_s=times,
        signal=signal,
        fitted_T2star_s=t2star,
        fitted_frequency_rad_s=frequency,
        gamma_phi_rad_s=gamma_phi,
        metadata={"sequence": "hahn_echo", **(metadata or {})},
    )


def photon_number_conditioned_ramsey(
    times_s: np.ndarray,
    *,
    nbar: float,
    kappa_rad_s: float,
    photon_number: int,
) -> RamseyResult:
    """Return analytic strong-dispersive photon-number-conditioned Ramsey decay."""

    times = np.asarray(times_s, dtype=float)
    gamma = gamma_phi_strong_dispersive_N(nbar, kappa_rad_s, photon_number)
    signal = np.exp(-gamma * times)
    return RamseyResult(
        times_s=times,
        signal=signal,
        fitted_T2star_s=Tphi_from_gamma(gamma),
        fitted_frequency_rad_s=0.0,
        gamma_phi_rad_s=gamma,
        metadata={"sequence": "photon_number_conditioned", "photon_number": int(photon_number)},
    )


def simulate_multimode_ramsey(
    model: DispersiveReadoutTransmonStorageModel,
    times_s: np.ndarray,
    *,
    kappa_storage_rad_s: float,
    nth_storage: float,
    kappa_readout_rad_s: float,
    nth_readout: float,
    frame: FrameSpec | None = None,
) -> RamseyResult:
    """Convenience wrapper for a three-mode storage/readout thermal Ramsey simulation."""

    noise = NoiseSpec(
        kappa_storage=float(kappa_storage_rad_s),
        nth_storage=float(nth_storage),
        kappa_readout=float(kappa_readout_rad_s),
        nth_readout=float(nth_readout),
    )
    return simulate_ramsey_with_residual_photons(
        model,
        times_s,
        noise,
        frame=frame,
        metadata={"mode": "storage+readout"},
    )


def simulate_noise_induced_dephasing(
    nadd_values: np.ndarray,
    nth: float,
    kappa_rad_s: float,
    chi_rad_s: float,
    T1_s: float,
    gamma_phi_other_rad_s: float = 0.0,
    measurement_noise: dict[str, Any] | None = None,
) -> NoiseInducedDephasingResult:
    """Generate synthetic Wang-style added-noise dephasing data."""

    nadd = np.asarray(nadd_values, dtype=float)
    if np.any(nadd < 0.0):
        raise ValueError("nadd_values must be nonnegative.")
    slope = gamma_phi_thermal(1.0, kappa_rad_s, chi_rad_s)
    gamma_phi = (nadd + float(nth)) * slope + float(gamma_phi_other_rad_s)
    Tphi = np.asarray([Tphi_from_gamma(value) for value in gamma_phi], dtype=float)
    T1_values = np.full_like(nadd, float(T1_s), dtype=float)
    T2e = np.asarray([T2_from_T1_Tphi(float(T1_s), value) for value in Tphi], dtype=float)

    if measurement_noise:
        rng = np.random.default_rng(measurement_noise.get("seed", 12345))
        relative = float(measurement_noise.get("relative_T2_sigma", 0.0))
        absolute = float(measurement_noise.get("absolute_T2_sigma_s", 0.0))
        if relative > 0.0 or absolute > 0.0:
            sigma = absolute + relative * T2e
            T2e = np.maximum(T2e + rng.normal(scale=sigma), np.finfo(float).eps)

    fitted = fit_noise_induced_dephasing(nadd, T1_values, T2e, kappa_rad_s, chi_rad_s)
    return NoiseInducedDephasingResult(
        nadd_values=nadd,
        T1_values_s=T1_values,
        T2e_values_s=T2e,
        Tphi_values_s=Tphi,
        fitted_nth=fitted.fitted_nth,
        fitted_slope_rad_s=fitted.fitted_slope_rad_s,
        confidence_interval=fitted.confidence_interval,
        metadata={
            "nth": float(nth),
            "gamma_phi_other_rad_s": float(gamma_phi_other_rad_s),
            "slope_rad_s_per_photon": float(slope),
            **fitted.metadata,
        },
    )


def fit_noise_induced_dephasing(
    nadd_values: np.ndarray,
    T1_values_s: np.ndarray,
    T2_values_s: np.ndarray,
    kappa_rad_s: float,
    chi_rad_s: float,
) -> NoiseInducedDephasingResult:
    """Fit added-noise dephasing data and infer the thermal offset."""

    nadd = np.asarray(nadd_values, dtype=float)
    T1 = np.asarray(T1_values_s, dtype=float)
    T2 = np.asarray(T2_values_s, dtype=float)
    if not (nadd.shape == T1.shape == T2.shape):
        raise ValueError("nadd_values, T1_values_s, and T2_values_s must have the same shape.")
    Tphi = np.asarray([Tphi_from_T1_T2(t1, t2) for t1, t2 in zip(T1, T2)], dtype=float)
    gamma_phi = np.divide(1.0, Tphi, out=np.zeros_like(Tphi, dtype=float), where=np.isfinite(Tphi))
    fit_slope, fit_intercept = np.polyfit(nadd, gamma_phi, 1)
    model_slope = gamma_phi_thermal(1.0, kappa_rad_s, chi_rad_s)
    fitted_nth = float(fit_intercept / model_slope) if model_slope > 0.0 else float("nan")

    residual = gamma_phi - (fit_slope * nadd + fit_intercept)
    if len(nadd) > 2:
        sigma = float(np.sqrt(np.sum(residual**2) / (len(nadd) - 2)))
        centered = nadd - np.mean(nadd)
        intercept_sigma = sigma * np.sqrt(1.0 / len(nadd) + np.mean(nadd) ** 2 / np.sum(centered**2))
        nth_sigma = intercept_sigma / model_slope if model_slope > 0.0 else float("nan")
    else:
        nth_sigma = 0.0
    ci = (float(fitted_nth - 1.96 * nth_sigma), float(fitted_nth + 1.96 * nth_sigma))
    return NoiseInducedDephasingResult(
        nadd_values=nadd,
        T1_values_s=T1,
        T2e_values_s=T2,
        Tphi_values_s=Tphi,
        fitted_nth=fitted_nth,
        fitted_slope_rad_s=float(fit_slope),
        confidence_interval=ci,
        metadata={"model_slope_rad_s_per_photon": float(model_slope)},
    )


__all__ = [
    "NoiseInducedDephasingResult",
    "RamseyResult",
    "fit_noise_induced_dephasing",
    "photon_number_conditioned_ramsey",
    "simulate_hahn_echo_with_residual_photons",
    "simulate_multimode_ramsey",
    "simulate_noise_induced_dephasing",
    "simulate_ramsey_with_residual_photons",
]
