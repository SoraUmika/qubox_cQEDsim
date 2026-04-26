from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.microwave_noise import (
    BathSpec,
    EffectiveCavityAttenuator,
    LosslessFilterStage,
    MicrowaveNoiseChain,
    ModeBathModel,
    PassiveLossStage,
    TwoModeCavityAttenuatorModel,
    T2_from_T1_Tphi,
    Tphi_from_T1_T2,
    Tphi_from_gamma,
    effective_temperature,
    gamma_phi_lorentzian_interpolation,
    gamma_phi_multimode,
    gamma_phi_strong_dispersive_N,
    gamma_phi_thermal,
    n_bose,
    n_bose_angular,
    required_internal_to_external_ratio,
    simulate_noise_induced_dephasing,
    simulate_ramsey_with_residual_photons,
    sweep_cavity_attenuator_design,
    thermal_lindblad_rates,
)
from cqed_sim.sim import NoiseSpec


def test_strict_bose_helpers_and_effective_temperature() -> None:
    freq_hz = 6.0e9
    assert n_bose(freq_hz, 0.0) == 0.0
    assert n_bose(freq_hz, 0.02) == pytest.approx(5.586578304e-7, rel=2.0e-10)
    assert n_bose_angular(2.0 * np.pi * freq_hz, 0.02) == pytest.approx(n_bose(freq_hz, 0.02))
    assert effective_temperature(freq_hz, n_bose(freq_hz, 0.06)) == pytest.approx(0.06)
    assert effective_temperature(freq_hz, 0.0) == 0.0
    with pytest.raises(ValueError):
        n_bose(freq_hz, -0.1)
    with pytest.raises(ValueError):
        n_bose(-freq_hz, 0.1)
    with pytest.raises(ValueError):
        effective_temperature(freq_hz, -1.0)


def test_bath_model_weighting_and_lindblad_rates() -> None:
    omega = 2.0 * np.pi * 7.5e9
    model = ModeBathModel(
        "readout",
        omega,
        [
            BathSpec("cold", 6.0, nbar=0.0, kind="cold_internal"),
            BathSpec("line", 1.0, nbar=0.7, kind="hot_line"),
        ],
    )
    assert model.total_kappa() == pytest.approx(7.0)
    assert model.effective_nbar() == pytest.approx(0.1)
    assert model.thermal_lindblad_rates() == pytest.approx((7.7, 0.7))
    assert thermal_lindblad_rates(5.0, 0.2) == pytest.approx((6.0, 1.0))
    with pytest.raises(ValueError):
        BathSpec("ambiguous", 1.0, temperature_K=0.02, nbar=0.0)


def test_dephasing_formulas_and_limits() -> None:
    nbar = 2.0e-4
    kappa = 2.0 * np.pi * 1.0e6
    strong_chi = 1.0e3 * kappa
    weak_chi = 1.0e-3 * kappa
    assert gamma_phi_thermal(nbar, kappa, strong_chi) == pytest.approx(nbar * kappa, rel=3.0e-4)
    assert gamma_phi_thermal(nbar, kappa, weak_chi, approximation="weak") == pytest.approx(
        4.0 * weak_chi**2 / kappa * nbar * (nbar + 1.0)
    )
    assert gamma_phi_lorentzian_interpolation(nbar, kappa, strong_chi) == pytest.approx(nbar * kappa, rel=2.0e-6)
    assert gamma_phi_strong_dispersive_N(0.0, kappa, 0) == pytest.approx(0.0)
    assert gamma_phi_strong_dispersive_N(0.2, kappa, 1) == pytest.approx(kappa * (1.0 + 3.0 * 0.2))
    assert Tphi_from_gamma(0.0) == float("inf")
    t2 = T2_from_T1_Tphi(100e-6, 200e-6)
    assert Tphi_from_T1_T2(100e-6, t2) == pytest.approx(200e-6)


def test_multimode_dephasing_adds() -> None:
    total, contributions = gamma_phi_multimode(
        [0.001, 0.002],
        [2.0 * np.pi * 1.0e6, 2.0 * np.pi * 0.5e6],
        [2.0 * np.pi * 0.2e6, 2.0 * np.pi * 0.1e6],
        mode_names=["storage", "readout"],
        approximation="weak",
    )
    assert len(contributions) == 2
    assert total == pytest.approx(sum(item.gamma_phi_rad_s for item in contributions))


def test_lossless_filter_does_not_reduce_in_band_occupation() -> None:
    freq_hz = 7.5e9
    source_temp = 0.06
    chain = MicrowaveNoiseChain(
        [LosslessFilterStage("control-filter", freq_hz, 20.0e6, 40.0)],
        input_temperature_K=source_temp,
    )
    assert chain.propagate_nbar(freq_hz) == pytest.approx(n_bose(freq_hz, source_temp))
    off_band = chain.propagate_nbar(freq_hz + 100.0e6)
    assert off_band < n_bose(freq_hz + 100.0e6, source_temp)


def test_passive_loss_stage_matches_beamsplitter_formula() -> None:
    freq_hz = 6.0e9
    n_in = n_bose(freq_hz, 300.0)
    stage = PassiveLossStage("4K", 20.0, 4.0)
    eta = 0.01
    assert stage.propagate_nbar(freq_hz, n_in) == pytest.approx(eta * n_in + (1.0 - eta) * n_bose(freq_hz, 4.0))


def test_effective_cavity_attenuator_wang_ratio() -> None:
    external = 7.0e-3
    model = EffectiveCavityAttenuator(
        omega_ro_rad_s=2.0 * np.pi * 7.573e9,
        kappa_internal_rad_s=6.0,
        kappa_external_rad_s=1.0,
        internal_nbar=0.0,
        external_nbar=external,
    )
    assert model.effective_nbar() == pytest.approx(external / 7.0)
    assert required_internal_to_external_ratio(1.0e-4, 7.0e-4) == pytest.approx(6.0)


def test_two_mode_hybridization_and_effective_rates() -> None:
    model = TwoModeCavityAttenuatorModel(
        omega_readout_rad_s=2.0 * np.pi * 7.5e9,
        omega_attenuator_rad_s=2.0 * np.pi * 7.5e9,
        coupling_J_rad_s=2.0 * np.pi * 10.0e6,
        kappa_readout_internal_rad_s=2.0,
        kappa_attenuator_internal_rad_s=10.0,
        kappa_external_rad_s=4.0,
        chi_bare_readout_rad_s=3.0,
    )
    parts = model.participation_ratios()
    assert parts["readout"] == pytest.approx([0.5, 0.5], rel=1.0e-12)
    assert parts["attenuator"] == pytest.approx([0.5, 0.5], rel=1.0e-12)
    assert model.effective_chi_per_mode() == pytest.approx([1.5, 1.5])
    assert model.effective_kappa_per_mode() == pytest.approx([8.0, 8.0])


def test_noise_induced_dephasing_fit_recovers_synthetic_offset() -> None:
    nadd = np.linspace(0.0, 0.08, 9)
    result = simulate_noise_induced_dephasing(
        nadd,
        nth=2.0e-4,
        kappa_rad_s=2.0 * np.pi * 1.0e6,
        chi_rad_s=2.0 * np.pi * 1.2e6,
        T1_s=100e-6,
    )
    assert result.fitted_nth == pytest.approx(2.0e-4, abs=2.0e-7)
    assert result.fitted_slope_rad_s > 0.0


def test_design_sweep_returns_monotonic_nbar_improvement() -> None:
    sweep = sweep_cavity_attenuator_design(
        np.array([0.0, 1.0, 6.0]),
        omega_ro_rad_s=2.0 * np.pi * 7.573e9,
        kappa_external_rad_s=2.0 * np.pi * 1.9e6,
        external_nbar=7.0e-4,
        chi_rad_s=2.0 * np.pi * 1.2e6,
        T1_s=100e-6,
        internal_nbar=0.0,
    )
    assert np.all(np.diff(sweep["nbar_eff"]) < 0.0)
    assert sweep["nbar_eff"][-1] == pytest.approx(1.0e-4)


def test_small_lindblad_ramsey_dephasing_is_close_to_analytic_strong_limit() -> None:
    kappa = 0.8
    chi = 20.0
    nth = 0.02
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=chi,
        kerr=0.0,
        n_cav=4,
        n_tr=2,
    )
    times = np.linspace(0.0, 7.0, 41)
    result = simulate_ramsey_with_residual_photons(
        model,
        times,
        NoiseSpec(kappa=kappa, nth=nth),
        frame=FrameSpec(),
    )
    expected = gamma_phi_thermal(nth, kappa, chi)
    assert result.gamma_phi_rad_s == pytest.approx(expected, rel=0.45, abs=0.03)
