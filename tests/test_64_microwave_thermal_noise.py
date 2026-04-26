from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.microwave_noise import (
    DistributedLine,
    DirectionalLoss,
    NoiseCascade,
    PassiveLoss,
    PassiveSMatrixComponent,
    bose_occupation,
    loss_db_to_power_transmission,
    occupation_to_effective_temperature,
    qubit_thermal_rates,
    resonator_lindblad_rates,
    resonator_thermal_occupation,
    sym_noise_temperature,
    thermal_photon_dephasing,
)


def test_bose_occupation_at_6ghz_regression_values() -> None:
    freq_hz = 6.0e9
    expected = {
        300.0: 1.041331036e3,
        4.0: 1.339707795e1,
        0.7: 1.965122911,
        0.1: 5.950190518e-2,
        0.06: 8.304373364e-3,
        0.02: 5.586578304e-7,
    }
    for temp_K, occupation in expected.items():
        assert bose_occupation(freq_hz, temp_K) == pytest.approx(occupation, rel=2.0e-10)

    vector = bose_occupation(np.array([freq_hz, freq_hz]), 0.0)
    assert np.all(vector == 0.0)


def test_temperature_and_symmetrized_noise_helpers() -> None:
    freq_hz = np.array([6.0e9, 7.0e9])
    n = bose_occupation(freq_hz, 0.1)
    assert occupation_to_effective_temperature(freq_hz, n) == pytest.approx(np.array([0.1, 0.1]))
    assert occupation_to_effective_temperature(6.0e9, 0.0) == 0.0
    assert loss_db_to_power_transmission(20.0) == pytest.approx(0.01)
    assert sym_noise_temperature(6.0e9, 0.0) > 0.0


def test_single_attenuator_limits() -> None:
    freq_hz = 6.0e9
    n_in = bose_occupation(freq_hz, 300.0)
    assert PassiveLoss("through", temp_K=4.0, loss_db=0.0).propagate(freq_hz, n_in)[0] == pytest.approx(n_in)

    thermal = bose_occupation(freq_hz, 4.0)
    assert PassiveLoss("thermalizer", temp_K=4.0, loss_db=300.0).propagate(freq_hz, n_in)[0] == pytest.approx(
        thermal
    )
    for loss_db in [0.1, 3.0, 20.0, 80.0]:
        assert PassiveLoss("equilibrium", temp_K=4.0, loss_db=loss_db).propagate(freq_hz, thermal)[0] == pytest.approx(
            thermal
        )


def test_20db_attenuator_at_4k_fed_by_300k_source() -> None:
    freq_hz = 6.0e9
    eta = 0.01
    expected = eta * bose_occupation(freq_hz, 300.0) + (1.0 - eta) * bose_occupation(freq_hz, 4.0)
    out = PassiveLoss("4K", temp_K=4.0, loss_db=20.0).propagate(freq_hz, bose_occupation(freq_hz, 300.0))[0]
    assert out == pytest.approx(expected, rel=1.0e-12)
    assert out == pytest.approx(23.67641753, abs=5.0e-9)


def test_two_attenuators_same_temperature_combine() -> None:
    freq_hz = 6.0e9
    n_in = bose_occupation(freq_hz, 300.0)
    temp_K = 0.7
    n_T = bose_occupation(freq_hz, temp_K)
    first = PassiveLoss("a", temp_K=temp_K, loss_db=7.0)
    second = PassiveLoss("b", temp_K=temp_K, loss_db=13.0)
    cascaded = second.propagate(freq_hz, first.propagate(freq_hz, n_in)[0])[0]
    eta_total = loss_db_to_power_transmission(20.0)
    expected = eta_total * n_in + (1.0 - eta_total) * n_T
    assert cascaded == pytest.approx(expected, rel=1.0e-12)


def test_previous_fridge_chain_regression_and_budget() -> None:
    freq_hz = 6.0e9
    cascade = NoiseCascade(
        [
            PassiveLoss("4K", temp_K=4.0, loss_db=20.0),
            PassiveLoss("still", temp_K=0.7, loss_db=10.0),
            PassiveLoss("100mK", temp_K=0.1, loss_db=0.01),
            PassiveLoss("MXC", temp_K=0.06, loss_db=40.0),
            PassiveLoss("MXC2", temp_K=0.02, loss_db=0.01),
        ]
    )
    result = cascade.propagate(freq_hz, source_temp_K=300.0)
    assert result.n_out == pytest.approx(8.6961850486e-3, rel=1.0e-12)
    assert result.effective_temperature == pytest.approx(6.0577e-2, rel=2.0e-5)

    expected_weights = {
        "source": 9.95405417e-8,
        "4K": 9.85451363e-6,
        "still": 8.95864876e-5,
        "100mK": 2.29464647e-7,
        "MXC": 9.97600294e-1,
        "MXC2": 2.29993618e-3,
    }
    expected_contributions = {
        "source": 1.03654655e-4,
        "4K": 1.32021687e-4,
        "still": 1.76048459e-4,
        "100mK": 1.36535837e-8,
        "MXC": 8.28444531e-3,
        "MXC2": 1.28487735e-9,
    }
    for label, expected in expected_weights.items():
        assert result.budget.weights[label] == pytest.approx(expected, rel=1.0e-8)
    for label, expected in expected_contributions.items():
        assert result.budget.contributions[label] == pytest.approx(expected, rel=1.0e-8)
    assert result.budget.weight_sum == pytest.approx(1.0, abs=1.0e-15)
    assert result.budget.contribution_sum == pytest.approx(result.n_out, rel=1.0e-15, abs=1.0e-15)


def test_distributed_line_constant_parameters_matches_closed_form() -> None:
    freq_hz = 6.0e9
    n_in = bose_occupation(freq_hz, 300.0)
    line = DistributedLine(
        "line",
        length_m=2.0,
        attenuation_db_per_m=0.5,
        temperature_K=0.7,
        num_slices=200,
    )
    out, trace = line.propagate(freq_hz, n_in)
    alpha = np.log(10.0) / 10.0 * 0.5
    eta = np.exp(-alpha * 2.0)
    expected = eta * n_in + (1.0 - eta) * bose_occupation(freq_hz, 0.7)
    assert out == pytest.approx(expected, rel=1.0e-12)
    assert trace.eta == pytest.approx(eta, rel=1.0e-12)


def test_distributed_line_varying_temperature_converges() -> None:
    freq_hz = 6.0e9

    def temp_profile(z_m: float) -> float:
        return 4.0 + (0.02 - 4.0) * z_m / 3.0

    coarse = DistributedLine("coarse", 3.0, 0.25, temp_profile, num_slices=100).propagate(freq_hz, 0.0)[0]
    fine = DistributedLine("fine", 3.0, 0.25, temp_profile, num_slices=1000).propagate(freq_hz, 0.0)[0]
    finer = DistributedLine("finer", 3.0, 0.25, temp_profile, num_slices=2000).propagate(freq_hz, 0.0)[0]
    assert fine == pytest.approx(finer, rel=2.0e-5)
    assert coarse == pytest.approx(fine, rel=2.0e-3)


def test_directional_loss_uses_forward_or_reverse_path() -> None:
    freq_hz = 6.0e9
    component = DirectionalLoss("iso", temp_K=0.06, forward_loss_db=0.5, reverse_isolation_db=20.0)
    forward = component.propagate(freq_hz, 1.0, direction="forward")[0]
    reverse = component.propagate(freq_hz, 1.0, direction="reverse")[0]
    assert forward > reverse


def test_passive_smatrix_scalar_matches_passive_loss() -> None:
    freq_hz = 6.0e9
    eta = 0.01
    n_in = bose_occupation(freq_hz, 300.0)
    temp_K = 4.0
    passive = PassiveSMatrixComponent(
        "scalar",
        temp_K=temp_K,
        S_matrix=lambda _freq: np.array([[np.sqrt(eta)]], dtype=complex),
    )
    c_out, trace = passive.propagate_covariance(freq_hz, np.array([[n_in]], dtype=complex))
    loss_out = PassiveLoss("loss", temp_K=temp_K, loss_db=20.0).propagate(freq_hz, n_in)[0]
    assert c_out[0, 0].real == pytest.approx(loss_out, rel=1.0e-12)
    assert trace.loss_covariance_eigenvalues[0] == pytest.approx(1.0 - eta)


def test_unitary_smatrix_adds_no_thermal_noise() -> None:
    freq_hz = 6.0e9
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    component = PassiveSMatrixComponent("swap", temp_K=300.0, S_matrix=lambda _freq: swap)
    c_in = np.diag([0.1, 0.3]).astype(complex)
    c_out, trace = component.propagate_covariance(freq_hz, c_in)
    assert c_out == pytest.approx(swap @ c_in @ swap.conj().T)
    assert trace.loss_covariance == pytest.approx(np.zeros((2, 2)))


def test_smatrix_rejects_active_nonpassive_matrix() -> None:
    component = PassiveSMatrixComponent(
        "gain",
        temp_K=4.0,
        S_matrix=lambda _freq: np.array([[1.01]], dtype=complex),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        component.propagate_covariance(6.0e9, np.array([[0.0]], dtype=complex))


def test_resonator_thermal_occupation() -> None:
    assert resonator_thermal_occupation([1.0, 4.0], [0.01, 0.001]) == pytest.approx(0.0028)
    assert resonator_lindblad_rates(5.0, 0.2) == pytest.approx((6.0, 1.0))


def test_qubit_detailed_balance() -> None:
    freq_hz = 6.0e9
    temp_K = 0.1
    n = bose_occupation(freq_hz, temp_K)
    gamma_down, gamma_up, gamma_1 = qubit_thermal_rates(100.0, n)
    assert gamma_up / gamma_down == pytest.approx(np.exp(-6.62607015e-34 * freq_hz / (1.380649e-23 * temp_K)))
    assert gamma_1 == pytest.approx(gamma_down + gamma_up)


def test_thermal_photon_dephasing_exact_and_approximations() -> None:
    kappa = 2.0 * np.pi * 5.0e6
    assert thermal_photon_dephasing(kappa, 2.0 * np.pi * 1.0e6, 0.0) == pytest.approx(0.0)

    weak_chi = 1.0e-3 * kappa
    n_cav = 0.7
    exact_weak = thermal_photon_dephasing(kappa, weak_chi, n_cav)
    approx_weak = thermal_photon_dephasing(kappa, weak_chi, n_cav, approximation="weak")
    assert exact_weak == pytest.approx(approx_weak, rel=3.0e-5)

    strong_chi = 100.0 * kappa
    low_n = 1.0e-4
    exact_strong = thermal_photon_dephasing(kappa, strong_chi, low_n)
    approx_strong = thermal_photon_dephasing(kappa, strong_chi, low_n, approximation="strong_low_occupation")
    assert exact_strong == pytest.approx(approx_strong, rel=3.0e-5)

    strong_estimate = thermal_photon_dephasing(
        kappa,
        strong_chi,
        8.696185e-3,
        approximation="strong_low_occupation",
    )
    assert strong_estimate / (2.0 * np.pi) == pytest.approx(43.48e3, rel=3.0e-4)
