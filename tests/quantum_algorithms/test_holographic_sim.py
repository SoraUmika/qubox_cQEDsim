from __future__ import annotations

import json

import numpy as np
import pytest

from cqed_sim.quantum_algorithms.holographic_sim import (
    BondNoiseChannel,
    HoloQUADSProgram,
    HoloVQEObjective,
    HolographicChannel,
    HolographicSampler,
    ObservableSchedule,
    TimeSlice,
    BurnInConfig,
    MatrixProductState,
    channel_diagnostics,
    hadamard_reference_channel,
    partial_swap_channel,
    pauli_z,
)
from cqed_sim.quantum_algorithms.holographic_sim.holographicSim import holographic_sim_bfs


def test_channel_from_unitary_infers_dimensions() -> None:
    hadamard = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    unitary = np.kron(hadamard, np.eye(2, dtype=np.complex128))
    channel = HolographicChannel.from_unitary(unitary, physical_dim=2)
    diagnostics = channel_diagnostics(channel)
    assert channel.physical_dim == 2
    assert channel.bond_dim == 2
    assert diagnostics.kraus_completeness_error < 1.0e-10


def test_invalid_unitary_rejected() -> None:
    with pytest.raises(ValueError):
        HolographicChannel.from_unitary(np.ones((4, 4), dtype=np.complex128), physical_dim=2)


def test_projective_branch_normalization() -> None:
    channel = hadamard_reference_channel()
    branches = channel.joint_output_state(np.array([[1.0]], dtype=np.complex128))
    assert branches.shape == (2, 2)
    step = HolographicSampler(channel).enumerate_correlator(ObservableSchedule([{"step": 1, "operator": pauli_z()}]))
    assert abs(step.branch_probability_sum - 1.0) < 1.0e-12
    assert abs(step.normalization_error) < 1.0e-12


def test_monte_carlo_vs_exact_agree_on_tiny_example() -> None:
    channel = hadamard_reference_channel()
    schedule = ObservableSchedule(
        [
            {"step": 1, "operator": pauli_z()},
            {"step": 2, "operator": pauli_z()},
        ]
    )
    sampler = HolographicSampler(channel)
    exact = sampler.enumerate_correlator(schedule)
    estimate = sampler.sample_correlator(schedule, shots=20_000, seed=7)
    assert abs(exact.mean) < 1.0e-12
    assert abs(estimate.mean - exact.mean) <= 6.0 * estimate.stderr


def test_burn_in_moves_partial_swap_toward_ground_state() -> None:
    channel = partial_swap_channel(theta=0.35)
    left_state = np.array([0.0, 1.0], dtype=np.complex128)
    sampler = HolographicSampler(channel, left_state=left_state, burn_in=BurnInConfig(steps=8))
    summary = sampler.summarize_burn_in()
    assert summary.steps == 8
    assert float(np.real(summary.final_state[0, 0])) > 0.5


def test_schedule_accepts_observable_alias() -> None:
    schedule = ObservableSchedule([{"step": 3, "observable": pauli_z()}], total_steps=5)
    assert schedule.insertion_for_step(3) is not None
    assert schedule.total_steps == 5


def test_legacy_bfs_wrapper_normalizes_probabilities() -> None:
    hadamard = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    branches = holographic_sim_bfs([hadamard], [pauli_z().matrix], d=2)
    prob_sum = sum(float(branch["prob"]) for branch in branches)
    assert len(branches) == 2
    assert abs(prob_sum - 1.0) < 1.0e-12


def test_holo_vqe_scaffold_returns_serializable_result(tmp_path) -> None:
    channel = hadamard_reference_channel()
    sampler = HolographicSampler(channel)
    objective = HoloVQEObjective(
        [
            {
                "coefficient": 1.0,
                "schedule": ObservableSchedule([{"step": 1, "operator": pauli_z()}]),
                "label": "onsite_z",
            }
        ]
    )
    result = objective.estimate(sampler, exact=True)
    saved = result.save(tmp_path / "energy.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["exact"] is True
    assert "onsite_z" in payload["terms"]


def test_holoquads_program_combines_slices() -> None:
    program = HoloQUADSProgram(
        [
            TimeSlice(steps=2, insertions=[{"step": 1, "operator": pauli_z()}], label="prep"),
            TimeSlice(steps=3, insertions=[{"step": 2, "operator": pauli_z()}], label="measure"),
        ]
    )
    schedule = program.combined_schedule()
    assert schedule.total_steps == 5
    assert schedule.measured_steps == (1, 4)


def test_mps_channel_round_trip() -> None:
    product_state = np.zeros((2, 2), dtype=np.complex128)
    product_state[0, 0] = 1.0
    mps = MatrixProductState(product_state)
    mps.make_right_canonical(cast_complete=True)
    channel = mps.to_holographic_channel()
    diagnostics = channel_diagnostics(channel)
    assert diagnostics.right_canonical_error < 1.0e-10


def test_channel_from_mps_state_matches_manual_construction() -> None:
    product_state = np.zeros((2, 2, 2), dtype=np.complex128)
    product_state[0, 0, 0] = 1.0
    manual_mps = MatrixProductState(product_state)
    manual_mps.make_right_canonical(cast_complete=True)
    manual_channel = manual_mps.to_holographic_channel(site=1)
    direct_channel = HolographicChannel.from_mps_state(product_state, site=1)

    assert direct_channel.physical_dim == manual_channel.physical_dim
    assert direct_channel.bond_dim == manual_channel.bond_dim
    for expected, observed in zip(manual_channel.kraus_ops, direct_channel.kraus_ops):
        assert np.allclose(observed, expected, atol=1.0e-10, rtol=0.0)


def test_sampler_from_mps_state_matches_direct_channel_sampler() -> None:
    product_state = np.zeros((2, 2), dtype=np.complex128)
    product_state[0, 0] = 1.0
    schedule = ObservableSchedule([{"step": 1, "operator": pauli_z()}], total_steps=1)

    direct_sampler = HolographicSampler(HolographicChannel.from_mps_state(product_state, site=0))
    convenience_sampler = HolographicSampler.from_mps_state(product_state, site=0)

    exact_direct = direct_sampler.enumerate_correlator(schedule)
    exact_convenience = convenience_sampler.enumerate_correlator(schedule)
    assert exact_direct.mean == pytest.approx(exact_convenience.mean)
    assert exact_direct.variance == pytest.approx(exact_convenience.variance)


def test_bond_dephasing_damps_offdiagonal_coherence_and_threads_through_burn_in() -> None:
    plus_state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_plus = np.outer(plus_state, plus_state.conj())
    bond_noise = BondNoiseChannel.dephasing(bond_dim=2, probability=0.25)
    expected = np.array([[0.5, 0.375], [0.375, 0.5]], dtype=np.complex128)

    assert np.allclose(bond_noise.apply(rho_plus), expected, atol=1.0e-12, rtol=0.0)
    assert bond_noise.kraus_completeness_error() < 1.0e-12

    identity_channel = HolographicChannel.from_kraus([np.eye(2, dtype=np.complex128)], physical_dim=1, bond_dim=2)
    sampler = HolographicSampler(
        identity_channel,
        left_state=plus_state,
        burn_in=BurnInConfig(steps=1),
        bond_noise=bond_noise,
    )
    summary = sampler.summarize_burn_in()
    assert np.allclose(summary.final_state, expected, atol=1.0e-12, rtol=0.0)


def test_bond_amplitude_damping_matches_qubit_formula() -> None:
    probability = 0.4
    bond_noise = BondNoiseChannel.amplitude_damping(bond_dim=2, probability=probability)

    excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    expected_excited = np.array([[probability, 0.0], [0.0, 1.0 - probability]], dtype=np.complex128)
    assert np.allclose(bond_noise.apply(excited), expected_excited, atol=1.0e-12, rtol=0.0)

    plus_state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_plus = np.outer(plus_state, plus_state.conj())
    expected_plus = np.array(
        [
            [0.5 * (1.0 + probability), 0.5 * np.sqrt(1.0 - probability)],
            [0.5 * np.sqrt(1.0 - probability), 0.5 * (1.0 - probability)],
        ],
        dtype=np.complex128,
    )
    assert np.allclose(bond_noise.apply(rho_plus), expected_plus, atol=1.0e-12, rtol=0.0)
    assert bond_noise.kraus_completeness_error() < 1.0e-12


def test_bond_depolarizing_matches_analytic_formula() -> None:
    probability = 0.3
    bond_noise = BondNoiseChannel.depolarizing(bond_dim=3, probability=probability)

    rho = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    expected = (1.0 - probability) * rho + probability * np.eye(3, dtype=np.complex128) / 3.0
    assert np.allclose(bond_noise.apply(rho), expected, atol=1.0e-12, rtol=0.0)
    assert bond_noise.kraus_completeness_error() < 1.0e-12


def test_bond_amplitude_damping_from_t1_metadata_matches_probability() -> None:
    duration = 2.5
    t1 = 10.0
    probability = 1.0 - np.exp(-duration / t1)
    from_t1 = BondNoiseChannel.amplitude_damping(bond_dim=2, duration=duration, t1=t1)
    direct = BondNoiseChannel.amplitude_damping(bond_dim=2, probability=probability)

    excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    assert np.allclose(from_t1.apply(excited), direct.apply(excited), atol=1.0e-12, rtol=0.0)
    assert from_t1.metadata["probability"] == pytest.approx(probability)


def test_bond_noise_from_qutip_super_matches_analytic_dephasing() -> None:
    qt = pytest.importorskip("qutip")

    probability = 0.3
    qeye = qt.qeye(2)
    projectors = [qt.basis(2, idx) * qt.basis(2, idx).dag() for idx in range(2)]
    superoperator = qt.kraus_to_super([np.sqrt(1.0 - probability) * qeye, *(np.sqrt(probability) * proj for proj in projectors)])
    from_super = BondNoiseChannel.from_qutip_super(superoperator)
    analytic = BondNoiseChannel.dephasing(bond_dim=2, probability=probability)

    plus_state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_plus = np.outer(plus_state, plus_state.conj())
    assert np.allclose(from_super.apply(rho_plus), analytic.apply(rho_plus), atol=1.0e-12, rtol=0.0)
    assert from_super.kraus_completeness_error() < 1.0e-10
    assert np.allclose(
        np.asarray(from_super.to_qutip_super().full(), dtype=np.complex128),
        np.asarray(superoperator.full(), dtype=np.complex128),
        atol=1.0e-10,
        rtol=0.0,
    )
