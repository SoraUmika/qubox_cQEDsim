from __future__ import annotations

import json

import numpy as np
import pytest

from cqed_sim.quantum_algorithms.holographic_sim import (
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
