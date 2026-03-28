from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.quantum_algorithms.holographic_sim import (
    HolographicChannel,
    HolographicSampler,
    MatrixProductState,
    contract_mps,
    pauli_x,
    ObservableSchedule,
    pauli_z,
    right_canonical_tensor_to_stinespring_unitary,
)
from cqed_sim.quantum_algorithms.holographic_sim.holographicSim import holographic_sim_bfs


ATOL = 1.0e-10


def _product_state(local_states: list[np.ndarray]) -> np.ndarray:
    state = np.asarray(local_states[0], dtype=np.complex128).reshape(-1)
    physical_dim = int(state.size)
    for local_state in local_states[1:]:
        vec = np.asarray(local_state, dtype=np.complex128).reshape(-1)
        if vec.size != physical_dim:
            raise ValueError("All local states must have the same physical dimension.")
        state = np.kron(state, vec)
    return state.reshape((physical_dim,) * len(local_states))


def _ghz_state(num_sites: int) -> np.ndarray:
    state = np.zeros(2**int(num_sites), dtype=np.complex128)
    state[0] = 1.0
    state[-1] = 1.0
    state /= np.linalg.norm(state)
    return state.reshape((2,) * int(num_sites))


def _w_state(num_sites: int) -> np.ndarray:
    state = np.zeros(2**int(num_sites), dtype=np.complex128)
    for site in range(int(num_sites)):
        state[1 << (int(num_sites) - 1 - site)] = 1.0
    state /= np.linalg.norm(state)
    return state.reshape((2,) * int(num_sites))


def _random_state(num_sites: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    state = rng.standard_normal(2**int(num_sites)) + 1j * rng.standard_normal(2**int(num_sites))
    state /= np.linalg.norm(state)
    return state.reshape((2,) * int(num_sites))


def _hadamard() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def _controlled_z() -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    full = np.array([[1.0 + 0.0j]])
    for idx in range(num_sites):
        full = np.kron(full, gate if idx == site else np.eye(2, dtype=np.complex128))
    return (full @ state.reshape(-1)).reshape((2,) * num_sites)


def _apply_adjacent_two_qubit_gate(state: np.ndarray, gate: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    tensor = state.reshape((2,) * num_sites)
    perm = [site, site + 1] + [idx for idx in range(num_sites) if idx not in (site, site + 1)]
    inverse = np.argsort(perm)
    permuted = np.transpose(tensor, perm)
    leading = permuted.reshape(4, -1)
    updated = (gate @ leading).reshape((2, 2) + tuple(2 for _ in range(num_sites - 2)))
    return np.transpose(updated, inverse)


def _cluster_state(num_sites: int) -> np.ndarray:
    state = np.zeros((2,) * num_sites, dtype=np.complex128)
    state[(0,) * num_sites] = 1.0
    for site in range(num_sites):
        state = _apply_single_qubit_gate(state, _hadamard(), site, num_sites)
    for site in range(num_sites - 1):
        state = _apply_adjacent_two_qubit_gate(state, _controlled_z(), site, num_sites)
    return state / np.linalg.norm(state)


def _cluster_stabilizer_operator_map(site: int, num_sites: int) -> dict[int, np.ndarray]:
    operator_map: dict[int, np.ndarray] = {int(site): pauli_x().matrix}
    if int(site) > 0:
        operator_map[int(site) - 1] = pauli_z().matrix
    if int(site) + 1 < int(num_sites):
        operator_map[int(site) + 1] = pauli_z().matrix
    return operator_map


def _state_cases() -> list[pytest.ParamSpecArgs]:
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    one = np.array([0.0, 1.0], dtype=np.complex128)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return [
        pytest.param("product_0000", _product_state([zero, zero, zero, zero]), id="product_0000"),
        pytest.param("product_+0+1", _product_state([plus, zero, plus, one]), id="product_+0+1"),
        pytest.param("ghz4", _ghz_state(4), id="ghz4"),
        pytest.param("cluster4", _cluster_state(4), id="cluster4"),
        pytest.param("w4", _w_state(4), id="w4"),
        pytest.param("random4", _random_state(4, seed=123), id="random4"),
    ]


OBSERVABLE_CASES: dict[str, dict[int, np.ndarray]] = {
    "Z0": {0: pauli_z().matrix},
    "Z3": {3: pauli_z().matrix},
    "ZZ01": {0: pauli_z().matrix, 1: pauli_z().matrix},
    "XX01": {0: pauli_x().matrix, 1: pauli_x().matrix},
    "X2": {2: pauli_x().matrix},
    "X0Z3": {0: pauli_x().matrix, 3: pauli_z().matrix},
}


def _dense_expectation(state: np.ndarray, operator_map: dict[int, np.ndarray]) -> complex:
    full_operator = np.array([[1.0 + 0.0j]])
    identity = np.eye(int(state.shape[0]), dtype=np.complex128)
    for site in range(int(state.ndim)):
        full_operator = np.kron(full_operator, operator_map.get(site, identity))
    flat_state = state.reshape(-1)
    return complex(np.vdot(flat_state, full_operator @ flat_state))


def _holographic_expectation(mps: MatrixProductState, operator_map: dict[int, np.ndarray]) -> complex:
    unitaries = [right_canonical_tensor_to_stinespring_unitary(mps.site_tensor(site, complete=True)) for site in range(mps.num_sites)]
    operator_list: list[np.ndarray | None] = [None] * mps.num_sites
    for site, operator in operator_map.items():
        operator_list[int(site)] = operator
    branches = holographic_sim_bfs(unitaries, operator_list, d=mps.physical_dim)
    probability_sum = sum(float(branch["prob"]) for branch in branches)
    assert probability_sum == pytest.approx(1.0, abs=ATOL)
    return sum(complex(branch["prob"]) * complex(branch["weight"]) for branch in branches)


def _schedule_from_operator_map(operator_map: dict[int, np.ndarray], *, total_steps: int) -> ObservableSchedule:
    return ObservableSchedule(
        [{"step": int(site) + 1, "operator": operator} for site, operator in sorted(operator_map.items())],
        total_steps=total_steps,
    )


@pytest.mark.parametrize(("state_name", "state"), _state_cases())
def test_right_canonical_tensor_stinespring_completion_matches_channel(state_name: str, state: np.ndarray) -> None:
    mps = MatrixProductState(state)
    mps.make_right_canonical(cast_complete=True)
    assert mps.tensors is not None
    assert mps.chi is not None
    reconstructed = contract_mps(mps.tensors)
    assert np.allclose(reconstructed, state, atol=ATOL, rtol=0.0), state_name

    for site in range(mps.num_sites):
        tensor = mps.site_tensor(site, complete=True)
        unitary = right_canonical_tensor_to_stinespring_unitary(tensor)
        channel_from_tensor = HolographicChannel.from_right_canonical_mps(tensor, label=f"{state_name}_site_{site}")
        channel_from_matrices = HolographicChannel.from_right_canonical_mps(
            tuple(tensor[:, outcome, :] for outcome in range(mps.physical_dim)),
            label=f"{state_name}_site_{site}_matrices",
        )
        channel_from_unitary = HolographicChannel.from_unitary(
            unitary,
            physical_dim=mps.physical_dim,
            bond_dim=mps.chi,
            label=f"{state_name}_site_{site}_unitary",
        )

        assert channel_from_tensor.kraus_completeness_error() < ATOL
        assert channel_from_tensor.right_canonical_error() < ATOL
        assert channel_from_matrices.kraus_completeness_error() < ATOL
        assert channel_from_matrices.right_canonical_error() < ATOL
        assert channel_from_unitary.kraus_completeness_error() < ATOL
        assert channel_from_unitary.right_canonical_error() < ATOL
        for expected, observed in zip(channel_from_tensor.kraus_ops, channel_from_unitary.kraus_ops):
            assert np.allclose(observed, expected, atol=ATOL, rtol=0.0), f"{state_name}: site={site}"
        for expected, observed in zip(channel_from_tensor.kraus_ops, channel_from_matrices.kraus_ops):
            assert np.allclose(observed, expected, atol=ATOL, rtol=0.0), f"{state_name}: matrix_sequence site={site}"


def test_matrix_product_state_site_stinespring_unitary_matches_public_helper() -> None:
    mps = MatrixProductState(_ghz_state(4))
    mps.make_right_canonical(cast_complete=True)

    for site in range(mps.num_sites):
        expected = right_canonical_tensor_to_stinespring_unitary(mps.site_tensor(site, complete=True))
        observed = mps.site_stinespring_unitary(site)
        assert np.allclose(observed, expected, atol=ATOL, rtol=0.0)


def test_public_stinespring_helper_accepts_matrix_sequences() -> None:
    mps = MatrixProductState(_ghz_state(4))
    mps.make_right_canonical(cast_complete=True)
    tensor = mps.site_tensor(1, complete=True)
    matrices = tuple(tensor[:, outcome, :] for outcome in range(mps.physical_dim))

    expected = right_canonical_tensor_to_stinespring_unitary(tensor)
    observed = right_canonical_tensor_to_stinespring_unitary(matrices)
    assert np.allclose(observed, expected, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize(("state_name", "state"), _state_cases())
def test_public_mps_sequence_sampler_matches_dense_expectations(state_name: str, state: np.ndarray) -> None:
    mps = MatrixProductState(state)
    mps.make_right_canonical(cast_complete=True)
    sampler = HolographicSampler.from_mps_sequence(state)

    for observable_name, operator_map in OBSERVABLE_CASES.items():
        dense_expectation = _dense_expectation(state, operator_map)
        exact = sampler.enumerate_correlator(_schedule_from_operator_map(operator_map, total_steps=mps.num_sites))
        assert np.allclose(exact.mean, dense_expectation, atol=ATOL, rtol=0.0), (
            f"{state_name}: public sequence sampler mismatch for {observable_name}"
        )


@pytest.mark.parametrize(("state_name", "state"), _state_cases())
def test_holographic_expectations_match_dense_known_states(state_name: str, state: np.ndarray) -> None:
    mps = MatrixProductState(state)
    mps.make_right_canonical(cast_complete=True)

    for observable_name, operator_map in OBSERVABLE_CASES.items():
        dense_expectation = _dense_expectation(state, operator_map)
        mps_expectation = mps.expect_operator_product(tuple(operator_map.items()))
        holographic_expectation = _holographic_expectation(mps, operator_map)

        assert np.allclose(mps_expectation, dense_expectation, atol=ATOL, rtol=0.0), (
            f"{state_name}: MPS expectation mismatch for {observable_name}"
        )
        assert np.allclose(holographic_expectation, dense_expectation, atol=ATOL, rtol=0.0), (
            f"{state_name}: holographic expectation mismatch for {observable_name}"
        )


def test_named_entangled_states_match_known_correlators() -> None:
    ghz = _ghz_state(4)
    ghz_sampler = HolographicSampler.from_mps_sequence(ghz)
    ghz_cases = {
        "Z0": ({0: pauli_z().matrix}, 0.0),
        "Z3": ({3: pauli_z().matrix}, 0.0),
        "Z0Z3": ({0: pauli_z().matrix, 3: pauli_z().matrix}, 1.0),
        "X0X1X2X3": ({0: pauli_x().matrix, 1: pauli_x().matrix, 2: pauli_x().matrix, 3: pauli_x().matrix}, 1.0),
    }
    for operator_map, expected in ghz_cases.values():
        exact = ghz_sampler.enumerate_correlator(_schedule_from_operator_map(operator_map, total_steps=4))
        assert np.allclose(exact.mean, expected, atol=ATOL, rtol=0.0)

    cluster = _cluster_state(4)
    cluster_sampler = HolographicSampler.from_mps_sequence(cluster)
    cluster_cases = {
        "X0Z1": {0: pauli_x().matrix, 1: pauli_z().matrix},
        "Z0X1Z2": {0: pauli_z().matrix, 1: pauli_x().matrix, 2: pauli_z().matrix},
        "Z1X2Z3": {1: pauli_z().matrix, 2: pauli_x().matrix, 3: pauli_z().matrix},
        "Z2X3": {2: pauli_z().matrix, 3: pauli_x().matrix},
    }
    for operator_map in cluster_cases.values():
        exact = cluster_sampler.enumerate_correlator(_schedule_from_operator_map(operator_map, total_steps=4))
        assert np.allclose(exact.mean, 1.0, atol=ATOL, rtol=0.0)


def test_ten_site_named_entangled_state_profiles_match_known_values() -> None:
    identity = np.eye(2, dtype=np.complex128)
    num_sites = 10

    ghz = _ghz_state(num_sites)
    ghz_sampler = HolographicSampler.from_mps_sequence(ghz)
    for site in range(num_sites):
        one_point = {int(site): pauli_z().matrix}
        one_point_exact = ghz_sampler.enumerate_correlator(_schedule_from_operator_map(one_point, total_steps=num_sites))
        assert np.allclose(one_point_exact.mean, 0.0, atol=ATOL, rtol=0.0)

        parity = {0: identity} if int(site) == 0 else {0: pauli_z().matrix, int(site): pauli_z().matrix}
        parity_exact = ghz_sampler.enumerate_correlator(_schedule_from_operator_map(parity, total_steps=num_sites))
        assert np.allclose(parity_exact.mean, 1.0, atol=ATOL, rtol=0.0)

    cluster = _cluster_state(num_sites)
    cluster_sampler = HolographicSampler.from_mps_sequence(cluster)
    for site in range(num_sites):
        stabilizer = _cluster_stabilizer_operator_map(int(site), num_sites)
        stabilizer_exact = cluster_sampler.enumerate_correlator(_schedule_from_operator_map(stabilizer, total_steps=num_sites))
        assert np.allclose(stabilizer_exact.mean, 1.0, atol=ATOL, rtol=0.0)