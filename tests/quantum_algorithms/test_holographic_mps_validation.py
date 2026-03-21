from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.quantum_algorithms.holographic_sim import (
    HolographicChannel,
    MatrixProductState,
    contract_mps,
    pauli_x,
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


def _state_cases() -> list[pytest.ParamSpecArgs]:
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    one = np.array([0.0, 1.0], dtype=np.complex128)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return [
        pytest.param("product_0000", _product_state([zero, zero, zero, zero]), id="product_0000"),
        pytest.param("product_+0+1", _product_state([plus, zero, plus, one]), id="product_+0+1"),
        pytest.param("ghz4", _ghz_state(4), id="ghz4"),
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