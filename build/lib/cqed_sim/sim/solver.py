from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import qutip as qt

from cqed_sim.backends.base_backend import BaseBackend
from cqed_sim.backends.numpy_backend import NumPyBackend


@dataclass
class DenseSolverResult:
    final_state: qt.Qobj
    states: list[qt.Qobj]
    expect: list[np.ndarray]
    backend_name: str


def _operator_to_dense(operator: qt.Qobj) -> np.ndarray:
    return np.asarray(operator.full(), dtype=np.complex128)


def _coeff_samples(coeff, tlist: np.ndarray) -> np.ndarray:
    if np.isscalar(coeff):
        return np.full(tlist.shape, complex(coeff), dtype=np.complex128)
    if callable(coeff):
        return np.asarray(coeff(tlist), dtype=np.complex128)
    values = np.asarray(coeff, dtype=np.complex128)
    if values.shape != tlist.shape:
        raise ValueError(f"Time-dependent coefficient shape {values.shape} does not match tlist shape {tlist.shape}.")
    return values


def _dense_hamiltonian_samples(hamiltonian: list, tlist: np.ndarray) -> list[np.ndarray]:
    samples = [_operator_to_dense(hamiltonian[0]).copy() for _ in range(tlist.size)]
    for operator, coeff in hamiltonian[1:]:
        operator_dense = _operator_to_dense(operator)
        coeffs = _coeff_samples(coeff, tlist)
        for idx, value in enumerate(coeffs):
            samples[idx] = samples[idx] + value * operator_dense
    return samples


def _vectorize_density_matrix(backend: BaseBackend, rho):
    return backend.reshape(np.asarray(backend.to_numpy(rho).T), (-1,))


def _devectorize_density_matrix(backend: BaseBackend, rho_vec, dim: int):
    return backend.reshape(rho_vec, (dim, dim)).T


def _as_qobj(state, template: qt.Qobj, is_density_matrix: bool) -> qt.Qobj:
    dense = np.asarray(state, dtype=np.complex128)
    if is_density_matrix:
        return qt.Qobj(dense, dims=template.dims)
    return qt.Qobj(dense.reshape((-1, 1)), dims=template.dims)


def solve_with_backend(
    hamiltonian: list,
    tlist: np.ndarray,
    initial_state: qt.Qobj,
    *,
    observables: Sequence[qt.Qobj] = (),
    collapse_ops: Sequence[qt.Qobj] = (),
    backend: BaseBackend | None = None,
    store_states: bool = False,
) -> DenseSolverResult:
    backend = NumPyBackend() if backend is None else backend
    dense_observables = [_operator_to_dense(operator) for operator in observables]
    dense_hamiltonians = _dense_hamiltonian_samples(hamiltonian, np.asarray(tlist, dtype=float))
    dense_collapse_ops = [_operator_to_dense(operator) for operator in collapse_ops]

    is_density_matrix = bool(initial_state.isoper or dense_collapse_ops)
    state_template = initial_state if not is_density_matrix else (initial_state if initial_state.isoper else initial_state.proj())
    if is_density_matrix:
        state = backend.asarray(initial_state.full() if initial_state.isoper else initial_state.proj().full())
    else:
        state = backend.asarray(np.asarray(initial_state.full(), dtype=np.complex128).ravel())

    states: list[qt.Qobj] = []
    expectations = [[] for _ in dense_observables]

    def record(current_state) -> None:
        state_qobj = _as_qobj(backend.to_numpy(current_state), state_template, is_density_matrix)
        if store_states:
            states.append(state_qobj)
        for idx, operator in enumerate(dense_observables):
            expectations[idx].append(complex(backend.expectation(operator, current_state)))

    record(state)
    for idx in range(len(tlist) - 1):
        dt = float(tlist[idx + 1] - tlist[idx])
        if dt <= 0.0:
            continue
        h_step = backend.asarray(dense_hamiltonians[idx])
        if is_density_matrix:
            liouvillian = backend.lindbladian(h_step, [backend.asarray(operator) for operator in dense_collapse_ops])
            propagator = backend.expm(liouvillian * dt)
            state_vec = backend.matmul(propagator, _vectorize_density_matrix(backend, state))
            state = _devectorize_density_matrix(backend, state_vec, h_step.shape[0])
        else:
            propagator = backend.expm((-1j * h_step) * dt)
            state = backend.matmul(propagator, state)
        record(state)

    final_state = _as_qobj(backend.to_numpy(state), state_template, is_density_matrix)
    return DenseSolverResult(
        final_state=final_state,
        states=states if store_states else [],
        expect=[np.asarray(values) for values in expectations],
        backend_name=backend.name,
    )


__all__ = ["DenseSolverResult", "solve_with_backend"]
