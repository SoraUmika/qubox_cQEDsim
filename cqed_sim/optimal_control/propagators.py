from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import qutip as qt
from scipy.linalg import expm, expm_frechet


@dataclass
class PropagationData:
    step_durations_s: np.ndarray
    slice_hamiltonians: tuple[np.ndarray, ...]
    slice_unitaries: tuple[np.ndarray, ...]
    forward_unitaries: tuple[np.ndarray, ...]
    derivative_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    @property
    def final_unitary(self) -> np.ndarray:
        return np.asarray(self.forward_unitaries[-1], dtype=np.complex128)

    def slice_derivative(self, step_index: int, control_index: int, control_operator: np.ndarray) -> np.ndarray:
        key = (int(step_index), int(control_index))
        cached = self.derivative_cache.get(key)
        if cached is not None:
            return cached
        h_step = self.slice_hamiltonians[int(step_index)]
        dt_s = float(self.step_durations_s[int(step_index)])
        derivative = expm_frechet(
            -1j * dt_s * h_step,
            -1j * dt_s * np.asarray(control_operator, dtype=np.complex128),
            compute_expm=False,
        )
        self.derivative_cache[key] = derivative
        return derivative


@dataclass
class LindbladPropagationData:
    step_durations_s: np.ndarray
    control_liouvillians: tuple[np.ndarray, ...]
    slice_liouvillians: tuple[np.ndarray, ...]
    slice_superoperators: tuple[np.ndarray, ...]
    forward_superoperators: tuple[np.ndarray, ...]
    derivative_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    @property
    def final_superoperator(self) -> np.ndarray:
        return np.asarray(self.forward_superoperators[-1], dtype=np.complex128)

    def slice_derivative(self, step_index: int, control_index: int) -> np.ndarray:
        key = (int(step_index), int(control_index))
        cached = self.derivative_cache.get(key)
        if cached is not None:
            return cached
        l_step = self.slice_liouvillians[int(step_index)]
        dl_step = self.control_liouvillians[int(control_index)]
        dt_s = float(self.step_durations_s[int(step_index)])
        derivative = expm_frechet(
            dt_s * l_step,
            dt_s * dl_step,
            compute_expm=False,
        )
        self.derivative_cache[key] = derivative
        return derivative


def _dense_liouvillian(hamiltonian: np.ndarray, collapse_ops: tuple[np.ndarray, ...]) -> np.ndarray:
    h_qobj = qt.Qobj(np.asarray(hamiltonian, dtype=np.complex128))
    c_qobjs = [qt.Qobj(np.asarray(operator, dtype=np.complex128)) for operator in collapse_ops]
    return np.asarray(qt.liouvillian(h_qobj, c_qobjs).full(), dtype=np.complex128)


def _vectorize_density_batch(densities: np.ndarray) -> np.ndarray:
    matrices = np.asarray(densities, dtype=np.complex128)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Density batches must have shape (n_states, dim, dim).")
    return np.transpose(matrices, (0, 2, 1)).reshape(matrices.shape[0], -1)


def devectorize_density_batch(vectors: np.ndarray, dim: int) -> np.ndarray:
    states = np.asarray(vectors, dtype=np.complex128)
    return np.transpose(states.reshape(states.shape[0], dim, dim), (0, 2, 1))


def build_propagation_data(
    drift_hamiltonian: np.ndarray,
    control_operators: tuple[np.ndarray, ...],
    control_values: np.ndarray,
    step_durations_s: np.ndarray,
) -> PropagationData:
    drift = np.asarray(drift_hamiltonian, dtype=np.complex128)
    values = np.asarray(control_values, dtype=float)
    durations = np.asarray(step_durations_s, dtype=float)
    dim = int(drift.shape[0])
    identity = np.eye(dim, dtype=np.complex128)

    slice_hamiltonians: list[np.ndarray] = []
    slice_unitaries: list[np.ndarray] = []
    forward_unitaries: list[np.ndarray] = [identity]

    for step_index in range(values.shape[1]):
        h_step = drift.copy()
        for control_index, operator in enumerate(control_operators):
            h_step = h_step + float(values[control_index, step_index]) * np.asarray(operator, dtype=np.complex128)
        u_step = expm(-1j * float(durations[step_index]) * h_step)
        slice_hamiltonians.append(h_step)
        slice_unitaries.append(u_step)
        forward_unitaries.append(u_step @ forward_unitaries[-1])

    return PropagationData(
        step_durations_s=durations,
        slice_hamiltonians=tuple(slice_hamiltonians),
        slice_unitaries=tuple(slice_unitaries),
        forward_unitaries=tuple(forward_unitaries),
    )


def build_lindblad_propagation_data(
    drift_hamiltonian: np.ndarray,
    control_operators: tuple[np.ndarray, ...],
    collapse_operators: tuple[np.ndarray, ...],
    control_values: np.ndarray,
    step_durations_s: np.ndarray,
) -> LindbladPropagationData:
    drift = np.asarray(drift_hamiltonian, dtype=np.complex128)
    values = np.asarray(control_values, dtype=float)
    durations = np.asarray(step_durations_s, dtype=float)
    dim = int(drift.shape[0])
    identity = np.eye(dim * dim, dtype=np.complex128)

    dense_collapse_ops = tuple(np.asarray(operator, dtype=np.complex128) for operator in collapse_operators)
    drift_liouvillian = _dense_liouvillian(drift, dense_collapse_ops)
    control_liouvillians = tuple(
        _dense_liouvillian(np.asarray(operator, dtype=np.complex128), ())
        for operator in control_operators
    )

    slice_liouvillians: list[np.ndarray] = []
    slice_superoperators: list[np.ndarray] = []
    forward_superoperators: list[np.ndarray] = [identity]

    for step_index in range(values.shape[1]):
        l_step = drift_liouvillian.copy()
        for control_index, control_liouvillian in enumerate(control_liouvillians):
            l_step = l_step + float(values[control_index, step_index]) * control_liouvillian
        s_step = expm(float(durations[step_index]) * l_step)
        slice_liouvillians.append(l_step)
        slice_superoperators.append(s_step)
        forward_superoperators.append(s_step @ forward_superoperators[-1])

    return LindbladPropagationData(
        step_durations_s=durations,
        control_liouvillians=control_liouvillians,
        slice_liouvillians=tuple(slice_liouvillians),
        slice_superoperators=tuple(slice_superoperators),
        forward_superoperators=tuple(forward_superoperators),
    )


def propagate_state_history(propagation: PropagationData, initial_states: np.ndarray) -> np.ndarray:
    states = np.asarray(initial_states, dtype=np.complex128)
    history = np.zeros((states.shape[0], len(propagation.slice_unitaries) + 1, states.shape[1]), dtype=np.complex128)
    history[:, 0, :] = states
    for step_index, u_step in enumerate(propagation.slice_unitaries):
        history[:, step_index + 1, :] = history[:, step_index, :] @ u_step.T
    return history


def backward_target_history(propagation: PropagationData, target_states: np.ndarray) -> np.ndarray:
    targets = np.asarray(target_states, dtype=np.complex128)
    history = np.zeros((targets.shape[0], len(propagation.slice_unitaries) + 1, targets.shape[1]), dtype=np.complex128)
    history[:, -1, :] = targets
    for step_index in range(len(propagation.slice_unitaries) - 1, -1, -1):
        u_step = propagation.slice_unitaries[step_index]
        history[:, step_index, :] = history[:, step_index + 1, :] @ u_step.conj()
    return history


def propagate_density_history(propagation: LindbladPropagationData, initial_densities: np.ndarray) -> np.ndarray:
    states = _vectorize_density_batch(initial_densities)
    history = np.zeros((states.shape[0], len(propagation.slice_superoperators) + 1, states.shape[1]), dtype=np.complex128)
    history[:, 0, :] = states
    for step_index, s_step in enumerate(propagation.slice_superoperators):
        history[:, step_index + 1, :] = history[:, step_index, :] @ s_step.T
    return history


def backward_density_target_history(propagation: LindbladPropagationData, target_densities: np.ndarray) -> np.ndarray:
    targets = _vectorize_density_batch(target_densities)
    history = np.zeros((targets.shape[0], len(propagation.slice_superoperators) + 1, targets.shape[1]), dtype=np.complex128)
    history[:, -1, :] = targets
    for step_index in range(len(propagation.slice_superoperators) - 1, -1, -1):
        s_step = propagation.slice_superoperators[step_index]
        history[:, step_index, :] = history[:, step_index + 1, :] @ s_step.conj()
    return history


__all__ = [
    "PropagationData",
    "LindbladPropagationData",
    "build_propagation_data",
    "build_lindblad_propagation_data",
    "propagate_state_history",
    "backward_target_history",
    "propagate_density_history",
    "backward_density_target_history",
    "devectorize_density_batch",
]