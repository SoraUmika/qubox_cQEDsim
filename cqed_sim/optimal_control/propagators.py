from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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


__all__ = [
    "PropagationData",
    "build_propagation_data",
    "propagate_state_history",
    "backward_target_history",
]