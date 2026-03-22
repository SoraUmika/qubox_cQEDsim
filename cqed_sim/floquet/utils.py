from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np
import qutip as qt


def angular_frequency_from_period(period: float) -> float:
    period_value = float(period)
    if period_value <= 0.0:
        raise ValueError("Period must be positive.")
    return float(2.0 * np.pi / period_value)


def sample_period_grid(period: float, n_samples: int, *, endpoint: bool = False) -> np.ndarray:
    n_points = int(n_samples)
    if n_points < 2:
        raise ValueError("At least two time samples are required per period.")
    return np.linspace(0.0, float(period), n_points, endpoint=endpoint, dtype=float)


def wrap_phase(phases: Sequence[float]) -> np.ndarray:
    values = np.asarray(phases, dtype=float)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def fold_quasienergies(
    quasienergies: Sequence[float],
    drive_angular_frequency: float,
    *,
    zone_center: float = 0.0,
) -> np.ndarray:
    omega = float(abs(drive_angular_frequency))
    values = np.asarray(quasienergies, dtype=float)
    if omega <= 0.0:
        return values.copy()
    return zone_center + ((values - zone_center + 0.5 * omega) % omega) - 0.5 * omega


def basis_level_tuples(subsystem_dims: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    dims = tuple(int(dim) for dim in subsystem_dims)
    if not dims:
        return ((),)
    return tuple(tuple(levels) for levels in product(*(range(dim) for dim in dims)))


def boundary_populations(modes: Iterable[qt.Qobj], subsystem_dims: Sequence[int]) -> np.ndarray:
    mode_list = list(modes)
    dims = tuple(int(dim) for dim in subsystem_dims)
    if not dims:
        return np.zeros(len(mode_list), dtype=float)
    basis_levels = basis_level_tuples(dims)
    edge_mask = np.asarray(
        [any(level == dim - 1 for level, dim in zip(levels, dims)) for levels in basis_levels],
        dtype=float,
    )
    populations: list[float] = []
    for mode in mode_list:
        coeffs = np.asarray(mode.full(), dtype=np.complex128).reshape(-1)
        populations.append(float(np.sum(np.abs(coeffs) ** 2 * edge_mask)))
    return np.asarray(populations, dtype=float)


def overlap_matrix(states_a: Sequence[qt.Qobj], states_b: Sequence[qt.Qobj]) -> np.ndarray:
    matrix = np.zeros((len(states_a), len(states_b)), dtype=float)
    for row, left in enumerate(states_a):
        for col, right in enumerate(states_b):
            matrix[row, col] = float(abs(left.overlap(right)) ** 2)
    return matrix


def bare_state_overlap_matrix(floquet_modes: Sequence[qt.Qobj], bare_states: Sequence[qt.Qobj]) -> np.ndarray:
    matrix = np.zeros((len(floquet_modes), len(bare_states)), dtype=float)
    for mode_index, mode in enumerate(floquet_modes):
        for bare_index, bare_state in enumerate(bare_states):
            matrix[mode_index, bare_index] = float(abs(bare_state.overlap(mode)) ** 2)
    return matrix


def unwrap_to_reference(value: float, reference: float, drive_angular_frequency: float) -> float:
    omega = float(abs(drive_angular_frequency))
    if omega <= 0.0:
        return float(value)
    shift = omega * np.round((reference - value) / omega)
    return float(value + shift)


def trapezoid(values: np.ndarray, grid: np.ndarray, axis: int = 0) -> np.ndarray:
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrator(values, grid, axis=axis)


__all__ = [
    "angular_frequency_from_period",
    "bare_state_overlap_matrix",
    "basis_level_tuples",
    "boundary_populations",
    "fold_quasienergies",
    "overlap_matrix",
    "sample_period_grid",
    "trapezoid",
    "unwrap_to_reference",
    "wrap_phase",
]