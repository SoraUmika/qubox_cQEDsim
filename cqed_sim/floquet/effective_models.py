from __future__ import annotations

import numpy as np
import qutip as qt

from .builders import compute_hamiltonian_fourier_components
from .utils import angular_frequency_from_period, fold_quasienergies


def _effective_hamiltonian_from_modes(quasienergies: np.ndarray, modes_0: tuple[qt.Qobj, ...]) -> qt.Qobj:
    if not modes_0:
        raise ValueError("At least one Floquet mode is required to build an effective Hamiltonian.")
    effective = 0.0 * modes_0[0] * modes_0[0].dag()
    for energy, mode in zip(np.asarray(quasienergies, dtype=float), modes_0):
        effective = effective + float(energy) * mode * mode.dag()
    return effective


def build_effective_floquet_hamiltonian(source, *, zone_center: float | None = None):
    from .core import FloquetResult, solve_floquet

    if isinstance(source, FloquetResult):
        if zone_center is None:
            return source.effective_hamiltonian if source.effective_hamiltonian is not None else _effective_hamiltonian_from_modes(source.quasienergies, source.floquet_modes_0)
        omega = angular_frequency_from_period(source.problem.period)
        quasienergies = fold_quasienergies(source.quasienergies, omega, zone_center=zone_center)
        return _effective_hamiltonian_from_modes(quasienergies, source.floquet_modes_0)
    result = solve_floquet(source)
    return build_effective_floquet_hamiltonian(result, zone_center=zone_center)


def build_sambe_hamiltonian(source, *, harmonic_cutoff: int | None = None, n_time_samples: int | None = None) -> qt.Qobj:
    from .core import FloquetProblem, FloquetResult

    if isinstance(source, FloquetResult):
        problem = source.problem
        harmonic_limit = source.config.sambe_harmonic_cutoff if harmonic_cutoff is None else int(harmonic_cutoff)
        sample_count = source.config.sambe_n_time_samples if n_time_samples is None else int(n_time_samples)
    elif isinstance(source, FloquetProblem):
        problem = source
        harmonic_limit = 3 if harmonic_cutoff is None else int(harmonic_cutoff)
        sample_count = 801 if n_time_samples is None else int(n_time_samples)
    else:
        raise TypeError(f"Unsupported source type '{type(source).__name__}'.")

    if harmonic_limit < 0:
        raise ValueError("The Sambe harmonic cutoff must be non-negative.")

    components = compute_hamiltonian_fourier_components(problem, harmonic_limit, n_time_samples=max(sample_count, 8))
    omega = angular_frequency_from_period(problem.period)
    static_hamiltonian = problem.static_hamiltonian
    base_matrix = np.asarray(static_hamiltonian.full(), dtype=np.complex128)
    dimension = base_matrix.shape[0]
    harmonic_indices = list(range(-harmonic_limit, harmonic_limit + 1))
    block_count = len(harmonic_indices)
    sambe = np.zeros((dimension * block_count, dimension * block_count), dtype=np.complex128)

    for row_block, row_harmonic in enumerate(harmonic_indices):
        for col_block, col_harmonic in enumerate(harmonic_indices):
            harmonic = row_harmonic - col_harmonic
            block = np.asarray(components.get(harmonic, 0.0 * static_hamiltonian).full(), dtype=np.complex128)
            if row_block == col_block:
                block = block + float(col_harmonic) * omega * np.eye(dimension, dtype=np.complex128)
            row_slice = slice(row_block * dimension, (row_block + 1) * dimension)
            col_slice = slice(col_block * dimension, (col_block + 1) * dimension)
            sambe[row_slice, col_slice] = block

    base_dims = [int(dim) for dim in static_hamiltonian.dims[0]]
    dims = [[block_count, *base_dims], [block_count, *base_dims]]
    return qt.Qobj(sambe, dims=dims)


def extract_sambe_quasienergies(
    sambe_hamiltonian: qt.Qobj,
    drive_angular_frequency: float,
    *,
    n_physical_states: int | None = None,
    zone_center: float = 0.0,
    cluster_tolerance: float = 1.0e-5,
) -> np.ndarray:
    folded = np.sort(
        fold_quasienergies(
            np.asarray(sambe_hamiltonian.eigenenergies(), dtype=float),
            drive_angular_frequency,
            zone_center=zone_center,
        )
    )
    if folded.size == 0:
        return folded

    clusters: list[list[float]] = [[float(folded[0])]]
    for value in folded[1:]:
        if abs(float(value) - clusters[-1][-1]) <= float(cluster_tolerance):
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])

    centers = np.asarray([np.mean(cluster) for cluster in clusters], dtype=float)
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=int)
    if n_physical_states is None:
        return np.sort(centers)

    target_count = int(n_physical_states)
    if target_count <= 0:
        raise ValueError("n_physical_states must be positive when provided.")

    order = sorted(range(len(clusters)), key=lambda index: (-sizes[index], centers[index]))
    selected = np.asarray([centers[index] for index in order[:target_count]], dtype=float)
    return np.sort(selected)


__all__ = [
    "build_effective_floquet_hamiltonian",
    "build_sambe_hamiltonian",
    "extract_sambe_quasienergies",
]