from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import qutip as qt
from scipy.optimize import linear_sum_assignment

from .builders import resolve_periodic_drive_operator
from .utils import (
    angular_frequency_from_period,
    bare_state_overlap_matrix,
    overlap_matrix,
    sample_period_grid,
    trapezoid,
    unwrap_to_reference,
)


@dataclass(frozen=True)
class FloquetTransitionStrength:
    initial_mode: int
    final_mode: int
    harmonic: int
    transition_angular_frequency: float
    matrix_element: complex
    strength: float


@dataclass(frozen=True)
class MultiphotonResonance:
    lower_state: int
    upper_state: int
    photon_order: int
    energy_difference: float
    resonance_angular_frequency: float
    detuning: float


@dataclass(frozen=True)
class FloquetSweepResult:
    results: tuple
    parameter_values: np.ndarray | None
    tracked_quasienergies: np.ndarray
    tracked_orders: tuple[np.ndarray, ...]
    overlap_scores: np.ndarray


def compute_period_propagator(source, *, config=None) -> qt.Qobj:
    from .core import FloquetResult, solve_floquet

    if isinstance(source, FloquetResult):
        return source.period_propagator
    return solve_floquet(source, config=config).period_propagator


def compute_quasienergies(source, *, config=None) -> np.ndarray:
    from .core import FloquetResult, solve_floquet

    if isinstance(source, FloquetResult):
        return np.asarray(source.quasienergies, dtype=float)
    return solve_floquet(source, config=config).quasienergies


def compute_floquet_modes(source, *, t: float = 0.0, config=None) -> tuple[qt.Qobj, ...]:
    from .core import FloquetResult, solve_floquet

    if not isinstance(source, FloquetResult):
        source = solve_floquet(source, config=config)
    return source.modes(t)


def compute_bare_state_overlaps(source, bare_states: Sequence[qt.Qobj] | None = None) -> np.ndarray:
    from .core import FloquetResult

    if isinstance(source, FloquetResult):
        return np.asarray(source.bare_state_overlaps, dtype=float)
    if bare_states is None:
        raise ValueError("Explicit bare states are required when not passing a FloquetResult.")
    return bare_state_overlap_matrix(tuple(source), tuple(bare_states))


def compute_floquet_transition_strengths(
    result,
    operator,
    *,
    harmonic_cutoff: int = 3,
    n_time_samples: int | None = None,
    quadrature: str = "x",
    min_strength: float = 0.0,
) -> tuple[FloquetTransitionStrength, ...]:
    sample_count = max(int(n_time_samples or result.config.n_time_samples), 8)
    grid = sample_period_grid(result.problem.period, sample_count, endpoint=False)
    omega = angular_frequency_from_period(result.problem.period)
    if isinstance(operator, qt.Qobj):
        probe_operator = operator
    else:
        probe_operator = resolve_periodic_drive_operator(result.problem, type("_Probe", (), {"operator": None, "target": operator, "quadrature": quadrature})())

    mode_samples = [result.modes(float(time)) for time in grid]
    strengths: list[FloquetTransitionStrength] = []
    for initial_index in range(len(result.quasienergies)):
        for final_index in range(len(result.quasienergies)):
            matrix_elements = np.asarray(
                [
                    (modes[initial_index].dag() * probe_operator * modes[final_index])[0, 0]
                    for modes in mode_samples
                ],
                dtype=np.complex128,
            )
            for harmonic in range(-int(harmonic_cutoff), int(harmonic_cutoff) + 1):
                phase = np.exp(-1j * float(harmonic) * omega * grid)
                averaged = trapezoid(matrix_elements * phase, grid) / float(result.problem.period)
                strength = float(abs(averaged) ** 2)
                if strength < float(min_strength):
                    continue
                transition_frequency = float(result.quasienergies[final_index] - result.quasienergies[initial_index] + harmonic * omega)
                strengths.append(
                    FloquetTransitionStrength(
                        initial_mode=initial_index,
                        final_mode=final_index,
                        harmonic=harmonic,
                        transition_angular_frequency=transition_frequency,
                        matrix_element=complex(averaged),
                        strength=strength,
                    )
                )
    strengths.sort(key=lambda item: item.strength, reverse=True)
    return tuple(strengths)


def identify_multiphoton_resonances(
    source,
    *,
    max_photon_order: int = 5,
    drive_angular_frequency: float | None = None,
    tolerance: float | None = None,
) -> tuple[MultiphotonResonance, ...]:
    from .core import FloquetResult

    if isinstance(source, FloquetResult):
        energies = np.asarray(source.bare_hamiltonian_eigenenergies, dtype=float)
        omega = angular_frequency_from_period(source.problem.period)
    else:
        energies = np.asarray(source, dtype=float)
        if energies.ndim != 1:
            raise ValueError("Energy input must be a one-dimensional array.")
        if drive_angular_frequency is None:
            raise ValueError("Passing bare energies directly requires drive_angular_frequency.")
        omega = float(drive_angular_frequency)

    resonance_tolerance = float(tolerance) if tolerance is not None else 0.05 * abs(omega)
    resonances: list[MultiphotonResonance] = []
    for lower in range(len(energies)):
        for upper in range(lower + 1, len(energies)):
            delta = float(energies[upper] - energies[lower])
            for order in range(1, int(max_photon_order) + 1):
                target = order * omega
                detuning = delta - target
                if abs(detuning) <= resonance_tolerance:
                    resonances.append(
                        MultiphotonResonance(
                            lower_state=lower,
                            upper_state=upper,
                            photon_order=order,
                            energy_difference=delta,
                            resonance_angular_frequency=target,
                            detuning=detuning,
                        )
                    )
    resonances.sort(key=lambda item: abs(item.detuning))
    return tuple(resonances)


def track_floquet_branches(results: Sequence, *, parameter_values: Sequence[float] | None = None, reference_time: float = 0.0) -> FloquetSweepResult:
    if not results:
        raise ValueError("At least one FloquetResult is required for branch tracking.")

    first_result = results[0]
    n_modes = len(first_result.quasienergies)
    tracked_quasienergies = np.zeros((len(results), n_modes), dtype=float)
    tracked_quasienergies[0] = np.asarray(first_result.quasienergies, dtype=float)
    tracked_orders: list[np.ndarray] = [np.arange(n_modes, dtype=int)]
    overlap_scores = np.ones((max(len(results) - 1, 0), n_modes), dtype=float)

    previous_modes = first_result.modes(reference_time)
    for result_index, result in enumerate(results[1:], start=1):
        current_modes = result.modes(reference_time)
        overlaps = overlap_matrix(previous_modes, current_modes)
        row_index, col_index = linear_sum_assignment(1.0 - overlaps)
        permutation = np.empty(n_modes, dtype=int)
        permutation[row_index] = col_index
        ordered_quasienergies = np.asarray(result.quasienergies, dtype=float)[permutation]
        omega = angular_frequency_from_period(result.problem.period)
        for branch in range(n_modes):
            ordered_quasienergies[branch] = unwrap_to_reference(
                ordered_quasienergies[branch],
                tracked_quasienergies[result_index - 1, branch],
                omega,
            )
        tracked_quasienergies[result_index] = ordered_quasienergies
        tracked_orders.append(permutation)
        overlap_scores[result_index - 1] = overlaps[np.arange(n_modes), permutation]
        previous_modes = tuple(current_modes[index] for index in permutation)

    parameter_array = None if parameter_values is None else np.asarray(parameter_values, dtype=float)
    return FloquetSweepResult(
        results=tuple(results),
        parameter_values=parameter_array,
        tracked_quasienergies=tracked_quasienergies,
        tracked_orders=tuple(tracked_orders),
        overlap_scores=overlap_scores,
    )


def run_floquet_sweep(problems: Sequence, *, parameter_values: Sequence[float] | None = None, config=None, reference_time: float = 0.0) -> FloquetSweepResult:
    from .core import solve_floquet

    results = tuple(solve_floquet(problem, config=config) for problem in problems)
    return track_floquet_branches(results, parameter_values=parameter_values, reference_time=reference_time)


__all__ = [
    "FloquetSweepResult",
    "FloquetTransitionStrength",
    "MultiphotonResonance",
    "compute_bare_state_overlaps",
    "compute_floquet_modes",
    "compute_floquet_transition_strengths",
    "compute_period_propagator",
    "compute_quasienergies",
    "identify_multiphoton_resonances",
    "run_floquet_sweep",
    "track_floquet_branches",
]