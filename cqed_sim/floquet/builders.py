from __future__ import annotations

from typing import Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.drive_targets import resolve_drive_target_operators

from .utils import angular_frequency_from_period, sample_period_grid, trapezoid


def resolve_periodic_drive_operator(problem, term) -> qt.Qobj:
    if term.operator is not None:
        return term.operator
    if problem.model is None:
        raise ValueError("A model is required when a periodic drive term uses a named or structured target.")
    raising, lowering = resolve_drive_target_operators(problem.model, term.target)
    quadrature = str(term.quadrature).strip().lower()
    if quadrature in {"x", "ix", "in_phase", "cos"}:
        return raising + lowering
    if quadrature in {"y", "quadrature", "sin"}:
        return -1j * (raising - lowering)
    raise ValueError(f"Unsupported Floquet quadrature '{term.quadrature}'.")


def build_floquet_hamiltonian(problem) -> qt.Qobj | qt.QobjEvo:
    static_hamiltonian = problem.static_hamiltonian
    pieces: list[qt.Qobj | list] = [static_hamiltonian]
    for term in problem.periodic_terms:
        operator = resolve_periodic_drive_operator(problem, term)
        if operator.dims != static_hamiltonian.dims:
            raise ValueError(
                f"Periodic drive term '{term.label or term.target or 'operator'}' has dims {operator.dims}, "
                f"but the static Hamiltonian uses dims {static_hamiltonian.dims}."
            )

        def coeff(time: float, _args=None, _term=term):
            return complex(_term.coefficient(time))

        pieces.append([operator, coeff])
    if len(pieces) == 1:
        return static_hamiltonian
    return qt.QobjEvo(pieces)


def compute_hamiltonian_fourier_components(problem, harmonic_cutoff: int, *, n_time_samples: int = 801) -> dict[int, qt.Qobj]:
    if int(harmonic_cutoff) < 0:
        raise ValueError("The Sambe harmonic cutoff must be non-negative.")
    grid = sample_period_grid(problem.period, max(int(n_time_samples), 8), endpoint=False)
    omega = angular_frequency_from_period(problem.period)
    static_hamiltonian = problem.static_hamiltonian
    harmonic_indices = range(-int(harmonic_cutoff), int(harmonic_cutoff) + 1)
    components: dict[int, qt.Qobj] = {index: 0.0 * static_hamiltonian for index in harmonic_indices}
    components[0] = static_hamiltonian.copy()

    for term in problem.periodic_terms:
        operator = resolve_periodic_drive_operator(problem, term)
        exact = term.exact_fourier_components(problem.period)
        if exact is not None:
            for harmonic, amplitude in exact.items():
                harmonic_int = int(harmonic)
                if harmonic_int not in components:
                    continue
                components[harmonic_int] = components[harmonic_int] + amplitude * operator
            continue

        samples = np.asarray(term.coefficient(grid), dtype=np.complex128)
        for harmonic in harmonic_indices:
            phase = np.exp(-1j * float(harmonic) * omega * grid)
            amplitude = trapezoid(samples * phase, grid) / float(problem.period)
            components[int(harmonic)] = components[int(harmonic)] + amplitude * operator
    return components


def harmonic_component_norms(problem, harmonic_cutoff: int, *, n_time_samples: int = 801) -> dict[int, float]:
    components = compute_hamiltonian_fourier_components(problem, harmonic_cutoff, n_time_samples=n_time_samples)
    return {harmonic: float(component.norm()) for harmonic, component in components.items()}


def build_target_drive_term(
    model,
    target,
    *,
    amplitude: complex,
    frequency: float,
    phase: float = 0.0,
    waveform="cos",
    quadrature: str = "x",
    label: str | None = None,
):
    from .core import PeriodicDriveTerm

    return PeriodicDriveTerm(
        target=target,
        quadrature=quadrature,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        waveform=waveform,
        label=label,
    )


def build_transmon_frequency_modulation_term(
    model,
    *,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
    waveform="cos",
    label: str | None = None,
):
    from .core import PeriodicDriveTerm

    return PeriodicDriveTerm(
        operator=model.transmon_number(),
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        waveform=waveform,
        label=label or "transmon_frequency_modulation",
    )


def build_mode_frequency_modulation_term(
    model,
    mode: str,
    *,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
    waveform="cos",
    label: str | None = None,
):
    from .core import PeriodicDriveTerm

    return PeriodicDriveTerm(
        operator=model.mode_number(mode),
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        waveform=waveform,
        label=label or f"{mode}_frequency_modulation",
    )


def build_dispersive_modulation_term(
    model,
    mode: str,
    *,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
    waveform="cos",
    label: str | None = None,
):
    from .core import PeriodicDriveTerm

    if not hasattr(model, "transmon_number"):
        raise ValueError("Dispersive modulation requires a model with a transmon subsystem.")
    return PeriodicDriveTerm(
        operator=model.mode_number(mode) * model.transmon_number(),
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        waveform=waveform,
        label=label or f"{mode}_dispersive_modulation",
    )


__all__ = [
    "build_dispersive_modulation_term",
    "build_floquet_hamiltonian",
    "build_mode_frequency_modulation_term",
    "build_target_drive_term",
    "build_transmon_frequency_modulation_term",
    "compute_hamiltonian_fourier_components",
    "harmonic_component_norms",
    "resolve_periodic_drive_operator",
]