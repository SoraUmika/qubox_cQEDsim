from __future__ import annotations

import numpy as np

from cqed_sim.optimal_control.readout_emptying import (
    _segment_terminal_coefficient,
    ReadoutEmptyingConstraints,
    ReadoutEmptyingSpec,
    build_emptying_constraint_matrix,
    compute_emptying_null_space,
    replay_linear_readout_branches,
    synthesize_readout_emptying_pulse,
)


def test_segment_terminal_coefficient_matches_direct_numerical_quadrature() -> None:
    detuning = 2.0 * np.pi * 1.3e6
    kappa = 2.0 * np.pi * 2.1e6
    t_start = 40e-9
    t_stop = 115e-9
    tau = 250e-9
    coefficient = _segment_terminal_coefficient(detuning, kappa, t_start, t_stop, tau)

    samples = np.linspace(t_start, t_stop, 200_001, dtype=float)
    integrand = -1j * np.exp(-(0.5 * kappa + 1j * detuning) * (tau - samples))
    reference = np.trapezoid(integrand, x=samples)

    assert np.allclose(coefficient, reference, rtol=1.0e-10, atol=1.0e-12)


def test_emptying_null_space_dimension_matches_expected_segment_count() -> None:
    spec_two = ReadoutEmptyingSpec(kappa=1.0, chi=0.2, tau=6.0, n_segments=2)
    spec_three = ReadoutEmptyingSpec(kappa=1.0, chi=0.2, tau=6.0, n_segments=3)

    null_two = compute_emptying_null_space(build_emptying_constraint_matrix(spec_two))
    null_three = compute_emptying_null_space(build_emptying_constraint_matrix(spec_three))

    assert null_two.shape == (2, 0)
    assert null_three.shape == (3, 1)


def test_synthesized_linear_waveform_satisfies_constraints_and_replay() -> None:
    spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=300e-9,
        n_segments=4,
    )
    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 8.0e6)
    result = synthesize_readout_emptying_pulse(spec, constraints)

    amplitudes = np.asarray(result.diagnostics["linear_segment_amplitudes"], dtype=np.complex128)
    constraint_matrix = build_emptying_constraint_matrix(spec)
    terminal = constraint_matrix @ amplitudes
    replay = replay_linear_readout_branches(spec, amplitudes)

    assert np.max(np.abs(terminal)) < 1.0e-12
    assert max(abs(value) for value in replay.final_alpha.values()) < 1.0e-12
    assert np.allclose(
        np.array(list(replay.final_alpha.values()), dtype=np.complex128),
        np.array(list(result.final_alpha.values()), dtype=np.complex128),
        atol=1.0e-12,
    )
