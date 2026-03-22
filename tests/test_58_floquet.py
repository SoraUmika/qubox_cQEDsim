from __future__ import annotations

import numpy as np

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FloquetConfig,
    FloquetProblem,
    PeriodicDriveTerm,
    TransmonModeSpec,
    UniversalCQEDModel,
    solve_floquet,
)
from cqed_sim.floquet import (
    build_sambe_hamiltonian,
    compute_hamiltonian_fourier_components,
    extract_sambe_quasienergies,
    fold_quasienergies,
    identify_multiphoton_resonances,
    run_floquet_sweep,
)


def _single_transmon_model(omega_q: float = 2.1, alpha: float = 0.0, dim: int = 2) -> UniversalCQEDModel:
    return UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=omega_q,
            dim=dim,
            alpha=alpha,
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(),
    )


def test_zero_drive_matches_static_spectrum_modulo_floquet_zone():
    drive_angular_frequency = 1.1
    model = DispersiveTransmonCavityModel(
        omega_c=0.7,
        omega_q=2.3,
        alpha=-0.25,
        chi=-0.04,
        kerr=0.0,
        n_cav=2,
        n_tr=3,
    )
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            PeriodicDriveTerm(target="qubit", amplitude=0.0, frequency=drive_angular_frequency, waveform="cos"),
        ),
        period=2.0 * np.pi / drive_angular_frequency,
    )

    result = solve_floquet(problem, FloquetConfig(n_time_samples=96))

    expected = np.sort(fold_quasienergies(problem.static_hamiltonian.eigenenergies(), drive_angular_frequency))
    assert np.allclose(np.sort(result.quasienergies), expected, atol=1.0e-5)
    assert np.allclose(result.bare_state_overlaps.sum(axis=1), 1.0, atol=1.0e-8)


def test_exact_cosine_harmonics_match_expected_fourier_components():
    drive_angular_frequency = 0.9
    amplitude = 0.32
    model = _single_transmon_model(omega_q=1.7, dim=2)
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            PeriodicDriveTerm(target="qubit", amplitude=amplitude, frequency=drive_angular_frequency, waveform="cos"),
        ),
        period=2.0 * np.pi / drive_angular_frequency,
    )

    components = compute_hamiltonian_fourier_components(problem, harmonic_cutoff=1, n_time_samples=96)
    expected_operator = model.transmon_lowering() + model.transmon_raising()

    assert (components[1] - 0.5 * amplitude * expected_operator).norm() < 1.0e-10
    assert (components[-1] - 0.5 * amplitude * expected_operator).norm() < 1.0e-10
    assert (components[0] - problem.static_hamiltonian).norm() < 1.0e-10


def test_period_propagator_eigenphases_match_returned_quasienergies():
    drive_angular_frequency = 1.3
    model = _single_transmon_model(omega_q=2.2, dim=2)
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            PeriodicDriveTerm(target="qubit", amplitude=0.25, frequency=drive_angular_frequency, phase=0.2, waveform="cos"),
        ),
        period=2.0 * np.pi / drive_angular_frequency,
    )
    result = solve_floquet(problem, FloquetConfig(n_time_samples=128))

    eigenphases = np.sort(np.angle(result.period_propagator.eigenenergies()))
    returned = np.sort(result.eigenphases)
    assert np.allclose(eigenphases, returned, atol=1.0e-7)


def test_sambe_truncation_converges_toward_propagator_result():
    drive_angular_frequency = 1.0
    model = _single_transmon_model(omega_q=2.3, dim=2)
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            PeriodicDriveTerm(target="qubit", amplitude=0.18, frequency=drive_angular_frequency, phase=0.1, waveform="cos"),
        ),
        period=2.0 * np.pi / drive_angular_frequency,
    )

    reference = np.sort(solve_floquet(problem, FloquetConfig(n_time_samples=128)).quasienergies)
    sambe_1 = extract_sambe_quasienergies(
        build_sambe_hamiltonian(problem, harmonic_cutoff=1, n_time_samples=256),
        drive_angular_frequency,
        n_physical_states=len(reference),
    )
    sambe_3 = extract_sambe_quasienergies(
        build_sambe_hamiltonian(problem, harmonic_cutoff=3, n_time_samples=512),
        drive_angular_frequency,
        n_physical_states=len(reference),
    )

    err_1 = np.max(np.abs(np.sort(sambe_1) - reference))
    err_3 = np.max(np.abs(np.sort(sambe_3) - reference))
    assert err_3 <= err_1 + 1.0e-8
    assert err_3 < 5.0e-2


def test_multiphoton_resonance_helper_identifies_transmon_two_photon_condition():
    drive_angular_frequency = 2.0
    model = _single_transmon_model(omega_q=2.0, alpha=0.0, dim=3)
    problem = FloquetProblem(model=model, periodic_terms=(), period=2.0 * np.pi / drive_angular_frequency)
    result = solve_floquet(problem, FloquetConfig(n_time_samples=64))

    resonances = identify_multiphoton_resonances(result, max_photon_order=3, tolerance=1.0e-8)
    assert any(
        resonance.lower_state == 0 and resonance.upper_state == 2 and resonance.photon_order == 2
        for resonance in resonances
    )


def test_branch_tracking_returns_continuous_quasienergy_sweep():
    drive_angular_frequency = 1.05
    model = _single_transmon_model(omega_q=2.4, dim=2)
    amplitudes = np.linspace(0.0, 0.3, 5)
    problems = [
        FloquetProblem(
            model=model,
            periodic_terms=(
                PeriodicDriveTerm(target="qubit", amplitude=float(amplitude), frequency=drive_angular_frequency, waveform="cos"),
            ),
            period=2.0 * np.pi / drive_angular_frequency,
        )
        for amplitude in amplitudes
    ]

    sweep = run_floquet_sweep(problems, parameter_values=amplitudes, config=FloquetConfig(n_time_samples=96))
    assert sweep.tracked_quasienergies.shape == (len(amplitudes), 2)
    assert np.all((sweep.overlap_scores >= 0.0) & (sweep.overlap_scores <= 1.0))
    assert np.max(np.abs(np.diff(sweep.tracked_quasienergies, axis=0))) < 0.6