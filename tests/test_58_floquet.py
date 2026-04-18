from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from scipy.special import jv

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FloquetConfig,
    FloquetMarkovConfig,
    FloquetProblem,
    NoiseSpec,
    PeriodicDriveTerm,
    SidebandDriveSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
    solve_floquet,
    solve_floquet_markov,
)
from cqed_sim.core import FrameSpec
from cqed_sim.floquet import (
    build_floquet_markov_baths,
    build_target_drive_term,
    build_transmon_frequency_modulation_term,
    build_sambe_hamiltonian,
    compute_floquet_modes,
    compute_floquet_transition_strengths,
    compute_hamiltonian_fourier_components,
    extract_sambe_quasienergies,
    flat_markov_spectrum,
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


def test_incommensurate_drive_frequency_is_rejected():
    model = _single_transmon_model(omega_q=2.2, dim=2)
    floquet_angular_frequency = 1.0
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            PeriodicDriveTerm(
                target="qubit",
                amplitude=0.18,
                frequency=np.sqrt(2.0) * floquet_angular_frequency,
                waveform="cos",
            ),
        ),
        period=2.0 * np.pi / floquet_angular_frequency,
    )

    with pytest.raises(ValueError, match="not commensurate"):
        solve_floquet(problem, FloquetConfig(n_time_samples=64))


def test_floquet_modes_are_periodic_and_states_pick_up_quasienergy_phase():
    drive_angular_frequency = 1.2
    model = _single_transmon_model(omega_q=2.1, dim=2)
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            PeriodicDriveTerm(
                target="qubit",
                amplitude=0.23,
                frequency=drive_angular_frequency,
                phase=0.15,
                waveform="cos",
            ),
        ),
        period=2.0 * np.pi / drive_angular_frequency,
    )

    result = solve_floquet(problem, FloquetConfig(n_time_samples=128))
    sample_time = 0.37 * problem.period
    modes_t = compute_floquet_modes(result, t=sample_time)
    modes_tp = compute_floquet_modes(result, t=sample_time + problem.period)
    states_t = result.states(sample_time)
    states_tp = result.states(sample_time + problem.period)

    for mode_index, quasienergy in enumerate(result.quasienergies):
        assert abs(modes_t[mode_index].overlap(modes_tp[mode_index])) == pytest.approx(1.0, abs=1.0e-6)
        phase = np.exp(-1j * float(quasienergy) * problem.period)
        assert (states_tp[mode_index] - phase * states_t[mode_index]).norm() < 1.0e-5


def test_longitudinal_frequency_modulation_transition_strengths_match_bessel_sidebands():
    # Validates Eq. (45) in Silveri et al., Rep. Prog. Phys. 80, 056002 (2017).
    modulation_angular_frequency = 1.0
    modulation_amplitude = 0.7
    # Keep the bare qubit transition inside the first Floquet zone so the
    # harmonic labels match the published sideband index directly.
    model = _single_transmon_model(omega_q=0.4, dim=2)
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            build_transmon_frequency_modulation_term(
                model,
                amplitude=modulation_amplitude,
                frequency=modulation_angular_frequency,
                waveform="cos",
            ),
        ),
        period=2.0 * np.pi / modulation_angular_frequency,
    )

    result = solve_floquet(problem, FloquetConfig(n_time_samples=256))
    probe_operator = model.transmon_lowering() + model.transmon_raising()
    strengths = compute_floquet_transition_strengths(
        result,
        probe_operator,
        harmonic_cutoff=2,
        n_time_samples=768,
        min_strength=0.0,
    )

    ground_index = int(np.flatnonzero(result.dominant_bare_state_indices == 0)[0])
    excited_index = int(np.flatnonzero(result.dominant_bare_state_indices == 1)[0])
    by_harmonic = {
        entry.harmonic: entry.strength
        for entry in strengths
        if entry.initial_mode == ground_index and entry.final_mode == excited_index
    }

    for harmonic in range(-2, 3):
        expected = float(abs(jv(harmonic, modulation_amplitude / modulation_angular_frequency)) ** 2)
        assert harmonic in by_harmonic
        assert abs(by_harmonic[harmonic] - expected) < 3.0e-3


def test_run_floquet_sweep_uses_config_overlap_reference_time_by_default(monkeypatch):
    model = _single_transmon_model(omega_q=2.2, dim=2)
    problems = [
        FloquetProblem(
            model=model,
            periodic_terms=(
                PeriodicDriveTerm(target="qubit", amplitude=0.1, frequency=1.0, waveform="cos"),
            ),
            period=2.0 * np.pi,
        )
        for _ in range(2)
    ]
    seen: dict[str, object] = {}

    def fake_solve_floquet(problem, config=None):
        seen.setdefault("solve_calls", 0)
        seen["solve_calls"] = int(seen["solve_calls"]) + 1
        return object()

    def fake_track(results, *, parameter_values=None, reference_time=0.0):
        seen["reference_time"] = reference_time
        seen["parameter_values"] = parameter_values
        seen["result_count"] = len(results)
        return "tracked"

    monkeypatch.setattr("cqed_sim.floquet.core.solve_floquet", fake_solve_floquet)
    monkeypatch.setattr("cqed_sim.floquet.analysis.track_floquet_branches", fake_track)

    config = FloquetConfig(n_time_samples=64, overlap_reference_time=0.37)
    output = run_floquet_sweep(problems, parameter_values=[0.1, 0.2], config=config)

    assert output == "tracked"
    assert seen["solve_calls"] == 2
    assert seen["result_count"] == 2
    assert seen["reference_time"] == pytest.approx(0.37)
    assert seen["parameter_values"] == [0.1, 0.2]


def test_run_floquet_sweep_explicit_reference_time_overrides_config(monkeypatch):
    model = _single_transmon_model(omega_q=2.2, dim=2)
    problems = [
        FloquetProblem(
            model=model,
            periodic_terms=(
                PeriodicDriveTerm(target="qubit", amplitude=0.1, frequency=1.0, waveform="cos"),
            ),
            period=2.0 * np.pi,
        )
        for _ in range(2)
    ]
    seen: dict[str, object] = {}

    def fake_solve_floquet(problem, config=None):
        return object()

    def fake_track(results, *, parameter_values=None, reference_time=0.0):
        seen["reference_time"] = reference_time
        return "tracked"

    monkeypatch.setattr("cqed_sim.floquet.core.solve_floquet", fake_solve_floquet)
    monkeypatch.setattr("cqed_sim.floquet.analysis.track_floquet_branches", fake_track)

    config = FloquetConfig(n_time_samples=64, overlap_reference_time=0.37)
    output = run_floquet_sweep(problems, config=config, reference_time=0.11)

    assert output == "tracked"
    assert seen["reference_time"] == pytest.approx(0.11)


def test_build_floquet_markov_baths_uses_noise_bridge_and_flat_spectrum() -> None:
    model = _single_transmon_model(omega_q=2.2, dim=2)
    problem = FloquetProblem(model=model, periodic_terms=(), period=2.0 * np.pi)

    baths = build_floquet_markov_baths(
        problem,
        NoiseSpec(tphi=4.0),
        spectrum=flat_markov_spectrum(2.5),
    )

    assert len(baths) == 1
    assert baths[0].operator.dims == problem.static_hamiltonian.dims
    np.testing.assert_allclose(baths[0].resolved_spectrum()(np.array([-1.0, 0.0, 1.0])), 2.5)


def test_floquet_markov_static_relaxation_matches_expected_decay() -> None:
    model = _single_transmon_model(omega_q=0.4, dim=2)
    problem = FloquetProblem(model=model, periodic_terms=(), period=2.0 * np.pi)
    tlist = np.linspace(0.0, 1.0, 11)
    baths = build_floquet_markov_baths(problem, NoiseSpec(t1=2.0))

    result = solve_floquet_markov(
        problem,
        qt.basis(2, 1),
        tlist,
        noise=NoiseSpec(t1=2.0),
        e_ops=[model.transmon_number()],
        config=FloquetMarkovConfig(store_states=True),
    )
    direct = qt.fmmesolve(
        problem.static_hamiltonian,
        qt.basis(2, 1),
        tlist,
        c_ops=[bath.operator for bath in baths],
        spectra_cb=[bath.resolved_spectrum() for bath in baths],
        T=problem.period,
        e_ops=[model.transmon_number()],
        options={"progress_bar": "", "store_states": True},
    )

    excited_population = np.asarray(result.expect[0], dtype=float)
    direct_population = np.asarray(direct.expect[0], dtype=float)

    assert result.metadata["used_noise_bridge"] is True
    assert len(result.states) == len(tlist)
    assert excited_population[-1] < excited_population[0]
    assert np.all(np.diff(excited_population) <= 1.0e-5)
    assert np.allclose(excited_population, direct_population, atol=1.0e-6)


def test_effective_red_sideband_hybridization_peaks_at_predicted_resonance():
    # Effective-Hamiltonian validation of the first-order sideband resonance
    # discussed by Beaudoin et al., Phys. Rev. A 86, 022305 (2012).
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.05,
        omega_q=2.0 * np.pi * 6.25,
        alpha=2.0 * np.pi * (-0.25),
        chi=2.0 * np.pi * (-0.015),
        kerr=0.0,
        n_cav=3,
        n_tr=3,
    )
    frame = FrameSpec()
    center_frequency = model.sideband_transition_frequency(
        cavity_level=0,
        lower_level=0,
        upper_level=1,
        sideband="red",
        frame=frame,
    )
    sideband = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red")
    drive_frequencies = center_frequency + 2.0 * np.pi * np.linspace(-0.12, 0.12, 13)
    problems = []
    for frequency in drive_frequencies:
        drive = build_target_drive_term(
            model,
            sideband,
            amplitude=2.0 * np.pi * 0.03,
            frequency=frequency,
            waveform="cos",
        )
        problems.append(FloquetProblem(model=model, periodic_terms=(drive,), period=2.0 * np.pi / frequency))

    sweep = run_floquet_sweep(
        problems,
        parameter_values=drive_frequencies / (2.0 * np.pi),
        config=FloquetConfig(n_time_samples=96, overlap_reference_time=0.17),
    )

    hybridization_scores = [
        float(np.max(np.minimum(result.bare_state_overlaps[:, 1], result.bare_state_overlaps[:, 2])))
        for result in sweep.results
    ]
    best_index = int(np.argmax(hybridization_scores))

    assert abs((drive_frequencies[best_index] - center_frequency) / (2.0 * np.pi)) <= 0.02
    assert hybridization_scores[best_index] > 0.45
