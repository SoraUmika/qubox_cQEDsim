from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from cqed_sim.metrics import ReadoutMetricSet
from cqed_sim.models import (
    ExplicitPurcellFilterMode,
    IQPulse,
    MISTScanConfig,
    MultilevelCQEDModel,
    ReadoutFrame,
    TransmonCosineSpec,
    add_explicit_purcell_filter,
    diagonalize_dressed_hamiltonian,
    diagonalize_transmon,
    scan_mist,
)
from cqed_sim.optimization import (
    LinearPointerSeedModel,
    PulseConstraints,
    StrongReadoutOptimizer,
    StrongReadoutOptimizerConfig,
)
from cqed_sim.pulses.clear import square_readout_seed
from cqed_sim.readout.classifiers import MatchedFilterClassifier
from cqed_sim.readout.input_output import linear_pointer_response, output_operator
from cqed_sim.solvers import (
    DressedDecay,
    MasterEquationConfig,
    collapse_operators_from_model,
    solve_master_equation,
)
from cqed_sim.solvers.trajectories import TrajectoryConfig, simulate_measurement_trajectories


def _small_readout_model(
    *,
    nr: int = 5,
    detuning: float = 0.0,
    coupling: float = 0.0,
    counter_rotating: bool = True,
) -> MultilevelCQEDModel:
    q_coupling = np.array(
        [
            [0.0, coupling, 0.0],
            [coupling, 0.0, np.sqrt(2.0) * coupling],
            [0.0, np.sqrt(2.0) * coupling, 0.0],
        ],
        dtype=np.complex128,
    )
    return MultilevelCQEDModel(
        transmon_energies=np.array([0.0, 5.0, 9.6], dtype=float),
        resonator_frequency=detuning,
        resonator_levels=nr,
        coupling_matrix=q_coupling,
        rotating_frame=ReadoutFrame(resonator_frequency=0.0),
        counter_rotating=counter_rotating,
    )


def test_cosine_transmon_operator_identities_and_transmon_limit() -> None:
    spec = TransmonCosineSpec(EJ=35.0, EC=0.60, ng=0.0, n_cut=11, levels=4)
    spectrum = diagonalize_transmon(spec)

    assert spectrum.charge_hamiltonian.isherm
    assert spectrum.charge_operator.isherm
    assert np.allclose(spectrum.n_matrix, spectrum.n_matrix.conj().T, atol=1.0e-12)
    assert np.all(np.diff(spectrum.energies) > 0.0)

    refined = diagonalize_transmon(TransmonCosineSpec(EJ=35.0, EC=0.60, ng=0.0, n_cut=13, levels=4))
    coarse_w01 = spectrum.shifted_energies[1]
    refined_w01 = refined.shifted_energies[1]
    assert abs(coarse_w01 - refined_w01) / refined_w01 < 2.0e-4

    w01_asymptotic = np.sqrt(8.0 * spec.EJ * spec.EC) - spec.EC
    anharmonicity = spectrum.shifted_energies[2] - 2.0 * spectrum.shifted_energies[1]
    assert spectrum.shifted_energies[1] == pytest.approx(w01_asymptotic, rel=0.08)
    assert anharmonicity == pytest.approx(-spec.EC, rel=0.35)


def test_multilevel_hamiltonian_dimensions_frames_and_rwa_terms() -> None:
    model = _small_readout_model(nr=3, detuning=7.0, coupling=0.04, counter_rotating=True)
    h = model.static_hamiltonian()

    assert h.isherm
    assert h.shape == (9, 9)
    assert h.dims == [[3, 3], [3, 3]]

    framed = MultilevelCQEDModel(
        transmon_energies=np.array([0.0, 5.0, 9.6], dtype=float),
        resonator_frequency=7.0,
        resonator_levels=2,
        coupling_matrix=np.zeros((3, 3), dtype=np.complex128),
        rotating_frame=ReadoutFrame(transmon_frequency=0.5, resonator_frequency=2.0),
    )
    state = framed.basis_state(2, 1)
    energy = complex(state.dag() * framed.static_hamiltonian() * state).real
    assert energy == pytest.approx(9.6 - 2.0 * 0.5 + 7.0 - 2.0)

    rwa = _small_readout_model(nr=2, coupling=0.12, counter_rotating=False)
    full = _small_readout_model(nr=2, coupling=0.12, counter_rotating=True)
    vacuum = rwa.basis_state(0, 0)
    excitation_pair = rwa.basis_state(1, 1)
    assert abs(complex(excitation_pair.dag() * rwa.static_hamiltonian() * vacuum)) < 1.0e-14
    assert abs(complex(excitation_pair.dag() * full.static_hamiltonian() * vacuum)) > 1.0e-3


@pytest.mark.parametrize("detuning", [-0.25, 0.0, 0.35])
def test_master_equation_preserves_density_matrix_and_matches_pointer(detuning: float) -> None:
    kappa = 0.65
    dt = 0.02
    samples = np.full(90, 0.020 - 0.004j, dtype=np.complex128)
    model = _small_readout_model(nr=8, detuning=detuning, coupling=0.0)
    pulse = IQPulse(samples=samples, dt=dt, drive_frequency=0.0)
    a_op = model.operators()["a"]
    result = solve_master_equation(
        model.build_hamiltonian(pulse),
        model.basis_state(0, 0),
        c_ops=[np.sqrt(kappa) * a_op],
        e_ops={"a": a_op},
        config=MasterEquationConfig(atol=1.0e-9, rtol=1.0e-8),
    )
    _t, alpha = linear_pointer_response(samples, dt=dt, kappa=kappa, detuning=detuning)
    eigenvalues = np.linalg.eigvalsh(np.asarray(result.final_state.full(), dtype=np.complex128))

    assert result.final_state.tr() == pytest.approx(1.0, abs=1.0e-10)
    assert np.min(eigenvalues) > -1.0e-8
    assert np.max(np.abs(result.expectations["a"] - alpha)) < 5.0e-3


def test_input_output_and_dressed_decay_channels_are_accounted_once() -> None:
    base = _small_readout_model(nr=3, coupling=0.03)
    bare_output = output_operator(base, kappa_r=0.7)
    assert (bare_output - np.sqrt(0.7) * base.operators()["a"]).norm() < 1.0e-12

    filtered = add_explicit_purcell_filter(
        base,
        ExplicitPurcellFilterMode(frequency=0.0, levels=2, coupling=0.05, kappa=0.4),
    )
    filter_output = output_operator(filtered, kappa_r=0.7, kappa_f=0.4)
    assert (filter_output - np.sqrt(0.4) * filtered.operators()["f"]).norm() < 1.0e-12
    with pytest.raises(ValueError, match="explicit filter output"):
        collapse_operators_from_model(filtered, include_purcell_qubit_decay=True)

    dressed = diagonalize_dressed_hamiltonian(base.static_hamiltonian(), levels=4)
    c_ops = collapse_operators_from_model(
        base,
        kappa_r=0.2,
        dressed_states=dressed.states,
        dressed_decays=[DressedDecay(source=1, target=0, rate=0.3)],
    )
    assert len(c_ops) == 2


def test_dressed_projectors_are_orthogonal_and_transition_columns_close() -> None:
    model = _small_readout_model(nr=3, coupling=0.05)
    dressed = diagonalize_dressed_hamiltonian(model.static_hamiltonian())
    projector_sum = sum(dressed.projectors.values(), 0 * dressed.retained_subspace_projector)

    assert (projector_sum - qt.qeye(projector_sum.dims[0])).norm() < 1.0e-10
    assert (dressed.qubit_projector(0) * dressed.qubit_projector(1)).norm() < 1.0e-10

    rho0 = model.basis_state(0, 0).proj()
    rho1 = model.basis_state(1, 0).proj()
    transitions = dressed.transition_matrix({0: rho0, 1: rho1}, measured_levels=(0, 1))

    assert np.all(np.sum(transitions.matrix, axis=0) <= 1.0 + 1.0e-12)
    assert np.all(transitions.missing_weight < 0.15)
    assert dressed.leakage_probability(model.basis_state(2, 0)) > 0.8


def test_mist_penalty_peaks_near_resonance_and_scales_with_amplitude() -> None:
    spec = TransmonCosineSpec(EJ=35.0, EC=0.60, ng=0.0, n_cut=9, levels=5)
    spectrum = diagonalize_transmon(spec)
    candidates = []
    for initial in (0, 1):
        for target in range(2, spec.levels):
            candidates.append((abs(spectrum.n_matrix[target, initial]), initial, target))
    _matrix_element, initial_level, target_level = max(candidates, key=lambda item: item[0])
    resonant_frequency = float(abs(spectrum.shifted_energies[target_level] - spectrum.shifted_energies[initial_level]))
    off_frequency = resonant_frequency + 1.0
    scan = scan_mist(
        MISTScanConfig(
            EJ=spec.EJ,
            EC=spec.EC,
            n_cut=spec.n_cut,
            levels=spec.levels,
            drive_amplitudes=[0.02, 0.20],
            drive_frequencies=[resonant_frequency, off_frequency],
            max_multiphoton_order=1,
            detuning_width=0.08,
        )
    )

    low_resonant = scan.penalty(0.02, resonant_frequency)
    high_resonant = scan.penalty(0.20, resonant_frequency)
    high_off = scan.penalty(0.20, off_frequency)

    assert high_resonant > 50.0 * low_resonant
    assert high_resonant > 10.0 * high_off
    assert any(
        item.initial_level == initial_level and item.target_level == target_level and item.photon_order == 1
        for item in scan.resonances
    )


def test_trajectory_records_are_seed_reproducible_and_scale_with_timestep() -> None:
    h = 0.0 * qt.qeye(2)
    rho0 = qt.basis(2, 0)
    out = 0.0 * qt.qeye(2)

    result_a = simulate_measurement_trajectories(
        h,
        rho0,
        tlist=np.arange(5, dtype=float) * 0.1,
        output_operator=out,
        config=TrajectoryConfig(ntraj=64, seed=123, heterodyne=True),
    )
    result_b = simulate_measurement_trajectories(
        h,
        rho0,
        tlist=np.arange(5, dtype=float) * 0.1,
        output_operator=out,
        config=TrajectoryConfig(ntraj=64, seed=123, heterodyne=True),
    )
    assert np.allclose(result_a.mean_I, result_b.mean_I)
    assert np.allclose(result_a.mean_Q, result_b.mean_Q)

    short_dt = simulate_measurement_trajectories(
        h,
        rho0,
        tlist=np.arange(5, dtype=float) * 0.05,
        output_operator=out,
        config=TrajectoryConfig(ntraj=800, seed=5, heterodyne=False),
    )
    long_dt = simulate_measurement_trajectories(
        h,
        rho0,
        tlist=np.arange(5, dtype=float) * 0.20,
        output_operator=out,
        config=TrajectoryConfig(ntraj=800, seed=5, heterodyne=False),
    )
    short_std = np.std([trajectory.I[1] for trajectory in short_dt.trajectories])
    long_std = np.std([trajectory.I[1] for trajectory in long_dt.trajectories])
    assert short_std > 1.7 * long_std


def test_matched_filter_uses_equal_noise_midpoint_boundary() -> None:
    classifier = MatchedFilterClassifier.fit({0: [0.0 + 0.0j], 1: [10.0 + 0.0j]})

    assert classifier.classify(np.array([[1.0 + 0.0j]]))[0] == 0
    assert classifier.classify(np.array([[9.0 + 0.0j]]))[0] == 1


def test_optimizer_certification_uses_late_stage_scorers() -> None:
    calls = {"stage_b": 0, "stage_c": 0}

    def stage_b(_pulse):
        calls["stage_b"] += 1
        return ReadoutMetricSet(
            assignment_fidelity=0.91,
            physical_qnd_fidelity=0.82,
            residual_resonator_photons=0.03,
        )

    def stage_c(_pulse):
        calls["stage_c"] += 1
        return ReadoutMetricSet(
            assignment_fidelity=0.73,
            physical_qnd_fidelity=0.61,
            leakage_probability=0.04,
            residual_resonator_photons=0.02,
        )

    optimizer = StrongReadoutOptimizer(
        linear_model=LinearPointerSeedModel(kappa=1.0, chi=0.2, noise_sigma=0.2),
        constraints=PulseConstraints(max_amplitude=0.4, fixed_total_duration=0.3),
        config=StrongReadoutOptimizerConfig(maxiter=0, n_candidates=1, include_clear_seed=False),
        stage_b_scorer=stage_b,
        stage_c_scorer=stage_c,
    )
    seed = square_readout_seed(amplitude=0.2, duration=0.3, dt=0.1, drive_frequency=0.0)
    result = optimizer.optimize(seeds=[seed])

    assert calls == {"stage_b": 1, "stage_c": 1}
    assert result.best.metrics.assignment_fidelity == pytest.approx(0.73)
    assert result.best.metrics.physical_qnd_fidelity == pytest.approx(0.61)
    assert result.best.stage_scores["B_master_equation"] == pytest.approx(0.0)
    assert result.best.stage_scores["C_trajectories"] == pytest.approx(0.0)
