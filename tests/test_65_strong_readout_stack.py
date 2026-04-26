from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.metrics import compute_readout_metrics
from cqed_sim.models import (
    ExplicitPurcellFilterMode,
    IQPulse,
    MultilevelCQEDModel,
    ReadoutFrame,
    TransmonCosineSpec,
    add_explicit_purcell_filter,
    diagonalize_dressed_hamiltonian,
    diagonalize_transmon,
)
from cqed_sim.optimization import LinearPointerSeedModel, PulseConstraints, StrongReadoutOptimizer, StrongReadoutOptimizerConfig, enforce_pulse_constraints
from cqed_sim.pulses.clear import clear_readout_seed, square_readout_seed
from cqed_sim.readout.input_output import linear_pointer_response
from cqed_sim.solvers import MasterEquationConfig, solve_master_equation


def _small_model(nr: int = 5, coupling: float = 0.0) -> MultilevelCQEDModel:
    return MultilevelCQEDModel(
        transmon_energies=np.array([0.0, 8.0, 15.5], dtype=float),
        resonator_frequency=0.0,
        resonator_levels=nr,
        coupling_matrix=np.zeros((3, 3), dtype=np.complex128) + np.diag([0.0, 0.0, 0.0]),
        rotating_frame=ReadoutFrame(resonator_frequency=0.0),
        counter_rotating=False,
    )


def test_weak_drive_master_equation_matches_linear_pointer_model() -> None:
    kappa = 0.7
    dt = 0.02
    samples = np.full(80, 0.025 + 0.005j, dtype=np.complex128)
    model = _small_model(nr=7)
    pulse = IQPulse(samples=samples, dt=dt, drive_frequency=0.0)
    hdata = model.build_hamiltonian(pulse)
    a = model.operators()["a"]
    result = solve_master_equation(
        hdata,
        model.basis_state(0, 0),
        c_ops=[np.sqrt(kappa) * a],
        e_ops={"a": a},
        config=MasterEquationConfig(atol=1.0e-9, rtol=1.0e-8),
    )
    _t, alpha = linear_pointer_response(samples, dt=dt, kappa=kappa, detuning=0.0)

    assert np.max(np.abs(result.expectations["a"] - alpha)) < 4.0e-3


def test_clear_seed_reduces_residual_photons_against_passive_square_ringdown() -> None:
    dt = 0.01
    kappa = 1.4
    square = square_readout_seed(amplitude=0.35, duration=1.0, dt=dt, drive_frequency=0.0)
    clear = clear_readout_seed(
        amplitude=0.35,
        duration=1.0,
        dt=dt,
        drive_frequency=0.0,
        kick_fraction=0.12,
        depletion_fraction=0.35,
        depletion_amplitude=-0.45,
    )
    _t, alpha_square = linear_pointer_response(square.samples, dt=dt, kappa=kappa)
    _t, alpha_clear = linear_pointer_response(clear.samples, dt=dt, kappa=kappa)

    assert abs(alpha_clear[-1]) ** 2 < abs(alpha_square[-1]) ** 2


def test_dressed_projectors_sum_to_identity_over_retained_subspace() -> None:
    model = MultilevelCQEDModel(
        transmon_energies=np.array([0.0, 5.0], dtype=float),
        resonator_frequency=4.8,
        resonator_levels=3,
        coupling_matrix=np.array([[0.0, 0.08], [0.08, 0.0]], dtype=np.complex128),
        counter_rotating=True,
    )
    dressed = diagonalize_dressed_hamiltonian(model.static_hamiltonian())
    projector_sum = sum(dressed.projectors.values(), 0 * dressed.retained_subspace_projector)

    assert (projector_sum - qt_identity_like(projector_sum)).norm() < 1.0e-10


def qt_identity_like(operator):
    import qutip as qt

    return qt.qeye(operator.dims[0])


def test_transition_matrix_columns_report_missing_leakage_weight() -> None:
    model = _small_model(nr=2)
    dressed = diagonalize_dressed_hamiltonian(model.static_hamiltonian())
    rho0 = model.basis_state(0, 0).proj()
    rho1 = model.basis_state(2, 0).proj()
    result = dressed.transition_matrix({0: rho0, 1: rho1}, measured_levels=(0, 1))

    assert np.all(np.sum(result.matrix, axis=0) <= 1.0 + 1.0e-12)
    assert result.missing_weight[0] < 1.0e-12
    assert result.missing_weight[1] > 0.9


def test_cutoff_convergence_changes_metrics_below_tolerance_for_weak_drive() -> None:
    def final_photon(nr: int) -> float:
        model = _small_model(nr=nr)
        samples = np.full(40, 0.015, dtype=np.complex128)
        pulse = IQPulse(samples=samples, dt=0.03, drive_frequency=0.0)
        a = model.operators()["a"]
        result = solve_master_equation(
            model.build_hamiltonian(pulse),
            model.basis_state(0, 0),
            c_ops=[np.sqrt(0.5) * a],
            e_ops={"n_r": model.operators()["n_r"]},
            config=MasterEquationConfig(atol=1.0e-9, rtol=1.0e-8),
        )
        return float(result.expectations["n_r"][-1])

    assert abs(final_photon(5) - final_photon(6)) < 1.0e-5


def test_pulse_constraints_are_enforced() -> None:
    pulse = square_readout_seed(amplitude=4.0, duration=1.0, dt=0.1, drive_frequency=0.0)
    constrained = enforce_pulse_constraints(
        pulse,
        PulseConstraints(max_amplitude=1.0, max_slew_rate=2.0, fixed_total_duration=1.0),
    )
    assert np.max(np.abs(constrained.samples)) <= 1.0 + 1.0e-12
    assert np.max(np.abs(np.diff(constrained.samples)) / constrained.dt) <= 2.0 + 1.0e-12


def test_optimizer_returns_ranked_candidates_and_keeps_qnd_in_objective() -> None:
    optimizer = StrongReadoutOptimizer(
        linear_model=LinearPointerSeedModel(kappa=1.0, chi=0.2, noise_sigma=0.3),
        constraints=PulseConstraints(max_amplitude=0.8, fixed_total_duration=0.5),
        config=StrongReadoutOptimizerConfig(maxiter=0, n_candidates=2, random_seed=1),
    )
    result = optimizer.optimize(amplitude=0.2, duration=0.5, dt=0.05, drive_frequency=0.0)

    assert len(result.candidates) >= 2
    assert result.candidates[0].objective <= result.candidates[-1].objective
    assert result.best.metrics.assignment_fidelity <= 1.0
    assert result.best.metrics.physical_qnd_fidelity == pytest.approx(1.0)


def test_explicit_filter_uses_filter_output_and_blocks_purcell_double_counting() -> None:
    base = _small_model(nr=3)
    filtered = add_explicit_purcell_filter(
        base,
        ExplicitPurcellFilterMode(frequency=0.0, levels=2, coupling=0.1, kappa=0.4),
    )
    c_ops = filtered.collapse_operators(kappa_r_internal=0.0)

    assert len(c_ops) == 1
    assert c_ops[0].dims == filtered.operators()["f"].dims
    with pytest.raises(ValueError, match="explicit filter output"):
        filtered.collapse_operators(include_purcell_qubit_decay=True)


def test_transmon_charge_basis_diagonalization_exports_charge_matrix() -> None:
    spectrum = diagonalize_transmon(TransmonCosineSpec(EJ=22.0, EC=0.25, ng=0.0, n_cut=5, levels=4))

    assert spectrum.n_matrix.shape == (4, 4)
    assert np.all(np.diff(spectrum.energies) > 0.0)
    assert np.allclose(spectrum.n_matrix, spectrum.n_matrix.conj().T)


def test_metric_helper_keeps_assignment_and_qnd_separate() -> None:
    metrics = compute_readout_metrics(
        confusion=np.array([[0.99, 0.02], [0.01, 0.98]]),
        transition_matrix=np.array([[0.90, 0.05], [0.08, 0.86]]),
        pulse_samples=np.ones(4),
        dt=0.1,
    )

    assert metrics.assignment_fidelity > metrics.physical_qnd_fidelity
    assert metrics.p_0_to_1 == pytest.approx(0.08)
    assert metrics.p_1_to_0 == pytest.approx(0.05)
