from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from scipy.linalg import expm

from cqed_sim import (
    ControlProblem,
    ControlSystem,
    ControlTerm,
    DensityMatrixTransferObjective,
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    SequenceCompiler,
    SimulationConfig,
    StateTransferObjective,
    UnitaryObjective,
    build_control_problem_from_model,
    simulate_sequence,
    state_preparation_objective,
)


def _rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def _qubit_only_cavity_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return model, frame


def _rotation_problem() -> tuple[DispersiveTransmonCavityModel, FrameSpec, ControlProblem]:
    model, frame = _qubit_only_cavity_model()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e8, 1.0e8),
                export_channel="qubit",
            ),
        ),
        objectives=(
            UnitaryObjective(
                target_operator=_rotation_y(np.pi / 2.0),
                ignore_global_phase=True,
                name="ry_pi_over_two",
            ),
        ),
        metadata={"test": "rotation_problem"},
    )
    return model, frame, problem


def test_grape_model_backed_unitary_rotation_reaches_high_fidelity() -> None:
    _model, _frame, problem = _rotation_problem()
    solver = GrapeSolver(GrapeConfig(maxiter=80, seed=11, random_scale=0.25))
    result = solver.solve(problem, initial_schedule=np.array([[8.0e6]], dtype=float))

    assert result.success
    assert result.metrics["nominal_fidelity"] > 0.999
    assert result.schedule.max_abs_amplitude() <= 1.0e8 + 1.0e-6
    assert result.system_metrics[0]["objectives"][0]["exact_unitary_fidelity"] > 0.999


def test_model_backed_q_quadrature_matches_standard_pauli_y() -> None:
    _model, _frame, problem = _rotation_problem()

    np.testing.assert_allclose(problem.control_terms[0].operator, qt.sigmay().full(), atol=1.0e-12)


def test_q_only_pulse_export_uses_positive_imaginary_baseband() -> None:
    _model, _frame, problem = _rotation_problem()

    pulses, drive_ops, metadata = problem.parameterization.to_pulses(np.array([[2.5e7]], dtype=float))

    assert drive_ops["qubit"] == "qubit"
    assert "c(t) = I(t) + i Q(t)" in metadata["mapping"]
    assert len(pulses) == 1
    assert np.isclose(pulses[0].amp, 2.5e7)
    assert np.isclose(pulses[0].phase, 0.5 * np.pi)


def test_grape_exported_pulses_replay_through_runtime_simulation() -> None:
    model, frame, problem = _rotation_problem()
    solver = GrapeSolver(GrapeConfig(maxiter=80, seed=11, random_scale=0.25))
    result = solver.solve(problem, initial_schedule=np.array([[8.0e6]], dtype=float))
    pulses, drive_ops, _meta = result.to_pulses()

    compiler = SequenceCompiler(dt=1.0e-9)
    compiled = compiler.compile(pulses, t_end=problem.time_grid.duration_s)
    initial_state = model.basis_state(0, 0)
    simulation = simulate_sequence(
        model,
        compiled,
        initial_state,
        drive_ops,
        config=SimulationConfig(frame=frame),
    )

    target_state = qt.Qobj(_rotation_y(np.pi / 2.0) @ np.array([1.0, 0.0], dtype=np.complex128), dims=initial_state.dims)
    fidelity = float(qt.metrics.fidelity(simulation.final_state, target_state))

    assert fidelity > 0.999


def test_grape_state_preparation_objective_reaches_target_state() -> None:
    model, frame = _qubit_only_cavity_model()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e8, 1.0e8),
            ),
        ),
        objectives=(
            state_preparation_objective(
                model.basis_state(0, 0),
                model.basis_state(1, 0),
            ),
        ),
    )

    result = GrapeSolver(GrapeConfig(maxiter=80, seed=17, random_scale=0.25)).solve(
        problem,
        initial_schedule=np.array([[3.5e7]], dtype=float),
    )

    assert result.success
    assert result.system_metrics[0]["objectives"][0]["fidelity_weighted"] > 0.999


def test_grape_worst_case_ensemble_improves_worst_case_transfer() -> None:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=1.0)
    parameterization = PiecewiseConstantParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-4.0 * np.pi, 4.0 * np.pi),
                quadrature="SCALAR",
            ),
        ),
    )
    objective = StateTransferObjective.single(
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
        name="flip",
    )
    nominal_problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x,),
                label="nominal",
            ),
        ),
        objectives=(objective,),
        ensemble_aggregate="mean",
    )
    robust_problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x,),
                label="scale_1_0",
            ),
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.35 * sigma_x,),
                label="scale_0_7",
            ),
        ),
        objectives=(objective,),
        ensemble_aggregate="worst",
    )

    solver = GrapeSolver(GrapeConfig(maxiter=60, seed=23, random_scale=0.4))
    nominal_result = solver.solve(nominal_problem)
    robust_result = solver.solve(robust_problem)

    def worst_case_fidelity(result) -> float:
        amplitude = float(result.schedule.values[0, 0])
        fidelities = []
        for scale in (1.0, 0.7):
            unitary = expm(-1j * amplitude * scale * 0.5 * sigma_x)
            final = unitary @ np.array([1.0, 0.0], dtype=np.complex128)
            fidelities.append(float(abs(final[1]) ** 2))
        return min(fidelities)

    assert nominal_result.success
    assert robust_result.success
    assert worst_case_fidelity(robust_result) > worst_case_fidelity(nominal_result)


def test_grape_noisy_state_transfer_matches_direct_liouvillian_overlap() -> None:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    collapse = np.sqrt(0.1) * sigma_minus
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=1.0)
    parameterization = PiecewiseConstantParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-4.0 * np.pi, 4.0 * np.pi),
                quadrature="SCALAR",
            ),
        ),
    )
    objective = StateTransferObjective.single(
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
        name="flip",
    )
    problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x,),
                collapse_operators=(collapse,),
                label="noisy",
            ),
        ),
        objectives=(objective,),
    )

    result = GrapeSolver(GrapeConfig(maxiter=60, seed=31, random_scale=0.4)).solve(problem)

    assert result.success
    assert result.nominal_final_unitary is None
    report = result.system_metrics[0]["objectives"][0]
    assert report["metric_type"] == "density_overlap"

    amplitude = float(result.schedule.values[0, 0])
    hamiltonian = amplitude * 0.5 * sigma_x
    identity = np.eye(2, dtype=np.complex128)
    cd_c = collapse.conj().T @ collapse
    liouvillian = -1j * (np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity))
    liouvillian += np.kron(collapse.conj(), collapse)
    liouvillian -= 0.5 * np.kron(identity, cd_c)
    liouvillian -= 0.5 * np.kron(cd_c.T, identity)
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho_final = (expm(liouvillian) @ rho0.T.reshape(-1)).reshape(2, 2).T
    target = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    target_overlap = float(np.real(np.trace(target @ rho_final)))

    assert report["fidelity_weighted"] == pytest.approx(target_overlap, abs=1.0e-6)
    assert target_overlap > 0.8


def test_density_matrix_transfer_objective_supports_closed_system_mixed_targets() -> None:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=1.0)
    parameterization = PiecewiseConstantParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-4.0 * np.pi, 4.0 * np.pi),
                quadrature="SCALAR",
            ),
        ),
    )
    maximally_mixed = 0.5 * np.eye(2, dtype=np.complex128)
    objective = DensityMatrixTransferObjective.single(
        maximally_mixed,
        maximally_mixed,
        name="preserve_mixed_state",
    )
    problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x,),
                label="closed_density",
            ),
        ),
        objectives=(objective,),
    )

    result = GrapeSolver(GrapeConfig(maxiter=5, seed=7, random_scale=0.1)).solve(
        problem,
        initial_schedule=np.array([[0.0]], dtype=float),
    )

    assert result.success
    report = result.system_metrics[0]["objectives"][0]
    assert report["metric_type"] == "density_overlap"
    assert report["fidelity_weighted"] == pytest.approx(1.0, abs=1.0e-10)