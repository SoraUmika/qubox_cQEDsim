from __future__ import annotations

import numpy as np
import qutip as qt
from scipy.linalg import expm

from cqed_sim import (
    ControlProblem,
    ControlSystem,
    ControlTerm,
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