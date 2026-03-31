from __future__ import annotations

import numpy as np

from cqed_sim import (
    CallableParameterization,
    CallablePulseFamily,
    ControlParameterSpec,
    ControlProblem,
    ControlSystem,
    ControlTerm,
    GateTimeOptimizationConfig,
    GrapeConfig,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    PulseParameterSpec,
    StateTransferObjective,
    StructuredControlChannel,
    StructuredControlConfig,
    StructuredPulseParameterization,
    GaussianDragPulseFamily,
    optimize_gate_time_with_grape,
    solve_grape,
    solve_structured_then_grape,
)


def _sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _sigma_y() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)


def _flip_objective() -> StateTransferObjective:
    return StateTransferObjective.single(
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
        name="flip",
    )


def test_callable_parameterization_supports_named_waveform_ansatz() -> None:
    sigma_x = _sigma_x()
    sigma_y = _sigma_y()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=1.0)

    def evaluator(values, _time_grid, _control_terms) -> np.ndarray:
        amplitude = float(values[0])
        phase = float(values[1])
        return np.array(
            [[amplitude * np.cos(phase)], [amplitude * np.sin(phase)]],
            dtype=float,
        )

    def pullback(gradient_command, values, _time_grid, _control_terms, _waveform) -> np.ndarray:
        amplitude = float(values[0])
        phase = float(values[1])
        gradient = np.asarray(gradient_command, dtype=float)
        return np.array(
            [
                gradient[0, 0] * np.cos(phase) + gradient[1, 0] * np.sin(phase),
                gradient[0, 0] * (-amplitude * np.sin(phase)) + gradient[1, 0] * (amplitude * np.cos(phase)),
            ],
            dtype=float,
        )

    parameterization = CallableParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-2.0 * np.pi, 2.0 * np.pi),
                quadrature="SCALAR",
            ),
            ControlTerm(
                name="y_drive",
                operator=0.5 * sigma_y,
                amplitude_bounds=(-2.0 * np.pi, 2.0 * np.pi),
                quadrature="SCALAR",
            ),
        ),
        parameter_specs=(
            ControlParameterSpec("amplitude", 0.0, 2.0 * np.pi, default=1.0),
            ControlParameterSpec("phase", -np.pi, np.pi, default=0.0, units="rad"),
        ),
        evaluator=evaluator,
        pullback_evaluator=pullback,
    )
    problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x, 0.5 * sigma_y),
                label="callable_parameterization_system",
            ),
        ),
        objectives=(_flip_objective(),),
    )

    result = solve_grape(
        problem,
        config=GrapeConfig(maxiter=45, seed=5, random_scale=0.2),
        initial_schedule=np.array([1.1, 0.7], dtype=float),
    )

    assert result.success
    assert result.metrics["nominal_fidelity"] > 0.999
    assert result.parameterization_metrics["parameterization"] == "CallableParameterization"
    assert result.parameterization_metrics["parameter_names"] == ["amplitude", "phase"]


def test_callable_pulse_family_integrates_with_structured_pullback() -> None:
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=10, dt_s=5.0e-9)
    control_terms = (
        ControlTerm(
            name="drive_I",
            operator=0.5 * _sigma_x(),
            amplitude_bounds=(-6.0e7, 6.0e7),
            export_channel="drive",
            quadrature="I",
        ),
        ControlTerm(
            name="drive_Q",
            operator=0.5 * _sigma_y(),
            amplitude_bounds=(-6.0e7, 6.0e7),
            export_channel="drive",
            quadrature="Q",
        ),
    )

    family = CallablePulseFamily(
        name="phasor",
        specs=(
            PulseParameterSpec("amplitude", 0.0, 5.0e7, 2.0e7, units="rad/s"),
            PulseParameterSpec("phase_rad", -np.pi, np.pi, 0.2, units="rad"),
        ),
        evaluator=lambda time_rel_s, _duration_s, values: np.asarray(
            float(values[0]) * np.exp(1j * float(values[1])) * np.ones_like(time_rel_s, dtype=np.complex128),
            dtype=np.complex128,
        ),
        jacobian_evaluator=lambda time_rel_s, _duration_s, values: np.asarray(
            [
                np.exp(1j * float(values[1])) * np.ones_like(time_rel_s, dtype=np.complex128),
                1j * float(values[0]) * np.exp(1j * float(values[1])) * np.ones_like(time_rel_s, dtype=np.complex128),
            ],
            dtype=np.complex128,
        ),
    )
    parameterization = StructuredPulseParameterization(
        time_grid=time_grid,
        control_terms=control_terms,
        channels=(StructuredControlChannel(name="phasor", pulse_family=family, export_channel="drive"),),
    )
    values = np.array([2.7e7, -0.35], dtype=float)
    weights = np.vstack(
        [
            np.linspace(0.3, -0.2, time_grid.steps),
            np.linspace(-0.1, 0.4, time_grid.steps),
        ]
    )

    analytic = parameterization.pullback(weights, values)
    finite_difference = np.zeros_like(analytic)
    for index in range(values.size):
        epsilon = 1.0e-6 * max(abs(float(values[index])), 1.0)
        plus = np.array(values, copy=True)
        minus = np.array(values, copy=True)
        plus[index] += epsilon
        minus[index] -= epsilon
        loss_plus = float(np.sum(parameterization.command_values(parameterization.clip(plus)) * weights))
        loss_minus = float(np.sum(parameterization.command_values(parameterization.clip(minus)) * weights))
        finite_difference[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

    assert np.allclose(analytic, finite_difference, atol=1.0e-4, rtol=2.0e-4)


def test_gate_time_optimization_with_grape_finds_feasible_duration() -> None:
    sigma_x = _sigma_x()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=1.0)
    parameterization = PiecewiseConstantParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-1.0, 1.0),
                quadrature="SCALAR",
            ),
        ),
    )
    problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x,),
                label="gate_time_system",
            ),
        ),
        objectives=(_flip_objective(),),
    )

    sweep = optimize_gate_time_with_grape(
        problem,
        durations_s=(2.0, 3.2, 4.0),
        config=GrapeConfig(maxiter=35, seed=3, random_scale=0.2),
        gate_time_config=GateTimeOptimizationConfig(warm_start_strategy="previous_best"),
        initial_schedule=np.array([[0.4]], dtype=float),
    )

    assert len(sweep.candidates) == 3
    assert sweep.candidates[0].result.metrics["nominal_fidelity"] < 0.9
    assert sweep.best_result.metrics["nominal_fidelity"] > 0.999
    assert sweep.best_duration_s > 3.0
    assert sweep.metrics["candidate_count"] == 3


def test_structured_then_grape_uses_command_warm_start() -> None:
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=12, dt_s=0.1)
    sigma_x = _sigma_x()
    sigma_y = _sigma_y()
    parameterization = StructuredPulseParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="drive_I",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-6.0, 6.0),
                export_channel="drive",
                quadrature="I",
            ),
            ControlTerm(
                name="drive_Q",
                operator=0.5 * sigma_y,
                amplitude_bounds=(-6.0, 6.0),
                export_channel="drive",
                quadrature="Q",
            ),
        ),
        channels=(
            StructuredControlChannel(
                name="gaussian_drag",
                pulse_family=GaussianDragPulseFamily(
                    amplitude_bounds=(0.0, 5.5),
                    sigma_fraction_bounds=(0.12, 0.3),
                    center_fraction_bounds=(0.35, 0.65),
                    phase_bounds=(-np.pi, np.pi),
                    drag_bounds=(-0.25, 0.25),
                    default_amplitude=2.5,
                    default_phase=0.0,
                ),
                export_channel="drive",
            ),
        ),
    )
    problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x, 0.5 * sigma_y),
                label="structured_then_grape_system",
            ),
        ),
        objectives=(_flip_objective(),),
    )

    hybrid = solve_structured_then_grape(
        problem,
        structured_config=StructuredControlConfig(maxiter=35, seed=4, initial_guess="random", random_scale=0.2),
        grape_config=GrapeConfig(maxiter=35, seed=4, random_scale=0.2),
    )

    assert hybrid.structured_result.success
    assert hybrid.grape_result.success
    assert np.allclose(hybrid.warm_start_schedule.values, hybrid.structured_result.command_values)
    assert hybrid.grape_result.objective_value <= hybrid.structured_result.objective_value + 1.0e-10
    assert hybrid.metrics["objective_improvement"] >= -1.0e-10