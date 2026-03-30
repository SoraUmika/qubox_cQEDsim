from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cqed_sim import (
    ControlProblem,
    ControlSystem,
    ControlTerm,
    CustomControlObjective,
    CustomObjectiveContext,
    CustomObjectiveEvaluation,
    DispersiveTransmonCavityModel,
    FirstOrderLowPassHardwareMap,
    FourierSeriesPulseFamily,
    FrameSpec,
    GainHardwareMap,
    GaussianDragPulseFamily,
    HardwareModel,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    StructuredControlChannel,
    StructuredControlConfig,
    StructuredPulseParameterization,
    build_structured_control_problem_from_model,
    save_structured_control_artifacts,
    solve_structured_control,
    state_preparation_objective,
)


def _qubit_only_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
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


def _finite_difference_waveform_jacobian(family, time_rel_s: np.ndarray, duration_s: float, values: np.ndarray) -> np.ndarray:
    base_values = family.clip(values)
    jacobian = np.zeros((len(family.parameter_specs), time_rel_s.size), dtype=np.complex128)
    for index, spec in enumerate(family.parameter_specs):
        epsilon = 1.0e-6 * max(abs(float(base_values[index])), abs(float(spec.upper_bound)), 1.0)
        plus = np.array(base_values, copy=True)
        minus = np.array(base_values, copy=True)
        plus[index] = spec.clip(float(base_values[index]) + epsilon)
        minus[index] = spec.clip(float(base_values[index]) - epsilon)
        delta = float(plus[index] - minus[index])
        jacobian[index, :] = (family.evaluate(time_rel_s, duration_s, plus) - family.evaluate(time_rel_s, duration_s, minus)) / delta
    return jacobian


def test_gaussian_drag_family_jacobian_matches_finite_difference() -> None:
    family = GaussianDragPulseFamily(
        amplitude_bounds=(0.0, 8.0e7),
        sigma_fraction_bounds=(0.1, 0.3),
        center_fraction_bounds=(0.4, 0.6),
        phase_bounds=(-np.pi, np.pi),
        drag_bounds=(-0.4, 0.4),
    )
    duration_s = 120.0e-9
    time_rel_s = np.linspace(0.5e-9, duration_s - 0.5e-9, 24)
    values = np.array([4.0e7, 0.18, 0.52, -0.4 * np.pi, 0.12], dtype=float)

    _waveform, analytic = family.waveform_and_jacobian(time_rel_s, duration_s, values)
    finite_difference = _finite_difference_waveform_jacobian(family, time_rel_s, duration_s, values)

    assert np.allclose(analytic, finite_difference, atol=5.0e-4, rtol=2.0e-4)


def test_fourier_family_jacobian_matches_finite_difference() -> None:
    family = FourierSeriesPulseFamily(n_modes=3, coefficient_bound=4.0e7)
    duration_s = 100.0e-9
    time_rel_s = np.linspace(0.5e-9, duration_s - 0.5e-9, 20)
    values = np.array([1.0e7, -2.5e6, 1.5e6, 3.0e6, -1.0e6, 5.0e6, -1.5e6, 2.0e6, -3.5e6, 2.5e6], dtype=float)

    _waveform, analytic = family.waveform_and_jacobian(time_rel_s, duration_s, values)
    finite_difference = _finite_difference_waveform_jacobian(family, time_rel_s, duration_s, values)

    assert np.allclose(analytic, finite_difference, atol=1.0e-8, rtol=1.0e-8)


def test_structured_parameterization_pullback_matches_finite_difference() -> None:
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=12, dt_s=10.0e-9)
    control_terms = (
        ControlTerm(
            name="qubit_I",
            operator=0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            amplitude_bounds=(-8.0e7, 8.0e7),
            export_channel="qubit",
            quadrature="I",
        ),
        ControlTerm(
            name="qubit_Q",
            operator=0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            amplitude_bounds=(-8.0e7, 8.0e7),
            export_channel="qubit",
            quadrature="Q",
        ),
    )
    parameterization = StructuredPulseParameterization(
        time_grid=time_grid,
        control_terms=control_terms,
        channels=(
            StructuredControlChannel(
                name="gaussian",
                pulse_family=GaussianDragPulseFamily(
                    amplitude_bounds=(0.0, 7.0e7),
                    sigma_fraction_bounds=(0.12, 0.22),
                    center_fraction_bounds=(0.45, 0.55),
                    phase_bounds=(-np.pi, np.pi),
                    drag_bounds=(-0.25, 0.25),
                ),
                export_channel="qubit",
            ),
        ),
    )
    values = np.array([3.2e7, 0.18, 0.5, -0.45 * np.pi, 0.08], dtype=float)
    weights = np.vstack(
        [
            np.linspace(0.2, -0.3, time_grid.steps),
            np.linspace(-0.1, 0.25, time_grid.steps),
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


def test_structured_fourier_q_quadrature_uses_positive_imaginary_baseband() -> None:
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=4, dt_s=5.0e-9)
    parameterization = StructuredPulseParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="qubit_I",
                operator=0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
                amplitude_bounds=(-4.0e7, 4.0e7),
                export_channel="qubit",
                quadrature="I",
            ),
            ControlTerm(
                name="qubit_Q",
                operator=0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
                amplitude_bounds=(-4.0e7, 4.0e7),
                export_channel="qubit",
                quadrature="Q",
            ),
        ),
        channels=(
            StructuredControlChannel(
                name="fourier",
                pulse_family=FourierSeriesPulseFamily(n_modes=1, coefficient_bound=4.0e7),
                export_channel="qubit",
            ),
        ),
    )

    values = np.array([0.0, 2.5e7], dtype=float)
    command = parameterization.command_values(values)
    pulses, _drive_ops, metadata = parameterization.to_pulses(values, waveform_values=command)

    assert np.allclose(command[0, :], 0.0)
    assert np.allclose(command[1, :], 2.5e7)
    assert "c(t) = I(t) + i Q(t)" in metadata["mapping"]
    assert len(pulses) == time_grid.steps
    assert np.isclose(pulses[0].phase, 0.5 * np.pi)


def test_structured_solver_runs_hardware_aware_model_workflow_and_saves_artifacts(tmp_path: Path) -> None:
    model, frame = _qubit_only_model()
    problem = build_structured_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=24, dt_s=6.0e-9),
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("I", "Q"),
                amplitude_bounds=(-8.0e7, 8.0e7),
                export_channel="qubit",
            ),
        ),
        structured_channels=(
            StructuredControlChannel(
                name="gaussian_drag",
                pulse_family=GaussianDragPulseFamily(
                    amplitude_bounds=(0.0, 7.0e7),
                    sigma_fraction_bounds=(0.12, 0.24),
                    center_fraction_bounds=(0.42, 0.58),
                    phase_bounds=(-np.pi, np.pi),
                    drag_bounds=(-0.25, 0.25),
                    default_phase=-0.5 * np.pi,
                ),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
        hardware_model=HardwareModel(
            maps=(
                FirstOrderLowPassHardwareMap(cutoff_hz=32.0e6, export_channels=("qubit",)),
                GainHardwareMap(gain=0.94, export_channels=("qubit",)),
            )
        ),
        metadata={"test": "structured_solver_hardware_aware"},
    )

    result = solve_structured_control(
        problem,
        config=StructuredControlConfig(maxiter=45, seed=3, initial_guess="random", random_scale=0.2),
    )

    assert result.success
    assert result.backend == "structured-control"
    assert result.metrics["nominal_physical_fidelity"] > 0.98
    assert not np.allclose(result.command_values, result.physical_values)

    artifacts = save_structured_control_artifacts(problem, result, tmp_path / "structured_demo")
    assert artifacts.result_json.exists()
    assert artifacts.parameters_csv.exists()
    assert artifacts.waveforms_csv.exists()
    assert artifacts.history_csv.exists()
    assert artifacts.waveform_plot.exists()
    assert artifacts.spectrum_plot.exists()
    assert artifacts.history_plot.exists()

    with artifacts.parameters_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(problem.parameterization.parameter_specs)


def test_custom_control_objective_runs_with_structured_solver() -> None:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=16, dt_s=1.0 / 16.0)
    parameterization = StructuredPulseParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-4.0 * np.pi, 4.0 * np.pi),
                quadrature="SCALAR",
            ),
        ),
        channels=(
            StructuredControlChannel(
                name="x_gaussian",
                pulse_family=GaussianDragPulseFamily(
                    amplitude_bounds=(0.0, 4.0 * np.pi),
                    sigma_fraction_bounds=(0.18, 0.18),
                    center_fraction_bounds=(0.5, 0.5),
                    phase_bounds=(0.0, 0.0),
                    drag_bounds=(0.0, 0.0),
                    default_amplitude=2.0 * np.pi,
                    default_phase=0.0,
                    default_drag=0.0,
                ),
                control_names=("x_drive",),
            ),
        ),
    )

    def evaluator(context: CustomObjectiveContext) -> CustomObjectiveEvaluation:
        dt = float(np.mean(context.problem.time_grid.step_durations_s))
        area = float(np.sum(context.resolved_waveforms.physical_values[0, :]) * dt)
        target_area = np.pi
        cost = float((area - target_area) ** 2)
        gradient = np.zeros_like(context.resolved_waveforms.physical_values, dtype=float)
        gradient[0, :] = 2.0 * (area - target_area) * dt
        return CustomObjectiveEvaluation(
            cost=cost,
            gradient_physical=gradient,
            metrics={"integrated_area": area, "target_area": target_area},
        )

    problem = ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x,),
                label="custom_metric_system",
            ),
        ),
        objectives=(CustomControlObjective(evaluator=evaluator, name="target_area"),),
    )

    result = solve_structured_control(problem, config=StructuredControlConfig(maxiter=35, initial_guess="defaults"))
    dt = float(np.mean(problem.time_grid.step_durations_s))
    area = float(np.sum(result.physical_values[0, :]) * dt)

    assert result.success
    assert abs(area - np.pi) < 1.0e-2
    assert result.system_metrics[0]["objectives"][0]["kind"] == "custom"