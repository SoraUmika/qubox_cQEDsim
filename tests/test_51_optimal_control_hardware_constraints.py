"""Tests for hardware-constrained GRAPE extensions.

Covers:
- HeldSampleParameterization expand / pullback
- Finite-difference gradient checks for all three HardwareMap types
- Backward compatibility: no hardware model → command == physical
- End-to-end hardware-aware GRAPE solve with waveform diagnostics
"""
from __future__ import annotations

import numpy as np
import pytest

from cqed_sim import (
    BoundaryWindowHardwareMap,
    ControlProblem,
    ControlSystem,
    ControlTerm,
    DispersiveTransmonCavityModel,
    FirstOrderLowPassHardwareMap,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    HardwareModel,
    HeldSampleParameterization,
    ModelControlChannelSpec,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    SmoothIQRadiusLimitHardwareMap,
    build_control_problem_from_model,
    resolve_control_schedule,
    state_preparation_objective,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


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


def _scalar_control_terms() -> tuple[ControlTerm, ...]:
    return (
        ControlTerm(
            name="x_drive",
            operator=0.5 * _sigma_x(),
            amplitude_bounds=(-1.0, 1.0),
            quadrature="SCALAR",
        ),
    )


def _iq_control_terms() -> tuple[ControlTerm, ...]:
    op = 0.5 * _sigma_x()
    return (
        ControlTerm(
            name="qubit_I",
            operator=op,
            amplitude_bounds=(-1.0, 1.0),
            quadrature="I",
            export_channel="qubit",
        ),
        ControlTerm(
            name="qubit_Q",
            operator=op,
            amplitude_bounds=(-1.0, 1.0),
            quadrature="Q",
            export_channel="qubit",
        ),
    )


def _finite_difference_pullback(
    hardware_model: HardwareModel,
    command: np.ndarray,
    weights: np.ndarray,
    control_terms: tuple[ControlTerm, ...],
    time_grid: PiecewiseConstantTimeGrid,
    *,
    epsilon: float = 1.0e-7,
) -> np.ndarray:
    """Central-difference estimate of the pullback (dL/d command) for L = sum(physical * weights)."""
    fd = np.zeros_like(command)
    for flat_index in range(command.size):
        perturb = np.zeros_like(command)
        perturb.reshape(-1)[flat_index] = epsilon
        physical_plus, _, _, _ = hardware_model.apply(
            command + perturb, control_terms=control_terms, time_grid=time_grid
        )
        physical_minus, _, _, _ = hardware_model.apply(
            command - perturb, control_terms=control_terms, time_grid=time_grid
        )
        loss_plus = float(np.sum(physical_plus * weights))
        loss_minus = float(np.sum(physical_minus * weights))
        fd.reshape(-1)[flat_index] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return fd


# ---------------------------------------------------------------------------
# HeldSampleParameterization
# ---------------------------------------------------------------------------


def test_held_sample_parameterization_expands_and_pullback_accumulates() -> None:
    parameterization = HeldSampleParameterization(
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=2.0e-9),
        control_terms=_scalar_control_terms(),
        sample_period_s=4.0e-9,
    )
    values = np.array([[1.0, 2.0, 3.0]], dtype=float)
    expanded = parameterization.command_values(values)
    gradient_command = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=float)
    reduced = parameterization.pullback(gradient_command, values)

    assert parameterization.n_slices == 3
    assert np.allclose(expanded, np.array([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]], dtype=float))
    assert np.allclose(reduced, np.array([[3.0, 7.0, 11.0]], dtype=float))


# ---------------------------------------------------------------------------
# FirstOrderLowPassHardwareMap — finite-difference gradient check
# ---------------------------------------------------------------------------


def test_first_order_lowpass_pullback_matches_finite_difference() -> None:
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=4, dt_s=5.0e-9)
    control_terms = _scalar_control_terms()
    command = np.array([[0.1, -0.25, 0.35, 0.2]], dtype=float)
    hardware_model = HardwareModel(maps=(FirstOrderLowPassHardwareMap(cutoff_hz=25.0e6),))
    _physical, _reports, _metrics, pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    weights = np.array([[0.7, -0.4, 0.2, 0.5]], dtype=float)

    analytic = pullback(weights)
    fd = _finite_difference_pullback(hardware_model, command, weights, control_terms, time_grid)

    assert np.allclose(analytic, fd, atol=1.0e-6, rtol=1.0e-5)


def test_first_order_lowpass_single_step_is_identity() -> None:
    """For a single time step the filter output must equal the input (no previous state)."""
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=20.0e-9)
    control_terms = _scalar_control_terms()
    command = np.array([[0.42]], dtype=float)
    hardware_model = HardwareModel(maps=(FirstOrderLowPassHardwareMap(cutoff_hz=50.0e6),))
    physical, _reports, _metrics, _pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    assert np.allclose(physical, command)


def test_first_order_lowpass_steady_state_converges() -> None:
    """A constant input signal should produce an output that converges to the input."""
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=50, dt_s=5.0e-9)
    control_terms = _scalar_control_terms()
    command = np.ones((1, 50), dtype=float) * 0.5
    hardware_model = HardwareModel(maps=(FirstOrderLowPassHardwareMap(cutoff_hz=25.0e6),))
    physical, _reports, metrics, _pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    # After 50 steps at 5 ns each (total 250 ns) with f_c = 25 MHz (tau ≈ 6.4 ns),
    # the filter should be very close to the input value.
    assert float(abs(physical[0, -1] - 0.5)) < 1.0e-3


# ---------------------------------------------------------------------------
# BoundaryWindowHardwareMap — finite-difference gradient check
# ---------------------------------------------------------------------------


def test_boundary_window_pullback_matches_finite_difference() -> None:
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=5, dt_s=5.0e-9)
    control_terms = _scalar_control_terms()
    command = np.array([[0.1, -0.3, 0.5, 0.2, -0.1]], dtype=float)
    hardware_model = HardwareModel(maps=(BoundaryWindowHardwareMap(ramp_slices=2),))
    _physical, _reports, _metrics, pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    weights = np.array([[0.4, -0.2, 0.7, 0.3, -0.5]], dtype=float)

    analytic = pullback(weights)
    fd = _finite_difference_pullback(hardware_model, command, weights, control_terms, time_grid)

    assert np.allclose(analytic, fd, atol=1.0e-6, rtol=1.0e-5)


def test_boundary_window_enforces_zero_endpoints() -> None:
    """ramp_slices=1 with apply_start=True/apply_end=True must zero both endpoints."""
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=4, dt_s=5.0e-9)
    control_terms = _scalar_control_terms()
    command = np.array([[1.0, 0.5, 0.5, 1.0]], dtype=float)
    hardware_model = HardwareModel(maps=(BoundaryWindowHardwareMap(ramp_slices=1),))
    physical, _reports, _metrics, _pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    assert physical[0, 0] == pytest.approx(0.0, abs=1.0e-12)
    assert physical[0, -1] == pytest.approx(0.0, abs=1.0e-12)
    assert float(abs(physical[0, 1])) > 0.0  # interior unchanged


# ---------------------------------------------------------------------------
# SmoothIQRadiusLimitHardwareMap — finite-difference gradient check
# ---------------------------------------------------------------------------


def test_smooth_iq_radius_pullback_matches_finite_difference() -> None:
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=3, dt_s=5.0e-9)
    control_terms = _iq_control_terms()
    # Deliberately exceed amplitude_max (0.8) in some slices to exercise the clipping branch.
    command = np.array([[0.7, 0.4, 0.3], [0.6, 0.8, 0.1]], dtype=float)
    amplitude_max = 0.8
    hardware_model = HardwareModel(
        maps=(SmoothIQRadiusLimitHardwareMap(amplitude_max=amplitude_max, export_channels=("qubit",)),)
    )
    _physical, _reports, _metrics, pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    weights = np.array([[0.3, 0.5, -0.2], [0.1, -0.4, 0.6]], dtype=float)

    analytic = pullback(weights)
    fd = _finite_difference_pullback(hardware_model, command, weights, control_terms, time_grid)

    assert np.allclose(analytic, fd, atol=1.0e-6, rtol=1.0e-5)


def test_smooth_iq_radius_below_limit_is_near_identity() -> None:
    """When all IQ radii are well below amplitude_max the physical waveform should be close to the command.

    The tanh-based saturation is never the true identity (tanh(z)/z < 1 for z > 0), but for
    r << amplitude_max the deviation is O((r/amplitude_max)^2).  Using amplitude_max = 10.0
    and r ≤ 0.3, we have z ≤ 0.03, so the relative error is < 0.03 %.
    """
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=3, dt_s=5.0e-9)
    control_terms = _iq_control_terms()
    command = np.array([[0.2, 0.1, 0.3], [0.1, 0.2, 0.0]], dtype=float)
    amplitude_max = 10.0  # r/amplitude_max ≤ 0.03 → deviation < 3e-4 relative
    hardware_model = HardwareModel(
        maps=(SmoothIQRadiusLimitHardwareMap(amplitude_max=amplitude_max, export_channels=("qubit",)),)
    )
    physical, reports, _metrics, _pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    assert np.allclose(physical, command, rtol=5.0e-4, atol=0.0)
    assert float(reports[0].metrics["clipping_fraction"]) == pytest.approx(0.0)


def test_smooth_iq_radius_clipping_fraction_reported() -> None:
    """The clipping_fraction metric (in the per-map report) reflects slices exceeding amplitude_max."""
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=4, dt_s=5.0e-9)
    control_terms = _iq_control_terms()
    # radii ≈ [0.98, 0.14, 1.27, 0.0] — slices 0 and 2 exceed amplitude_max=0.5
    command = np.array([[0.9, 0.1, 0.9, 0.0], [0.4, 0.1, 0.9, 0.0]], dtype=float)
    amplitude_max = 0.5
    hardware_model = HardwareModel(
        maps=(SmoothIQRadiusLimitHardwareMap(amplitude_max=amplitude_max, export_channels=("qubit",)),)
    )
    _physical, reports, _aggregate_metrics, _pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    assert len(reports) == 1
    assert reports[0].name == "SmoothIQRadiusLimitHardwareMap"
    # clipping_fraction is in the per-map report (not the aggregate HardwareModel metrics)
    map_metrics = reports[0].metrics
    # 2 of 4 slices exceed the limit → fraction = 0.5
    assert float(map_metrics["clipping_fraction"]) == pytest.approx(0.5, abs=0.01)
    assert float(map_metrics["max_command_radius"]) > amplitude_max
    assert float(map_metrics["max_physical_radius"]) <= amplitude_max + 0.01


# ---------------------------------------------------------------------------
# Backward compatibility: no hardware model
# ---------------------------------------------------------------------------


def test_no_hardware_model_command_equals_physical() -> None:
    """When no hardware model is attached, command_values must equal physical_values."""
    model, frame = _qubit_only_model()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=4, dt_s=10.0e-9)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("I", "Q"),
                amplitude_bounds=(-5.0e7, 5.0e7),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
        # no hardware_model
    )
    assert problem.hardware_model is None

    schedule = problem.parameterization.zero_schedule()
    resolved = resolve_control_schedule(problem, schedule, apply_hardware=True)

    assert np.allclose(resolved.command_values, resolved.physical_values)
    assert not resolved.hardware_metrics.get("hardware_active", True)
    assert resolved.hardware_metrics.get("hardware_map_count", -1) == 0


def test_backward_compat_piecewise_constant_no_hardware_grape() -> None:
    """PiecewiseConstantParameterization without a HardwareModel: result.command_values == physical_values."""
    model, frame = _qubit_only_model()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=4, dt_s=10.0e-9)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("I", "Q"),
                amplitude_bounds=(-5.0e7, 5.0e7),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
    )
    result = GrapeSolver(GrapeConfig(maxiter=5, seed=0)).solve(problem)

    assert result.command_values is not None
    assert result.physical_values is not None
    assert result.command_values.shape == (2, 4)
    assert result.physical_values.shape == (2, 4)
    assert np.allclose(result.command_values, result.physical_values), (
        "Without a hardware model, command and physical waveforms must be identical."
    )
    assert not result.hardware_metrics.get("hardware_active", True)
    assert len(result.hardware_reports) == 0


# ---------------------------------------------------------------------------
# Hardware-aware GRAPE: end-to-end solve with diagnostics
# ---------------------------------------------------------------------------


def _hardware_aware_problem() -> tuple[DispersiveTransmonCavityModel, FrameSpec, object]:
    model, frame = _qubit_only_model()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=20.0e-9)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("I", "Q"),
                amplitude_bounds=(-8.0e7, 8.0e7),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
        parameterization_cls=HeldSampleParameterization,
        parameterization_kwargs={"sample_period_s": 40.0e-9},
        hardware_model=HardwareModel(
            maps=(
                FirstOrderLowPassHardwareMap(cutoff_hz=25.0e6, export_channels=("qubit",)),
                SmoothIQRadiusLimitHardwareMap(amplitude_max=6.0e7, export_channels=("qubit",)),
                BoundaryWindowHardwareMap(ramp_slices=1, export_channels=("qubit",)),
            )
        ),
    )
    return model, frame, problem


def test_hardware_aware_grape_records_waveforms_and_supports_replay_modes() -> None:
    model, frame, problem = _hardware_aware_problem()
    initial_schedule = np.array(
        [[0.0, 0.0, 0.0], [3.0e7, 3.0e7, 3.0e7]],
        dtype=float,
    )
    result = GrapeSolver(
        GrapeConfig(
            maxiter=80,
            seed=5,
            random_scale=0.15,
            apply_hardware_in_forward_model=True,
            report_command_reference=True,
        )
    ).solve(problem, initial_schedule=initial_schedule)

    command_radii = np.sqrt(np.square(result.command_values[0, :]) + np.square(result.command_values[1, :]))
    physical_radii = np.sqrt(np.square(result.physical_values[0, :]) + np.square(result.physical_values[1, :]))

    command_replay = result.evaluate_with_simulator(
        problem, model=model, frame=frame, compiler_dt_s=1.0e-9, waveform_mode="command"
    )
    physical_replay = result.evaluate_with_simulator(
        problem, model=model, frame=frame, compiler_dt_s=1.0e-9, waveform_mode="physical"
    )

    # Schedule shape: HeldSampleParameterization with n_slices=3 coarse samples
    assert result.success
    assert result.schedule.values.shape == (2, 3)
    # Command and physical live on the propagation grid (6 time slices)
    assert result.command_values.shape == (2, 6)
    assert result.physical_values.shape == (2, 6)
    # Parameterization and hardware metadata are populated
    assert result.parameterization_metrics["parameterization"] == "HeldSampleParameterization"
    assert len(result.hardware_reports) == 3
    # Physical fidelity should be high after 80 iterations
    assert result.metrics["nominal_physical_fidelity"] > 0.98
    # Command-domain fidelity is finite (computed as a reference)
    assert np.isfinite(result.metrics["nominal_command_fidelity"])
    # IQ radius limit enforced: all physical radii ≤ amplitude_max (6e7) within tolerance
    assert np.max(physical_radii) <= 6.0e7 + 1.0e-6
    # Boundary window: first and last physical samples must be (near) zero
    assert result.physical_values[0, 0] == pytest.approx(0.0, abs=1.0e-10)
    assert result.physical_values[1, 0] == pytest.approx(0.0, abs=1.0e-10)
    assert result.physical_values[0, -1] == pytest.approx(0.0, abs=1.0e-10)
    assert result.physical_values[1, -1] == pytest.approx(0.0, abs=1.0e-10)
    # Command and physical radii differ (hardware transforms are active)
    assert not np.allclose(command_radii, physical_radii)
    # Replay metadata
    assert command_replay.waveform_mode == "command"
    assert physical_replay.waveform_mode == "physical"
    assert physical_replay.metrics["aggregate_fidelity"] > 0.98


def test_hardware_aware_grape_hardware_reports_are_named() -> None:
    """hardware_reports tuple must contain named entries for each applied map."""
    _model, _frame, problem = _hardware_aware_problem()
    result = GrapeSolver(GrapeConfig(maxiter=5, seed=0)).solve(problem)

    names = [report["name"] for report in result.hardware_reports]
    assert names == [
        "FirstOrderLowPassHardwareMap",
        "SmoothIQRadiusLimitHardwareMap",
        "BoundaryWindowHardwareMap",
    ]
    for report in result.hardware_reports:
        assert isinstance(report["metrics"], dict)


def test_hardware_model_chaining_preserves_shape() -> None:
    """Composing multiple HardwareMap instances must not change the waveform shape."""
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=8, dt_s=5.0e-9)
    control_terms = _iq_control_terms()
    command = np.random.default_rng(42).standard_normal((2, 8)) * 0.3
    hardware_model = HardwareModel(
        maps=(
            FirstOrderLowPassHardwareMap(cutoff_hz=50.0e6),
            SmoothIQRadiusLimitHardwareMap(amplitude_max=0.5, export_channels=("qubit",)),
            BoundaryWindowHardwareMap(ramp_slices=2),
        )
    )
    physical, reports, metrics, pullback = hardware_model.apply(
        command, control_terms=control_terms, time_grid=time_grid
    )
    assert physical.shape == command.shape
    assert len(reports) == 3
    assert metrics["hardware_active"] is True
    assert metrics["hardware_map_count"] == 3

    # Pullback must also preserve the shape
    grad_physical = np.ones_like(command)
    grad_command = pullback(grad_physical)
    assert grad_command.shape == command.shape
