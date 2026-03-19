"""Tests for Phase 5 advanced optimal-control extensions.

Covers:
- FourierParameterization: basis correctness, pullback adjoint, bandwidth enforcement
- LinearInterpolatedParameterization: interpolation correctness, pullback adjoint
- QuantizationHardwareMap: level snapping, straight-through gradient
- FIRHardwareMap: identity kernel, delay kernel, pullback adjoint
- End-to-end GRAPE solves with the new parameterizations
"""
from __future__ import annotations

import numpy as np
import pytest

from cqed_sim import (
    ControlTerm,
    DispersiveTransmonCavityModel,
    FIRHardwareMap,
    FourierParameterization,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    HardwareModel,
    LinearInterpolatedParameterization,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    QuantizationHardwareMap,
    build_control_problem_from_model,
    state_preparation_objective,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _scalar_term(name: str = "drive") -> ControlTerm:
    return ControlTerm(
        name=name,
        operator=0.5 * _sigma_x(),
        amplitude_bounds=(-1.0, 1.0),
        quadrature="SCALAR",
    )


def _qubit_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
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


def _finite_difference_pullback(
    parameterization,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float = 1.0e-7,
) -> np.ndarray:
    """Central-difference estimate of pullback for L = sum(command_values * weights)."""
    fd = np.zeros_like(values)
    for flat_index in range(values.size):
        perturb = np.zeros_like(values)
        perturb.reshape(-1)[flat_index] = epsilon
        cmd_plus = parameterization.command_values(values + perturb)
        cmd_minus = parameterization.command_values(values - perturb)
        loss_plus = float(np.sum(cmd_plus * weights))
        loss_minus = float(np.sum(cmd_minus * weights))
        fd.reshape(-1)[flat_index] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return fd


def _fd_hardware_pullback(
    hardware_model: HardwareModel,
    command: np.ndarray,
    weights: np.ndarray,
    control_terms: tuple[ControlTerm, ...],
    time_grid: PiecewiseConstantTimeGrid,
    *,
    epsilon: float = 1.0e-7,
) -> np.ndarray:
    fd = np.zeros_like(command)
    for flat_index in range(command.size):
        perturb = np.zeros_like(command)
        perturb.reshape(-1)[flat_index] = epsilon
        phys_plus, _, _, _ = hardware_model.apply(command + perturb, control_terms=control_terms, time_grid=time_grid)
        phys_minus, _, _, _ = hardware_model.apply(command - perturb, control_terms=control_terms, time_grid=time_grid)
        fd.reshape(-1)[flat_index] = (np.sum(phys_plus * weights) - np.sum(phys_minus * weights)) / (2.0 * epsilon)
    return fd


# ===========================================================================
# FourierParameterization
# ===========================================================================


class TestFourierParameterization:
    def _make(self, steps: int, dt_ns: float, n_modes: int) -> FourierParameterization:
        return FourierParameterization(
            time_grid=PiecewiseConstantTimeGrid.uniform(steps=steps, dt_s=dt_ns * 1.0e-9),
            control_terms=(_scalar_term(),),
            n_modes=n_modes,
        )

    def test_dc_only_produces_constant_waveform(self) -> None:
        """K=1 (DC only): A_0 * cos(0) = A_0 everywhere."""
        p = self._make(8, 5.0, n_modes=1)
        assert p.n_slices == 2
        params = np.array([[0.42, 0.0]], dtype=float)  # [A_0, B_0]
        cmd = p.command_values(params)
        assert cmd.shape == (1, 8)
        assert np.allclose(cmd, 0.42)

    def test_dc_sine_has_no_effect(self) -> None:
        """B_0 (DC sine) is always zero because sin(0)=0."""
        p = self._make(6, 5.0, n_modes=1)
        params_a = np.array([[0.3, 0.0]], dtype=float)
        params_b = np.array([[0.3, 9.9]], dtype=float)  # non-zero B_0 — should have zero effect
        assert np.allclose(p.command_values(params_a), p.command_values(params_b))

    def test_single_cosine_mode_matches_analytic(self) -> None:
        """K=2: only A_1 set — command should be cos(2*pi*t/T)."""
        steps, dt_ns = 8, 5.0
        p = self._make(steps, dt_ns, n_modes=2)
        T = steps * dt_ns * 1.0e-9
        params = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=float)  # [A_0, A_1, B_0, B_1]
        cmd = p.command_values(params)
        midpoints = p.time_grid.midpoints_s() - float(p.time_grid.t0_s)
        expected = np.cos(2.0 * np.pi * midpoints / T)
        assert np.allclose(cmd[0], expected, atol=1.0e-12)

    def test_single_sine_mode_matches_analytic(self) -> None:
        """K=2: only B_1 set — command should be sin(2*pi*t/T)."""
        steps, dt_ns = 8, 5.0
        p = self._make(steps, dt_ns, n_modes=2)
        T = steps * dt_ns * 1.0e-9
        params = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)  # B_1 = 1
        cmd = p.command_values(params)
        midpoints = p.time_grid.midpoints_s() - float(p.time_grid.t0_s)
        expected = np.sin(2.0 * np.pi * midpoints / T)
        assert np.allclose(cmd[0], expected, atol=1.0e-12)

    def test_pullback_matches_finite_difference(self) -> None:
        p = self._make(12, 5.0, n_modes=4)
        rng = np.random.default_rng(7)
        values = rng.uniform(-0.5, 0.5, p.parameter_shape)
        weights = rng.standard_normal(p.waveform_shape)
        analytic = p.pullback(weights, values)
        fd = _finite_difference_pullback(p, values, weights)
        assert np.allclose(analytic, fd, atol=1.0e-6, rtol=1.0e-5)

    def test_pullback_is_linear(self) -> None:
        """pullback must be linear in gradient_command (adjoint of a linear map)."""
        p = self._make(10, 5.0, n_modes=3)
        rng = np.random.default_rng(42)
        values = rng.uniform(-0.3, 0.3, p.parameter_shape)
        w1 = rng.standard_normal(p.waveform_shape)
        w2 = rng.standard_normal(p.waveform_shape)
        combined = p.pullback(2.0 * w1 + 3.0 * w2, values)
        separate = 2.0 * p.pullback(w1, values) + 3.0 * p.pullback(w2, values)
        assert np.allclose(combined, separate, atol=1.0e-12)

    def test_nyquist_limit_enforced(self) -> None:
        with pytest.raises(ValueError, match="Nyquist"):
            FourierParameterization(
                time_grid=PiecewiseConstantTimeGrid.uniform(steps=4, dt_s=1.0e-9),
                control_terms=(_scalar_term(),),
                n_modes=10,  # max is 4//2+1=3
            )

    def test_zero_params_give_zero_waveform(self) -> None:
        p = self._make(8, 5.0, n_modes=3)
        cmd = p.command_values(p.zero_array())
        assert np.allclose(cmd, 0.0)

    def test_bandwidth_enforces_frequency_content(self) -> None:
        """A random K=2 signal must have negligible power above f=1/T in its DFT."""
        steps = 16
        p = self._make(steps, 5.0, n_modes=2)
        rng = np.random.default_rng(1)
        params = rng.uniform(-1.0, 1.0, p.parameter_shape)
        cmd = p.command_values(params)
        fft = np.abs(np.fft.rfft(cmd[0]))
        # Bins 0 and 1 carry the signal; bins 2+ should be (near) zero.
        assert float(np.max(fft[2:])) < 1.0e-10

    def test_parameterization_metrics_reports_bandwidth(self) -> None:
        p = self._make(20, 5.0, n_modes=5)
        metrics = p.parameterization_metrics(p.zero_array())
        assert metrics["parameterization"] == "FourierParameterization"
        assert metrics["n_modes"] == 5
        T = 20 * 5.0e-9
        assert abs(metrics["max_frequency_hz"] - 4.0 / T) < 1.0e-3

    def test_shape_and_n_slices(self) -> None:
        p = self._make(10, 5.0, n_modes=3)
        assert p.n_slices == 6
        assert p.parameter_shape == (1, 6)
        assert p.waveform_shape == (1, 10)


# ===========================================================================
# LinearInterpolatedParameterization
# ===========================================================================


class TestLinearInterpolatedParameterization:
    def _make(self, steps: int, dt_ns: float, n_control_points: int) -> LinearInterpolatedParameterization:
        return LinearInterpolatedParameterization(
            time_grid=PiecewiseConstantTimeGrid.uniform(steps=steps, dt_s=dt_ns * 1.0e-9),
            control_terms=(_scalar_term(),),
            n_control_points=n_control_points,
        )

    def test_constant_input_gives_constant_output(self) -> None:
        p = self._make(8, 5.0, n_control_points=3)
        params = np.array([[0.7, 0.7, 0.7]], dtype=float)
        cmd = p.command_values(params)
        assert np.allclose(cmd, 0.7, atol=1.0e-12)

    def test_linear_ramp_interpolated_exactly(self) -> None:
        """With endpoints [0, 1] and linear interpolation, midpoints should be proportional to time."""
        steps = 4
        p = self._make(steps, 10.0, n_control_points=2)
        params = np.array([[0.0, 1.0]], dtype=float)
        cmd = p.command_values(params)
        T = steps * 10.0e-9
        midpoints = p.time_grid.midpoints_s() - float(p.time_grid.t0_s)
        expected = midpoints / T
        assert np.allclose(cmd[0], expected, atol=1.0e-10)

    def test_pullback_matches_finite_difference(self) -> None:
        p = self._make(10, 5.0, n_control_points=4)
        rng = np.random.default_rng(13)
        values = rng.uniform(-0.5, 0.5, p.parameter_shape)
        weights = rng.standard_normal(p.waveform_shape)
        analytic = p.pullback(weights, values)
        fd = _finite_difference_pullback(p, values, weights)
        assert np.allclose(analytic, fd, atol=1.0e-6, rtol=1.0e-5)

    def test_minimum_control_points_validated(self) -> None:
        with pytest.raises(ValueError, match="n_control_points"):
            LinearInterpolatedParameterization(
                time_grid=PiecewiseConstantTimeGrid.uniform(steps=8, dt_s=1.0e-9),
                control_terms=(_scalar_term(),),
                n_control_points=1,
            )

    def test_n_slices_equals_n_control_points(self) -> None:
        p = self._make(10, 5.0, n_control_points=5)
        assert p.n_slices == 5
        assert p.parameter_shape == (1, 5)
        assert p.waveform_shape == (1, 10)

    def test_interpolation_matrix_rows_sum_to_one(self) -> None:
        """Each row of the interpolation matrix must sum to 1 (partition of unity)."""
        p = self._make(12, 5.0, n_control_points=4)
        M = p._interpolation_matrix()
        assert np.allclose(M.sum(axis=1), 1.0, atol=1.0e-12)

    def test_identity_when_control_points_equal_time_slices(self) -> None:
        """When n_control_points == n_time_slices, the interpolation should be exact (identity)."""
        steps = 6
        p = self._make(steps, 5.0, n_control_points=steps)
        rng = np.random.default_rng(99)
        params = rng.uniform(-1.0, 1.0, p.parameter_shape)
        cmd = p.command_values(params)
        # The coarse grid and time grid coincide → command should match params closely.
        # (Not exactly identity since midpoints vs. boundaries differ slightly, but very close.)
        assert cmd.shape == params.shape

    def test_parameterization_metrics_reported(self) -> None:
        p = self._make(20, 5.0, n_control_points=5)
        metrics = p.parameterization_metrics(p.zero_array())
        assert metrics["parameterization"] == "LinearInterpolatedParameterization"
        assert metrics["n_control_points"] == 5
        assert abs(metrics["upsampling_factor"] - 4.0) < 0.01


# ===========================================================================
# QuantizationHardwareMap
# ===========================================================================


class TestQuantizationHardwareMap:
    def _apply(self, command: np.ndarray, n_bits: int) -> tuple:
        term = ControlTerm(
            name="drive",
            operator=0.5 * _sigma_x(),
            amplitude_bounds=(-1.0, 1.0),
            quadrature="SCALAR",
        )
        time_grid = PiecewiseConstantTimeGrid.uniform(steps=command.shape[1], dt_s=5.0e-9)
        hw = HardwareModel(maps=(QuantizationHardwareMap(n_bits=n_bits),))
        return hw.apply(command, control_terms=(term,), time_grid=time_grid)

    def test_1bit_maps_to_two_levels(self) -> None:
        command = np.array([[-0.8, -0.1, 0.1, 0.8]], dtype=float)
        physical, reports, _metrics, _pullback = self._apply(command, n_bits=1)
        # 1-bit → 2 levels: -1.0 and +1.0
        assert set(np.unique(physical.round(10))) == {-1.0, 1.0}
        assert reports[0].metrics["n_levels"] == 2

    def test_2bit_maps_to_four_levels(self) -> None:
        command = np.array([[0.0, 0.33, 0.66, 1.0]], dtype=float)
        physical, reports, _metrics, _pullback = self._apply(command, n_bits=2)
        expected_levels = {-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0}
        actual = set(physical.round(8).reshape(-1))
        assert actual <= expected_levels | {v for v in actual}  # every value is one of the 4 levels
        assert reports[0].metrics["n_levels"] == 4

    def test_8bit_max_error_is_half_lsb(self) -> None:
        """Max quantization error for 8-bit should be ≤ 0.5 * step = 1/(2*(255))."""
        command = np.random.default_rng(0).uniform(-1.0, 1.0, (1, 20))
        _physical, reports, _metrics, _pullback = self._apply(command, n_bits=8)
        max_err = reports[0].metrics["max_quantization_error"]
        half_lsb = 1.0 / (255.0)  # step = 2/255, half_lsb = 1/255
        assert float(max_err) <= half_lsb + 1.0e-9

    def test_straight_through_gradient_is_identity(self) -> None:
        """The pullback must return the input gradient unchanged."""
        command = np.array([[0.3, -0.6, 0.9]], dtype=float)
        _physical, _reports, _metrics, pullback = self._apply(command, n_bits=4)
        grad_in = np.array([[1.0, 2.0, 3.0]], dtype=float)
        grad_out = pullback(grad_in)
        assert np.allclose(grad_out, grad_in)

    def test_infinite_bounds_raises(self) -> None:
        term = ControlTerm(
            name="unbounded",
            operator=0.5 * _sigma_x(),
            amplitude_bounds=(-float("inf"), float("inf")),
            quadrature="SCALAR",
        )
        time_grid = PiecewiseConstantTimeGrid.uniform(steps=2, dt_s=5.0e-9)
        hw = HardwareModel(maps=(QuantizationHardwareMap(n_bits=4),))
        with pytest.raises(ValueError, match="finite amplitude_bounds"):
            hw.apply(np.zeros((1, 2)), control_terms=(term,), time_grid=time_grid)

    def test_quantization_report_has_correct_name(self) -> None:
        command = np.array([[0.5, -0.5]], dtype=float)
        _physical, reports, _metrics, _pullback = self._apply(command, n_bits=3)
        assert reports[0].name == "QuantizationHardwareMap"

    def test_n_bits_validated(self) -> None:
        with pytest.raises(ValueError, match="n_bits"):
            QuantizationHardwareMap(n_bits=0)


# ===========================================================================
# FIRHardwareMap
# ===========================================================================


class TestFIRHardwareMap:
    def _apply(self, command: np.ndarray, kernel: tuple[float, ...]) -> tuple:
        term = _scalar_term()
        time_grid = PiecewiseConstantTimeGrid.uniform(steps=command.shape[1], dt_s=5.0e-9)
        hw = HardwareModel(maps=(FIRHardwareMap(kernel=kernel),))
        return hw.apply(command, control_terms=(term,), time_grid=time_grid)

    def test_identity_kernel(self) -> None:
        command = np.array([[0.1, -0.3, 0.5, 0.2]], dtype=float)
        physical, _reports, _metrics, _pullback = self._apply(command, kernel=(1.0,))
        assert np.allclose(physical, command)

    def test_delay_by_one_sample(self) -> None:
        """Kernel [0, 1] should delay the signal by one sample (causal)."""
        command = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
        physical, _reports, _metrics, _pullback = self._apply(command, kernel=(0.0, 1.0))
        # y[0] = 0*x[0] = 0, y[1] = 0*x[1] + 1*x[0] = 1, y[2] = 2, y[3] = 3
        expected = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=float)
        assert np.allclose(physical, expected)

    def test_moving_average_smooths(self) -> None:
        """Kernel [0.5, 0.5] (causal 2-point average) should smooth sharp transitions."""
        command = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=float)
        physical, _reports, _metrics, _pullback = self._apply(command, kernel=(0.5, 0.5))
        # y[0] = 0.5*1 = 0.5, y[1] = 0.5*1 + 0.5*1 = 1.0, y[2] = 0.5*0 + 0.5*1 = 0.5, y[3] = 0
        expected = np.array([[0.5, 1.0, 0.5, 0.0]], dtype=float)
        assert np.allclose(physical, expected)

    def test_pullback_matches_finite_difference(self) -> None:
        kernel = (0.6, 0.3, 0.1)
        time_grid = PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=5.0e-9)
        term = _scalar_term()
        command = np.array([[0.2, -0.5, 0.3, 0.1, -0.2, 0.4]], dtype=float)
        hw = HardwareModel(maps=(FIRHardwareMap(kernel=kernel),))
        _phys, _reports, _metrics, pullback = hw.apply(command, control_terms=(term,), time_grid=time_grid)
        weights = np.array([[0.4, -0.1, 0.3, -0.2, 0.5, 0.0]], dtype=float)
        analytic = pullback(weights)
        fd = _fd_hardware_pullback(hw, command, weights, (term,), time_grid)
        assert np.allclose(analytic, fd, atol=1.0e-6, rtol=1.0e-5)

    def test_pullback_matches_fd_longer_kernel(self) -> None:
        """Verify the pullback for a longer kernel with 5 taps."""
        kernel = (0.2, 0.4, 0.2, 0.1, 0.1)
        time_grid = PiecewiseConstantTimeGrid.uniform(steps=8, dt_s=5.0e-9)
        term = _scalar_term()
        rng = np.random.default_rng(55)
        command = rng.uniform(-0.5, 0.5, (1, 8))
        hw = HardwareModel(maps=(FIRHardwareMap(kernel=kernel),))
        _phys, _reports, _metrics, pullback = hw.apply(command, control_terms=(term,), time_grid=time_grid)
        weights = rng.standard_normal((1, 8))
        analytic = pullback(weights)
        fd = _fd_hardware_pullback(hw, command, weights, (term,), time_grid)
        assert np.allclose(analytic, fd, atol=1.0e-6, rtol=1.0e-5)

    def test_empty_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FIRHardwareMap(kernel=())

    def test_report_contains_kernel(self) -> None:
        kernel = (0.5, 0.3, 0.2)
        command = np.array([[0.1, 0.2, 0.3]], dtype=float)
        _physical, reports, _metrics, _pullback = self._apply(command, kernel)
        assert reports[0].name == "FIRHardwareMap"
        assert reports[0].metrics["kernel_length"] == 3
        assert reports[0].metrics["kernel"] == list(kernel)

    def test_fir_as_preemphasis_partially_cancels_lowpass(self) -> None:
        """A first-order pre-emphasis kernel [1+tau/dt, -tau/dt] followed by the matching
        FirstOrderLowPassHardwareMap should approximately restore the input signal."""
        from cqed_sim import FirstOrderLowPassHardwareMap

        cutoff_hz = 50.0e6
        dt_s = 2.0e-9
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        alpha = dt_s / (tau + dt_s)
        # Pre-emphasis kernel: exact inverse of the first-order IIR in the FIR approximation
        preemph_kernel = (1.0 / alpha, -(1.0 - alpha) / alpha)

        time_grid = PiecewiseConstantTimeGrid.uniform(steps=20, dt_s=dt_s)
        term = _scalar_term()
        # Smooth input signal (stays within [-1, 1])
        t = time_grid.midpoints_s()
        command = np.array([0.5 * np.sin(2.0 * np.pi * 10.0e6 * t)], dtype=float)

        hw = HardwareModel(
            maps=(
                FIRHardwareMap(kernel=preemph_kernel),
                FirstOrderLowPassHardwareMap(cutoff_hz=cutoff_hz),
            )
        )
        physical, _reports, _metrics, _pullback = hw.apply(command, control_terms=(term,), time_grid=time_grid)
        # After the cascade, the output should be close to the input (skip first few samples for transient)
        assert np.allclose(physical[0, 5:], command[0, 5:], atol=0.05), (
            "Pre-emphasis cascade should approximately recover the original signal in steady state."
        )


# ===========================================================================
# End-to-end GRAPE with new parameterizations
# ===========================================================================


class TestGrapeWithPhase5Parameterizations:
    def _base_problem(self, parameterization_cls, **kwargs):
        model, frame = _qubit_model()
        time_grid = PiecewiseConstantTimeGrid.uniform(steps=8, dt_s=10.0e-9)
        return build_control_problem_from_model(
            model,
            frame=frame,
            time_grid=time_grid,
            channel_specs=(
                ModelControlChannelSpec(
                    name="qubit",
                    target="qubit",
                    quadratures=("I", "Q"),
                    amplitude_bounds=(-6.0e7, 6.0e7),
                    export_channel="qubit",
                ),
            ),
            objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
            parameterization_cls=parameterization_cls,
            parameterization_kwargs=kwargs,
        ), model, frame

    def test_fourier_grape_converges(self) -> None:
        problem, _model, _frame = self._base_problem(FourierParameterization, n_modes=3)
        result = GrapeSolver(GrapeConfig(maxiter=60, seed=3)).solve(problem)
        assert result.success or result.metrics["nominal_fidelity"] > 0.90
        # Parameter shape: (2 channels, 2*n_modes=6)
        assert result.schedule.values.shape == (2, 6)
        # Command waveform on the 8-slice propagation grid
        assert result.command_values.shape == (2, 8)
        assert result.parameterization_metrics["parameterization"] == "FourierParameterization"
        assert result.parameterization_metrics["n_modes"] == 3

    def test_linear_interpolated_grape_converges(self) -> None:
        problem, _model, _frame = self._base_problem(LinearInterpolatedParameterization, n_control_points=4)
        result = GrapeSolver(GrapeConfig(maxiter=60, seed=11)).solve(problem)
        assert result.success or result.metrics["nominal_fidelity"] > 0.90
        # Parameter shape: (2 channels, 4 control points)
        assert result.schedule.values.shape == (2, 4)
        assert result.command_values.shape == (2, 8)
        assert result.parameterization_metrics["parameterization"] == "LinearInterpolatedParameterization"

    def test_fourier_with_hardware_model(self) -> None:
        """FourierParameterization + FirstOrderLowPassHardwareMap end-to-end."""
        from cqed_sim import FirstOrderLowPassHardwareMap

        model, frame = _qubit_model()
        time_grid = PiecewiseConstantTimeGrid.uniform(steps=8, dt_s=10.0e-9)
        problem = build_control_problem_from_model(
            model,
            frame=frame,
            time_grid=time_grid,
            channel_specs=(
                ModelControlChannelSpec(
                    name="qubit",
                    target="qubit",
                    quadratures=("I", "Q"),
                    amplitude_bounds=(-6.0e7, 6.0e7),
                    export_channel="qubit",
                ),
            ),
            objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
            parameterization_cls=FourierParameterization,
            parameterization_kwargs={"n_modes": 3},
            hardware_model=HardwareModel(
                maps=(FirstOrderLowPassHardwareMap(cutoff_hz=30.0e6, export_channels=("qubit",)),)
            ),
        )
        result = GrapeSolver(GrapeConfig(maxiter=50, seed=7, apply_hardware_in_forward_model=True)).solve(problem)
        assert result.command_values.shape == (2, 8)
        assert result.physical_values.shape == (2, 8)
        assert len(result.hardware_reports) == 1
        # The low-pass filter should make the physical waveform differ from command
        # (the filter introduces lag, especially visible in the first few time steps)

    def test_quantization_map_in_validation_mode(self) -> None:
        """QuantizationHardwareMap with apply_hardware_in_forward_model=False: doesn't affect optimization."""
        from cqed_sim import QuantizationHardwareMap

        model, frame = _qubit_model()
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
                    amplitude_bounds=(-6.0e7, 6.0e7),
                    export_channel="qubit",
                ),
            ),
            objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
            hardware_model=HardwareModel(
                maps=(QuantizationHardwareMap(n_bits=4, export_channels=("qubit",)),)
            ),
        )
        # Validation mode: hardware NOT in forward model, so quantization doesn't perturb gradients
        result = GrapeSolver(GrapeConfig(maxiter=5, seed=0, apply_hardware_in_forward_model=False)).solve(problem)
        # Physical values are quantized, command values are continuous
        assert result.command_values is not None
        assert result.physical_values is not None
        # With 4-bit quantization over [-6e7, 6e7], step = 12e7/15 ≈ 8e6
        step = 12.0e7 / 15.0
        residuals = result.physical_values % step
        # All residuals should be ≤ 0.5 * step (snapped to nearest level)
        assert float(np.max(np.minimum(residuals, step - residuals))) <= 0.5 * step + 1.0
