"""Tests for the hardware-aware control transfer layer.

Coverage:
 11.1  identity    – ControlLine with no maps matches raw waveform
 11.2  gain        – GainHardwareMap / calibration_gain scales waveforms
 11.3  delay       – DelayHardwareMap shifts waveform correctly
 11.4  FIR         – FIRHardwareMap convolves correctly
 11.5  per-line    – different lines produce different outputs from same input
 11.6  parity      – manually pre-transformed waveform gives identical simulation result
 11.7  GRAPE post  – postprocess_grape_waveforms applies hardware via Mode A
 11.8  regression  – no hardware_context → behaviour unchanged
 11.9  GainHardwareMap gradient check
 11.10 DelayHardwareMap gradient check
 11.11 HardwareContext.as_hardware_model() builds correct combined model
 11.12 SequenceCompiler with hardware_context integrates end-to-end
 11.13 make_three_line_cqed_context produces a valid HardwareContext
 11.14 delay_samples_from_time helper
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports from the package
# ---------------------------------------------------------------------------
from cqed_sim.control import (
    ControlLine,
    HardwareContext,
    delay_samples_from_time,
    postprocess_grape_waveforms,
)
from cqed_sim.control.cqed_device import make_three_line_cqed_context
from cqed_sim.optimal_control.hardware import (
    DelayHardwareMap,
    FIRHardwareMap,
    FirstOrderLowPassHardwareMap,
    GainHardwareMap,
    HardwareModel,
    _AppliedHardwareMap,
)
from cqed_sim.optimal_control.parameterizations import PiecewiseConstantTimeGrid
from cqed_sim.optimal_control.problems import ControlTerm
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def _make_dummy_control_terms(n: int = 1, *, quadrature: str = "SCALAR", export_channel: str = "ch") -> tuple[ControlTerm, ...]:
    op = np.eye(2, dtype=np.complex128)
    return tuple(
        ControlTerm(
            name=f"ctrl_{i}",
            operator=op,
            quadrature=quadrature,
            export_channel=export_channel,
        )
        for i in range(n)
    )


def _uniform_grid(n: int, dt: float = 1e-9) -> PiecewiseConstantTimeGrid:
    return PiecewiseConstantTimeGrid.uniform(steps=n, dt_s=dt)


# ---------------------------------------------------------------------------
# 11.1  Identity transfer
# ---------------------------------------------------------------------------

class TestIdentityTransfer:
    def test_no_maps_passthrough(self):
        """ControlLine with empty transfer_maps returns calibration_gain * waveform."""
        waveform = np.array([1.0 + 2j, 3.0 - 1j, 0.5 + 0j], dtype=np.complex128)
        line = ControlLine(name="q", transfer_maps=(), calibration_gain=1.0)
        out = line.apply_to_waveform(waveform, dt=1e-9)
        np.testing.assert_allclose(out, waveform, atol=1e-14)

    def test_identity_gain_passthrough(self):
        """GainHardwareMap(1.0) is an exact identity."""
        waveform = np.random.randn(50) + 1j * np.random.randn(50)
        line = ControlLine(name="q", transfer_maps=(GainHardwareMap(gain=1.0),))
        out = line.apply_to_waveform(waveform, dt=1e-9)
        np.testing.assert_allclose(out, waveform, atol=1e-14)

    def test_empty_context_passthrough_in_compiler(self):
        """SequenceCompiler with an empty HardwareContext leaves distorted unchanged."""
        dt = 0.5
        ctx = HardwareContext(lines={})
        pulses = [Pulse("ch", 0.0, 4.0, _square, amp=1.0)]
        compiled_without = SequenceCompiler(dt=dt).compile(pulses)
        compiled_with = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses)
        np.testing.assert_array_equal(
            compiled_without.channels["ch"].distorted,
            compiled_with.channels["ch"].distorted,
        )


# ---------------------------------------------------------------------------
# 11.2  Gain transfer
# ---------------------------------------------------------------------------

class TestGainTransfer:
    def test_scalar_gain(self):
        """GainHardwareMap scales the waveform by the expected factor."""
        waveform = np.ones(10, dtype=np.complex128) * (1.0 + 1j)
        g = 2.5
        line = ControlLine(name="q", transfer_maps=(GainHardwareMap(gain=g),))
        out = line.apply_to_waveform(waveform, dt=1e-9)
        np.testing.assert_allclose(out, g * waveform, atol=1e-13)

    def test_calibration_gain_applied_after_maps(self):
        """calibration_gain multiplies AFTER transfer_maps."""
        waveform = np.array([1.0, 2.0, 3.0], dtype=complex)
        hw_gain = 2.0
        cal_gain = 3.0
        line = ControlLine(
            name="q",
            transfer_maps=(GainHardwareMap(gain=hw_gain),),
            calibration_gain=cal_gain,
        )
        out = line.apply_to_waveform(waveform, dt=1e-9)
        expected = waveform * hw_gain * cal_gain
        np.testing.assert_allclose(out, expected, atol=1e-13)

    def test_calibration_gain_only(self):
        """calibration_gain without any transfer map scales the waveform."""
        waveform = np.array([1.0 + 0j, 0.0 + 1j], dtype=np.complex128)
        alpha = 2e9 * np.pi
        line = ControlLine(name="q", calibration_gain=alpha)
        out = line.apply_to_waveform(waveform, dt=1e-9)
        np.testing.assert_allclose(out, alpha * waveform, atol=1e-5)

    def test_gain_in_compiler(self):
        """SequenceCompiler with a GainHardwareMap context scales .distorted."""
        dt = 0.5
        g = 2.0
        ctx = HardwareContext(lines={
            "ch": ControlLine("ch", transfer_maps=(GainHardwareMap(gain=g),)),
        })
        pulses = [Pulse("ch", 0.0, 3.0, _square, amp=1.0)]
        compiled_base = SequenceCompiler(dt=dt).compile(pulses)
        compiled_ctx = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses)
        np.testing.assert_allclose(
            compiled_ctx.channels["ch"].distorted,
            g * compiled_base.channels["ch"].distorted,
            atol=1e-13,
        )


# ---------------------------------------------------------------------------
# 11.3  Delay transfer
# ---------------------------------------------------------------------------

class TestDelayTransfer:
    def test_shift_by_n_samples(self):
        """DelayHardwareMap shifts the waveform by the correct number of samples."""
        n = 10
        delay = 3
        waveform = np.arange(n, dtype=complex)
        line = ControlLine(name="q", transfer_maps=(DelayHardwareMap(delay_samples=delay),))
        out = line.apply_to_waveform(waveform, dt=1e-9)
        # First `delay` samples should be zero; rest shifted
        np.testing.assert_allclose(out[:delay], 0.0, atol=1e-14)
        np.testing.assert_allclose(out[delay:], waveform[: n - delay], atol=1e-14)

    def test_zero_delay_is_identity(self):
        """DelayHardwareMap(0) is an identity."""
        waveform = np.exp(1j * np.linspace(0, 2 * np.pi, 20))
        line = ControlLine(name="q", transfer_maps=(DelayHardwareMap(delay_samples=0),))
        out = line.apply_to_waveform(waveform, dt=1e-9)
        np.testing.assert_allclose(out, waveform, atol=1e-14)

    def test_delay_exceeds_length_gives_zeros(self):
        """When delay ≥ n_steps the output should be all zeros."""
        n = 5
        waveform = np.ones(n, dtype=complex)
        line = ControlLine(name="q", transfer_maps=(DelayHardwareMap(delay_samples=n),))
        out = line.apply_to_waveform(waveform, dt=1e-9)
        np.testing.assert_allclose(out, 0.0, atol=1e-14)

    def test_delay_samples_from_time_helper(self):
        """delay_samples_from_time converts correctly."""
        assert delay_samples_from_time(3e-9, 1e-9) == 3
        assert delay_samples_from_time(2.4e-9, 1e-9) == 2
        assert delay_samples_from_time(2.6e-9, 1e-9) == 3
        assert delay_samples_from_time(0.0, 1e-9) == 0


# ---------------------------------------------------------------------------
# 11.4  FIR filter
# ---------------------------------------------------------------------------

class TestFIRTransfer:
    def test_single_tap_identity_kernel(self):
        """FIRHardwareMap with kernel [1.0] is an identity."""
        waveform = np.array([1.0, -1.0, 0.5, 0.0], dtype=complex)
        line = ControlLine(name="q", transfer_maps=(FIRHardwareMap(kernel=(1.0,)),))
        out = line.apply_to_waveform(waveform, dt=1e-9)
        np.testing.assert_allclose(out, waveform, atol=1e-14)

    def test_averaging_kernel(self):
        """FIRHardwareMap with kernel [0.5, 0.5] computes running average."""
        waveform = np.array([2.0, 0.0, 2.0, 0.0], dtype=float) + 0j
        # Expected: y[0]=1.0, y[1]=1.0, y[2]=1.0, y[3]=1.0
        # causal: y[0]=h[0]*x[0]=1.0, y[1]=h[0]*x[1]+h[1]*x[0]=0+1=1, etc.
        line = ControlLine(name="q", transfer_maps=(FIRHardwareMap(kernel=(0.5, 0.5)),))
        out = line.apply_to_waveform(waveform, dt=1e-9)
        expected = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        np.testing.assert_allclose(out.real, expected, atol=1e-13)

    def test_fir_complex_waveform(self):
        """FIR with complex input applies correctly to real and imaginary parts."""
        kernel = (0.6, 0.4)
        n = 8
        real_in = np.random.randn(n)
        imag_in = np.random.randn(n)
        waveform = real_in + 1j * imag_in

        line = ControlLine(name="q", transfer_maps=(FIRHardwareMap(kernel=kernel),))
        out = line.apply_to_waveform(waveform, dt=1e-9)

        # Manually compute expected FIR on real and imag separately
        h = np.asarray(kernel, dtype=float)
        def fir_manual(x):
            y = np.zeros_like(x)
            for l, c in enumerate(h):
                y[l:] += c * x[:n - l]
            return y

        np.testing.assert_allclose(out.real, fir_manual(real_in), atol=1e-13)
        np.testing.assert_allclose(out.imag, fir_manual(imag_in), atol=1e-13)


# ---------------------------------------------------------------------------
# 11.5  Per-line independence
# ---------------------------------------------------------------------------

class TestPerLineIndependence:
    def test_different_lines_different_outputs(self):
        """Two differently configured lines produce different outputs from the same input."""
        waveform = np.ones(10, dtype=complex)
        ctx = HardwareContext(lines={
            "qubit":   ControlLine("qubit",   transfer_maps=(GainHardwareMap(2.0),)),
            "storage": ControlLine("storage", transfer_maps=(GainHardwareMap(0.5),)),
        })
        out_q = ctx.apply_to_waveform("qubit", waveform, dt=1e-9)
        out_s = ctx.apply_to_waveform("storage", waveform, dt=1e-9)
        assert not np.allclose(out_q, out_s)
        np.testing.assert_allclose(out_q, 2.0 * waveform, atol=1e-13)
        np.testing.assert_allclose(out_s, 0.5 * waveform, atol=1e-13)

    def test_missing_line_in_compiler_unchanged(self):
        """Channels without a matching ControlLine are left unchanged by the context."""
        dt = 0.5
        ctx = HardwareContext(lines={
            "q": ControlLine("q", transfer_maps=(GainHardwareMap(3.0),)),
        })
        pulses = [
            Pulse("q",    0.0, 2.0, _square, amp=1.0),
            Pulse("cav",  0.0, 2.0, _square, amp=1.0),
        ]
        compiled = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses)
        compiled_base = SequenceCompiler(dt=dt).compile(pulses)
        # "cav" should be unchanged
        np.testing.assert_array_equal(
            compiled.channels["cav"].distorted,
            compiled_base.channels["cav"].distorted,
        )
        # "q" should be scaled
        np.testing.assert_allclose(
            compiled.channels["q"].distorted,
            3.0 * compiled_base.channels["q"].distorted,
            atol=1e-13,
        )


# ---------------------------------------------------------------------------
# 11.6  Simulation parity
# ---------------------------------------------------------------------------

class TestSimulationParity:
    """Manually pre-transforming the waveform gives the same compiled result
    as using hardware_context in the SequenceCompiler."""

    def test_gain_parity(self):
        """Manually scaling amp by g == hardware_context with GainHardwareMap(g)."""
        dt = 0.5
        g = 1.7
        pulses_raw = [Pulse("q", 0.0, 4.0, _square, amp=1.0)]
        pulses_scaled = [Pulse("q", 0.0, 4.0, _square, amp=g)]

        ctx = HardwareContext(lines={"q": ControlLine("q", transfer_maps=(GainHardwareMap(g),))})
        compiled_ctx = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses_raw)
        compiled_manual = SequenceCompiler(dt=dt).compile(pulses_scaled)

        np.testing.assert_allclose(
            compiled_ctx.channels["q"].distorted,
            compiled_manual.channels["q"].distorted,
            atol=1e-13,
        )


# ---------------------------------------------------------------------------
# 11.7  GRAPE Mode A postprocessing
# ---------------------------------------------------------------------------

class TestGrapePostprocessing:
    """postprocess_grape_waveforms applies hardware context to GRAPE physical values."""

    def test_scalar_control_gain(self):
        """Scalar control with gain: physical_values row is scaled by gain."""
        n = 20
        physical_values = np.random.randn(1, n)
        control_terms = (
            ControlTerm(
                name="x_drive",
                operator=np.eye(2, dtype=np.complex128),
                quadrature="SCALAR",
                export_channel="qubit",
            ),
        )
        ctx = HardwareContext(lines={
            "qubit": ControlLine("qubit", transfer_maps=(GainHardwareMap(2.5),)),
        })
        result = postprocess_grape_waveforms(ctx, physical_values, control_terms, dt=1e-9)
        np.testing.assert_allclose(result[0], 2.5 * physical_values[0], atol=1e-13)

    def test_iq_pair_gain(self):
        """IQ pair on same export channel: complex phasor is scaled."""
        n = 16
        op = np.eye(2, dtype=np.complex128)
        control_terms = (
            ControlTerm(name="q_I", operator=op, quadrature="I", export_channel="qubit"),
            ControlTerm(name="q_Q", operator=op, quadrature="Q", export_channel="qubit"),
        )
        physical_values = np.random.randn(2, n)
        g = 1.5
        ctx = HardwareContext(lines={
            "qubit": ControlLine("qubit", transfer_maps=(GainHardwareMap(g),)),
        })
        result = postprocess_grape_waveforms(ctx, physical_values, control_terms, dt=1e-9)
        np.testing.assert_allclose(result[0], g * physical_values[0], atol=1e-13)
        np.testing.assert_allclose(result[1], g * physical_values[1], atol=1e-13)

    def test_unmatched_channel_unchanged(self):
        """Controls without a matching context line are left unchanged."""
        n = 10
        physical_values = np.random.randn(1, n)
        control_terms = (
            ControlTerm(
                name="x", operator=np.eye(2, dtype=np.complex128),
                quadrature="SCALAR", export_channel="other_ch",
            ),
        )
        ctx = HardwareContext(lines={
            "qubit": ControlLine("qubit", transfer_maps=(GainHardwareMap(5.0),)),
        })
        result = postprocess_grape_waveforms(ctx, physical_values, control_terms, dt=1e-9)
        np.testing.assert_array_equal(result, physical_values)

    def test_delay_in_postprocess(self):
        """Delay map shifts control in Mode A postprocessing."""
        n = 10
        delay = 2
        x = np.ones((1, n))
        control_terms = (
            ControlTerm(
                name="c", operator=np.eye(2, dtype=np.complex128),
                quadrature="SCALAR", export_channel="ch",
            ),
        )
        ctx = HardwareContext(lines={
            "ch": ControlLine("ch", transfer_maps=(DelayHardwareMap(delay_samples=delay),)),
        })
        result = postprocess_grape_waveforms(ctx, x, control_terms, dt=1e-9)
        np.testing.assert_allclose(result[0, :delay], 0.0, atol=1e-14)
        np.testing.assert_allclose(result[0, delay:], 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 11.8  Regression: no hardware_context → behaviour unchanged
# ---------------------------------------------------------------------------

class TestRegression:
    def test_sequence_compiler_backward_compat(self):
        """SequenceCompiler without hardware_context behaves exactly as before."""
        dt = 0.5
        pulses = [Pulse("q", 0.0, 3.0, _square, amp=2.0)]
        old_api = SequenceCompiler(dt=dt)
        new_api = SequenceCompiler(dt=dt, hardware_context=None)
        c_old = old_api.compile(pulses)
        c_new = new_api.compile(pulses)
        np.testing.assert_array_equal(c_old.channels["q"].distorted, c_new.channels["q"].distorted)
        np.testing.assert_array_equal(c_old.tlist, c_new.tlist)

    def test_identity_hardware_context_unchanged(self):
        """A HardwareContext with all-identity lines gives the same result as no context."""
        dt = 0.5
        ctx = HardwareContext(lines={
            "q": ControlLine("q", transfer_maps=(), calibration_gain=1.0),
        })
        pulses = [Pulse("q", 0.0, 3.0, _square, amp=1.5)]
        c_base = SequenceCompiler(dt=dt).compile(pulses)
        c_ctx = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses)
        np.testing.assert_allclose(
            c_ctx.channels["q"].distorted,
            c_base.channels["q"].distorted,
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# 11.9  GainHardwareMap gradient check
# ---------------------------------------------------------------------------

class TestGainHardwareMapGradient:
    """GainHardwareMap pullback is exact: dJ/dx = gain * dJ/dy."""

    def _run_map(self, values, gain):
        n_controls, n_steps = values.shape
        op = np.eye(2, dtype=np.complex128)
        control_terms = tuple(
            ControlTerm(name=f"c{i}", operator=op, quadrature="SCALAR")
            for i in range(n_controls)
        )
        grid = PiecewiseConstantTimeGrid.uniform(steps=n_steps, dt_s=1e-9)
        hw_map = GainHardwareMap(gain=gain)
        return hw_map.apply(values, control_terms=control_terms, time_grid=grid)

    def test_pullback_equals_gain_times_gradient(self):
        rng = np.random.default_rng(42)
        values = rng.standard_normal((2, 15))
        grad_output = rng.standard_normal((2, 15))
        gain = 3.7
        applied = self._run_map(values, gain)
        grad_input = applied.pullback(grad_output)
        np.testing.assert_allclose(grad_input, gain * grad_output, atol=1e-13)

    def test_output_is_gain_times_input(self):
        values = np.array([[1.0, 2.0, 3.0]])
        applied = self._run_map(values, gain=4.0)
        np.testing.assert_allclose(applied.values[0], [4.0, 8.0, 12.0], atol=1e-13)


# ---------------------------------------------------------------------------
# 11.10  DelayHardwareMap gradient check
# ---------------------------------------------------------------------------

class TestDelayHardwareMapGradient:
    """DelayHardwareMap pullback is exact: the transpose of the forward shift."""

    def _run_map(self, values, delay):
        n_controls, n_steps = values.shape
        op = np.eye(2, dtype=np.complex128)
        control_terms = tuple(
            ControlTerm(name=f"c{i}", operator=op, quadrature="SCALAR")
            for i in range(n_controls)
        )
        grid = PiecewiseConstantTimeGrid.uniform(steps=n_steps, dt_s=1e-9)
        hw_map = DelayHardwareMap(delay_samples=delay)
        return hw_map.apply(values, control_terms=control_terms, time_grid=grid)

    def test_finite_difference_gradient(self):
        """Pullback matches finite-difference gradient of L2 loss."""
        rng = np.random.default_rng(7)
        n = 20
        delay = 4
        x = rng.standard_normal((1, n))

        def forward(x_in):
            applied = self._run_map(x_in, delay)
            return applied.values.copy()

        # L2 loss: L = 0.5 * ||y||^2,  dL/dy = y
        y = forward(x)
        grad_y = y.copy()   # dL/dy

        applied = self._run_map(x, delay)
        grad_x_pullback = applied.pullback(grad_y)

        # Finite differences
        eps = 1e-5
        grad_x_fd = np.zeros_like(x)
        for i in range(n):
            x_plus = x.copy(); x_plus[0, i] += eps
            x_minus = x.copy(); x_minus[0, i] -= eps
            loss_plus = 0.5 * np.sum(forward(x_plus) ** 2)
            loss_minus = 0.5 * np.sum(forward(x_minus) ** 2)
            grad_x_fd[0, i] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(grad_x_pullback, grad_x_fd, atol=1e-7)

    def test_delay_forward_values(self):
        x = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
        applied = self._run_map(x, delay=2)
        np.testing.assert_allclose(applied.values[0], [0.0, 0.0, 0.0, 1.0, 2.0], atol=1e-14)


# ---------------------------------------------------------------------------
# 11.11  HardwareContext.as_hardware_model
# ---------------------------------------------------------------------------

class TestAsHardwareModel:
    def test_returns_hardware_model(self):
        ctx = HardwareContext(lines={
            "q": ControlLine("q", transfer_maps=(GainHardwareMap(2.0),)),
        })
        model = ctx.as_hardware_model()
        assert isinstance(model, HardwareModel)

    def test_gain_applied_via_hardware_model(self):
        """as_hardware_model produces the same result as apply_to_waveform for gain."""
        n = 12
        rng = np.random.default_rng(1)
        values = rng.standard_normal((1, n))
        g = 2.5

        op = np.eye(2, dtype=np.complex128)
        control_terms = (
            ControlTerm(name="c0", operator=op, quadrature="SCALAR", export_channel="q"),
        )
        grid = PiecewiseConstantTimeGrid.uniform(steps=n, dt_s=1e-9)

        ctx = HardwareContext(lines={
            "q": ControlLine("q", transfer_maps=(GainHardwareMap(gain=g),)),
        })
        model = ctx.as_hardware_model()
        physical, _, _, _ = model.apply(values, control_terms=control_terms, time_grid=grid)
        np.testing.assert_allclose(physical[0], g * values[0], atol=1e-13)

    def test_calibration_gain_in_model(self):
        """calibration_gain is appended as GainHardwareMap in the model."""
        n = 5
        values = np.ones((1, n))
        op = np.eye(2, dtype=np.complex128)
        control_terms = (
            ControlTerm(name="c0", operator=op, quadrature="SCALAR", export_channel="q"),
        )
        grid = PiecewiseConstantTimeGrid.uniform(steps=n, dt_s=1e-9)
        cal = 3.0

        ctx = HardwareContext(lines={"q": ControlLine("q", calibration_gain=cal)})
        model = ctx.as_hardware_model()
        physical, _, _, _ = model.apply(values, control_terms=control_terms, time_grid=grid)
        np.testing.assert_allclose(physical[0], cal * np.ones(n), atol=1e-13)

    def test_multi_line_channel_isolation(self):
        """Maps from different lines do not interfere with each other."""
        n = 8
        op = np.eye(2, dtype=np.complex128)
        control_terms = (
            ControlTerm(name="q_I", operator=op, quadrature="SCALAR", export_channel="qubit"),
            ControlTerm(name="s_I", operator=op, quadrature="SCALAR", export_channel="storage"),
        )
        grid = PiecewiseConstantTimeGrid.uniform(steps=n, dt_s=1e-9)
        values = np.ones((2, n))

        ctx = HardwareContext(lines={
            "qubit":   ControlLine("qubit",   transfer_maps=(GainHardwareMap(2.0),)),
            "storage": ControlLine("storage", transfer_maps=(GainHardwareMap(5.0),)),
        })
        model = ctx.as_hardware_model()
        physical, _, _, _ = model.apply(values, control_terms=control_terms, time_grid=grid)
        np.testing.assert_allclose(physical[0], 2.0 * np.ones(n), atol=1e-13)  # qubit
        np.testing.assert_allclose(physical[1], 5.0 * np.ones(n), atol=1e-13)  # storage


# ---------------------------------------------------------------------------
# 11.12  End-to-end SequenceCompiler integration
# ---------------------------------------------------------------------------

class TestSequenceCompilerIntegration:
    def test_hardware_context_with_fir_in_compiler(self):
        """FIRHardwareMap in a ControlLine is applied correctly by the compiler."""
        dt = 1.0
        kernel = (0.5, 0.5)
        ctx = HardwareContext(lines={
            "q": ControlLine("q", transfer_maps=(FIRHardwareMap(kernel=kernel),)),
        })
        # Square pulse: samples should be averaged by the FIR
        pulses = [Pulse("q", 0.0, 6.0, _square, amp=2.0)]
        compiled = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses, t_end=6.0)
        dist = compiled.channels["q"].distorted
        # First sample: 0.5 * amp = 1.0 (only h[0] * x[0])
        # Subsequent samples: 0.5 * amp + 0.5 * amp = 2.0
        assert abs(dist[0].real - 1.0) < 1e-12
        assert abs(dist[1].real - 2.0) < 1e-12

    def test_delay_in_compiler(self):
        """DelayHardwareMap shifts the channel waveform in the compiled sequence."""
        dt = 1.0
        delay = 2
        ctx = HardwareContext(lines={
            "q": ControlLine("q", transfer_maps=(DelayHardwareMap(delay_samples=delay),)),
        })
        pulses = [Pulse("q", 0.0, 5.0, _square, amp=1.0)]
        compiled = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses, t_end=6.0)
        dist = compiled.channels["q"].distorted
        # First `delay` samples should be near 0
        np.testing.assert_allclose(dist[:delay].real, 0.0, atol=1e-12)
        # Samples after delay should be 1
        np.testing.assert_allclose(dist[delay:delay + 5].real, 1.0, atol=1e-12)

    def test_baseband_unchanged_by_context(self):
        """The hardware context does NOT modify .baseband, only .distorted."""
        dt = 0.5
        ctx = HardwareContext(lines={
            "q": ControlLine("q", transfer_maps=(GainHardwareMap(10.0),)),
        })
        pulses = [Pulse("q", 0.0, 3.0, _square, amp=1.0)]
        compiled_base = SequenceCompiler(dt=dt).compile(pulses)
        compiled_ctx = SequenceCompiler(dt=dt, hardware_context=ctx).compile(pulses)
        # baseband should be identical
        np.testing.assert_array_equal(
            compiled_base.channels["q"].baseband,
            compiled_ctx.channels["q"].baseband,
        )


# ---------------------------------------------------------------------------
# 11.13  make_three_line_cqed_context
# ---------------------------------------------------------------------------

class TestMakeThreeLineCqedContext:
    def test_default_context_has_three_lines(self):
        ctx = make_three_line_cqed_context()
        assert set(ctx.line_names) == {"qubit", "storage", "readout"}

    def test_custom_names(self):
        ctx = make_three_line_cqed_context(
            qubit_name="transmon", storage_name="cavity", readout_name="resonator"
        )
        assert set(ctx.line_names) == {"transmon", "cavity", "resonator"}

    def test_identity_lines_no_maps(self):
        """Default (all unit gains, zero delays, no filters) → all ControlLines have empty maps."""
        ctx = make_three_line_cqed_context()
        for line in ctx.lines.values():
            assert len(line.transfer_maps) == 0

    def test_gain_map_added_when_nonunity(self):
        ctx = make_three_line_cqed_context(qubit_gain=0.9)
        maps = ctx.lines["qubit"].transfer_maps
        assert any(isinstance(m, GainHardwareMap) for m in maps)

    def test_delay_map_added_when_nonzero(self):
        ctx = make_three_line_cqed_context(storage_delay_s=3e-9, dt=1e-9)
        maps = ctx.lines["storage"].transfer_maps
        assert any(isinstance(m, DelayHardwareMap) for m in maps)
        delay_map = next(m for m in maps if isinstance(m, DelayHardwareMap))
        assert delay_map.delay_samples == 3

    def test_lowpass_map_added(self):
        ctx = make_three_line_cqed_context(readout_lowpass_hz=100e6)
        maps = ctx.lines["readout"].transfer_maps
        assert any(isinstance(m, FirstOrderLowPassHardwareMap) for m in maps)

    def test_calibration_gain_stored(self):
        ctx = make_three_line_cqed_context(qubit_calibration_gain=2e9 * np.pi)
        assert abs(ctx.lines["qubit"].calibration_gain - 2e9 * np.pi) < 1.0

    def test_context_applies_to_waveform(self):
        """Context with gain can be applied to a waveform."""
        g = 1.5
        ctx = make_three_line_cqed_context(qubit_gain=g)
        waveform = np.ones(10, dtype=complex)
        out = ctx.apply_to_waveform("qubit", waveform, dt=1e-9)
        np.testing.assert_allclose(out, g * waveform, atol=1e-12)


# ---------------------------------------------------------------------------
# 11.14  delay_samples_from_time helper (standalone)
# ---------------------------------------------------------------------------

class TestDelayHelper:
    def test_exact(self):
        assert delay_samples_from_time(5e-9, 1e-9) == 5

    def test_rounding_down(self):
        assert delay_samples_from_time(1.4e-9, 1e-9) == 1

    def test_rounding_up(self):
        assert delay_samples_from_time(1.6e-9, 1e-9) == 2

    def test_zero(self):
        assert delay_samples_from_time(0.0, 1e-9) == 0


# ---------------------------------------------------------------------------
# Additional: ControlLine construction validation
# ---------------------------------------------------------------------------

class TestControlLineValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ControlLine(name="")

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DelayHardwareMap(delay_samples=-1)

    def test_frozen_immutable(self):
        line = ControlLine(name="q")
        with pytest.raises((AttributeError, TypeError)):
            line.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Additional: HardwareContext key mismatch validation
# ---------------------------------------------------------------------------

class TestHardwareContextValidation:
    def test_key_name_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            HardwareContext(lines={
                "wrong_key": ControlLine("actual_name"),
            })

    def test_line_names_property(self):
        ctx = HardwareContext(lines={
            "a": ControlLine("a"),
            "b": ControlLine("b"),
        })
        assert set(ctx.line_names) == {"a", "b"}
