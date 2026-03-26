"""Tests for the extended hardware-aware control layer (Phase 5 extensions).

Coverage:
 12.1  LinearCalibrationMap: apply, equality, as_hardware_map, serialization
 12.2  CallableCalibrationMap: apply, no gradient, no serialization, equality/hash
 12.3  ControlLine with calibration_map: backward compat + new behavior
 12.4  ControlLine unit/coupling metadata: programmed_unit, device_unit, etc.
 12.5  FrequencyResponseHardwareMap: FIR kernel computation, DC gain, apply
 12.6  FrequencyResponseHardwareMap gradient (finite-difference)
 12.7  ControlLine.to_dict() / from_dict() round-trip
 12.8  HardwareContext.to_dict() / from_dict() round-trip
 12.9  HardwareContext.save() / load() file round-trip
 12.10 Non-serializable calibrations raise TypeError
 12.11 Unknown types in from_dict raise ValueError/KeyError
 12.12 Integration: serialized context produces same transforms as original
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cqed_sim.control import (
    ControlLine,
    HardwareContext,
    LinearCalibrationMap,
    CallableCalibrationMap,
    CalibrationMap,
    calibration_map_from_dict,
    hardware_map_to_dict,
    hardware_map_from_dict,
    delay_samples_from_time,
    postprocess_grape_waveforms,
)
from cqed_sim.optimal_control.hardware import (
    GainHardwareMap,
    DelayHardwareMap,
    FIRHardwareMap,
    FirstOrderLowPassHardwareMap,
    BoundaryWindowHardwareMap,
    QuantizationHardwareMap,
    SmoothIQRadiusLimitHardwareMap,
    FrequencyResponseHardwareMap,
    HardwareModel,
)
from cqed_sim.optimal_control.parameterizations import PiecewiseConstantTimeGrid
from cqed_sim.optimal_control.problems import ControlTerm
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_waveform(n: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)


def _make_ctrl_term(name: str = "c", ch: str = "ch") -> ControlTerm:
    return ControlTerm(
        name=name,
        operator=np.eye(2, dtype=complex),
        export_channel=ch,
    )


def _uniform_grid(n: int, dt: float = 1e-9) -> PiecewiseConstantTimeGrid:
    return PiecewiseConstantTimeGrid.uniform(steps=n, dt_s=dt)


def _rect(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


# ---------------------------------------------------------------------------
# 12.1  LinearCalibrationMap
# ---------------------------------------------------------------------------

class TestLinearCalibrationMap:
    def test_unity_gain_identity(self):
        m = LinearCalibrationMap(gain=1.0)
        w = _make_waveform()
        np.testing.assert_array_equal(m.apply(w), w)

    def test_gain_scales_amplitude(self):
        m = LinearCalibrationMap(gain=2.0)
        w = _make_waveform()
        np.testing.assert_allclose(m.apply(w), 2.0 * w)

    def test_negative_gain(self):
        m = LinearCalibrationMap(gain=-1.0)
        w = _make_waveform()
        np.testing.assert_allclose(m.apply(w), -w)

    def test_zero_gain(self):
        m = LinearCalibrationMap(gain=0.0)
        w = _make_waveform()
        np.testing.assert_allclose(m.apply(w), np.zeros_like(w))

    def test_apply_preserves_shape(self):
        m = LinearCalibrationMap(gain=3.0)
        for shape in [(5,), (2, 5), (3, 4, 5)]:
            w = np.ones(shape, dtype=complex)
            assert m.apply(w).shape == shape

    def test_as_hardware_map_unity_returns_none(self):
        m = LinearCalibrationMap(gain=1.0)
        assert m.as_hardware_map() is None

    def test_as_hardware_map_nonunity_returns_gain_map(self):
        m = LinearCalibrationMap(gain=2.5)
        hw = m.as_hardware_map(export_channels=("ch",))
        assert isinstance(hw, GainHardwareMap)
        assert hw.gain == pytest.approx(2.5)
        assert hw.export_channels == ("ch",)

    def test_serialization_roundtrip(self):
        m = LinearCalibrationMap(gain=3.14)
        d = m.to_dict()
        assert d["type"] == "LinearCalibrationMap"
        assert d["gain"] == pytest.approx(3.14)
        m2 = LinearCalibrationMap.from_dict(d)
        assert m2.gain == pytest.approx(3.14)
        assert m == m2

    def test_calibration_map_from_dict(self):
        m = LinearCalibrationMap(gain=0.5)
        m2 = calibration_map_from_dict(m.to_dict())
        assert isinstance(m2, LinearCalibrationMap)
        assert m2.gain == pytest.approx(0.5)

    def test_equality_and_hash(self):
        m1 = LinearCalibrationMap(gain=2.0)
        m2 = LinearCalibrationMap(gain=2.0)
        m3 = LinearCalibrationMap(gain=3.0)
        assert m1 == m2
        assert m1 != m3
        assert hash(m1) == hash(m2)

    def test_is_calibration_map_subclass(self):
        assert isinstance(LinearCalibrationMap(), CalibrationMap)


# ---------------------------------------------------------------------------
# 12.2  CallableCalibrationMap
# ---------------------------------------------------------------------------

class TestCallableCalibrationMap:
    def test_apply_uses_function(self):
        fn = lambda x: x * 2.0
        m = CallableCalibrationMap(fn, label="double")
        w = _make_waveform()
        np.testing.assert_allclose(m.apply(w), 2.0 * w.astype(complex))

    def test_label_stored(self):
        m = CallableCalibrationMap(lambda x: x, label="my_map")
        assert m.label == "my_map"

    def test_default_label(self):
        m = CallableCalibrationMap(lambda x: x)
        assert m.label == "callable"

    def test_apply_nonlinear(self):
        fn = lambda x: np.abs(x) ** 0.5 * np.exp(1j * np.angle(x))
        m = CallableCalibrationMap(fn)
        w = np.array([4.0 + 0j, 9.0 + 0j])
        result = m.apply(w)
        np.testing.assert_allclose(np.abs(result), [2.0, 3.0])

    def test_as_hardware_map_returns_none(self):
        m = CallableCalibrationMap(lambda x: x * 1.5)
        assert m.as_hardware_map() is None

    def test_to_dict_raises_type_error(self):
        m = CallableCalibrationMap(lambda x: x)
        with pytest.raises(TypeError, match="serialized"):
            m.to_dict()

    def test_equality_same_fn_object(self):
        fn = lambda x: x
        m1 = CallableCalibrationMap(fn)
        m2 = CallableCalibrationMap(fn)
        assert m1 == m2

    def test_equality_different_fn_object(self):
        m1 = CallableCalibrationMap(lambda x: x)
        m2 = CallableCalibrationMap(lambda x: x)
        assert m1 != m2

    def test_hash_based_on_fn_identity(self):
        fn = lambda x: x
        m1 = CallableCalibrationMap(fn)
        m2 = CallableCalibrationMap(fn)
        assert hash(m1) == hash(m2)

    def test_is_calibration_map_subclass(self):
        m = CallableCalibrationMap(lambda x: x)
        assert isinstance(m, CalibrationMap)

    def test_calibration_map_from_dict_raises_for_callable(self):
        with pytest.raises(ValueError, match="Unknown"):
            calibration_map_from_dict({"type": "CallableCalibrationMap"})


# ---------------------------------------------------------------------------
# 12.3  ControlLine with calibration_map: backward compat + new behavior
# ---------------------------------------------------------------------------

class TestControlLineCalibrationMap:
    def test_default_creates_linear_map_from_gain(self):
        """calibration_map is auto-created from calibration_gain=2.0."""
        line = ControlLine("q", calibration_gain=2.0)
        assert line.calibration_map is not None
        assert isinstance(line.calibration_map, LinearCalibrationMap)
        assert line.calibration_map.gain == pytest.approx(2.0)

    def test_explicit_linear_map_used(self):
        line = ControlLine("q", calibration_map=LinearCalibrationMap(gain=3.0))
        assert isinstance(line.calibration_map, LinearCalibrationMap)
        assert line.calibration_map.gain == pytest.approx(3.0)

    def test_callable_map_used(self):
        fn = lambda x: x * 5.0
        cm = CallableCalibrationMap(fn, label="x5")
        line = ControlLine("q", calibration_map=cm)
        assert line.calibration_map is cm

    def test_apply_to_waveform_uses_calibration_map(self):
        w = np.array([1.0 + 0j, 2.0 + 0j], dtype=complex)
        line = ControlLine("q", calibration_map=LinearCalibrationMap(gain=3.0))
        result = line.apply_to_waveform(w, dt=1e-9)
        np.testing.assert_allclose(result, 3.0 * w)

    def test_apply_to_waveform_callable_map(self):
        w = np.array([4.0 + 0j, 9.0 + 0j], dtype=complex)
        fn = lambda x: x ** 2
        line = ControlLine("q", calibration_map=CallableCalibrationMap(fn))
        result = line.apply_to_waveform(w, dt=1e-9)
        # fn applied to combined signal = (x**2 real + 0j)
        np.testing.assert_allclose(np.abs(result), [16.0, 81.0])

    def test_backward_compat_calibration_gain_float(self):
        """Old code passing calibration_gain as float should still work."""
        line = ControlLine("q", calibration_gain=2.0)
        w = np.array([1.0 + 0j, 2.0 + 0j], dtype=complex)
        result = line.apply_to_waveform(w, dt=1e-9)
        np.testing.assert_allclose(result, 2.0 * w)

    def test_as_hardware_model_includes_linear_calibration(self):
        line = ControlLine("q", calibration_map=LinearCalibrationMap(gain=2.0))
        model = line.as_hardware_model()
        # Should include a GainHardwareMap for the calibration
        gain_maps = [m for m in model.maps if isinstance(m, GainHardwareMap)]
        assert any(abs(m.gain - 2.0) < 1e-10 for m in gain_maps)

    def test_as_hardware_model_unity_calibration_no_extra_map(self):
        line = ControlLine("q", calibration_map=LinearCalibrationMap(gain=1.0))
        model = line.as_hardware_model()
        # Unity calibration should not add a GainHardwareMap
        assert len(model.maps) == 0

    def test_as_hardware_model_callable_calibration_skipped(self):
        """CallableCalibrationMap should NOT add any map to the GRAPE model."""
        fn = lambda x: x * 2.0
        line = ControlLine(
            "q",
            transfer_maps=(GainHardwareMap(gain=1.5),),
            calibration_map=CallableCalibrationMap(fn),
        )
        model = line.as_hardware_model()
        # Only the GainHardwareMap from transfer_maps; no calibration map
        assert len(model.maps) == 1


# ---------------------------------------------------------------------------
# 12.4  ControlLine unit/coupling metadata
# ---------------------------------------------------------------------------

class TestControlLineMetadata:
    def test_default_metadata_is_none(self):
        line = ControlLine("q")
        assert line.programmed_unit is None
        assert line.device_unit is None
        assert line.coefficient_unit is None
        assert line.operator_label is None
        assert line.frame is None

    def test_metadata_stored_correctly(self):
        line = ControlLine(
            name="qubit",
            programmed_unit="V",
            device_unit="V",
            coefficient_unit="rad/s",
            operator_label="σ_x / 2",
            frame="rotating_qubit",
        )
        assert line.programmed_unit == "V"
        assert line.device_unit == "V"
        assert line.coefficient_unit == "rad/s"
        assert line.operator_label == "σ_x / 2"
        assert line.frame == "rotating_qubit"

    def test_metadata_round_trips_via_to_dict(self):
        line = ControlLine(
            name="q",
            programmed_unit="normalized",
            device_unit="V",
            coefficient_unit="rad/s",
            operator_label="(a + a†) / 2",
            frame="interaction",
        )
        d = line.to_dict()
        assert d["programmed_unit"] == "normalized"
        assert d["device_unit"] == "V"
        assert d["coefficient_unit"] == "rad/s"
        assert d["operator_label"] == "(a + a†) / 2"
        assert d["frame"] == "interaction"
        line2 = ControlLine.from_dict(d)
        assert line2.programmed_unit == "normalized"
        assert line2.operator_label == "(a + a†) / 2"
        assert line2.frame == "interaction"

    def test_partial_metadata_stored(self):
        line = ControlLine("q", programmed_unit="V")
        assert line.programmed_unit == "V"
        assert line.device_unit is None

    def test_free_form_metadata_dict(self):
        line = ControlLine("q", metadata={"gain_dB": -1.0, "calibrated": True})
        assert line.metadata["gain_dB"] == pytest.approx(-1.0)
        assert line.metadata["calibrated"] is True


# ---------------------------------------------------------------------------
# 12.5  FrequencyResponseHardwareMap: kernel, DC gain, apply
# ---------------------------------------------------------------------------

class TestFrequencyResponseHardwareMap:
    def _make_flat_response(self, dc_gain: float = 1.0, n_freqs: int = 8) -> FrequencyResponseHardwareMap:
        """All-pass response with unity gain."""
        freqs = tuple(float(f) for f in np.linspace(0, 500e6, n_freqs))
        resp  = tuple(complex(dc_gain) for _ in range(n_freqs))
        return FrequencyResponseHardwareMap(
            frequencies_hz=freqs,
            response=resp,
            n_taps=32,
            dt_s=1e-9,
        )

    def test_construction_valid(self):
        m = self._make_flat_response()
        assert m.n_taps == 32
        assert len(m.frequencies_hz) == 8
        assert len(m.response) == 8

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            FrequencyResponseHardwareMap(
                frequencies_hz=(0.0, 1e8),
                response=(1.0 + 0j,),
                n_taps=4,
                dt_s=1e-9,
            )

    def test_too_few_freq_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            FrequencyResponseHardwareMap(
                frequencies_hz=(0.0,),
                response=(1.0 + 0j,),
                n_taps=4,
                dt_s=1e-9,
            )

    def test_n_taps_zero_raises(self):
        with pytest.raises(ValueError, match="n_taps"):
            FrequencyResponseHardwareMap(
                frequencies_hz=(0.0, 1e8),
                response=(1.0 + 0j, 1.0 + 0j),
                n_taps=0,
                dt_s=1e-9,
            )

    def test_dt_s_non_positive_raises(self):
        with pytest.raises(ValueError, match="dt_s"):
            FrequencyResponseHardwareMap(
                frequencies_hz=(0.0, 1e8),
                response=(1.0 + 0j, 1.0 + 0j),
                n_taps=4,
                dt_s=0.0,
            )

    def test_dc_gain_preserved(self):
        """For a flat response with gain g, the FIR kernel sums to ≈ g."""
        for dc_gain in [0.5, 1.0, 2.0]:
            m = self._make_flat_response(dc_gain=dc_gain, n_freqs=16)
            kernel = m.fir_kernel()
            assert abs(kernel.sum() - dc_gain) < 0.15, (
                f"DC gain {dc_gain}: kernel sum = {kernel.sum():.4f}"
            )

    def test_kernel_length(self):
        m = self._make_flat_response(n_freqs=8)
        assert len(m.fir_kernel()) == 32

    def test_apply_returns_correct_shape(self):
        m = self._make_flat_response()
        n = 20
        ctrl = (_make_ctrl_term("c", "ch"),)
        grid = _uniform_grid(n)
        values = np.ones((1, n), dtype=float)
        applied = m.apply(values, control_terms=ctrl, time_grid=grid)
        assert applied.values.shape == (1, n)

    def test_apply_flat_response_passes_through(self):
        """Near-flat response (high n_taps) should approximately pass signal through."""
        m = self._make_flat_response(dc_gain=1.0, n_freqs=16)
        # Replace with more taps for better accuracy
        m = FrequencyResponseHardwareMap(
            frequencies_hz=m.frequencies_hz,
            response=m.response,
            n_taps=128,
            dt_s=1e-9,
        )
        n = 50
        ctrl = (_make_ctrl_term("c", "ch"),)
        grid = _uniform_grid(n)
        values = np.ones((1, n), dtype=float)
        applied = m.apply(values, control_terms=ctrl, time_grid=grid)
        # Output should converge to ~1.0 for DC input after transient
        assert float(np.mean(applied.values[0, 10:])) == pytest.approx(1.0, abs=0.15)

    def test_apply_respects_channel_filter(self):
        """Map with export_channels filter should not affect other channels."""
        m = FrequencyResponseHardwareMap(
            frequencies_hz=(0.0, 5e8),
            response=(2.0 + 0j, 2.0 + 0j),
            n_taps=4,
            dt_s=1e-9,
            export_channels=("target",),
        )
        ctrl_other = (_make_ctrl_term("c", "other"),)
        grid = _uniform_grid(5)
        values = np.ones((1, 5), dtype=float)
        applied = m.apply(values, control_terms=ctrl_other, time_grid=grid)
        np.testing.assert_array_equal(applied.values, values)

    def test_fir_kernel_matches_fir_hardware_map(self):
        """For a response matching a known FIR kernel, FrequencyResponseHardwareMap
        should produce approximately the same output as FIRHardwareMap."""
        # Known FIR: simple two-tap filter [0.7, 0.3]
        # Its frequency response: H(f) = 0.7 + 0.3 * exp(-2πi*f*dt)
        dt = 1e-9
        freqs = np.linspace(0, 1.0 / (2 * dt), 64)
        response = 0.7 + 0.3 * np.exp(-2j * np.pi * freqs * dt)
        m_freq = FrequencyResponseHardwareMap(
            frequencies_hz=tuple(float(f) for f in freqs),
            response=tuple(complex(r) for r in response),
            n_taps=2,
            dt_s=dt,
        )
        kernel = m_freq.fir_kernel()
        # Should be close to [0.7, 0.3]
        assert abs(kernel[0] - 0.7) < 0.05
        assert abs(kernel[1] - 0.3) < 0.05


# ---------------------------------------------------------------------------
# 12.6  FrequencyResponseHardwareMap gradient (finite difference)
# ---------------------------------------------------------------------------

class TestFrequencyResponseHardwareMapGradient:
    def _make_map(self) -> FrequencyResponseHardwareMap:
        freqs = (0.0, 1e8, 2e8, 3e8, 4e8, 5e8)
        resp  = (1.0+0j, 0.9-0.05j, 0.7-0.1j, 0.5-0.15j, 0.3-0.1j, 0.1-0.02j)
        return FrequencyResponseHardwareMap(
            frequencies_hz=freqs, response=resp, n_taps=8, dt_s=1e-9,
        )

    def test_pullback_finite_difference(self):
        m = self._make_map()
        n = 10
        ctrl = (_make_ctrl_term("c", "ch"),)
        grid = _uniform_grid(n)
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal((1, n))

        applied = m.apply(x0, control_terms=ctrl, time_grid=grid)
        grad_out = rng.standard_normal((1, n))
        grad_in = applied.pullback(grad_out)

        eps = 1e-5
        fd_grad = np.zeros_like(x0)
        for i in range(n):
            xp = x0.copy(); xp[0, i] += eps
            xm = x0.copy(); xm[0, i] -= eps
            yp = m.apply(xp, control_terms=ctrl, time_grid=grid).values
            ym = m.apply(xm, control_terms=ctrl, time_grid=grid).values
            fd_grad[0, i] = np.sum(grad_out * (yp - ym) / (2 * eps))

        np.testing.assert_allclose(grad_in, fd_grad, rtol=1e-3, atol=1e-8)


# ---------------------------------------------------------------------------
# 12.7  ControlLine.to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------

class TestControlLineSerializationRoundtrip:
    def test_simple_line(self):
        line = ControlLine(
            name="qubit",
            transfer_maps=(GainHardwareMap(gain=0.9),),
            calibration_map=LinearCalibrationMap(gain=2.0),
        )
        d = line.to_dict()
        line2 = ControlLine.from_dict(d)
        assert line2.name == "qubit"
        assert isinstance(line2.calibration_map, LinearCalibrationMap)
        assert line2.calibration_map.gain == pytest.approx(2.0)
        assert len(line2.transfer_maps) == 1
        assert isinstance(line2.transfer_maps[0], GainHardwareMap)
        assert line2.transfer_maps[0].gain == pytest.approx(0.9)

    def test_all_map_types_serialize(self):
        """Verify hardware_map_to_dict / from_dict for each built-in type."""
        maps = [
            GainHardwareMap(gain=2.0, control_names=("a",), export_channels=("ch",)),
            DelayHardwareMap(delay_samples=3),
            FIRHardwareMap(kernel=(0.5, 0.3, 0.2)),
            FirstOrderLowPassHardwareMap(cutoff_hz=80e6),
            BoundaryWindowHardwareMap(ramp_slices=5),
            QuantizationHardwareMap(n_bits=14),
            SmoothIQRadiusLimitHardwareMap(amplitude_max=1.0),
            FrequencyResponseHardwareMap(
                frequencies_hz=(0.0, 5e8),
                response=(1.0+0j, 0.5-0.1j),
                n_taps=8,
                dt_s=1e-9,
            ),
        ]
        for m in maps:
            d = hardware_map_to_dict(m)
            m2 = hardware_map_from_dict(d)
            assert type(m2) == type(m), f"Type mismatch for {type(m).__name__}"

    def test_gain_map_roundtrip_values(self):
        m = GainHardwareMap(gain=3.14, control_names=("x",), export_channels=("ch",))
        d = hardware_map_to_dict(m)
        m2 = hardware_map_from_dict(d)
        assert m2.gain == pytest.approx(3.14)
        assert m2.control_names == ("x",)
        assert m2.export_channels == ("ch",)

    def test_delay_map_roundtrip_values(self):
        m = DelayHardwareMap(delay_samples=7)
        m2 = hardware_map_from_dict(hardware_map_to_dict(m))
        assert m2.delay_samples == 7

    def test_fir_map_roundtrip_values(self):
        kernel = (0.6, 0.25, 0.15)
        m = FIRHardwareMap(kernel=kernel)
        m2 = hardware_map_from_dict(hardware_map_to_dict(m))
        np.testing.assert_allclose(m2.kernel, kernel)

    def test_freq_response_map_roundtrip_values(self):
        freqs = (0.0, 1e8, 2e8, 5e8)
        resp  = (1.0+0j, 0.9-0.05j, 0.7-0.1j, 0.1-0.01j)
        m = FrequencyResponseHardwareMap(
            frequencies_hz=freqs, response=resp, n_taps=16, dt_s=2e-9,
        )
        d = hardware_map_to_dict(m)
        m2 = hardware_map_from_dict(d)
        assert m2.n_taps == 16
        assert m2.dt_s == pytest.approx(2e-9)
        np.testing.assert_allclose(m2.frequencies_hz, freqs)
        np.testing.assert_allclose(m2.response, resp)

    def test_line_with_all_metadata_roundtrip(self):
        line = ControlLine(
            name="q",
            transfer_maps=(DelayHardwareMap(delay_samples=2),),
            calibration_map=LinearCalibrationMap(gain=1.5),
            programmed_unit="V",
            device_unit="V",
            coefficient_unit="rad/s",
            operator_label="σ_x / 2",
            frame="rotating",
            metadata={"note": "example"},
        )
        d = line.to_dict()
        line2 = ControlLine.from_dict(d)
        assert line2.programmed_unit == "V"
        assert line2.device_unit == "V"
        assert line2.coefficient_unit == "rad/s"
        assert line2.operator_label == "σ_x / 2"
        assert line2.frame == "rotating"
        assert line2.metadata["note"] == "example"
        assert line2.calibration_map.gain == pytest.approx(1.5)  # type: ignore[union-attr]

    def test_line_with_no_maps_roundtrip(self):
        line = ControlLine("empty")
        d = line.to_dict()
        line2 = ControlLine.from_dict(d)
        assert line2.name == "empty"
        assert len(line2.transfer_maps) == 0


# ---------------------------------------------------------------------------
# 12.8  HardwareContext.to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------

class TestHardwareContextSerializationRoundtrip:
    def _make_ctx(self) -> HardwareContext:
        return HardwareContext(
            lines={
                "qubit": ControlLine(
                    name="qubit",
                    transfer_maps=(
                        GainHardwareMap(gain=0.89),
                        DelayHardwareMap(delay_samples=2),
                    ),
                    calibration_map=LinearCalibrationMap(gain=1.0),
                    programmed_unit="V",
                    coefficient_unit="rad/s",
                    operator_label="σ_x / 2",
                    frame="rotating_qubit",
                ),
                "storage": ControlLine(
                    name="storage",
                    transfer_maps=(FirstOrderLowPassHardwareMap(cutoff_hz=60e6),),
                    calibration_gain=2.0,
                ),
            },
            metadata={"device": "test_v1"},
        )

    def test_to_dict_structure(self):
        ctx = self._make_ctx()
        d = ctx.to_dict()
        assert d["version"] == "1.0"
        assert "lines" in d
        assert "metadata" in d
        assert set(d["lines"].keys()) == {"qubit", "storage"}

    def test_from_dict_restores_lines(self):
        ctx = self._make_ctx()
        ctx2 = HardwareContext.from_dict(ctx.to_dict())
        assert set(ctx2.line_names) == {"qubit", "storage"}

    def test_line_calibration_map_preserved(self):
        ctx = self._make_ctx()
        ctx2 = HardwareContext.from_dict(ctx.to_dict())
        assert isinstance(ctx2.lines["qubit"].calibration_map, LinearCalibrationMap)
        assert ctx2.lines["qubit"].calibration_map.gain == pytest.approx(1.0)

    def test_storage_calibration_gain_preserved(self):
        ctx = self._make_ctx()
        ctx2 = HardwareContext.from_dict(ctx.to_dict())
        # storage was built with calibration_gain=2.0 → LinearCalibrationMap(gain=2.0)
        cm = ctx2.lines["storage"].calibration_map
        assert isinstance(cm, LinearCalibrationMap)
        assert cm.gain == pytest.approx(2.0)

    def test_transfer_maps_preserved(self):
        ctx = self._make_ctx()
        ctx2 = HardwareContext.from_dict(ctx.to_dict())
        q_maps = ctx2.lines["qubit"].transfer_maps
        assert len(q_maps) == 2
        assert isinstance(q_maps[0], GainHardwareMap)
        assert isinstance(q_maps[1], DelayHardwareMap)
        assert q_maps[0].gain == pytest.approx(0.89)
        assert q_maps[1].delay_samples == 2

    def test_metadata_preserved(self):
        ctx = self._make_ctx()
        ctx2 = HardwareContext.from_dict(ctx.to_dict())
        assert ctx2.metadata["device"] == "test_v1"

    def test_operator_label_frame_preserved(self):
        ctx = self._make_ctx()
        ctx2 = HardwareContext.from_dict(ctx.to_dict())
        assert ctx2.lines["qubit"].operator_label == "σ_x / 2"
        assert ctx2.lines["qubit"].frame == "rotating_qubit"


# ---------------------------------------------------------------------------
# 12.9  HardwareContext.save() / load() file round-trip
# ---------------------------------------------------------------------------

class TestHardwareContextFileRoundtrip:
    def _make_ctx(self) -> HardwareContext:
        return HardwareContext(
            lines={
                "q": ControlLine(
                    name="q",
                    transfer_maps=(GainHardwareMap(gain=0.95), DelayHardwareMap(delay_samples=1)),
                    calibration_map=LinearCalibrationMap(gain=1.5),
                    programmed_unit="normalized",
                    coefficient_unit="rad/s",
                ),
            },
            metadata={"label": "save_load_test"},
        )

    def test_save_creates_json_file(self):
        ctx = self._make_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ctx.json"
            returned = ctx.save(path)
            assert returned == path
            assert path.exists()
            payload = json.loads(path.read_text())
            assert payload["version"] == "1.0"

    def test_load_restores_context(self):
        ctx = self._make_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ctx.json"
            ctx.save(path)
            ctx2 = HardwareContext.load(path)
            assert set(ctx2.line_names) == {"q"}
            cm = ctx2.lines["q"].calibration_map
            assert isinstance(cm, LinearCalibrationMap)
            assert cm.gain == pytest.approx(1.5)

    def test_save_load_string_path(self):
        ctx = self._make_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_str = str(Path(tmpdir) / "ctx.json")
            ctx.save(path_str)
            ctx2 = HardwareContext.load(path_str)
            assert ctx2.line_names == ("q",)

    def test_file_is_valid_json(self):
        ctx = self._make_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ctx.json"
            ctx.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict)

    def test_save_creates_parent_dirs(self):
        ctx = self._make_ctx()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "ctx.json"
            ctx.save(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# 12.10  Non-serializable calibrations raise TypeError
# ---------------------------------------------------------------------------

class TestNonSerializableCalibration:
    def test_callable_map_to_dict_raises(self):
        m = CallableCalibrationMap(lambda x: x * 2)
        with pytest.raises(TypeError):
            m.to_dict()

    def test_control_line_with_callable_raises_on_to_dict(self):
        line = ControlLine("q", calibration_map=CallableCalibrationMap(lambda x: x))
        with pytest.raises(TypeError):
            line.to_dict()

    def test_hardware_context_with_callable_raises_on_to_dict(self):
        ctx = HardwareContext(
            lines={"q": ControlLine("q", calibration_map=CallableCalibrationMap(lambda x: x))}
        )
        with pytest.raises(TypeError):
            ctx.to_dict()

    def test_hardware_context_with_callable_raises_on_save(self):
        ctx = HardwareContext(
            lines={"q": ControlLine("q", calibration_map=CallableCalibrationMap(lambda x: x))}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(TypeError):
                ctx.save(Path(tmpdir) / "ctx.json")

    def test_custom_hardware_map_to_dict_raises(self):
        """Custom (non-built-in) HardwareMap should raise TypeError."""
        class MyMap(GainHardwareMap):
            pass

        m = MyMap(gain=1.0)
        # MyMap is not in the known-serializable set
        # Note: since MyMap is a subclass of GainHardwareMap and isinstance check
        # uses the parent, this actually serializes. Test the custom class scenario:
        # We test by passing something truly unknown.
        from cqed_sim.optimal_control.hardware import HardwareMap
        from abc import abstractmethod

        class Unknown(HardwareMap):
            def apply(self, values, *, control_terms, time_grid):
                return values  # type: ignore[return-value]

        # hardware_map_to_dict should raise for Unknown
        with pytest.raises(TypeError, match="Cannot serialize"):
            hardware_map_to_dict(Unknown())


# ---------------------------------------------------------------------------
# 12.11  Unknown types in from_dict raise ValueError
# ---------------------------------------------------------------------------

class TestUnknownTypeErrors:
    def test_hardware_map_from_dict_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown HardwareMap type"):
            hardware_map_from_dict({"type": "MadeUpHardwareMap", "value": 1})

    def test_calibration_map_from_dict_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown CalibrationMap type"):
            calibration_map_from_dict({"type": "MadeUpCalibrationMap"})

    def test_calibration_map_from_dict_empty_type(self):
        with pytest.raises(ValueError, match="Unknown CalibrationMap type"):
            calibration_map_from_dict({"type": ""})

    def test_control_line_from_dict_unknown_map(self):
        d = {
            "name": "q",
            "transfer_maps": [{"type": "UnknownMap", "x": 1}],
            "calibration_gain": 1.0,
            "calibration_map": {"type": "LinearCalibrationMap", "gain": 1.0},
        }
        with pytest.raises(ValueError):
            ControlLine.from_dict(d)


# ---------------------------------------------------------------------------
# 12.12  Integration: serialized context produces same transforms as original
# ---------------------------------------------------------------------------

class TestIntegrationSerializedTransforms:
    def _make_ctx(self) -> HardwareContext:
        return HardwareContext(
            lines={
                "qubit": ControlLine(
                    name="qubit",
                    transfer_maps=(
                        GainHardwareMap(gain=0.85),
                        DelayHardwareMap(delay_samples=2),
                        FirstOrderLowPassHardwareMap(cutoff_hz=100e6),
                    ),
                    calibration_map=LinearCalibrationMap(gain=1.5),
                    programmed_unit="V",
                    coefficient_unit="rad/s",
                    operator_label="σ_x / 2",
                    frame="rotating",
                ),
                "storage": ControlLine(
                    name="storage",
                    transfer_maps=(FIRHardwareMap(kernel=(0.6, 0.3, 0.1)),),
                    calibration_map=LinearCalibrationMap(gain=2.0),
                ),
            }
        )

    def _make_pulses(self) -> list[Pulse]:
        return [
            Pulse(channel="qubit",   t0=0.0, duration=100e-9, envelope=_rect, carrier=0.0, amp=1.0),
            Pulse(channel="storage", t0=0.0, duration=100e-9, envelope=_rect, carrier=0.0, amp=0.5),
        ]

    def test_waveforms_identical_after_roundtrip(self):
        ctx = self._make_ctx()
        pulses = self._make_pulses()
        dt = 1e-9

        compiler_orig = SequenceCompiler(dt=dt, hardware_context=ctx)
        seq_orig = compiler_orig.compile(pulses)

        # Serialize and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ctx.json"
            ctx.save(path)
            ctx_loaded = HardwareContext.load(path)

        compiler_loaded = SequenceCompiler(dt=dt, hardware_context=ctx_loaded)
        seq_loaded = compiler_loaded.compile(pulses)

        for ch in ["qubit", "storage"]:
            w_orig   = seq_orig.channels[ch].distorted
            w_loaded = seq_loaded.channels[ch].distorted
            np.testing.assert_allclose(
                w_loaded, w_orig, rtol=1e-10, atol=1e-12,
                err_msg=f"Channel '{ch}' waveform mismatch after serialization round-trip."
            )

    def test_grape_mode_a_postprocess_roundtrip(self):
        """Mode A postprocess gives same result before/after context serialization."""
        ctx = self._make_ctx()
        dt = 1e-9

        ctrl_i = ControlTerm(
            name="ctrl_I", operator=np.eye(2, dtype=complex),
            amplitude_bounds=(-1.0, 1.0), export_channel="qubit", quadrature="I",
        )
        ctrl_q = ControlTerm(
            name="ctrl_Q", operator=np.eye(2, dtype=complex),
            amplitude_bounds=(-1.0, 1.0), export_channel="qubit", quadrature="Q",
        )
        control_terms = (ctrl_i, ctrl_q)
        rng = np.random.default_rng(99)
        physical_values = rng.standard_normal((2, 30))

        result_orig = postprocess_grape_waveforms(ctx, physical_values, control_terms, dt=dt)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ctx.json"
            ctx.save(path)
            ctx_loaded = HardwareContext.load(path)

        result_loaded = postprocess_grape_waveforms(ctx_loaded, physical_values, control_terms, dt=dt)
        np.testing.assert_allclose(result_loaded, result_orig, rtol=1e-10, atol=1e-12)

    def test_as_hardware_model_after_roundtrip(self):
        """GRAPE Mode B hardware model has same number of maps after roundtrip."""
        ctx = self._make_ctx()
        model_orig = ctx.as_hardware_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ctx.json"
            ctx.save(path)
            ctx_loaded = HardwareContext.load(path)

        model_loaded = ctx_loaded.as_hardware_model()
        assert len(model_loaded.maps) == len(model_orig.maps)
