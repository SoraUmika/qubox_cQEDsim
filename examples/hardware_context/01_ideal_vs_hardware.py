"""Example 01: Ideal vs hardware-aware pulse compilation.

Demonstrates how GainHardwareMap, DelayHardwareMap, and
FirstOrderLowPassHardwareMap distort a compiled control waveform relative to
the ideal case, and how ControlLine unit/coupling metadata is recorded.

The hardware model represents a typical qubit XY drive line with:
  * 1 dB insertion loss (gain ≈ 0.891)
  * 3 ns cable propagation delay
  * 80 MHz first-order bandwidth limit

The example also shows round-trip JSON serialization of the HardwareContext.

Run time: < 1 second.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cqed_sim.control import (
    ControlLine,
    HardwareContext,
    LinearCalibrationMap,
)
from cqed_sim.optimal_control.hardware import (
    DelayHardwareMap,
    FirstOrderLowPassHardwareMap,
    GainHardwareMap,
)
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler

# ── Parameters ───────────────────────────────────────────────────────────────

DT_S = 1e-9           # 1 ns sample period
T_GATE_S = 100e-9     # 100 ns pulse duration
GAIN_dB = -1.0        # insertion loss in dB
DELAY_S = 3e-9        # cable delay
LOWPASS_HZ = 80e6     # bandwidth


# ── Define pulses ─────────────────────────────────────────────────────────────
# Flat-top (rectangle) envelope on channel "qubit".

def _rect(t_rel: np.ndarray) -> np.ndarray:
    """Unit rectangle envelope (full amplitude for all t_rel in [0, 1))."""
    return np.ones_like(t_rel, dtype=np.complex128)


pulse = Pulse(
    channel="qubit",
    t0=0.0,
    duration=T_GATE_S,
    envelope=_rect,
    carrier=0.0,
    amp=1.0,
)


# ── Compiler 1: ideal (no hardware context) ───────────────────────────────────

compiler_ideal = SequenceCompiler(dt=DT_S)
seq_ideal = compiler_ideal.compile([pulse])
w_ideal = seq_ideal.channels["qubit"].distorted


# ── Compiler 2: hardware-aware ────────────────────────────────────────────────
# Build a ControlLine with explicit unit / coupling metadata.

gain_linear = 10 ** (GAIN_dB / 20)          # ≈ 0.891
delay_samples = int(round(DELAY_S / DT_S))  # 3 samples

ctx = HardwareContext(
    lines={
        "qubit": ControlLine(
            name="qubit",
            transfer_maps=(
                GainHardwareMap(gain=gain_linear),
                DelayHardwareMap(delay_samples=delay_samples),
                FirstOrderLowPassHardwareMap(cutoff_hz=LOWPASS_HZ),
            ),
            calibration_map=LinearCalibrationMap(gain=1.0),
            # Unit / coupling metadata (documentation, not enforced)
            programmed_unit="normalized",
            device_unit="V",
            coefficient_unit="rad/s",
            operator_label="σ_x / 2",
            frame="rotating_qubit",
            metadata={
                "gain_dB": GAIN_dB,
                "delay_s": DELAY_S,
                "lowpass_hz": LOWPASS_HZ,
            },
        )
    },
    metadata={"description": "single-qubit XY drive line"},
)

compiler_hw = SequenceCompiler(dt=DT_S, hardware_context=ctx)
seq_hw = compiler_hw.compile([pulse])
w_hw = seq_hw.channels["qubit"].distorted


# ── Compare waveforms ─────────────────────────────────────────────────────────

peak_ideal = float(np.max(np.abs(w_ideal)))
peak_hw    = float(np.max(np.abs(w_hw)))
rms_ideal  = float(np.sqrt(np.mean(np.abs(w_ideal) ** 2)))
rms_hw     = float(np.sqrt(np.mean(np.abs(w_hw) ** 2)))

print("=== Example 01: Ideal vs hardware-aware ===")
print(f"Ideal      — peak: {peak_ideal:.4f}, RMS: {rms_ideal:.4f}")
print(f"Hardware   — peak: {peak_hw:.4f}, RMS: {rms_hw:.4f}")
print(
    f"Peak change: {(peak_hw / peak_ideal - 1.0) * 100:+.1f}%  "
    f"(expected ≈ {(gain_linear - 1.0) * 100:+.1f}% from gain alone)"
)

# Verify: samples [0, delay_samples) should be zero (delay effect)
print(f"\nFirst {delay_samples} samples of hardware waveform (should be ~0):")
print(f"  max abs = {float(np.max(np.abs(w_hw[:delay_samples]))):.2e}")


# ── Inspect ControlLine metadata ─────────────────────────────────────────────

line = ctx.lines["qubit"]
print("\n--- ControlLine metadata ---")
print(f"  programmed_unit : {line.programmed_unit}")
print(f"  device_unit     : {line.device_unit}")
print(f"  coefficient_unit: {line.coefficient_unit}")
print(f"  operator_label  : {line.operator_label}")
print(f"  frame           : {line.frame}")
print(f"  calibration_map : {line.calibration_map}")


# ── Round-trip JSON serialization ─────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "ctx.json"
    ctx.save(path)
    ctx_loaded = HardwareContext.load(path)

    # Verify the loaded context produces identical waveforms
    compiler_loaded = SequenceCompiler(dt=DT_S, hardware_context=ctx_loaded)
    seq_loaded = compiler_loaded.compile([pulse])
    w_loaded = seq_loaded.channels["qubit"].distorted

    max_diff = float(np.max(np.abs(w_loaded - w_hw)))
    print(f"\nSerialization round-trip max waveform diff: {max_diff:.2e}")

    # Print the serialized JSON structure (partial)
    payload = json.loads(path.read_text())
    print(f"\nSerialized JSON keys: {list(payload.keys())}")
    qubit_data = payload["lines"]["qubit"]
    print(f"  line.name           : {qubit_data['name']}")
    print(f"  line.programmed_unit: {qubit_data['programmed_unit']}")
    print(f"  line.operator_label : {qubit_data['operator_label']}")
    print(f"  number of maps      : {len(qubit_data['transfer_maps'])}")

print("\nExample 01 complete.")
