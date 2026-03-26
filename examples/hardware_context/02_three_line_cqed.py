"""Example 02: Three-line cQED device hardware context.

Demonstrates :func:`~cqed_sim.control.cqed_device.make_three_line_cqed_context`
for a typical qubit / storage cavity / readout resonator device with
physical per-line imperfections.

Hardware parameters chosen to represent realistic room-temperature coax lines:

==========  ========  =========  ============
Line        Loss (dB) Delay (ns) Bandwidth
==========  ========  =========  ============
qubit       1.0       2          80 MHz
storage     1.5       3          60 MHz
readout     0.5       1          200 MHz
==========  ========  =========  ============

Shows:
* How to build the context with the factory function.
* How to inspect per-line transfer maps.
* How to use ``FrequencyResponseHardwareMap`` for a VNA-measured response.
* How to serialize and reload the context.

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
    hardware_map_to_dict,
)
from cqed_sim.control.cqed_device import make_three_line_cqed_context
from cqed_sim.optimal_control.hardware import (
    FrequencyResponseHardwareMap,
    GainHardwareMap,
)
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler

DT_S = 1e-9  # 1 ns


# ── Build the three-line cQED context ────────────────────────────────────────

ctx = make_three_line_cqed_context(
    qubit_gain=10 ** (-1.0 / 20),    # 1 dB loss
    qubit_delay_s=2e-9,
    qubit_lowpass_hz=80e6,
    qubit_calibration_gain=1.0,

    storage_gain=10 ** (-1.5 / 20),  # 1.5 dB loss
    storage_delay_s=3e-9,
    storage_lowpass_hz=60e6,
    storage_calibration_gain=1.0,

    readout_gain=10 ** (-0.5 / 20),  # 0.5 dB loss
    readout_delay_s=1e-9,
    readout_lowpass_hz=200e6,
    readout_calibration_gain=1.0,

    dt=DT_S,
    extra_metadata={"device": "example_chip_v1", "measured_date": "2026-03-25"},
)

print("=== Example 02: Three-line cQED context ===")
print(f"Lines: {list(ctx.lines.keys())}")
print(f"Context metadata: {ctx.metadata}")

for name, line in ctx.lines.items():
    maps_summary = [type(m).__name__ for m in line.transfer_maps]
    print(f"\n  [{name}]")
    print(f"    calibration_gain  : {line.calibration_gain:.4f}")
    print(f"    transfer chain    : {maps_summary}")
    print(f"    calibration_map   : {line.calibration_map}")
    print(f"    metadata.delay_s  : {line.metadata.get('delay_s', 0)*1e9:.1f} ns")


# ── Apply context in SequenceCompiler ────────────────────────────────────────

def _rect(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)

pulses = [
    Pulse(channel="qubit",   t0=0.0,   duration=100e-9, envelope=_rect, carrier=0.0, amp=1.0),
    Pulse(channel="storage", t0=0.0,   duration=100e-9, envelope=_rect, carrier=0.0, amp=0.5),
    Pulse(channel="readout", t0=100e-9, duration=50e-9,  envelope=_rect, carrier=0.0, amp=0.3),
]

compiler_ideal = SequenceCompiler(dt=DT_S)
compiler_hw    = SequenceCompiler(dt=DT_S, hardware_context=ctx)

seq_ideal = compiler_ideal.compile(pulses)
seq_hw    = compiler_hw.compile(pulses)

for ch in ["qubit", "storage", "readout"]:
    w_i = seq_ideal.channels[ch].distorted
    w_h = seq_hw.channels[ch].distorted
    peak_i = float(np.max(np.abs(w_i)))
    peak_h = float(np.max(np.abs(w_h)))
    print(f"\n  {ch}: ideal peak={peak_i:.4f}, hardware peak={peak_h:.4f}")


# ── Use a FrequencyResponseHardwareMap (VNA-measured transfer function) ───────
# Simulate a measured low-pass / band-shaping response on the readout line.

freqs_hz = np.array([0, 50e6, 100e6, 200e6, 300e6, 400e6, 500e6])
# Complex transfer function (magnitude + some phase tilt)
response = np.array([
    1.00 + 0j,
    0.98 - 0.02j,
    0.92 - 0.05j,
    0.75 - 0.10j,
    0.50 - 0.15j,
    0.25 - 0.10j,
    0.08 - 0.03j,
])

vna_map = FrequencyResponseHardwareMap(
    frequencies_hz=tuple(float(f) for f in freqs_hz),
    response=tuple(complex(r) for r in response),
    n_taps=32,
    dt_s=DT_S,
)

# Build a custom readout line using VNA data instead of first-order lowpass
readout_line_vna = ControlLine(
    name="readout",
    transfer_maps=(
        GainHardwareMap(gain=10 ** (-0.5 / 20)),
        vna_map,
    ),
    calibration_map=LinearCalibrationMap(gain=1.0),
    programmed_unit="normalized",
    device_unit="V",
    coefficient_unit="rad/s",
    operator_label="readout drive",
    frame="rotating_readout",
)

print("\n--- VNA-derived FIR taps (first 8) ---")
kernel = vna_map.fir_kernel()
print(f"  {kernel[:8]}")
print(f"  DC gain (sum of taps): {kernel.sum():.4f}  (expected ≈ {abs(response[0]):.4f})")

ctx_vna = HardwareContext(
    lines={**{k: v for k, v in ctx.lines.items() if k != "readout"},
           "readout": readout_line_vna},
)
print(f"\nVNA context lines: {list(ctx_vna.lines.keys())}")


# ── Serialize and reload ─────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "three_line_ctx.json"
    ctx.save(path)

    # Verify file is valid JSON
    payload = json.loads(path.read_text())
    assert payload["version"] == "1.0"
    assert set(payload["lines"].keys()) == {"qubit", "storage", "readout"}

    ctx_loaded = HardwareContext.load(path)
    assert set(ctx_loaded.line_names) == {"qubit", "storage", "readout"}

    # Check one line round-trips
    q_orig = ctx.lines["qubit"]
    q_load = ctx_loaded.lines["qubit"]
    assert type(q_load.calibration_map) == type(q_orig.calibration_map)
    assert len(q_load.transfer_maps) == len(q_orig.transfer_maps)

    # Apply loaded context and compare waveforms
    compiler_reloaded = SequenceCompiler(dt=DT_S, hardware_context=ctx_loaded)
    seq_reloaded = compiler_reloaded.compile(pulses)
    max_diff = max(
        float(np.max(np.abs(
            seq_reloaded.channels[ch].distorted - seq_hw.channels[ch].distorted
        )))
        for ch in ["qubit", "storage", "readout"]
    )
    print(f"\nRound-trip max waveform diff (all channels): {max_diff:.2e}")

print("\nExample 02 complete.")
