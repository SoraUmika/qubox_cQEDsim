# API Reference — Hardware & Control (`cqed_sim.control`, `cqed_sim.optimal_control.hardware`)

This page documents the hardware-aware control pipeline: how programmed waveforms are transformed into physical device signals, and how those transformations integrate with GRAPE optimal control.

!!! note "Context"
    This layer was designed for hardware-faithful simulation workflows where the pulse optimizer must account for the physical effects of the AWG/DAC signal chain. It is **not required** for basic pulse-level simulation.

---

## Overview: The Control Pipeline

The hardware-aware pipeline maps programmed waveform values to physical Hamiltonian coefficients:

```
u_prog(t)  ──[ControlLine.transfer_maps]──▶  u_dev(t)  ──[CalibrationMap]──▶  c(t)
                                                                                 │
                                                              H(t) = H₀ + Σⱼ cⱼ(t) Oⱼ
```

- **`u_prog(t)`** — programmed AWG output (e.g., in Volts or normalized units)
- **`u_dev(t)`** — device-side signal after the transfer chain (filtering, gain, delay)
- **`c(t)`** — Hamiltonian coefficient (typically in rad/s)
- **`Oⱼ`** — drive operator (e.g., `(a + a†)/2`)

---

## `ControlLine`

**Module path:** `cqed_sim.control.ControlLine`

A single logical control line: one waveform channel → one Hamiltonian operator.

```python
from cqed_sim.control import ControlLine
from cqed_sim.optimal_control.hardware import FirstOrderLowPassHardwareMap

line = ControlLine(
    name="qubit_I",
    operator=qubit_drive_op,
    calibration_gain=2 * np.pi * 1e6,        # rad/s per unit of u_dev
    transfer_maps=[FirstOrderLowPassHardwareMap(cutoff_hz=200e6, dt_s=2e-9)],
    programmed_unit="V",
    device_unit="V",
    coefficient_unit="rad/s",
    operator_label="(b + b†) / 2",
    frame="rotating_qubit",
)
```

### Fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Unique identifier for this control channel |
| `operator` | `qt.Qobj` | Drive operator `Oⱼ` in the Hamiltonian |
| `calibration_gain` | `float \| None` | Scalar gain: `c(t) = gain × u_dev(t)` |
| `calibration_map` | `CalibrationMap \| None` | Explicit calibration map (overrides `calibration_gain` if set) |
| `transfer_maps` | `list[HardwareMap]` | Ordered chain of hardware distortion maps applied to `u_prog` |
| `programmed_unit` | `str \| None` | Unit of `u_prog` (e.g., `"V"`, `"normalized"`, `"rad/s"`) |
| `device_unit` | `str \| None` | Unit of `u_dev` after transfer chain |
| `coefficient_unit` | `str \| None` | Unit of `c(t)` (typically `"rad/s"`) |
| `operator_label` | `str \| None` | Human-readable label for `Oⱼ` |
| `frame` | `str \| None` | Rotating frame label (e.g., `"lab"`, `"rotating_qubit"`) |

### Methods

```python
# Apply the full pipeline: u_prog → u_dev → c(t)
c = line.apply_pipeline(u_prog_samples, dt_s=2e-9)

# Serialization
d = line.to_dict()
line2 = ControlLine.from_dict(d)
```

---

## `HardwareContext`

**Module path:** `cqed_sim.control.HardwareContext`

A collection of `ControlLine` objects representing all control channels in an experiment.

```python
from cqed_sim.control import HardwareContext

ctx = HardwareContext(lines=[line_I, line_Q, line_readout])
```

### Creating a standard cQED context

```python
from cqed_sim.control.cqed_device import make_three_line_cqed_context

ctx = make_three_line_cqed_context(
    qubit_op=model.drive_coupling_operators(frame)["qubit"],
    cavity_op=model.drive_coupling_operators(frame)["cavity"],
    readout_op=model.drive_coupling_operators(frame)["readout"],
    calibration_gains={
        "qubit_I": 2 * np.pi * 1e6,
        "qubit_Q": 2 * np.pi * 1e6,
        "cavity":  2 * np.pi * 0.5e6,
    },
)
```

### Methods

| Method | Description |
|---|---|
| `ctx.line(name)` | Retrieve a `ControlLine` by name |
| `ctx.apply_all(u_dict, dt_s)` | Apply each line's pipeline; returns `{name: c_array}` |
| `ctx.as_hardware_model()` | Convert to `HardwareModel` for use in GRAPE |
| `ctx.to_dict()` | Serialize to JSON-compatible dict |
| `ctx.save(path)` | Write to a JSON file |
| `HardwareContext.from_dict(d)` | Deserialize |
| `HardwareContext.load(path)` | Load from a JSON file |

### Serialization example

```python
ctx.save("device.json")
ctx2 = HardwareContext.load("device.json")
```

!!! warning "CallableCalibrationMap not serializable"
    `CallableCalibrationMap` raises `TypeError` on `to_dict()`. Use `LinearCalibrationMap`
    or implement a custom serializable subclass for JSON roundtrips.

---

## `CalibrationMap`

**Module path:** `cqed_sim.control.calibration`

Abstract base class for the `u_dev(t) → c(t)` mapping.

### `LinearCalibrationMap`

Scalar gain: `c(t) = gain × u_dev(t)`.

```python
from cqed_sim.control.calibration import LinearCalibrationMap

cal = LinearCalibrationMap(gain=2 * np.pi * 1e6)
c = cal.apply(u_dev_samples)
hw_map = cal.as_hardware_map()   # Returns GainHardwareMap
d = cal.to_dict()                # JSON serializable
```

### `CallableCalibrationMap`

Arbitrary function: `c(t) = fn(u_dev(t))`. Supports nonlinear calibration curves.

```python
from cqed_sim.control.calibration import CallableCalibrationMap

cal = CallableCalibrationMap(fn=lambda u: np.tanh(u / 0.5), label="tanh_saturation")
```

!!! note
    `CallableCalibrationMap` has no gradient and is not JSON-serializable.
    For GRAPE Mode B (gradient through hardware), use `LinearCalibrationMap` or a gradient-supported subclass.

### `calibration_map_from_dict`

Deserialize a calibration map from a dict:

```python
from cqed_sim.control.calibration import calibration_map_from_dict

cal = calibration_map_from_dict({"type": "linear", "gain": 6283185.0})
```

---

## Hardware Maps (`cqed_sim.optimal_control.hardware`)

Hardware maps implement the `u_prog → u_dev` transfer chain. Each map transforms an array of samples and provides a gradient pullback for use inside GRAPE.

```python
from cqed_sim.optimal_control.hardware import (
    HardwareMap,
    HardwareModel,
    FirstOrderLowPassHardwareMap,
    BoundaryWindowHardwareMap,
    SmoothIQRadiusLimitHardwareMap,
    QuantizationHardwareMap,
    FIRHardwareMap,
    GainHardwareMap,
    DelayHardwareMap,
    FrequencyResponseHardwareMap,
)
```

### `HardwareMap` (abstract base)

```python
class HardwareMap:
    def apply(self, u: np.ndarray) -> np.ndarray: ...
    def vjp(self, u: np.ndarray, g: np.ndarray) -> np.ndarray: ...
    def to_dict(self) -> dict: ...
```

`vjp` is the vector-Jacobian product (gradient pullback), used by GRAPE to propagate gradients through the hardware chain.

### `FirstOrderLowPassHardwareMap`

Single-pole low-pass IIR filter.

```python
filt = FirstOrderLowPassHardwareMap(cutoff_hz=200e6, dt_s=2e-9)
```

| Parameter | Description |
|---|---|
| `cutoff_hz` | -3 dB cutoff frequency in Hz |
| `dt_s` | Sample period in seconds |

### `FIRHardwareMap`

Arbitrary FIR filter specified by tap coefficients.

```python
fir = FIRHardwareMap(taps=np.array([0.25, 0.5, 0.25]))
```

### `FrequencyResponseHardwareMap`

Converts a measured complex frequency response `H(f)` to an FIR filter via IFFT. Useful when you have a VNA measurement of a control line.

```python
from cqed_sim.optimal_control.hardware import FrequencyResponseHardwareMap

hwmap = FrequencyResponseHardwareMap(
    frequencies_hz=freqs,     # 1-D array of frequency points
    response=H_measured,      # Complex array, same length as frequencies_hz
    n_taps=64,                # Number of FIR taps in the resulting filter
    dt_s=2e-9,
)
```

Supports exact gradient pullback and is JSON serializable.

### `BoundaryWindowHardwareMap`

Applies a smooth envelope that tapers waveforms to zero at the start and end. Prevents sharp edges from generating out-of-band spectral content.

```python
win = BoundaryWindowHardwareMap(rise_samples=10, fall_samples=10)
```

### `SmoothIQRadiusLimitHardwareMap`

Differentiable soft-clip on the I/Q amplitude. Prevents the optimized waveform from exceeding the AWG power limit while remaining gradient-compatible.

```python
lim = SmoothIQRadiusLimitHardwareMap(max_radius=1.0, softness=0.05)
```

### `QuantizationHardwareMap`

Quantizes the waveform to a finite number of DAC levels. The gradient is approximated by a straight-through estimator.

```python
quant = QuantizationHardwareMap(n_bits=14, v_range=1.0)
```

### `GainHardwareMap`

Scalar gain map. Used when the device has a known amplification or attenuation factor.

```python
gain_map = GainHardwareMap(gain=0.8)
```

### `DelayHardwareMap`

Integer sample delay. Useful for modeling cable delays or mismatched trigger timing.

```python
delay = DelayHardwareMap(n_samples=3)
```

### `HardwareModel`

Encapsulates the full hardware pipeline for use inside a `ControlProblem`. Returned by `HardwareContext.as_hardware_model()`.

```python
hardware_model = ctx.as_hardware_model()

from cqed_sim.optimal_control import ControlProblem
problem = ControlProblem(
    ...
    hardware_model=hardware_model,
)
```

---

## Serialization helpers

```python
from cqed_sim.optimal_control.hardware import hardware_map_to_dict, hardware_map_from_dict

d = hardware_map_to_dict(filt)
filt2 = hardware_map_from_dict(d)
```

All built-in `HardwareMap` subclasses are supported. Custom subclasses must implement `to_dict()` and register a corresponding factory in `hardware_map_from_dict`.

---

## Integration with GRAPE

There are two modes for hardware-aware optimal control:

### Mode A — Postprocessing (recommended for nonlinear calibrations)

GRAPE optimizes waveforms without hardware effects. After convergence, apply the hardware pipeline to produce the physical waveform:

```python
from cqed_sim.control import postprocess_grape_waveforms

physical_waveforms = postprocess_grape_waveforms(
    optimized_waveforms,   # dict: channel_name → array
    hardware_context=ctx,
    dt_s=2e-9,
)
```

Use this mode when `CallableCalibrationMap` or any non-differentiable element is present.

### Mode B — Gradient through hardware

Build a `HardwareModel` from the context and pass it to `ControlProblem`. GRAPE propagates gradients back through the hardware chain at each iteration.

```python
problem = ControlProblem(
    parameterization=parameterization,
    systems=[system],
    objectives=[objective],
    hardware_model=ctx.as_hardware_model(),
)
```

Mode B requires all maps in the pipeline to implement `vjp`. `CallableCalibrationMap` is silently skipped; use `LinearCalibrationMap` or gradient-supported maps for full gradient coverage.

---

## Utility functions

```python
from cqed_sim.control import delay_samples_from_time

n = delay_samples_from_time(delay_s=5e-9, dt_s=2e-9)  # Returns 2 (nearest integer)
```

---

## See Also

- [Optimal Control API](optimal_control.md) — `ControlProblem`, `GrapeSolver`, objectives, and penalties
- [Sequence Compilation](sequence.md) — `SequenceCompiler` with `hardware_context` parameter
- [Physics & Conventions](../physics_conventions.md#optimal-control-conventions) — drive sign and frame conventions that apply inside GRAPE
- Example scripts: `examples/hardware_context/01_ideal_vs_hardware.py`, `02_three_line_cqed.py`, `03_grape_hardware_comparison.py`
