# `cqed_sim.control` — Hardware-Aware Control Transfer Layer

## What It Does

The `control` module models the signal path from programmed waveforms to the
Hamiltonian coefficients seen by the simulator.  For each physical control
line *j* the chain is:

```
u_prog_j(t)  --[T_j]-->  u_dev_j(t)  --[f_j]-->  c_j(t)
```

where **T_j** is the hardware transfer model (gain, delay, FIR filter, …) and
**f_j** is the calibration map converting hardware units to rad/s.

## Why It Exists

Without an explicit control layer the optimizer and the simulator would
silently disagree about what waveform reaches the device.  This module
makes the programmed → device → Hamiltonian mapping explicit, serializable,
and composable with GRAPE postprocessing.

## Key Classes and Functions

| Symbol | Role |
|---|---|
| `ControlLine` | One named hardware line with transfer model, calibration map, and unit metadata |
| `HardwareContext` | Collection of `ControlLine` objects; provides a unified transform and JSON serialization |
| `CalibrationMap` | Abstract base for hardware-to-Hamiltonian unit conversion |
| `LinearCalibrationMap` | Proportional gain + offset calibration |
| `CallableCalibrationMap` | Arbitrary Python callable calibration |
| `postprocess_grape_waveforms(...)` | Apply a `HardwareContext` to GRAPE output waveforms |
| `delay_samples_from_time(delay_s, dt)` | Convert a physical time delay to an integer sample count |
| `hardware_map_to_dict` / `hardware_map_from_dict` | Serialize / deserialize individual `HardwareMap` objects |

The companion submodule `cqed_sim.control.cqed_device` provides
`make_three_line_cqed_context(...)`, a convenience factory that builds a
three-line `HardwareContext` (qubit drive, cavity drive, readout) from typical
cQED parameters.  This factory is re-exported at the top-level
`cqed_sim.make_three_line_cqed_context` for easy access.

## Units

The module is unit-coherent.  The recommended convention is rad/s for
frequencies and seconds for times.  Each `ControlLine` carries explicit
`programmed_unit`, `device_unit`, and `coefficient_unit` fields for
documentation.

## How It Fits In

- **GRAPE / optimal control:** `postprocess_grape_waveforms` distorts
  optimizer output through the hardware transfer model before it reaches the
  simulator, enabling hardware-aware optimization (Mode A).
- **Simulation:** `HardwareContext` plugs into `SimulationSession` to apply
  the transfer chain during Hamiltonian assembly.
- **Serialization:** `HardwareContext.save()` / `.load()` round-trip the full
  hardware model to JSON for reproducible notebooks and experiment logs.

## Typical Usage

```python
from cqed_sim.control import ControlLine, HardwareContext, LinearCalibrationMap

qubit_line = ControlLine(
    name="qubit_drive",
    calibration=LinearCalibrationMap(gain=2 * np.pi * 50e6),
)
ctx = HardwareContext(lines=[qubit_line])
ctx.save("my_hardware.json")
```

Or use the convenience factory:

```python
from cqed_sim import make_three_line_cqed_context

ctx = make_three_line_cqed_context(model, frame=frame)
```

## Limitations

- Only linear time-invariant (LTI) transfer models are currently supported.
- Nonlinear mixer or amplifier saturation effects are not modeled.
