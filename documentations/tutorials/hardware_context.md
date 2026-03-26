# Hardware-Aware Control Tutorial

This page covers the hardware-aware simulation and optimal-control pipeline: how `cqed_sim` models the physical signal chain from AWG output to Hamiltonian coefficients, and how that chain integrates with GRAPE.

---

## Motivation

Real cQED experiments don't apply ideal waveforms. Between the programmed values and the qubit, there are:

- **Low-pass filters** (cable bandwidth, amplifier roll-off)
- **DAC quantization** (finite bit depth)
- **IQ amplitude limits** (mixer or amplifier saturation)
- **Cable delays** and trigger timing mismatches
- **Crosstalk** between control lines
- **Calibration curves** (nonlinear amplitude-to-drive-strength mapping)

`cqed_sim` provides the `ControlLine` and `HardwareContext` abstraction to model these effects explicitly, and integrates them into the GRAPE solver so that optimized pulses already account for hardware distortion.

---

## Step 1: Define a Hardware Context

A `HardwareContext` collects all control lines for a device:

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.control import ControlLine, HardwareContext
from cqed_sim.optimal_control.hardware import FirstOrderLowPassHardwareMap

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9, omega_q=2*np.pi*6e9,
    alpha=2*np.pi*(-200e6), chi=2*np.pi*(-2.84e6),
    kerr=2*np.pi*(-2e3), n_cav=8, n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

# Build a qubit I-channel with a 200 MHz low-pass filter
qubit_I = ControlLine(
    name="qubit_I",
    operator=model.drive_coupling_operators(frame)["qubit"],
    calibration_gain=2*np.pi*1e6,           # rad/s per Volt
    transfer_maps=[
        FirstOrderLowPassHardwareMap(cutoff_hz=200e6, dt_s=2e-9),
    ],
    programmed_unit="V",
    device_unit="V",
    coefficient_unit="rad/s",
    operator_label="(b + b†) / 2",
    frame="rotating_qubit",
)
```

For a standard three-line dispersive readout + storage + qubit setup, use the factory:

```python
from cqed_sim.control.cqed_device import make_three_line_cqed_context

ctx = make_three_line_cqed_context(
    qubit_op=model.drive_coupling_operators(frame)["qubit"],
    cavity_op=model.drive_coupling_operators(frame)["cavity"],
    readout_op=model.drive_coupling_operators(frame)["readout"],
    calibration_gains={
        "qubit_I": 2*np.pi*1e6,
        "qubit_Q": 2*np.pi*1e6,
        "cavity":  2*np.pi*0.5e6,
    },
)
```

---

## Step 2: Apply the Pipeline

After constructing a `HardwareContext`, apply it to transform programmed waveforms into physical Hamiltonian coefficients:

```python
u_prog = {"qubit_I": np.sin(np.linspace(0, np.pi, 50))}
c_physical = ctx.apply_all(u_prog, dt_s=2e-9)
```

The result is the physical coefficient `c(t)` that enters the Hamiltonian:

$$H(t) = H_0 + \sum_j c_j(t) \, O_j$$

---

## Step 3: Serialize a Device Configuration

Save and reload a hardware context for reproducibility:

```python
ctx.save("device.json")
ctx2 = HardwareContext.load("device.json")
```

All built-in hardware maps and `LinearCalibrationMap` are JSON-serializable. `CallableCalibrationMap` (arbitrary Python functions) cannot be serialized.

---

## Step 4: Hardware-Aware GRAPE

There are two modes for hardware-aware optimal control:

### Mode A — Postprocessing

GRAPE optimizes unconstrained waveforms, then hardware effects are applied after convergence:

```python
from cqed_sim.control import postprocess_grape_waveforms

physical_waveforms = postprocess_grape_waveforms(
    optimized_waveforms=result.schedule_dict(),
    hardware_context=ctx,
    dt_s=2e-9,
)
```

This works for any calibration, including nonlinear maps.

### Mode B — Gradient Through Hardware

Attach a `HardwareModel` to the GRAPE problem so the solver accounts for filtering during optimization:

```python
from cqed_sim.optimal_control import (
    build_control_problem_from_model,
    GrapeSolver, GrapeConfig,
    HardwareModel, HeldSampleParameterization,
    PiecewiseConstantTimeGrid, ModelControlChannelSpec,
    state_preparation_objective,
)
from cqed_sim.optimal_control.hardware import (
    FirstOrderLowPassHardwareMap,
    SmoothIQRadiusLimitHardwareMap,
    BoundaryWindowHardwareMap,
)

problem = build_control_problem_from_model(
    model, frame=frame,
    time_grid=PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=20e-9),
    channel_specs=(ModelControlChannelSpec(
        name="qubit", target="qubit", quadratures=("I", "Q"),
        amplitude_bounds=(-8e7, 8e7), export_channel="qubit",
    ),),
    objectives=(state_preparation_objective(
        model.basis_state(0, 0), model.basis_state(1, 0)
    ),),
    parameterization_cls=HeldSampleParameterization,
    parameterization_kwargs={"sample_period_s": 40e-9},
    hardware_model=HardwareModel(maps=(
        FirstOrderLowPassHardwareMap(cutoff_hz=25e6, export_channels=("qubit",)),
        SmoothIQRadiusLimitHardwareMap(amplitude_max=6e7, export_channels=("qubit",)),
        BoundaryWindowHardwareMap(ramp_slices=1, export_channels=("qubit",)),
    )),
)

result = GrapeSolver(GrapeConfig(maxiter=80, seed=7)).solve(problem)
```

Mode B requires all maps to have gradient support (`vjp`).

---

## Step 5: Replay and Verify

After optimization, replay both the command and physical waveforms through the full time-domain simulator:

```python
command_replay = result.evaluate_with_simulator(
    problem, model=model, frame=frame,
    compiler_dt_s=1e-9, waveform_mode="command",
)
physical_replay = result.evaluate_with_simulator(
    problem, model=model, frame=frame,
    compiler_dt_s=1e-9, waveform_mode="physical",
)
print(f"Command fidelity: {command_replay.metrics['aggregate_fidelity']:.6f}")
print(f"Physical fidelity: {physical_replay.metrics['aggregate_fidelity']:.6f}")
```

!!! note
    The physical-replay fidelity includes all hardware distortion. If the physical fidelity is significantly worse than the command fidelity, the hardware model is actively limiting performance and may need compensation or a longer pulse duration.

---

## Available Hardware Maps

| Map | Effect | Gradient |
|---|---|---|
| `FirstOrderLowPassHardwareMap` | Single-pole IIR low-pass filter | Yes |
| `FIRHardwareMap` | Arbitrary FIR filter by tap coefficients | Yes |
| `FrequencyResponseHardwareMap` | Measured H(f) → FIR via IFFT | Yes |
| `BoundaryWindowHardwareMap` | Smooth taper at pulse start/end | Yes |
| `SmoothIQRadiusLimitHardwareMap` | Soft-clip I/Q amplitude | Yes |
| `QuantizationHardwareMap` | DAC quantization (straight-through gradient) | Approximate |
| `GainHardwareMap` | Scalar gain | Yes |
| `DelayHardwareMap` | Integer sample delay | Yes |

---

## Example Scripts

| Script | Description |
|---|---|
| `examples/hardware_context/01_ideal_vs_hardware.py` | Side-by-side ideal vs hardware-filtered simulation |
| `examples/hardware_context/02_three_line_cqed.py` | Standard three-line dispersive context |
| `examples/hardware_context/03_grape_hardware_comparison.py` | GRAPE with and without hardware maps |
| `examples/hardware_constrained_grape_demo.py` | Full hardware-aware GRAPE demo with replay |

---

## See Also

- [Hardware & Control API](../api/hardware.md) — full API reference
- [Optimal Control API](../api/optimal_control.md) — `ControlProblem`, `GrapeSolver`, objectives
- [Physics & Conventions](../physics_conventions.md#optimal-control-conventions) — drive sign and frame conventions
