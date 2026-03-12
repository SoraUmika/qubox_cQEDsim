# Getting Started

This page introduces the key concepts and mental model for working with `cqed_sim`.

---

## What Is cqed_sim?

`cqed_sim` is a pulse-level simulator for circuit quantum electrodynamics systems in the dispersive regime. It is built on [QuTiP](https://qutip.org/) and designed for:

- Simulating transmon–cavity interactions under explicit drive pulses
- Modeling realistic hardware effects (IQ distortion, DAC quantization, filtering)
- Running calibration and tomography protocols in simulation
- Optimizing gate sequences via unitary synthesis

The package is intended for **experimental physicists**, **graduate students**, and **developers** working with cQED systems.

---

## Core Mental Model

A simulation in `cqed_sim` follows a pipeline:

```
Model → Frame → Pulses → Compile → Simulate → Extract
```

### 1. Model

A **model** defines the physical system: mode frequencies, dispersive shifts, Kerr nonlinearities, and Hilbert-space truncation dimensions.

There are three model classes:

| Class | Use Case |
|---|---|
| `DispersiveTransmonCavityModel` | Two-mode (qubit + storage/cavity) |
| `DispersiveReadoutTransmonStorageModel` | Three-mode (qubit + storage + readout) |
| `UniversalCQEDModel` | General multilevel transmon with arbitrary bosonic modes |

The two-mode and three-mode models are convenience wrappers around `UniversalCQEDModel`.

### 2. Frame

A `FrameSpec` specifies the rotating frame in which the simulation runs. Setting frame frequencies equal to the bare mode frequencies removes fast rotations from the dynamics.

### 3. Pulses

`Pulse` objects describe drive waveforms applied to channels (`"qubit"`, `"storage"`, `"readout"`, etc.). Pulses carry an envelope function, carrier frequency, amplitude, phase, and optional DRAG correction.

Convenience builders construct common pulse types:

- `build_rotation_pulse(...)` — qubit rotations
- `build_displacement_pulse(...)` — cavity displacements
- `build_sideband_pulse(...)` — sideband transitions
- `build_sqr_multitone_pulse(...)` — selective qubit rotations

### 4. Compile

`SequenceCompiler` samples pulses onto a uniform time grid and applies hardware processing (crosstalk, IQ distortion, filtering, quantization).

### 5. Simulate

`simulate_sequence(...)` runs the time-dependent Schrödinger or Lindblad master equation using QuTiP's ODE solvers (or an optional dense-matrix backend).

### 6. Extract

After simulation, extract physical quantities from the final state:

- Partial traces (qubit, cavity, storage, readout)
- Bloch vectors (joint or Fock-conditioned)
- Photon numbers and cavity moments
- Wigner functions
- Qubit measurement outcomes

---

## Convenience Wrapper: SimulationExperiment

For common workflows, `SimulationExperiment` wraps the entire pipeline into a single object:

```python
from cqed_sim.experiment import (
    SimulationExperiment,
    StatePreparationSpec,
    QubitMeasurementSpec,
    qubit_state,
    fock_state,
)

experiment = SimulationExperiment(
    model=model,
    pulses=[pulse],
    drive_ops={"q": "qubit"},
    dt=2e-9,
    frame=frame,
    state_prep=StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=fock_state(0),
    ),
    measurement=QubitMeasurementSpec(shots=2048, seed=42),
)

result = experiment.run()
print(result.measurement.probabilities)
```

---

## What's Next

- [Installation](installation.md) — set up the package
- [Physics & Conventions](physics_conventions.md) — understand the Hamiltonian and sign conventions
- [Defining Models](user_guides/defining_models.md) — build your first model
- [Tutorials](tutorials/displacement_spectroscopy.md) — complete worked examples
