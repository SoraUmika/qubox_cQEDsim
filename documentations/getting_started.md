# Getting Started

This page introduces the core mental model for working with `cqed_sim` as a reusable simulation library.

---

## What Is `cqed_sim`?

`cqed_sim` is a pulse-level circuit-QED simulator built on [QuTiP](https://qutip.org/). The package focuses on reusable building blocks:

- model construction and rotating frames
- pulse construction and sequence compilation
- solver execution and extractor utilities
- state preparation primitives
- qubit-measurement and readout-chain primitives
- calibration, tomography, and synthesis helpers

High-level workflow recipes now live under `examples/`.

---

## Core Mental Model

The canonical library path is:

```
Model -> Frame -> Prepare -> Pulses -> Compile -> Simulate -> Extract / Measure
```

### 1. Model

Use `DispersiveTransmonCavityModel`, `DispersiveReadoutTransmonStorageModel`, or `UniversalCQEDModel` from `cqed_sim.core`.

### 2. Frame

`FrameSpec` defines the rotating frame. Matching frame frequencies to bare mode frequencies removes bare rotations.

### 3. Prepare

Use `StatePreparationSpec` and `prepare_state(...)` from `cqed_sim.core` to build model-consistent tensor-product initial states.

### 4. Pulses and Compile

Create `Pulse` objects directly or with builders, then compile them with `SequenceCompiler`.

### 5. Simulate

Run `simulate_sequence(...)` for one-off trajectories or `prepare_simulation(...)` for repeated runs.

### 6. Extract and Measure

Use `cqed_sim.sim` extractors for reduced states, moments, and Wigner functions. Use `measure_qubit(...)` and `cqed_sim.measurement.ReadoutChain` when you need qubit readout or synthetic I/Q.

---

## Direct Workflow Example

```python
from cqed_sim.core import FrameSpec, StatePreparationSpec, fock_state, prepare_state, qubit_state
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

initial = prepare_state(
    model,
    StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=fock_state(0),
    ),
)

compiled = SequenceCompiler(dt=2e-9).compile(pulses, t_end=t_end)
result = simulate_sequence(model, compiled, initial, drive_ops, config=SimulationConfig(frame=frame))
measurement = measure_qubit(result.final_state, QubitMeasurementSpec(shots=2048, seed=42))
```

For repo-side end-to-end recipes, see:

- `examples/protocol_style_simulation.py`
- `examples/kerr_free_evolution.py`
- `examples/sequential_sideband_reset.py`

---

## What's Next

- [Installation](installation.md) — set up the package
- [Physics & Conventions](physics_conventions.md) — understand the Hamiltonian and sign conventions
- [Defining Models](user_guides/defining_models.md) — build your first model
- [Tutorials](tutorials/displacement_spectroscopy.md) — complete worked examples
