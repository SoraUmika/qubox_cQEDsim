# API Reference — Overview

This section provides a complete reference for the `cqed_sim` public API, organized by module.

---

## Package Summary

`cqed_sim` is a hardware-faithful time-domain circuit-QED pulse simulator built on QuTiP. It models qubit–storage and qubit–storage–readout systems in the dispersive regime with explicit pulse-level drive schedules, Lindblad open-system dynamics, and calibration / tomography helpers.

The package also includes a periodic-drive Floquet layer for closed-system quasienergy analysis of strongly driven and parametrically modulated cQED Hamiltonians.

**Dependencies:** NumPy ≥ 1.24, SciPy ≥ 1.10, QuTiP ≥ 5.0. Optional: JAX for the dense-matrix backend path.

**Internal units:** Hamiltonian coefficients and rotating-frame frequencies are in **rad/s**; times are in **seconds**. All user-facing constructors accept these units unless suffixed otherwise (e.g., `_hz`, `_ns`).

---

## Module Map

| Module | Purpose |
|---|---|
| [`core`](core.md) | Hilbert-space conventions, models, frames, ideal gates |
| [`gates`](gates.md) | Ideal gate operators: qubit, bosonic, entangling, two-qubit |
| [`pulses`](pulses.md) | Pulse dataclass, envelopes, builders, calibration formulas, hardware |
| [`sequence`](sequence.md) | SequenceCompiler, compiled-channel timeline |
| [`simulation`](simulation.md) | Hamiltonian assembly, solver, noise, extractors, couplings |
| [`floquet`](floquet.md) | Periodic-drive Floquet analysis, quasienergies, Sambe builders, and branch tracking |
| [`measurement`](measurement.md) | State preparation, qubit measurement, readout-chain modeling |
| [`optimal_control`](optimal_control.md) | Direct-control problems, held-sample and piecewise-constant schedules, GRAPE, hardware maps, penalties, and waveform export |
| [`control` / hardware](hardware.md) | ControlLine, HardwareContext, CalibrationMap, hardware transfer pipeline |
| [`unitary_synthesis`](unitary_synthesis.md) | Subspace targeting, gate sequences, optimization, constraints |
| [`rl_control`](rl_control.md) | RL environments, task registry, action/observation/reward layers, randomization |
| [`calibration`](calibration.md) | SQR gate calibration |
| [`calibration_targets`](calibration_targets.md) | Spectroscopy, Rabi, Ramsey, T₁, T₂ echo, DRAG tuning |
| [`analysis`](analysis.md) | Parameter translation (bare → dressed) |
| [`observables`](observables.md) | Bloch, Fock-resolved, phase, trajectory, Wigner diagnostics |
| [`plotting`](plotting.md) | Bloch tracks, calibration, gate diagnostics, Wigner grids |
| [`operators`](operators.md) | Pauli, cavity ladder, embedding helpers |
| [`gate_io`](gate_io.md) | Gate sequence JSON I/O |
| [`tomography`](tomography.md) | Fock-resolved tomography, all-XY, leakage calibration |
| [`backends`](backends.md) | Dense NumPy/JAX solver backends |
| [`quantum_algorithms.holographic_sim`](holographic_sim.md) | Generic holographic channel sampling, exact branch estimation, and MPS-inspired abstractions |

---

## Main Simulation Path

```
Model → FrameSpec → Pulses → SequenceCompiler → simulate_sequence → Extractors
```

Alternative periodic-drive path:

```
Model → PeriodicDriveTerm → FloquetProblem → solve_floquet → Floquet analysis helpers
```

1. Build a model (`UniversalCQEDModel`, `DispersiveTransmonCavityModel`, or `DispersiveReadoutTransmonStorageModel`)
2. Define a `FrameSpec`
3. Construct `Pulse` objects (directly or via builders)
4. Compile with `SequenceCompiler`
5. Simulate with `simulate_sequence(...)`
6. Extract results with extractors or `measure_qubit(...)`

For strictly periodic closed-system problems, replace steps 3-6 with `PeriodicDriveTerm` plus `FloquetProblem` and call `solve_floquet(...)` instead.

---

## Canonical Reference

This API reference is synchronized with `API_REFERENCE.md` in the repository root, which serves as the single source of truth. For physics conventions, sign definitions, and Hamiltonian algebra see the [Physics & Conventions](../physics_conventions.md) page.
