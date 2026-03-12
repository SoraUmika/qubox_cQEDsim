# Architecture

This page describes the internal architecture of `cqed_sim` for developers and advanced users.

---

## Package Layout

```
cqed_sim/
├── core/                 # Models, frames, conventions, ideal gates, frequency helpers
├── pulses/               # Pulse dataclass, envelopes, builders, calibration formulas, hardware
├── sequence/             # SequenceCompiler, compiled-channel timeline
├── sim/                  # Hamiltonian assembly, solver, noise, extractors, couplings
├── experiment/           # State preparation, measurement, readout chain, protocols
├── analysis/             # Parameter translation (bare → dressed)
├── backends/             # Dense NumPy/JAX solver backends
├── calibration/          # SQR gate calibration
├── calibration_targets/  # Spectroscopy, Rabi, Ramsey, T₁, T₂ echo, DRAG tuning
├── io/                   # Gate sequence JSON I/O
├── observables/          # Bloch, Fock-resolved, phase, trajectory, Wigner diagnostics
├── operators/            # Pauli, cavity ladder, embedding helpers
├── plotting/             # Visualization (Bloch, calibration, gate diagnostics, Wigner)
├── tomo/                 # Fock-resolved tomography, all-XY calibration
└── unitary_synthesis/    # Subspace optimization, gate sequences, constraints
```

---

## Module Responsibilities

### `core` — System Definition

The foundation of the package. Contains:

- **Model classes:** `UniversalCQEDModel` (general), `DispersiveTransmonCavityModel` (two-mode), `DispersiveReadoutTransmonStorageModel` (three-mode)
- **Subsystem specs:** `TransmonModeSpec`, `BosonicModeSpec`, `DispersiveCouplingSpec`
- **Frame:** `FrameSpec` for rotating-frame definition
- **Conventions:** Hilbert-space dimension helpers, basis indexing
- **Frequency helpers:** Manifold transition frequencies, carrier conversion
- **Ideal gates:** Qubit rotations, displacement, SNAP, SQR, beamsplitter
- **Coupling specs:** `CrossKerrSpec`, `SelfKerrSpec`, `ExchangeSpec`
- **Drive targets:** `TransmonTransitionDriveSpec`, `SidebandDriveSpec`

The two-mode and three-mode models are **compatibility wrappers** that delegate to `UniversalCQEDModel`. New code should prefer `UniversalCQEDModel` for flexibility.

### `pulses` — Waveform Construction

- **`Pulse`** dataclass: channel, timing, envelope, carrier, amplitude, phase, DRAG
- **Envelopes:** `square_envelope`, `gaussian_envelope`, `cosine_rise_envelope`, `normalized_gaussian`, `multitone_gaussian_envelope`
- **Builders:** `build_rotation_pulse`, `build_displacement_pulse`, `build_sideband_pulse`, `build_sqr_multitone_pulse`
- **Calibration formulas:** Amplitude scaling for displacement, rotation, SQR
- **`HardwareConfig`:** IQ distortion, quantization, filtering model

### `sequence` — Timeline Assembly

`SequenceCompiler` takes a list of `Pulse` objects and produces a `CompiledSequence` with per-channel sampled waveforms. The compilation pipeline includes:

1. Uniform time grid construction
2. Per-pulse envelope sampling with carrier
3. Crosstalk mixing between channels
4. Per-channel hardware processing (ZOH, lowpass, quantization, IQ distortion)

### `sim` — Solver and Extractors

The simulation engine:

- **`simulate_sequence()`** — main entry point
- **`SimulationSession` / `prepare_simulation()`** — session reuse for sweeps
- **`NoiseSpec`** — Lindblad noise specification
- **State extractors:** Partial traces, Bloch coordinates, photon numbers, Wigner functions, conditioned observables
- **Couplings:** `cross_kerr()`, `self_kerr()`, `exchange()`, `TunableCoupler`
- **Diagnostics:** Channel norms, instantaneous frequency analysis

### `experiment` — Protocol Layer

High-level experiment abstractions:

- **State preparation:** `prepare_state()`, `StatePreparationSpec`
- **Measurement:** `measure_qubit()`, `QubitMeasurementSpec`
- **Readout chain:** `ReadoutResonator`, `PurcellFilter`, `AmplifierChain`, `ReadoutChain`
- **Protocol wrapper:** `SimulationExperiment` combines prepare → compile → simulate → measure
- **Kerr free evolution:** Specialized cavity free-evolution workflow

### `analysis` — Parameter Translation

Translates between bare transmon parameters (E_J, E_C, g) and dressed dispersive parameters (ω_q, α, χ) used at runtime.

### `backends` — Alternative Solvers

Dense piecewise-constant solver backends (`NumPyBackend`, `JaxBackend`) for small-system checks and backend parity validation. Not intended as primary solvers for large systems.

### `calibration` — SQR Gate Calibration

Per-manifold amplitude, phase, and frequency correction optimization for Selective Qubit Rotation gates.

### `calibration_targets` — Surrogate Experiments

Lightweight surrogate-model calibration sweeps: spectroscopy, Rabi, Ramsey, T₁, T₂ echo, DRAG tuning. Each returns a `CalibrationResult` with fitted parameters.

### `io` — Gate Sequence I/O

JSON-based gate sequence loading and serialization. Gate types: `DisplacementGate`, `RotationGate`, `SQRGate`.

### `observables` — Diagnostic Observables

Post-simulation analysis: Fock-resolved Bloch diagnostics, relative phase families, Bloch trajectories, weakness metrics, Wigner functions.

### `operators` — Primitive Operators

Pauli matrices, cavity ladder operators, embedding helpers, state constructors.

### `plotting` — Visualization

Matplotlib-based plotting for Bloch tracks, calibration results, gate diagnostics, phase evolution, weakness comparisons, and Wigner grids.

### `tomo` — Tomography

Fock-resolved tomography protocol, all-XY calibration, selective π-pulses, leakage matrix calibration.

### `unitary_synthesis` — Gate Optimization

Gradient-free optimization of gate sequences (Displacement, Rotation, SQR, SNAP) to implement target unitaries within qubit–cavity subspaces. Includes subspace targeting, leakage penalties, time constraints, and progress reporting.

---

## Data Flow

```
UniversalCQEDModel
        │
        ├── FrameSpec ──► static_hamiltonian(frame)
        │
        ├── Pulse objects ──► SequenceCompiler.compile() ──► CompiledSequence
        │                                                          │
        │                                   ┌──────────────────────┘
        │                                   ▼
        └── simulate_sequence(model, compiled, initial_state, drive_ops, config)
                                            │
                                            ▼
                                    SimulationResult
                                            │
                    ┌───────────────────────┼──────────────────┐
                    ▼                       ▼                  ▼
            reduced_qubit_state()   cavity_wigner()    measure_qubit()
            bloch_xyz_from_joint()  cavity_moments()   QubitMeasurementResult
```

---

## Stable Public API vs Internal

### Public API

Everything exported through `cqed_sim/__init__.py` and subpackage `__init__.py` files is part of the public API. The full reference is in [API Reference](api/overview.md) and `API_REFERENCE.md`.

### Internal / Semi-Internal

The following are implementation details and should not be relied upon:

- `_operators_cache`, `_static_h_cache` on model dataclasses
- `_legacy_drive_couplings()` in `sim.runner`
- `DenseSolverResult` in `sim.solver`
- `coupling_term_key()`, `resolve_operator()` in `core.hamiltonian`
- `PHASE_FAMILY_SPECS` in `observables.fock`

---

## Extension Points

- **New models:** Subclass or compose using `UniversalCQEDModel` with custom `TransmonModeSpec` and `BosonicModeSpec` configurations
- **New pulse envelopes:** Pass any callable `f(t_rel) -> ndarray` as the envelope argument to `Pulse`
- **New backends:** Implement `BaseBackend` abstract interface
- **New calibration targets:** Follow the pattern in `calibration_targets/` returning `CalibrationResult`
- **New gate types:** For unitary synthesis, implement `GateBase` interface
