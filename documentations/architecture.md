# Architecture

This page describes the internal architecture of `cqed_sim` for developers and advanced users.

---

## Package Layout

```
cqed_sim/
|-- core/                 # Models, frames, conventions, ideal gates, state-prep primitives
|-- measurement/          # Qubit measurement and readout-chain modeling
|-- pulses/               # Pulse dataclass, envelopes, builders, calibration formulas, hardware
|-- sequence/             # SequenceCompiler, compiled-channel timeline
|-- sim/                  # Hamiltonian assembly, solver, noise, extractors, couplings
|-- analysis/             # Parameter translation (bare -> dressed)
|-- backends/             # Dense NumPy/JAX solver backends
|-- calibration/          # SQR gate calibration
|-- calibration_targets/  # Spectroscopy, Rabi, Ramsey, T1, T2 echo, DRAG tuning
|-- rl_control/           # RL environments, action/observation/reward layers, diagnostics, tasks
|-- system_id/            # Calibration-informed priors and randomization hooks
|-- io/                   # Gate sequence JSON I/O
|-- observables/          # Bloch, Fock-resolved, phase, trajectory, Wigner diagnostics
|-- operators/            # Pauli, cavity ladder, embedding helpers
|-- plotting/             # Visualization (Bloch, calibration, gate diagnostics, Wigner)
|-- tomo/                 # Fock-resolved tomography, all-XY calibration
`-- unitary_synthesis/    # Subspace optimization, gate sequences, constraints
`-- optimal_control/      # Direct-control problems, GRAPE, schedule export, and model-backed builders
```

Guided notebook tutorials now live under the top-level `tutorials/` directory. Standalone scripts, audits, studies, and specialized workflow helpers live under `examples/`, not inside the import package.

---

## Module Responsibilities

### `core` — System Definition and State Preparation

The foundation of the package. Contains:

- model classes: `UniversalCQEDModel`, `DispersiveTransmonCavityModel`, `DispersiveReadoutTransmonStorageModel`
- subsystem specs: `TransmonModeSpec`, `BosonicModeSpec`, `DispersiveCouplingSpec`
- `FrameSpec` for rotating-frame definition
- Hilbert-space conventions and basis indexing
- frequency helpers and ideal gates
- structured drive targets
- state-preparation primitives: `StatePreparationSpec`, `prepare_state(...)`, and subsystem-state helpers

### `measurement` — Reusable Readout Primitives

Contains the reusable measurement layer:

- `QubitMeasurementSpec` and `measure_qubit(...)`
- `ReadoutResonator`, `PurcellFilter`, `AmplifierChain`, `ReadoutChain`
- synthetic I/Q generation, nearest-center classification, and readout backaction helpers

### `pulses` — Waveform Construction

- `Pulse` dataclass
- standard envelopes
- common pulse builders
- calibration formulas
- `HardwareConfig`

### `sequence` — Timeline Assembly

`SequenceCompiler` samples pulses onto a uniform grid and applies crosstalk and hardware processing.

### `sim` — Solver and Extractors

The simulation engine:

- `simulate_sequence(...)`
- `SimulationSession` / `prepare_simulation(...)`
- `NoiseSpec` and collapse-operator generation
- reduced states, Bloch coordinates, photon numbers, Wigner functions, conditioned observables
- coupling helpers such as `cross_kerr()`, `self_kerr()`, `exchange()`

### Other Reusable Library Areas

- `analysis` for parameter translation
- `calibration` and `calibration_targets` for reusable calibration helpers
- `rl_control` for benchmark-task definitions, environment wrappers, domain randomization, and measurement-aware observations/rewards
- `system_id` for lightweight posterior/prior hooks that feed the RL randomization layer
- `tomo` for tomography primitives
- `observables` and `plotting` for diagnostics and visualization
- `unitary_synthesis` for gate-sequence optimization
- `optimal_control` for piecewise-constant direct-control optimization and GRAPE-based waveform design

---

## Example Boundary

`cqed_sim` intentionally keeps user-facing educational notebooks and specialized repo workflows outside the import package:

- numbered tutorial notebooks under `tutorials/`
- typical standalone end-to-end workflow recipes
- Kerr free-evolution demonstrations
- sequential sideband-reset recipes
- paper reproductions, audits, and one-off studies

Representative example entry points:

- `tutorials/README.md`
- `tutorials/00_tutorial_index.ipynb`
- `examples/protocol_style_simulation.py`
- `examples/kerr_free_evolution.py`
- `examples/kerr_sign_verification.py`
- `examples/sequential_sideband_reset.py`

---

## Data Flow

```
UniversalCQEDModel / wrapper models
        |
        +--> FrameSpec --> static_hamiltonian(frame)
        |
        +--> prepare_state(...)
        |
        +--> Pulse objects --> SequenceCompiler.compile() --> CompiledSequence
                                                           |
                                                           v
                              simulate_sequence(model, compiled, initial_state, drive_ops, config)
                                                           |
                                                           v
                                                   SimulationResult
                                                           |
                               +---------------------------+---------------------------+
                               v                           v                           v
                     reduced_qubit_state()        cavity_wigner()             measure_qubit()
                     bloch_xyz_from_joint()       mode_moments()              ReadoutChain
```

---

## Stable Public API vs Internal

Everything exported through `cqed_sim/__init__.py` and the active subpackage `__init__.py` files is part of the public API. The notebooks under `tutorials/` and the workflow modules under `examples/` are repository-side teaching or study assets, not part of the installed library surface.
