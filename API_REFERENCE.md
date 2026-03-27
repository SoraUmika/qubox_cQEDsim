# `cqed_sim` — API Reference

> **Canonical source of truth for the public API of the `cqed_sim` library.**
>
> This document reflects the current implementation. Where behavior is inferred rather
> than explicitly documented in docstrings, that is noted. For physics conventions,
> sign definitions, and Hamiltonian algebra see `physics_and_conventions/physics_conventions_report.tex`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Package Architecture](#2-package-architecture)
3. [Core Abstractions (`cqed_sim.core`)](#3-core-abstractions)
   - [UniversalCQEDModel and Subsystem Specs](#universalcqedmodel-and-subsystem-specs)
   - 3.1 [DispersiveTransmonCavityModel](#31-dispersivetransmoncavitymodel)
   - 3.2 [DispersiveReadoutTransmonStorageModel](#32-dispersivereadouttransmonstoragemodel)
   - 3.3 [FrameSpec](#33-framespec)
   - 3.4 [Coupling Specifications](#34-coupling-specifications)
   - 3.5 [Basis Conventions](#35-basis-conventions)
   - 3.6 [Frequency Helpers](#36-frequency-helpers)
   - 3.7 [Ideal Gate Operators](#37-ideal-gate-operators)
   - 3.8 [Energy Spectrum](#38-energy-spectrum)
   - 3.9 [Gate Library (`cqed_sim.gates`)](#39-gate-library-cqed_simgates)
4. [Pulse System (`cqed_sim.pulses`)](#4-pulse-system)
   - 4.1 [Pulse](#41-pulse)
   - 4.2 [Envelopes](#42-envelopes)
   - 4.3 [Pulse Builders](#43-pulse-builders)
   - 4.4 [Calibration Formulas](#44-calibration-formulas)
   - 4.5 [HardwareConfig](#45-hardwareconfig)
5. [Sequence Compilation (`cqed_sim.sequence`)](#5-sequence-compilation)
   - 5.1 [SequenceCompiler](#51-sequencecompiler)
   - 5.2 [CompiledSequence / CompiledChannel](#52-compiledsequence--compiledchannel)
6. [Simulation Engine (`cqed_sim.sim`)](#6-simulation-engine)
   - 6.1 [simulate_sequence](#61-simulate_sequence)
   - 6.2 [SimulationConfig](#62-simulationconfig)
   - 6.3 [SimulationResult](#63-simulationresult)
   - 6.4 [SimulationSession and Prepared Simulation](#64-simulationsession-and-prepared-simulation)
   - 6.5 [NoiseSpec](#65-noisespec)
   - 6.6 [Coupling Helpers](#66-coupling-helpers)
   - 6.7 [State Extractors](#67-state-extractors)
   - 6.8 [Diagnostics](#68-diagnostics)
7. [State Preparation and Measurement (`cqed_sim.core`, `cqed_sim.measurement`)](#7-state-preparation-and-measurement)
   - 7.1 [State Preparation](#71-state-preparation)
   - 7.2 [Qubit Measurement](#72-qubit-measurement)
   - 7.3 [Readout Chain](#73-readout-chain)
8. [Gate I/O (`cqed_sim.io`)](#8-gate-io)
9. [Analysis (`cqed_sim.analysis`)](#9-analysis)
9A. [Floquet Analysis (`cqed_sim.floquet`)](#9a-floquet-analysis-cqed_simfloquet)
10. [Backends (`cqed_sim.backends`)](#10-backends)
11. [SQR Calibration And Multitone Validation (`cqed_sim.calibration`)](#11-sqr-calibration)
12. [Calibration Targets (`cqed_sim.calibration_targets`)](#12-calibration-targets)
13. [Tomography (`cqed_sim.tomo`)](#13-tomography)
14. [Observables (`cqed_sim.observables`)](#14-observables)
15. [Operators (`cqed_sim.operators`)](#15-operators)
16. [Plotting (`cqed_sim.plotting`)](#16-plotting)
17. [Unitary Synthesis (`cqed_sim.unitary_synthesis`)](#17-unitary-synthesis)
    - 17.1 [Subspace](#171-subspace)
    - 17.2 [System Backends](#172-system-backends)
    - 17.3 [Core Targets](#173-core-targets)
    - 17.4 [Gate Primitives and Sequences](#174-gate-primitives-and-sequences)
    - 17.5 [Waveform Bridge](#175-waveform-bridge)
    - 17.6 [Phase 2 Configuration Objects](#176-phase-2-configuration-objects)
    - 17.7 [UnitarySynthesizer](#177-unitarysynthesizer)
    - 17.8 [Results and Metrics](#178-results-and-metrics)
17A. [Holographic Quantum Algorithms (`cqed_sim.quantum_algorithms.holographic_sim`)](#17a-holographic-quantum-algorithms-cqed_simquantum_algorithmsholographic_sim)
17B. [RL Control And System Identification (`cqed_sim.rl_control`, `cqed_sim.system_id`)](#17b-rl-control-and-system-identification-cqed_simrl_control-cqed_simsystem_id)
17C. [Optimal Control (`cqed_sim.optimal_control`)](#17c-optimal-control-cqed_simoptimal_control)
18. [Simulation Workflows / Common Usage Patterns](#18-simulation-workflows)
19. [Physics-Facing API and Conventions](#19-physics-facing-api-and-conventions)
20. [Notes on Internal Utilities](#20-notes-on-internal-utilities)
21. [Ambiguities / Gaps / Known Mismatches](#21-ambiguities--gaps--known-mismatches)

---

## 1. Overview

`cqed_sim` is a hardware-faithful time-domain circuit-QED pulse simulator built on
QuTiP. It models qubit–storage and qubit–storage–readout systems in the dispersive
regime with explicit pulse-level drive schedules, Lindblad open-system dynamics, and
calibration / tomography helpers.

The library also includes a periodic-drive Floquet layer for closed cQED Hamiltonians,
including quasienergy extraction, one-period propagators, optional Sambe-space
harmonic constructions, resonance detection, and overlap-based quasienergy branch
tracking.

**Dependencies:** NumPy ≥ 1.24, SciPy ≥ 1.10, QuTiP ≥ 5.0. Optional: JAX for the
dense-matrix backend path. The current top-level package import also relies on
matplotlib ≥ 3.8 and pandas ≥ 2.0 because progress-reporting utilities are part of
the public import surface. The packaged runtime also includes the local
`physics_and_conventions` module because several public APIs import it directly.

**Units:** The library is unit-coherent: it does not enforce specific physical units
for frequencies or times. Any internally consistent unit system is valid (for example,
rad/s with times in seconds, or rad/ns with times in nanoseconds). The recommended
convention used in the main examples and calibration function naming is rad/s and
seconds. User-facing constructors with suffixes such as `_hz` or `_ns` are explicit
exceptions that accept those specific units.

---

## 2. Package Architecture

```
cqed_sim/
├── core/            # Hilbert-space conventions, models, frames, ideal gates, state-prep primitives
├── pulses/          # Pulse dataclass, envelopes, builders, calibration formulas, hardware
├── sequence/        # SequenceCompiler, compiled-channel timeline
├── sim/             # Hamiltonian assembly, solver, noise, extractors, couplings
├── floquet/         # Periodic-drive Floquet analysis, Sambe builders, resonance helpers, branch tracking
├── measurement/     # Qubit measurement and readout-chain modeling
├── analysis/        # Parameter translation (bare → dressed)
├── backends/        # Dense NumPy/JAX solver backends
├── calibration/     # SQR gate calibration, reduced multitone checks, full logical-subspace multitone validation
├── calibration_targets/  # Spectroscopy, Rabi, Ramsey, T1, T2 echo, DRAG tuning
├── rl_control/      # RL environments, task registry, action/observation/reward layers, randomization
├── system_id/       # Calibration-informed priors and fit-then-randomize hooks
├── io/              # Gate sequence JSON I/O
├── observables/     # Bloch, Fock-resolved, phase, trajectory, Wigner diagnostics
├── operators/       # Pauli, cavity ladder, embedding helpers
├── plotting/        # Bloch tracks, calibration, gate diagnostics, Wigner grids
├── tomo/            # Fock-resolved tomography, all-XY, leakage calibration
├── unitary_synthesis/  # Subspace targeting, gate sequences, optimization, constraints
├── optimal_control/  # Direct-control problems, piecewise-constant schedules, GRAPE, penalties, pulse export
└── quantum_algorithms/  # Generic holographic quantum-algorithm utilities
```

The calibration package now contains two complementary multitone validation layers in addition to the guarded SQR helpers:

- `conditioned_multitone`: reduced conditioned-qubit reachability and correction optimization.
- `targeted_subspace_multitone`: full logical-subspace validation and optimization, including restricted logical-process metrics, state-transfer checks, cavity-block preservation, and leakage accounting.

**Main simulation path for a typical user:**

1. Build a model (`UniversalCQEDModel`, `DispersiveTransmonCavityModel`, or `DispersiveReadoutTransmonStorageModel`).
2. Define a `FrameSpec`.
3. Construct `Pulse` objects (directly or via builders).
4. Compile with `SequenceCompiler`.
5. Simulate with `simulate_sequence(...)`.
6. Extract results with extractors or `measure_qubit(...)`.

---

## 3. Core Abstractions

**Module:** `cqed_sim.core`

### UniversalCQEDModel and Subsystem Specs

`UniversalCQEDModel` is the generalized model-layer abstraction used by the
current wrapper models. It centralizes operator construction, basis helpers,
static Hamiltonian assembly, multilevel ancilla transitions, and bosonic-mode
transition helpers.

```python
@dataclass(frozen=True)
class TransmonModeSpec:
    omega: float
    dim: int = 3
    alpha: float = 0.0
    label: str = "qubit"
    aliases: Sequence[str] = ("qubit", "transmon")
    frame_channel: str = "q"

@dataclass(frozen=True)
class BosonicModeSpec:
    label: str
    omega: float
    dim: int
    kerr: float = 0.0
    kerr_higher: Sequence[float] = ()
    aliases: Sequence[str] = ()
    frame_channel: str = "c"

@dataclass(frozen=True)
class DispersiveCouplingSpec:
    mode: str
    chi: float = 0.0
    chi_higher: Sequence[float] = ()
    transmon: str = "qubit"

@dataclass
class UniversalCQEDModel:
    transmon: TransmonModeSpec | None = None
    bosonic_modes: Sequence[BosonicModeSpec] = ()
    dispersive_couplings: Sequence[DispersiveCouplingSpec] = ()
    cross_kerr_terms: Sequence[CrossKerrSpec] = ()
    self_kerr_terms: Sequence[SelfKerrSpec] = ()
    exchange_terms: Sequence[ExchangeSpec] = ()
```

#### Notes

- `UniversalCQEDModel` supports transmon-only, cavity-only, and multilevel transmon-plus-multimode systems.
- Tensor ordering is transmon first when present, followed by bosonic modes in declaration order.
- The existing `DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel` are compatibility frontends that delegate to this shared core.

#### Common methods

| Method | Signature | Description |
|---|---|---|
| `operators()` | `-> dict[str, qt.Qobj]` | Cached full-space ladder and number operators, including compatibility aliases |
| `hamiltonian(frame=None)` | `-> qt.Qobj` | Alias of `static_hamiltonian(frame)` |
| `energy_spectrum(frame=None, levels=None)` | `-> EnergySpectrum` | Exact diagonalization of the static Hamiltonian with vacuum-referenced energies |
| `basis_state(*levels)` | `-> qt.Qobj` | Basis ket using subsystem declaration order |
| `basis_energy(*levels, frame=None)` | `-> float` | Diagonal basis energy under the static Hamiltonian |
| `transmon_lowering()` | `-> qt.Qobj` | Full-space transmon lowering operator |
| `mode_annihilation(mode)` | `(str) -> qt.Qobj` | Full-space bosonic lowering operator for the named mode |
| `transmon_transition_frequency(...)` | `-> float` | Multilevel transmon transition at fixed bosonic occupations |
| `mode_transition_frequency(...)` | `-> float` | Bosonic ladder spacing at fixed occupations |
| `sideband_transition_frequency(...)` | `-> float` | Effective red/blue sideband manifold transition |

### 3.1 DispersiveTransmonCavityModel

**Module path:** `cqed_sim.core.model.DispersiveTransmonCavityModel`

Two-mode (qubit + cavity/storage) dispersive system model. This class is a
thin compatibility wrapper around `UniversalCQEDModel`.

```python
@dataclass
class DispersiveTransmonCavityModel:
    omega_c: float              # Cavity/storage frequency (rad/s)
    omega_q: float              # Qubit frequency (rad/s)
    alpha: float                # Transmon anharmonicity (rad/s), typically negative
    chi: float = 0.0            # First-order dispersive shift (rad/s)
    chi_higher: Sequence[float] = ()  # Higher-order dispersive: chi_2, chi_3, ...
    kerr: float = 0.0           # Cavity self-Kerr (rad/s)
    kerr_higher: Sequence[float] = ()  # Higher-order Kerr: K_2, ...
    cross_kerr_terms: Sequence[CrossKerrSpec] = ()
    self_kerr_terms: Sequence[SelfKerrSpec] = ()
    exchange_terms: Sequence[ExchangeSpec] = ()
    n_cav: int = 12             # Cavity Hilbert-space dimension
    n_tr: int = 3               # Transmon Hilbert-space dimension
    subsystem_labels: tuple[str, ...] = ("qubit", "storage")
```

**Physics convention:** Positive `chi` means the qubit |g⟩→|e⟩ transition frequency
**increases** with increasing cavity photon number:

$$\omega_{ge}(n) = \omega_{ge}(0) + \chi \cdot n + \chi_2 \cdot n(n-1) + \cdots$$

The static Hamiltonian in the rotating frame is:

$$H_0/\hbar = \delta_c\, n_c + \delta_q\, n_q + \frac{\alpha}{2}\, b^{\dagger 2} b^2 + \frac{K}{2}\, n_c(n_c - 1) + \chi\, n_c\, n_q + \chi_2\, n_c(n_c-1)\, n_q \;+\;\cdots$$

where δ_c = ω_c − ω_c^frame, δ_q = ω_q − ω_q^frame.

Higher-order coefficients use **falling-factorial** form: `chi_higher[i]` multiplies
n_c(n_c−1)···(n_c−i), **not** n_c^(i+1).

#### Properties

| Property | Type | Description |
|---|---|---|
| `subsystem_dims` | `tuple[int, ...]` | `(n_tr, n_cav)` |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `operators()` | `-> dict[str, qt.Qobj]` | Cached dict of ladder/number operators: `a`, `adag`, `b`, `bdag`, `n_c`, `n_q` |
| `drive_coupling_operators()` | `-> dict[str, tuple[qt.Qobj, qt.Qobj]]` | `(raising, lowering)` pairs keyed by `"cavity"`, `"storage"`, `"qubit"`, `"sideband"` |
| `transmon_level_projector(level)` | `(int) -> qt.Qobj` | Projector onto a transmon level in the full tensor-product space |
| `transmon_transition_operators(lower_level, upper_level)` | `(int, int) -> tuple[qt.Qobj, qt.Qobj]` | Full-space raising/lowering operators for a selected transmon ladder transition |
| `mode_operators(mode="storage")` | `(str) -> tuple[qt.Qobj, qt.Qobj]` | Retrieve `(annihilation, creation)` operators for the named bosonic mode |
| `sideband_drive_operators(mode="storage", lower_level=0, upper_level=1, sideband="red")` | `(str, int, int, str) -> tuple[qt.Qobj, qt.Qobj]` | Effective multilevel sideband coupling operators |
| `static_hamiltonian(frame)` | `(frame: FrameSpec \| None = None) -> qt.Qobj` | Full static Hamiltonian including all coupling terms. Cached by frame parameters. |
| `basis_state(q_level, cavity_level)` | `(int, int) -> qt.Qobj` | Returns \|q⟩⊗\|n⟩ ket |
| `basis_energy(q_level, cavity_level, frame)` | `(int, int, FrameSpec \| None) -> float` | Eigenvalue of \|q,n⟩ under static Hamiltonian |
| `coherent_qubit_superposition(n_cav)` | `(int) -> qt.Qobj` | (&#124;g,n⟩ + &#124;e,n⟩) / √2 |
| `manifold_transition_frequency(n, frame)` | `(int, FrameSpec \| None) -> float` | &#124;g,n⟩ ↔ &#124;e,n⟩ transition in given frame |
| `transmon_transition_frequency(cavity_level=0, lower_level=0, upper_level=1, frame=None)` | `(int, int, int, FrameSpec \| None) -> float` | General transmon transition frequency at fixed cavity Fock number |
| `sideband_transition_frequency(cavity_level=0, lower_level=0, upper_level=1, sideband="red", frame=None)` | `(int, int, int, str, FrameSpec \| None) -> float` | Rotating-frame frequency addressed by the effective sideband transition |
| `hamiltonian(frame=None)` | `(FrameSpec \| None) -> qt.Qobj` | Alias of `static_hamiltonian(frame)` |
| `energy_spectrum(frame=None, levels=None)` | `(FrameSpec \| None, int \| None) -> EnergySpectrum` | Vacuum-referenced eigenspectrum of the static Hamiltonian |
| `transmon_lowering()` / `transmon_raising()` / `transmon_number()` | `-> qt.Qobj` | Convenience accessors for the full-space transmon operators |
| `cavity_annihilation()` / `cavity_creation()` / `cavity_number()` | `-> qt.Qobj` | Convenience accessors for the storage/cavity mode |
| `as_universal_model()` | `-> UniversalCQEDModel` | Return the delegated universal model instance |

---

### 3.2 DispersiveReadoutTransmonStorageModel

**Module path:** `cqed_sim.core.readout_model.DispersiveReadoutTransmonStorageModel`

Three-mode (qubit + storage + readout) dispersive system. This class is a thin
compatibility wrapper around `UniversalCQEDModel`.

```python
@dataclass
class DispersiveReadoutTransmonStorageModel:
    omega_s: float              # Storage frequency (rad/s)
    omega_r: float              # Readout resonator frequency (rad/s)
    omega_q: float              # Qubit frequency (rad/s)
    alpha: float                # Anharmonicity (rad/s)
    chi_s: float = 0.0          # Storage–qubit dispersive shift (rad/s)
    chi_r: float = 0.0          # Readout–qubit dispersive shift (rad/s)
    chi_sr: float = 0.0         # Storage–readout cross-Kerr (rad/s)
    kerr_s: float = 0.0         # Storage self-Kerr (rad/s)
    kerr_r: float = 0.0         # Readout self-Kerr (rad/s)
    cross_kerr_terms: tuple[CrossKerrSpec, ...] = ()
    self_kerr_terms: tuple[SelfKerrSpec, ...] = ()
    exchange_terms: tuple[ExchangeSpec, ...] = ()
    n_storage: int = 12
    n_readout: int = 8
    n_tr: int = 3
    subsystem_labels: tuple[str, ...] = ("qubit", "storage", "readout")
```

**Static Hamiltonian:**

$$H_0/\hbar = \delta_s\, n_s + \delta_r\, n_r + \delta_q\, n_q + \frac{\alpha}{2}\, b^{\dagger 2}b^2 + \chi_s\, n_s\, n_q + \chi_r\, n_r\, n_q + \chi_{sr}\, n_s\, n_r + \frac{K_s}{2}\, n_s(n_s-1) + \frac{K_r}{2}\, n_r(n_r-1)$$

**Note:** positive `chi_sr` raises both bosonic mode frequencies with occupancy of the other mode, while positive `chi_s` and `chi_r` raise the qubit transition frequency with storage/readout occupancy.

#### Methods

| Method | Signature | Description |
|---|---|---|
| `operators()` | `-> dict[str, qt.Qobj]` | `b`, `bdag`, `a_s`, `adag_s`, `a_r`, `adag_r`, `n_q`, `n_s`, `n_r` |
| `drive_coupling_operators()` | `-> dict[str, tuple[qt.Qobj, qt.Qobj]]` | Keys: `"storage"`, `"cavity"`, `"qubit"`, `"transmon"`, `"readout"` |
| `transmon_level_projector(level)` | `(int) -> qt.Qobj` | Projector onto a transmon level in the full tensor-product space |
| `transmon_transition_operators(lower_level, upper_level)` | `(int, int) -> tuple[qt.Qobj, qt.Qobj]` | Full-space raising/lowering operators for a selected ancilla transition |
| `mode_operators(mode="storage")` | `(str) -> tuple[qt.Qobj, qt.Qobj]` | Retrieve `(annihilation, creation)` operators for `"storage"` or `"readout"` |
| `sideband_drive_operators(mode="storage", lower_level=0, upper_level=1, sideband="red")` | `(str, int, int, str) -> tuple[qt.Qobj, qt.Qobj]` | Effective multilevel sideband coupling operators on the selected bosonic mode |
| `static_hamiltonian(frame)` | `(FrameSpec \| None) -> qt.Qobj` | Full three-mode static Hamiltonian |
| `basis_state(q, ns, nr)` | `(int, int, int) -> qt.Qobj` | \|q⟩⊗\|n_s⟩⊗\|n_r⟩ |
| `basis_energy(q, ns, nr, frame)` | `(int, int, int, FrameSpec \| None) -> float` | Eigenvalue for given basis state |
| `qubit_transition_frequency(ns, nr, q, frame)` | `(...) -> float` | Qubit transition at given storage/readout occupation |
| `storage_transition_frequency(q, ns, nr, frame)` | `(...) -> float` | Storage transition frequency |
| `readout_transition_frequency(q, ns, nr, frame)` | `(...) -> float` | Readout transition frequency |
| `transmon_transition_frequency(storage_level=0, readout_level=0, lower_level=0, upper_level=1, frame=None)` | `(...) -> float` | General transmon transition frequency in the three-mode model |
| `sideband_transition_frequency(mode="storage", storage_level=0, readout_level=0, lower_level=0, upper_level=1, sideband="red", frame=None)` | `(...) -> float` | Rotating-frame frequency addressed by the effective sideband transition |
| `hamiltonian(frame=None)` | `(FrameSpec \| None) -> qt.Qobj` | Alias of `static_hamiltonian(frame)` |
| `energy_spectrum(frame=None, levels=None)` | `(FrameSpec \| None, int \| None) -> EnergySpectrum` | Vacuum-referenced eigenspectrum of the static Hamiltonian |
| `transmon_lowering()` / `transmon_raising()` / `transmon_number()` | `-> qt.Qobj` | Convenience accessors for the full-space transmon operators |
| `storage_annihilation()` / `storage_creation()` / `storage_number()` | `-> qt.Qobj` | Convenience accessors for the storage mode |
| `readout_annihilation()` / `readout_creation()` / `readout_number()` | `-> qt.Qobj` | Convenience accessors for the readout mode |
| `as_universal_model()` | `-> UniversalCQEDModel` | Return the delegated universal model instance |

---

### 3.3 FrameSpec

**Module path:** `cqed_sim.core.frame.FrameSpec`

```python
@dataclass(frozen=True)
class FrameSpec:
    omega_c_frame: float = 0.0   # Cavity/storage rotating-frame frequency (rad/s)
    omega_q_frame: float = 0.0   # Qubit rotating-frame frequency (rad/s)
    omega_r_frame: float = 0.0   # Readout rotating-frame frequency (rad/s)
```

| Property | Type | Description |
|---|---|---|
| `omega_s_frame` | `float` | Alias for `omega_c_frame` (three-mode compatibility) |

**Convention:** FrameSpec(0, 0, 0) is the lab frame (no rotation). Setting
frame frequencies equal to the model frequencies removes bare rotations.

---

### 3.4 Coupling Specifications

**Module path:** `cqed_sim.core.hamiltonian`

These frozen dataclasses specify additional coupling terms appended to the
static Hamiltonian via `additional_coupling_terms()`.

```python
@dataclass(frozen=True)
class CrossKerrSpec:
    left: str       # Mode name (e.g., "a", "b")
    right: str      # Mode name
    chi: float      # Coupling coefficient (rad/s)

@dataclass(frozen=True)
class SelfKerrSpec:
    mode: str       # Mode name
    kerr: float     # Kerr coefficient (rad/s)

@dataclass(frozen=True)
class ExchangeSpec:
    left: str                   # Mode name
    right: str                  # Mode name
    coupling: float | complex   # Coupling strength (rad/s)
```

**Helper functions:**

| Function | Signature | Description |
|---|---|---|
| `additional_coupling_terms(operators, *, cross_kerr_terms, self_kerr_terms, exchange_terms)` | `-> list[qt.Qobj]` | Convert specs to Hamiltonian operator terms |
| `assemble_static_hamiltonian(base, operators, *, ...)` | `-> qt.Qobj` | Add coupling terms to a base Hamiltonian |

Structured drive targets live in `cqed_sim.core.drive_targets`:

```python
@dataclass(frozen=True)
class TransmonTransitionDriveSpec:
    lower_level: int
    upper_level: int

@dataclass(frozen=True)
class SidebandDriveSpec:
    mode: str = "storage"
    lower_level: int = 0
    upper_level: int = 1
    sideband: str = "red"   # "red" or "blue"
```

These specs let `simulate_sequence(...)` and the pulse builders target an explicit multilevel ancilla transition or an effective sideband manifold without introducing ad hoc notebook-only operator wiring.

---

### 3.5 Basis Conventions

**Module path:** `cqed_sim.core.conventions`

**Two-mode tensor ordering:** qubit ⊗ cavity → |q, n⟩ = |q⟩⊗|n⟩

**Three-mode tensor ordering:** qubit ⊗ storage ⊗ readout → |q, n_s, n_r⟩

| Function | Signature | Description |
|---|---|---|
| `qubit_cavity_dims(n_qubit, n_cav)` | `-> list[list[int]]` | QuTiP dims for 2-mode: `[[n_q, n_c], [n_q, n_c]]` |
| `qubit_cavity_index(n_cav, qubit_level, cavity_level)` | `-> int` | Flat index = `q * n_cav + n` |
| `qubit_cavity_block_indices(n_cav, cavity_level)` | `-> tuple[int, int]` | `(index_g, index_e)` for Fock level n |
| `qubit_storage_readout_dims(n_q, n_s, n_r)` | `-> list[list[int]]` | QuTiP dims for 3-mode |
| `qubit_storage_readout_index(n_s, n_r, q, ns, nr)` | `-> int` | Flat index for 3-mode |
| `qubit_storage_readout_block_indices(n_s, n_r, ns, nr)` | `-> tuple[int, int]` | `(index_g, index_e)` for 3-mode |

**Computational basis:** |g⟩ = |0⟩, |e⟩ = |1⟩. QuTiP σ_z signs are unmodified:
σ_z|g⟩ = +|g⟩, σ_z|e⟩ = −|e⟩.

---

### 3.6 Frequency Helpers

**Module path:** `cqed_sim.core.frequencies`

| Function | Signature | Description |
|---|---|---|
| `manifold_transition_frequency(model, n, frame)` | `(DispersiveTransmonCavityModel, int, FrameSpec \| None) -> float` | |g,n⟩↔|e,n⟩ transition frequency including all chi_higher terms |
| `transmon_transition_frequency(model, ..., frame)` | `(model, ..., FrameSpec \| None) -> float` | General multilevel transmon transition frequency helper for two- and three-mode models |
| `sideband_transition_frequency(model, ..., frame)` | `(model, ..., FrameSpec \| None) -> float` | Transition frequency for the effective sideband manifold |
| `effective_sideband_rabi_frequency(coupling, detuning)` | `(float, float) -> float` | `sqrt((2g)^2 + delta^2)` for the effective detuned sideband model |
| `falling_factorial_scalar(n, order)` | `(int, int) -> float` | n(n−1)(n−2)···(n−order+1) |
| `carrier_for_transition_frequency(transition_frequency)` | `(float) -> float` | Convert a rotating-frame transition frequency into the resonant `Pulse.carrier` value |
| `transition_frequency_from_carrier(carrier)` | `(float) -> float` | Convert a `Pulse.carrier` back into the rotating-frame transition frequency it addresses |

---

### 3.7 Ideal Gate Operators

**Module path:** `cqed_sim.core.ideal_gates`

All functions return QuTiP `Qobj` unitaries.

| Function | Signature | Description |
|---|---|---|
| `qubit_rotation_xy(theta, phi)` | `(float, float) -> qt.Qobj` | exp(−iθ/2 [cos φ σ_x + sin φ σ_y]). 2×2 unitary. |
| `qubit_rotation_axis(theta, axis)` | `(float, str) -> qt.Qobj` | Rotation around `"x"`, `"y"`, or `"z"` axis. |
| `displacement_op(n_cav, alpha)` | `(int, complex) -> qt.Qobj` | D(α) = exp(α a† − α* a). n_cav × n_cav. |
| `cavity_block_phase_op(phases, *, fock_levels=None, cavity_dim=None)` | `(array, ...) -> qt.Qobj` | Cavity-only diagonal phase layer. Omitted levels remain identity. |
| `logical_block_phase_op(phases, *, fock_levels=None, cavity_dim=None, qubit_dim=2)` | `(array, ...) -> qt.Qobj` | Qubit-first embedding `I_q ⊗ cavity_block_phase_op(...)`. |
| `snap_op(phases)` | `(array) -> qt.Qobj` | Contiguous cavity block-phase gate diag(e^{iφ_0}, e^{iφ_1}, ...). Dimension = len(phases). |
| `sqr_op(thetas, phis)` | `(array, array) -> qt.Qobj` | Σ_n \|n⟩⟨n\| ⊗ R(θ_n, φ_n). Two-mode unitary. |
| `embed_qubit_op(op_q, n_cav)` | `(qt.Qobj, int) -> qt.Qobj` | op_q ⊗ I_cav |
| `embed_cavity_op(op_c, n_tr)` | `(qt.Qobj, int) -> qt.Qobj` | I_qubit ⊗ op_c. Default `n_tr=2`. |
| `beamsplitter_unitary(n_a, n_b, theta)` | `(int, int, float) -> qt.Qobj` | exp[−iθ(a b† + a† b)]. Full two-mode unitary. |

### 3.8 Energy Spectrum

**Module path:** `cqed_sim.core.spectrum`

```python
@dataclass(frozen=True)
class EnergyLevel:
    index: int
    energy: float
    raw_energy: float
    eigenstate: qt.Qobj
    dominant_basis_levels: tuple[int, ...]
    dominant_basis_label: str
    dominant_basis_overlap: float

@dataclass(frozen=True)
class EnergySpectrum:
    hamiltonian: qt.Qobj
    frame: FrameSpec
    levels: tuple[EnergyLevel, ...]
    vacuum_energy: float
    vacuum_basis_levels: tuple[int, ...]
    vacuum_basis_label: str
    vacuum_level_index: int | None
    vacuum_level_overlap: float
    vacuum_residual_norm: float
    subsystem_labels: tuple[str, ...]
    subsystem_dims: tuple[int, ...]
    basis_levels: tuple[tuple[int, ...], ...]
    basis_labels: tuple[str, ...]
```

```python
def compute_energy_spectrum(
    model,
    *,
    frame: FrameSpec | None = None,
    levels: int | None = None,
    sort: str = "low",
) -> EnergySpectrum
```

The helper diagonalizes the model's static Hamiltonian with QuTiP's native eigensolver.
Reported `EnergySpectrum.energies` are always shifted so the bare vacuum basis state has
energy `0`, while `raw_energy` retains the unshifted eigenvalue in the selected frame.

For ladder-style plots or physically intuitive absolute level spacings, the lab frame
`FrameSpec()` is usually the clearest choice. In rotating frames, vacuum-referenced
energies can still be negative because only the zero of energy is shifted.

| Symbol | Description |
|---|---|
| `EnergySpectrum.energies` | Vacuum-referenced eigenenergies as a NumPy array |
| `EnergySpectrum.raw_energies` | Unshifted eigenenergies as a NumPy array |
| `EnergySpectrum.eigenstates` | Tuple of QuTiP eigenkets |
| `EnergySpectrum.find_level(label)` | Retrieve a level by its dominant bare-basis label |
| `EnergySpectrum.level_rows(max_levels=None)` | Compact list-of-dicts summary for tables or notebooks |

---

### 3.9 Gate Library (`cqed_sim.gates`)

**Module path:** `cqed_sim.gates`

A structured library of ideal unitary gates organised by subsystem type.
All functions return QuTiP `qt.Qobj` unitaries. All angles are in **radians**;
couplings and frequencies are in **rad/s**; times are in **seconds**.

**Subsystem-ordering conventions** follow the package-wide rule:
**qubit/transmon first, cavity second** for qubit-cavity composite gates.
Two-qubit gates use **control first, target second**.

**Submodules:**

| Submodule | Content |
|---|---|
| `cqed_sim.gates.qubit` | Single-qubit named and parameterised gates |
| `cqed_sim.gates.transmon` | Multilevel transition-selective rotations |
| `cqed_sim.gates.bosonic` | Bosonic cavity gates |
| `cqed_sim.gates.coupled` | Qubit-cavity conditional and interaction gates |
| `cqed_sim.gates.two_qubit` | Two-qubit gates |

All public names are re-exported at the top level of `cqed_sim`.

---

#### 3.9.1 Single-Qubit Gates (`cqed_sim.gates.qubit`)

Basis: `|g⟩ = basis(2,0)`, `|e⟩ = basis(2,1)`.
Convention: `R_n(θ) = exp(−iθ/2 n̂·σ)`.

| Function | Signature | Description |
|---|---|---|
| `rx(theta)` | `(float) -> qt.Qobj` | `exp(−iθ/2 X)` |
| `ry(theta)` | `(float) -> qt.Qobj` | `exp(−iθ/2 Y)` |
| `rz(theta)` | `(float) -> qt.Qobj` | `exp(−iθ/2 Z)` |
| `rphi(theta, phi)` | `(float, float) -> qt.Qobj` | `exp[−iθ/2 (cos φ X + sin φ Y)]`. At `phi=0` = `rx`; at `phi=π/2` = `ry`. |
| `identity_gate()` | `() -> qt.Qobj` | 2×2 identity |
| `x_gate()` | `() -> qt.Qobj` | Pauli X (NOT) |
| `y_gate()` | `() -> qt.Qobj` | Pauli Y |
| `z_gate()` | `() -> qt.Qobj` | Pauli Z |
| `h_gate()` | `() -> qt.Qobj` | Hadamard: `(X+Z)/√2` |
| `s_gate()` | `() -> qt.Qobj` | Phase gate: `diag(1, i)`. Satisfies `S²=Z`. |
| `s_dag_gate()` | `() -> qt.Qobj` | `S†`: `diag(1, −i)` |
| `t_gate()` | `() -> qt.Qobj` | T gate: `diag(1, exp(iπ/4))`. Satisfies `T²=S`. |
| `t_dag_gate()` | `() -> qt.Qobj` | `T†`: `diag(1, exp(−iπ/4))` |

---

#### 3.9.2 Multilevel Transmon Gates (`cqed_sim.gates.transmon`)

Acts on a `dim`-dimensional transmon Hilbert space. Levels are 0-based:
`|g⟩=|0⟩`, `|e⟩=|1⟩`, `|f⟩=|2⟩`, …

| Function | Signature | Description |
|---|---|---|
| `transition_rotation(dim, level_a, level_b, theta, phi=0.0)` | `(int, int, int, float, float) -> qt.Qobj` | `exp[−iθ/2 (e^{−iφ}\|a⟩⟨b\| + e^{iφ}\|b⟩⟨a\|)]`. Identity on all other levels. |
| `r_ge(theta, phi=0.0, dim=3)` | `(float, float, int) -> qt.Qobj` | g↔e selective rotation. At `dim=2` matches `rphi`. |
| `r_ef(theta, phi=0.0, dim=3)` | `(float, float, int) -> qt.Qobj` | e↔f selective rotation. Requires `dim≥3`. |

**Notes:**

- Non-adjacent level pairs `(level_a, level_b)` are supported mathematically.
- The physical realisation of non-adjacent transitions requires a resonant drive mechanism.

---

#### 3.9.3 Bosonic Cavity Gates (`cqed_sim.gates.bosonic`)

All operate on a single bosonic mode with Fock-space dimension `dim`.

| Function | Signature | Description |
|---|---|---|
| `displacement(alpha, dim)` | `(complex, int) -> qt.Qobj` | `D(α) = exp(α a† − α* a)`. |
| `oscillator_rotation(theta, dim)` | `(float, int) -> qt.Qobj` | `R(θ) = exp(−iθ a†a)`. Acts as `\|n⟩ → e^{−inθ}\|n⟩`. |
| `parity(dim)` | `(int) -> qt.Qobj` | `Π = exp(iπ a†a)`. Acts as `\|n⟩ → (−1)^n\|n⟩`. Equal to `oscillator_rotation(−π, dim)`. |
| `squeeze(zeta, dim)` | `(complex, int) -> qt.Qobj` | `S(ζ) = exp(½ ζ* a² − ½ ζ (a†)²)`. |
| `kerr_evolution(kerr, time, dim)` | `(float, float, int) -> qt.Qobj` | `U_K(t) = exp[−iK/2 t n̂(n̂−1)]`. Diagonal in the Fock basis. **Different from** `self_kerr()` in `sim.couplings`, which returns the Hamiltonian term. |
| `snap(phases, dim)` | `(dict or array, int) -> qt.Qobj` | `Σ_n e^{iφ_n}\|n⟩⟨n\|`. Accepts sparse `{n: phase}` dict or dense array. Unspecified levels receive zero phase. |

**`snap` input formats:**

```python
# Sparse dict: only specified Fock levels receive a phase
U = snap({0: 0.0, 1: np.pi/2, 2: np.pi}, dim=10)

# Dense array: entry k is the phase for Fock level k
U = snap(np.linspace(0, 2*np.pi, 10), dim=10)
```

---

#### 3.9.4 Qubit-Cavity Conditional and Interaction Gates (`cqed_sim.gates.coupled`)

Composite Hilbert space with **qubit first, cavity second**.
Basis state ordering: `|g,0⟩, |g,1⟩, …, |g,N−1⟩, |e,0⟩, …, |e,N−1⟩`.

##### Conditional gates

| Function | Signature | Description |
|---|---|---|
| `dispersive_phase(chi, time, cavity_dim, qubit_dim=2, convention="n_e")` | `(float, float, int, int, str) -> qt.Qobj` | Dispersive-shift unitary. See conventions below. |
| `conditional_rotation(theta, cavity_dim, qubit_dim=2, control_state="e")` | `(float, int, int, str) -> qt.Qobj` | `\|g⟩⟨g\| ⊗ I + \|e⟩⟨e\| ⊗ exp(−iθn̂)` (for `control_state="e"`). |
| `conditional_displacement(alpha=None, *, alpha_g=None, alpha_e=None, cavity_dim, qubit_dim=2)` | see below | Conditional displacement; see forms below. |
| `controlled_parity(cavity_dim, qubit_dim=2)` | `(int, int) -> qt.Qobj` | `\|g⟩⟨g\| ⊗ I + \|e⟩⟨e\| ⊗ Π` |
| `controlled_snap(phases, cavity_dim, qubit_dim=2)` | `(dict or array, int, int) -> qt.Qobj` | `\|g⟩⟨g\| ⊗ I + \|e⟩⟨e\| ⊗ SNAP({φ_n})` |

**`dispersive_phase` conventions:**

```text
convention="n_e"  (default, matches UniversalCQEDModel):
    H = χ n̂_cav |e⟩⟨e|
    U = |g⟩⟨g| ⊗ I  +  |e⟩⟨e| ⊗ exp(−iχt n̂)

convention="z"  (Pauli-Z style):
    H = (χ/2) n̂_cav Z
    U = exp(−iχt/2 n̂) ⊗ |g⟩⟨g|  +  exp(+iχt/2 n̂) ⊗ |e⟩⟨e|
```

**`conditional_displacement` call forms:**

```python
# Symmetric: |g⟩ branch → D(+α), |e⟩ branch → D(−α)
U = conditional_displacement(alpha=0.5+0.1j, cavity_dim=20)

# General: independent displacements per branch
U = conditional_displacement(alpha_g=0.4, alpha_e=-0.3j, cavity_dim=20)
```

##### Number-selective qubit rotation (SQR)

Tensor ordering for `sqr` and `multi_sqr` is **qubit first, cavity second**,
consistent with all other qubit-cavity gates in the package.

| Function | Signature | Description |
|---|---|---|
| `sqr(theta, phi, n, cavity_dim, qubit_dim=2)` | `(float, float, int, int, int) -> qt.Qobj` | `R_φ(θ) ⊗ \|n⟩⟨n\| + I ⊗ Σ_{m≠n} \|m⟩⟨m\|`. Single-Fock-level SQR. |
| `multi_sqr(theta_by_n, phi_by_n, cavity_dim, qubit_dim=2)` | `(dict or array, dict or array, int, int) -> qt.Qobj` | `Σ_n R_{φ_n}(θ_n) ⊗ \|n⟩⟨n\|`. Dict `{n: angle}` or dense array. |

Note: the existing `sqr_op(thetas, phis)` in `cqed_sim.core.ideal_gates` also uses
**qubit-first** ordering and provides a dense array interface.

##### Interaction gates (Jaynes–Cummings, blue sideband, beam splitter)

Composite space with **qubit first, cavity second** (for JC and blue sideband);
**mode a first, mode b second** (for beam splitter).

| Function | Signature | Description |
|---|---|---|
| `jaynes_cummings(g, time, cavity_dim, qubit_dim=2)` | `(float, float, int, int) -> qt.Qobj` | `exp[−it g (σ+ ⊗ a + σ− ⊗ a†)]`. Red-sideband exchange. |
| `blue_sideband(g, time, cavity_dim, qubit_dim=2)` | `(float, float, int, int) -> qt.Qobj` | `exp[−it g (σ+ ⊗ a† + σ− ⊗ a)]`. Blue-sideband / two-photon exchange. |
| `beam_splitter(g, time, dim_a, dim_b)` | `(float, float, int, int) -> qt.Qobj` | `exp[−it g (a†b + ab†)]`. Two-mode beamsplitter. Equivalent to `beamsplitter_unitary(dim_a, dim_b, g*time)`. |

---

#### 3.9.5 Two-Qubit Gates (`cqed_sim.gates.two_qubit`)

Standard 4×4 unitaries on a two-qubit Hilbert space. Tensor ordering:
**control qubit first, target qubit second**.
Basis order: `|gg⟩, |ge⟩, |eg⟩, |ee⟩`.

| Function | Signature | Description |
|---|---|---|
| `cnot_gate()` | `() -> qt.Qobj` | CNOT: `\|g⟩⟨g\| ⊗ I + \|e⟩⟨e\| ⊗ X` |
| `cz_gate()` | `() -> qt.Qobj` | CZ: `diag(1, 1, 1, −1)` |
| `controlled_phase(phi)` | `(float) -> qt.Qobj` | CP(φ): `diag(1, 1, 1, e^{iφ})`. At `phi=π` equals `cz_gate()`. |
| `swap_gate()` | `() -> qt.Qobj` | SWAP: exchanges the two qubit states |
| `iswap_gate()` | `() -> qt.Qobj` | iSWAP: `diag(1,0,0,1)` on outer states; `i` factors on `\|ge⟩↔\|eg⟩` |
| `sqrt_iswap_gate()` | `() -> qt.Qobj` | √iSWAP. Satisfies `√iSWAP · √iSWAP = iSWAP`. |

**Matrix forms:**

```text
CZ  = diag(1, 1, 1, −1)
CP(φ) = diag(1, 1, 1, e^{iφ})

SWAP = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]

iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]

√iSWAP = [[1, 0,      0,      0],
           [0, 1/√2,  i/√2,   0],
           [0, i/√2,  1/√2,   0],
           [0, 0,      0,      1]]
```

---

#### 3.9.6 Usage Examples

```python
import numpy as np
import cqed_sim as cs

# --- single-qubit ---
Rx_pi = cs.rx(np.pi)                      # π rotation about X
Rph   = cs.rphi(np.pi/2, phi=np.pi/4)    # equatorial rotation

# --- transmon ---
U_ge  = cs.r_ge(np.pi, phi=0.0, dim=3)   # π-pulse on g-e transition
U_ef  = cs.r_ef(np.pi/2, dim=3)          # π/2-pulse on e-f transition

# --- bosonic ---
D    = cs.displacement(0.5 + 0.1j, dim=20)
R    = cs.oscillator_rotation(np.pi/4, dim=20)
P    = cs.parity(dim=20)
Sq   = cs.squeeze(0.3, dim=20)
UK   = cs.kerr_evolution(kerr=-2e6, time=5e-6, dim=20)

# sparse dict SNAP
U_snap = cs.snap({0: 0.0, 1: np.pi/2, 2: np.pi}, dim=10)
# dense array SNAP
U_snap = cs.snap(np.linspace(0, 2*np.pi, 10), dim=10)

# --- qubit-cavity conditional ---
# dispersive phase (default n_e convention)
U_chi  = cs.dispersive_phase(chi=-2e6, time=np.pi/2e6, cavity_dim=30)

# conditional displacement — symmetric and general forms
CD_sym = cs.conditional_displacement(alpha=1.0, cavity_dim=30)
CD_gen = cs.conditional_displacement(alpha_g=0.5, alpha_e=-0.3j, cavity_dim=30)

# controlled parity (for Wigner function readout)
U_cp   = cs.controlled_parity(cavity_dim=30)

# SQR: rotate qubit only when cavity is in Fock |3⟩
U_sqr  = cs.sqr(theta=np.pi, phi=0.0, n=3, cavity_dim=12)

# multi-SQR via dict
U_msqr = cs.multi_sqr(
    theta_by_n={1: np.pi/2, 3: np.pi},
    phi_by_n={1: 0.0, 3: np.pi/4},
    cavity_dim=12,
)

# Jaynes–Cummings half swap
g_vac = 20e6   # 20 MHz vacuum Rabi rate (rad/s)
t_half = np.pi / (2 * g_vac)
U_jc   = cs.jaynes_cummings(g=g_vac, time=t_half, cavity_dim=10)

# beam splitter between two modes
U_bs   = cs.beam_splitter(g=10e6, time=np.pi/(2*10e6), dim_a=8, dim_b=8)

# --- two-qubit ---
CZ    = cs.cz_gate()
CNOT  = cs.cnot_gate()
iSWAP = cs.iswap_gate()
```

---

## 4. Pulse System

**Module:** `cqed_sim.pulses`

### 4.1 Pulse

**Module path:** `cqed_sim.pulses.pulse.Pulse`

```python
@dataclass(frozen=True)
class Pulse:
    channel: str                          # Drive channel: "qubit", "storage", etc.
    t0: float                             # Start time (s)
    duration: float                       # Duration (s)
    envelope: Callable[[ndarray], ndarray] | ndarray  # Analytic or pre-sampled
    carrier: float = 0.0                  # Carrier frequency (rad/s)
    phase: float = 0.0                    # Phase offset (rad)
    amp: float = 1.0                      # Amplitude scaling
    drag: float = 0.0                     # DRAG coefficient
    sample_rate: float | None = None      # For discrete envelopes (Hz)
    label: str | None = None              # Optional identifier
```

| Property / Method | Returns | Description |
|---|---|---|
| `t1` | `float` | End time: `t0 + duration` |
| `sample(t)` | `ndarray` | Sample pulse at arbitrary time points. Routes to analytic or discrete path. |

**Waveform formula:**

$$\epsilon(t) = \text{amp} \cdot \text{envelope}(t_\text{rel}) \cdot e^{i(\text{carrier} \cdot t + \text{phase})}$$

where t_rel = (t − t0) / duration. The **exp(+i·ω·t)** sign convention is used
throughout the repository.
Because the drive Hamiltonian is assembled as `epsilon(t) * raising + epsilon*(t) * lowering`,
the resonant rotating-frame transition frequency is `-carrier`. Use
`carrier_for_transition_frequency(...)` when you want the user-facing detuning axis to match
the physical transition frequency in the chosen frame.

If `drag ≠ 0`, a quadrature correction is added: the envelope derivative scaled by
`drag` is added in the imaginary channel.

---

### 4.2 Envelopes

**Module path:** `cqed_sim.pulses.envelopes`

| Function | Signature | Description |
|---|---|---|
| `square_envelope(t_rel)` | `ndarray -> ndarray` | Constant 1.0 |
| `gaussian_envelope(t_rel, sigma, center=0.5)` | `-> ndarray` | exp(−(t−c)²/(2σ²)) |
| `cosine_rise_envelope(t_rel, rise_fraction=0.1)` | `-> ndarray` | Flat-top with cosine edges |
| `normalized_gaussian(t_rel, sigma_fraction)` | `-> ndarray` | Gaussian with unit area |
| `gaussian_area_fraction(sigma_fraction, n_pts=4097)` | `-> float` | Numerical area of Gaussian |
| `multitone_gaussian_envelope(t_rel, duration_s, sigma_fraction, tone_specs)` | `-> ndarray` | Multi-tone Gaussian modulated envelope |

#### MultitoneTone

```python
@dataclass(frozen=True)
class MultitoneTone:
    manifold: int          # Fock level n
    omega_rad_s: float     # Tone frequency (rad/s)
    amp_rad_s: float       # Tone amplitude (rad/s)
    phase_rad: float       # Tone phase (rad)
```

The multitone envelope formula is:

$$w(t) = \text{env}(t_\text{rel}) \cdot \sum_n a_n \, e^{i(\phi_n + \omega_n \cdot t)}$$

---

### 4.3 Pulse Builders

**Module path:** `cqed_sim.pulses.builders`

All builders return `(pulse list, drive-operator mapping, metadata)`. The drive mapping is `dict[str, str]` for the original displacement/rotation/SQR builders and `dict[str, SidebandDriveSpec]` for `build_sideband_pulse(...)`.

#### `build_displacement_pulse`

```python
def build_displacement_pulse(
    gate: DisplacementGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]
```

**Config keys:** `"duration_displacement_s"` (float, seconds).

Creates a square-envelope pulse on channel `"storage"`. Drive mapping: `{"storage": "cavity"}`.

Amplitude: ε = iα / T (rotating-frame calibration).

#### `build_rotation_pulse`

```python
def build_rotation_pulse(
    gate: RotationGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]
```

**Config keys:** `"duration_rotation_s"`, `"rotation_sigma_fraction"`.

Creates a normalized-Gaussian pulse on channel `"q"`. Drive mapping: `{"q": "qubit"}`.

Amplitude: Ω = θ / (2T) under the RWA.

#### `build_sideband_pulse`

```python
def build_sideband_pulse(
    target: SidebandDriveSpec,
    *,
    duration_s: float,
    amplitude_rad_s: float,
    channel: str = "sideband",
    carrier: float = 0.0,
    phase: float = 0.0,
    sigma_fraction: float | None = None,
    label: str | None = None,
) -> tuple[list[Pulse], dict[str, SidebandDriveSpec], dict[str, Any]]
```

Builds an effective multilevel sideband pulse with either a square envelope or a normalized Gaussian envelope (`sigma_fraction`). The target specifies the ancilla transition (`lower_level`, `upper_level`), the bosonic mode, and whether the coupling is the red or blue sideband. The implementation is an effective rotating-wave Hamiltonian, not a microscopic flux-drive derivation.

#### `build_sqr_multitone_pulse`

```python
def build_sqr_multitone_pulse(
    gate: SQRGate,
    model: DispersiveTransmonCavityModel,
    config: Mapping[str, Any],
    *,
    frame: FrameSpec | None = None,
    calibration: SQRCalibrationResult | None = None,
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]
```

**Config keys:** `"duration_sqr_s"`, `"sqr_sigma_fraction"`, `"sqr_theta_cutoff"`,
optionally `"use_rotating_frame"`, `"fock_fqs_hz"`.

Creates a multitone Gaussian pulse on channel `"q"`. If `calibration` is provided,
per-manifold corrections (d_lambda, d_alpha, d_omega) are applied.

---

### 4.4 Calibration Formulas

**Module path:** `cqed_sim.pulses.calibration`

| Function | Signature | Description |
|---|---|---|
| `displacement_square_amplitude(alpha, duration_s)` | `(complex, float) -> complex` | ε = iα / T |
| `rotation_gaussian_amplitude(theta, duration_s)` | `(float, float) -> float` | Ω = θ / (2T) |
| `sqr_lambda0_rad_s(duration_s)` | `(float) -> float` | λ₀ = π / (2T) |
| `sqr_rotation_coefficient(theta, d_lambda_norm)` | `(float, float) -> float` | s = θ/π + d_lambda_norm |
| `sqr_tone_amplitude_rad_s(theta, duration_s, d_lambda_norm)` | `(float, float, float) -> float` | a = λ₀·s = θ/(2T) + λ₀·d_lambda_norm |
| `pad_parameter_array(values, n_cav)` | `-> ndarray` | Pad/truncate to n_cav |
| `pad_sqr_angles(thetas, phis, n_cav)` | `-> tuple[ndarray, ndarray]` | Pad both arrays |
| `build_sqr_tone_specs(model, frame, thetas, phis, duration_s, ...)` | `-> list[MultitoneTone]` | Build tone specs with frequencies and amplitudes for each active manifold |

**`build_sqr_tone_specs` additional parameters:**
- `d_lambda_values: list[float] | None` — per-manifold corrections
- `fock_fqs_hz: list[float] | None` — empirical Fock-level frequencies
- `include_all_levels: bool = False` — include zero-amplitude tones
- `tone_cutoff: float = 1e-10` — amplitude threshold

**SQR frequency convention:** tone frequencies are the **negative** of the manifold
transition frequency in the rotating frame, aligning with the exp(+iωt) waveform
convention. The implementation uses `carrier_for_transition_frequency(...)` for this mapping.

---

### 4.5 HardwareConfig

**Module path:** `cqed_sim.pulses.hardware.HardwareConfig`

Models IQ distortion, quantization, and filtering for realistic waveform generation.

```python
@dataclass(frozen=True)
class HardwareConfig:
    lo_freq: float = 0.0             # LO frequency (rad/s)
    if_freq: float = 0.0             # IF frequency (rad/s)
    gain_i: float = 1.0              # I-channel gain
    gain_q: float = 1.0              # Q-channel gain
    quadrature_skew: float = 0.0     # IQ phase skew (rad)
    dc_i: float = 0.0                # DC offset on I
    dc_q: float = 0.0                # DC offset on Q
    image_leakage: float = 0.0       # Image sideband leakage
    channel_gain: float = 1.0        # Overall channel gain
    zoh_samples: int = 1             # Zero-order hold samples
    lowpass_bw: float | None = None  # Lowpass bandwidth (Hz)
    detuning: float = 0.0            # Extra frequency detuning (rad/s)
    timing_quantum: float | None = None  # Timing resolution (s)
    amplitude_bits: int | None = None    # DAC bit depth
```

**Hardware processing helpers:**

| Function | Description |
|---|---|
| `apply_timing_quantization(t0, quantum)` | Round time to nearest quantum |
| `apply_zoh(x, zoh_samples)` | Zero-order hold interpolation |
| `apply_first_order_lowpass(x, dt, bw)` | First-order IIR lowpass filter |
| `apply_amplitude_quantization(x, bits)` | DAC quantization of I/Q |
| `apply_iq_distortion(baseband, t, hw)` | Full IQ distortion chain → (distorted, RF) |
| `image_ratio_db(gain_i, gain_q)` | Image suppression in dB |

---

## 5. Sequence Compilation

**Module:** `cqed_sim.sequence`

### 5.1 SequenceCompiler

**Module path:** `cqed_sim.sequence.scheduler.SequenceCompiler`

```python
class SequenceCompiler:
    def __init__(
        self,
        dt: float,
        hardware: dict[str, HardwareConfig] | None = None,
        crosstalk_matrix: dict[str, dict[str, float]] | None = None,
        enable_cache: bool = False,
    )
```

| Parameter | Type | Description |
|---|---|---|
| `dt` | `float` | Global sample time step (seconds) |
| `hardware` | `dict[str, HardwareConfig] \| None` | Per-channel hardware configs |
| `crosstalk_matrix` | `dict[str, dict[str, float]] \| None` | source → {dest: coefficient} |
| `enable_cache` | `bool` | Memoize compiled sequences |

#### `compile`

```python
def compile(self, pulses: list[Pulse], t_end: float | None = None) -> CompiledSequence
```

**Processing pipeline:**

1. Build uniform time grid from 0 to max pulse end (or explicit `t_end`).
2. Per-pulse: apply timing quantization, carrier IF offset, sample and accumulate to baseband.
3. Apply crosstalk mixing between channels if configured.
4. Per-channel hardware processing: ZOH → lowpass → amplitude quantization → IQ distortion.
5. Return `CompiledSequence` with per-channel baseband, distorted, and RF waveforms.

---

### 5.2 CompiledSequence / CompiledChannel

```python
@dataclass
class CompiledChannel:
    baseband: np.ndarray    # Complex baseband after signal processing
    distorted: np.ndarray   # Complex baseband after IQ distortion
    rf: np.ndarray          # Real RF waveform (upconverted)

@dataclass
class CompiledSequence:
    tlist: np.ndarray                    # Time grid (starts at 0.0)
    dt: float                            # Sample step (s)
    channels: dict[str, CompiledChannel] # Per-channel compiled waveforms
```

---

## 6. Simulation Engine

**Module:** `cqed_sim.sim`

### 6.1 simulate_sequence

**Module path:** `cqed_sim.sim.runner.simulate_sequence`

```python
def simulate_sequence(
    model,
    compiled: CompiledSequence,
    initial_state: qt.Qobj,
    drive_ops: dict[str, str | TransmonTransitionDriveSpec | SidebandDriveSpec],
    config: SimulationConfig | None = None,
    c_ops: Sequence[qt.Qobj] | None = None,
    noise: NoiseSpec | None = None,
    e_ops: dict[str, qt.Qobj] | None = None,
) -> SimulationResult
```

**High-level entry point.** Creates a `SimulationSession` and runs a single trajectory.

| Parameter | Type | Description |
|---|---|---|
| `model` | any model | Must have `operators()`, `subsystem_dims`, `static_hamiltonian()`, `drive_coupling_operators()` |
| `compiled` | `CompiledSequence` | Timeline from `SequenceCompiler.compile()` |
| `initial_state` | `qt.Qobj` | Initial ket or density matrix |
| `drive_ops` | `dict[str, str \| TransmonTransitionDriveSpec \| SidebandDriveSpec]` | Maps pulse channel names to string targets or structured multilevel targets |
| `config` | `SimulationConfig \| None` | Solver configuration |
| `c_ops` | `Sequence[qt.Qobj] \| None` | Additional collapse operators |
| `noise` | `NoiseSpec \| None` | Lindblad noise specification |
| `e_ops` | `dict[str, qt.Qobj] \| None` | Custom observables; defaults to `default_observables(model)` |

**Solver selection:** If `config.backend` is set, uses the dense piecewise-constant
solver. Otherwise, uses QuTiP's `sesolve` (pure state) or `mesolve` (density matrix /
open system).

---

### 6.2 SimulationConfig

```python
@dataclass(frozen=True)
class SimulationConfig:
    frame: FrameSpec = FrameSpec()
    atol: float = 1e-8
    rtol: float = 1e-7
    max_step: float | None = None
    store_states: bool = False
    backend: BaseBackend | None = None   # None = use QuTiP path
```

---

### 6.3 SimulationResult

```python
@dataclass
class SimulationResult:
    final_state: qt.Qobj                     # State at end of simulation
    states: list[qt.Qobj] | None             # Trajectory if store_states=True
    expectations: dict[str, np.ndarray]       # Time series per observable name
    solver_result: Any                        # Raw QuTiP solver result
```

Also importable directly as `cqed_sim.SimulationResult`.

> **Disambiguation:** The unitary-synthesis package has a separate
> `cqed_sim.unitary_synthesis.SimulationResult` with different fields;
> see [§17.8](#178-results-and-metrics).

---

### 6.4 SimulationSession and Prepared Simulation

For high-throughput workloads (parameter sweeps, multiple initial states), prepare
the session once and reuse it.

```python
def prepare_simulation(
    model, compiled, drive_ops, *,
    config=None, c_ops=None, noise=None, e_ops=None
) -> SimulationSession
```

```python
@dataclass
class SimulationSession:
    model: Any
    compiled: CompiledSequence
    drive_ops: dict[str, str]
    config: SimulationConfig
    c_ops: Sequence[qt.Qobj] | None
    noise: NoiseSpec | None
    e_ops: dict[str, qt.Qobj] | None
    # Computed in __post_init__:
    hamiltonian: list          # Time-dependent Hamiltonian
    effective_c_ops: tuple     # Combined collapse operators
    observables: dict          # Expanded observable dict
    solver_options: dict       # QuTiP solver options
```

| Method | Signature | Description |
|---|---|---|
| `run(initial_state)` | `(qt.Qobj) -> SimulationResult` | Single-trajectory simulation |
| `run_many(initial_states, *, max_workers=1)` | `(Iterable, ...) -> list[SimulationResult]` | Parallel execution via ProcessPoolExecutor |

```python
def simulate_batch(
    session: SimulationSession,
    initial_states: Iterable[qt.Qobj],
    *,
    max_workers: int = 1,
    mp_context: str = "spawn",
) -> list[SimulationResult]
```

**Supporting functions:**

| Function | Signature | Description |
|---|---|---|
| `default_observables(model)` | `-> dict[str, qt.Qobj]` | P_e projector, mode quadratures & photon numbers |
| `hamiltonian_time_slices(model, compiled, drive_ops, frame)` | `-> list` | `[H_0, [O_1⁺, coeff_1], [O_1⁻, conj_1], ...]` QuTiP format |

---

### 6.5 NoiseSpec

**Module path:** `cqed_sim.sim.noise.NoiseSpec`

```python
@dataclass(frozen=True)
class NoiseSpec:
    t1: float | None = None              # T₁ relaxation (s)
    transmon_t1: tuple[float | None, ...] | None = None  # Explicit ladder T1 values: (T1_ge, T1_fe, ...)
    tphi: float | None = None            # T_φ dephasing (s)
    tphi_storage: float | None = None    # Storage pure dephasing time (s)
    tphi_readout: float | None = None    # Readout pure dephasing time (s)
    kappa: float | None = None           # Cavity decay rate (1/s)
    nth: float = 0.0                     # Cavity thermal occupation
    kappa_storage: float | None = None   # Storage decay (defaults to kappa)
    kappa_readout: float | None = None   # Readout decay (1/s)
    nth_storage: float | None = None     # Storage thermal occupation
    nth_readout: float | None = None     # Readout thermal occupation
```

| Property | Returns | Description |
|---|---|---|
| `gamma1` | `float` | 1/t1 if set, else 0.0 |
| `gamma_phi` | `float` | 1/(2·tphi) if set, else 0.0 |
| `gamma_phi_storage` | `float` | 1/tphi_storage if set, else 0.0 |
| `gamma_phi_readout` | `float` | 1/tphi_readout if set, else 0.0 |

> **Dephasing-rate convention:** The qubit dephasing rate uses γ_φ = 1/(2·T_φ)
> because the jump operator is σ_z, whose square contributes a factor of 2 in the
> Lindblad master equation. Bosonic modes (storage, readout) use γ_φ = 1/T_φ
> because their jump operator is n̂, where the factor of 2 does not arise.
> Both conventions are physically correct; see the `NoiseSpec` class docstring
> in `cqed_sim/sim/noise.py` for a detailed derivation.

```python
def collapse_operators(model, noise: NoiseSpec | None) -> list[qt.Qobj]
```

Returns Lindblad jump operators:
- √γ₁ · b (legacy aggregate transmon relaxation when `transmon_t1` is not supplied)
- `sqrt(1/T1_j) * |j-1><j|` for each explicit transmon ladder transition listed in `transmon_t1`
- √γ_φ · σ_z (dephasing for 2-level) or √γ_φ · n_q (multi-level)
- √(κ(n_th+1)) · a and √(κ·n_th) · a† per bosonic mode
- `sqrt(1/tphi_storage) * n_s` for storage pure dephasing when `tphi_storage` is set
- `sqrt(1/tphi_readout) * n_r` for readout pure dephasing when `tphi_readout` is set

---

### 6.6 Coupling Helpers

**Module path:** `cqed_sim.sim.couplings`

| Function / Class | Signature | Description |
|---|---|---|
| `cross_kerr(a, b, chi)` | `(Qobj, Qobj, float) -> Qobj` | χ · a†a · b†b |
| `self_kerr(a, kerr)` | `(Qobj, float) -> Qobj` | +(K/2) · a†²a² |
| `exchange(a, b, coupling)` | `(Qobj, Qobj, float\|complex) -> Qobj` | J · (a†b + ab†) |
| `TunableCoupler` | frozen dataclass | Flux-tunable coupler: `j_max`, `flux_period`, `phase_offset`, `dc_offset` |
| `TunableCoupler.exchange_rate(flux)` | `(float) -> float` | dc_offset + j_max·cos(2π·flux/flux_period + phase_offset) |
| `TunableCoupler.operator(a, b, flux)` | `(Qobj, Qobj, float) -> Qobj` | exchange(a, b, self.exchange_rate(flux)) |

---

### 6.7 State Extractors

**Module path:** `cqed_sim.sim.extractors`

#### Partial-Trace Extractors

| Function | Signature | Description |
|---|---|---|
| `reduced_qubit_state(state)` | `(Qobj) -> Qobj` | Trace out everything except qubit (index 0) |
| `reduced_transmon_state(state)` | `(Qobj) -> Qobj` | Alias for `reduced_qubit_state` |
| `reduced_cavity_state(state)` | `(Qobj) -> Qobj` | Trace out qubit (2-mode systems only) |
| `reduced_storage_state(state)` | `(Qobj) -> Qobj` | Storage subsystem (index 1) |
| `reduced_readout_state(state)` | `(Qobj) -> Qobj` | Readout subsystem (index 2) |
| `reduced_subsystem_state(state, subsystem)` | `(Qobj, int\|str) -> Qobj` | General partial trace. String aliases: `"qubit"`, `"transmon"`, `"cavity"`, `"storage"`, `"readout"` |

#### Bloch Coordinates

| Function | Signature | Returns |
|---|---|---|
| `bloch_xyz_from_joint(state)` | `(Qobj) -> (x, y, z)` | Bloch vector from joint state; requires a 2-level reduced transmon state |
| `conditioned_bloch_xyz(state, n, fallback="nan")` | `(Qobj, int, str) -> (x, y, z, p_n, valid)` | Bloch vector conditioned on cavity Fock level n |
| `conditioned_qubit_state(state, n, fallback="nan")` | `(Qobj, int, str) -> (rho_q, p_n, valid)` | Normalized qubit state conditioned on Fock n |
| `conditioned_population(state, n)` | `(Qobj, int) -> float` | Probability of cavity being in Fock state n |

#### Multilevel Population Helpers

| Function | Signature | Returns |
|---|---|---|
| `subsystem_level_population(state, subsystem, level)` | `(Qobj, str\|int, int) -> float` | Population in a chosen level of a reduced subsystem |
| `transmon_level_populations(state)` | `(Qobj) -> dict[int, float]` | All transmon level populations in the reduced ancilla state |
| `compute_shelving_leakage(initial_state, final_state, shelved_level=1, subsystem="transmon")` | `(Qobj, Qobj, int, str\|int) -> float` | Absolute population change of a shelved ancilla level |

#### Mode Moments and Photon Numbers

| Function | Signature | Returns |
|---|---|---|
| `mode_moments(state, subsystem, dim)` | `-> dict` | `{"a": ⟨a⟩, "adag_a": ⟨a†a⟩, "n": ⟨n⟩}` |
| `cavity_moments(state, n_cav)` | `-> dict` | Cavity mode moments |
| `storage_moments(state, n_storage)` | `-> dict` | Storage mode moments |
| `readout_moments(state, n_readout)` | `-> dict` | Readout mode moments |
| `storage_photon_number(state)` | `-> float` | ⟨n_s⟩ |
| `readout_photon_number(state)` | `-> float` | ⟨n_r⟩ |
| `joint_expectation(state, operator)` | `-> complex` | Tr[ρ · O] |

#### Qubit-Conditioned Extractors

| Function | Signature | Returns |
|---|---|---|
| `qubit_conditioned_subsystem_state(state, subsystem, qubit_level, fallback)` | `-> (rho, p, valid)` | Subsystem state conditioned on qubit level |
| `qubit_conditioned_mode_moments(state, subsystem, qubit_level)` | `-> dict` | Moments of a mode conditioned on qubit state |
| `readout_response_by_qubit_state(state)` | `-> dict[int, dict]` | `{0: moments_g, 1: moments_e}` |

#### Wigner Function

```python
def cavity_wigner(
    rho_c: qt.Qobj,
    xvec=None, yvec=None,
    n_points: int = 41,
    extent: float = 4.0,
    coordinate: str = "quadrature",  # or "alpha"/"coherent"
) -> tuple[ndarray, ndarray, ndarray]
```

Returns `(xvec, yvec, W)`. The `"quadrature"` coordinate uses natural units;
`"alpha"` scales by √2.

**Performance note:** Subsystem projectors and mode annihilation operators are
cached via `@lru_cache`, so repeated calls with the same dimensions/levels
avoid redundant QuTiP tensor-product construction.

---

### 6.8 Diagnostics

**Module path:** `cqed_sim.sim.diagnostics`

| Function | Signature | Description |
|---|---|---|
| `channel_norms(compiled)` | `(CompiledSequence) -> dict[str, float]` | L2 norm of distorted waveform per channel |
| `instantaneous_phase_frequency(signal, dt)` | `(ndarray, float) -> (phase, omega)` | Unwrapped phase and instantaneous frequency from complex signal |

---

## 7. State Preparation and Measurement

**Modules:** `cqed_sim.core.state_prep`, `cqed_sim.measurement`

### 7.1 State Preparation

**Module path:** `cqed_sim.core.state_prep`

#### SubsystemStateSpec

```python
@dataclass(frozen=True)
class SubsystemStateSpec:
    kind: str
    label: str | None = None
    level: int | None = None
    alpha: complex | None = None
    amplitudes: Any | None = None
    density_matrix: Any | None = None
```

#### StatePreparationSpec

```python
@dataclass(frozen=True)
class StatePreparationSpec:
    qubit: SubsystemStateSpec = qubit_state("g")
    storage: SubsystemStateSpec = vacuum_state()
    readout: SubsystemStateSpec | None = None
```

#### State Constructor Helpers

| Function | Signature | Description |
|---|---|---|
| `qubit_state(label)` | `(str) -> SubsystemStateSpec` | Labels: `"g"`, `"e"`, `"+x"`, `"-x"`, `"+y"`, `"-y"` |
| `qubit_level(level)` | `(int) -> SubsystemStateSpec` | Qubit by level index |
| `vacuum_state()` | `() -> SubsystemStateSpec` | Bosonic vacuum |
| `fock_state(level)` | `(int) -> SubsystemStateSpec` | Bosonic Fock state |
| `coherent_state(alpha)` | `(complex) -> SubsystemStateSpec` | Bosonic coherent state |
| `amplitude_state(amplitudes)` | `(Any) -> SubsystemStateSpec` | Arbitrary state from amplitudes |
| `density_matrix_state(rho)` | `(Any) -> SubsystemStateSpec` | State from density matrix |

#### Preparation Functions

```python
def prepare_state(model, spec: StatePreparationSpec | None = None) -> qt.Qobj
def prepare_ground_state(model) -> qt.Qobj
```

`prepare_state(...)` builds a tensor-product state that follows model subsystem ordering automatically.

---

### 7.2 Qubit Measurement

**Module path:** `cqed_sim.measurement.qubit`

#### QubitMeasurementSpec

```python
@dataclass(frozen=True)
class QubitMeasurementSpec:
    shots: int | None = None
    confusion_matrix: np.ndarray | None = None
    iq_sigma: float | None = None
    seed: int | None = None
    lump_other_into: str = "e"
    readout_chain: ReadoutChain | None = None
    readout_duration: float | None = None
    readout_dt: float | None = None
    readout_drive_frequency: float | None = None
    readout_chi: float | None = None
    qubit_frequency: float | None = None
    include_filter: bool = True
    include_measurement_dephasing: bool = False
    include_purcell_relaxation: bool = False
    classify_from_iq: bool = False
```

#### QubitMeasurementResult

```python
@dataclass
class QubitMeasurementResult:
    probabilities: dict[str, float]
    observed_probabilities: dict[str, float]
    expectation_z: float
    counts: dict[str, int] | None = None
    samples: np.ndarray | None = None
    iq_samples: np.ndarray | None = None
    post_measurement_state: qt.Qobj | None = None
    readout_centers: dict[str, np.ndarray] | None = None
    readout_metadata: dict[str, float] | None = None
```

#### measure_qubit

```python
def measure_qubit(state: qt.Qobj, spec: QubitMeasurementSpec | None = None) -> QubitMeasurementResult
```

Pipeline:

1. Extract the reduced qubit state.
2. Compute latent probabilities `(p_g, p_e)`.
3. Apply the optional confusion matrix.
4. Apply readout-chain backaction and/or I/Q generation when requested.
5. Sample outcomes when `shots` is set.

Confusion-matrix convention: `p_observed = M @ p_latent` with `(g, e)` ordering.

---

### 7.3 Readout Chain

**Module path:** `cqed_sim.measurement.readout_chain`

| Dataclass | Description |
|---|---|
| `ReadoutResonator` | Single-pole dispersive readout resonator model |
| `PurcellFilter` | Frequency-selective linewidth suppression model |
| `AmplifierChain` | Linear gain plus additive thermal noise |
| `ReadoutChain` | Resonator + optional filter + amplifier + integration settings |
| `ReadoutTrace` | Time-domain cavity/output/voltage/I/Q record |

Common methods:

| API | Description |
|---|---|
| `ReadoutChain.simulate_trace(...)` | Time-domain resonator response and downconverted trace |
| `ReadoutChain.simulate_waveform(...)` | Time-domain replay for an arbitrary complex drive waveform |
| `ReadoutChain.iq_centers(...)` | Noiseless I/Q centers for `g` and `e` |
| `ReadoutChain.sample_iq(...)` | Noisy I/Q sampling from latent labels |
| `ReadoutChain.classify_iq(...)` | Nearest-center classification |
| `ReadoutChain.apply_backaction(...)` | Measurement dephasing and optional Purcell relaxation |
| `ReadoutChain.gamma_meas(...)` | Measurement-induced dephasing rate |
| `ReadoutChain.purcell_rate(...)` | Purcell decay rate |
| `ReadoutChain.purcell_limited_t1(...)` | Purcell-limited `T1` |

### 7.4 Continuous Readout Replay

**Module path:** `cqed_sim.measurement.stochastic`

| Dataclass | Description |
|---|---|
| `ContinuousReadoutSpec` | SME replay options: frame, monitored subsystem, number of trajectories, storage policy |
| `ContinuousReadoutTrajectory` | One trajectory's measurement record, final state, optional states, and expectations |
| `ContinuousReadoutResult` | Aggregate replay result with average expectations and all trajectories |

Common APIs:

| API | Description |
|---|---|
| `simulate_continuous_readout(...)` | QuTiP `smesolve(...)` wrapper using `cqed_sim` drive/noise conventions |
| `integrate_measurement_record(...)` | Integrate a homodyne or heterodyne record over its final time axis |

The monitored path is built from `split_collapse_operators(...)`: one selected bosonic emission channel is promoted to the stochastic measurement path, while relaxation, thermal excitation, and dephasing remain ordinary Lindblad terms.

### 7.5 Strong-Readout Disturbance Helpers

**Module path:** `cqed_sim.measurement.strong_readout`

| Dataclass | Description |
|---|---|
| `StrongReadoutMixingSpec` | Occupancy- and slew-activated phenomenological strong-readout model |
| `StrongReadoutDisturbance` | Returned envelopes, activation profile, and occupancy estimate |

Common APIs:

| API | Description |
|---|---|
| `build_strong_readout_disturbance(...)` | Build auxiliary `g-e` / `e-f` disturbance envelopes from a readout waveform |
| `strong_readout_drive_targets(...)` | Matching `TransmonTransitionDriveSpec` mapping for those channels |
| `infer_dispersive_coupling(...)` | Infer `g` from dispersive parameters |
| `estimate_dispersive_critical_photon_number(...)` | Estimate `n_crit = (Delta / 2g)^2` |

`StrongReadoutMixingSpec` also supports a simple higher-ladder continuation through
`higher_ladder_scales`, `higher_ladder_start_level`, and `higher_channel_prefix`.
When `higher_ladder_scales` is non-empty, `strong_readout_drive_targets(...)` can emit
additional channels such as `mix_high_2_3`, `mix_high_3_4`, ... up to the optional
`max_transmon_level`, while `build_strong_readout_disturbance(...)` returns the matching
scaled envelopes in `StrongReadoutDisturbance.higher_envelopes`.

Workflow boundary:

- high-level orchestration no longer lives in `cqed_sim`
- guided notebook tutorials now live under `tutorials/`
- standalone protocol recipes now live under `examples/`
- the reusable helper `pure_dephasing_time_from_t1_t2(...)` lives in `cqed_sim.sim.noise`
- stochastic replay reuses `split_collapse_operators(...)` from `cqed_sim.sim`

Repository-side workflow entry points:

- `tutorials/README.md`
- `tutorials/00_tutorial_index.ipynb`
- `tutorials/03_cavity_displacement_basics.ipynb`
- `tutorials/17_readout_resonator_response.ipynb`
- `README_AGENT_WORKFLOW.md`
- `agent_workflow/README.md`
- `tools/run_agent_workflow.py`
- `run_agent_workflow.ps1`
- `examples/protocol_style_simulation.py`
- `examples/continuous_readout_replay_demo.py`
- `examples/kerr_free_evolution.py`
- `examples/kerr_sign_verification.py`
- `examples/logical_block_phase_targeted_subspace_demo.py`
- `examples/sequential_sideband_reset.py`

---

## 8. Gate I/O

**Module:** `cqed_sim.io`

**Module path:** `cqed_sim.io.gates`

### Gate Dataclasses

```python
@dataclass(frozen=True)
class DisplacementGate:
    index: int; name: str; re: float; im: float
    # Properties: type="Displacement", target="storage", alpha=complex(re,im), params

@dataclass(frozen=True)
class RotationGate:
    index: int; name: str; theta: float; phi: float
    # Properties: type="Rotation", target="qubit", params

@dataclass(frozen=True)
class SQRGate:
    index: int; name: str; theta: tuple[float, ...]; phi: tuple[float, ...]
    # Properties: type="SQR", target="qubit", params
```

**Union type:** `Gate = DisplacementGate | RotationGate | SQRGate`

### Functions

| Function | Signature | Description |
|---|---|---|
| `load_gate_sequence(path_like)` | `(str \| Path) -> tuple[Path, list[Gate]]` | Load and validate JSON gate sequence. Typo-tolerant for `.json`/`.josn` extensions. |
| `render_gate_table(gates, max_rows=20)` | `(list[Gate], int) -> None` | Print formatted ASCII gate table |
| `gate_to_record(gate)` | `(Gate) -> dict` | Convert gate to dict record |
| `gate_summary_text(gate)` | `(Gate) -> str` | One-line summary of gate params |

**JSON format:** Array of objects, each with `"type"`, `"target"`, `"name"` (optional),
`"params"` (dict). SQR gates accept both `"theta"`/`"thetas"` and `"phi"`/`"phis"` keys.

---

## 9. Analysis

**Module:** `cqed_sim.analysis`

**Module path:** `cqed_sim.analysis.parameter_translation`

Translates between bare transmon parameters, measured dispersive parameters, and the
runtime convention.

### HamiltonianParams

```python
@dataclass(frozen=True)
class HamiltonianParams:
    omega_q: float          # Transmon frequency (rad/s)
    omega_r: float          # Resonator frequency (rad/s)
    alpha: float            # Anharmonicity (rad/s), typically negative
    chi: float              # First-order dispersive shift (rad/s)
    chi_2: float            # Second-order dispersive shift (rad/s)
    g: float                # Coupling strength (rad/s)
    delta: float            # Detuning ω_q − ω_r (rad/s)
    ec: float               # Charging energy (rad/s)
    ej: float               # Josephson energy (rad/s)
    synthesis_chi: float    # Same canonical chi exposed for synthesis callers
    synthesis_chi_2: float  # Same canonical chi_2 exposed for synthesis callers
    regime: str = "dispersive"
    metadata: dict = field(default_factory=dict)
```

### Translation Functions

```python
def from_transmon_params(
    ej: float, ec: float, g: float, omega_r: float,
    *, resonator_dim: int = 5, transmon_dim: int = 6,
) -> HamiltonianParams
```

Translates bare transmon parameters (E_J, E_C, g, ω_r) to dressed runtime
coefficients. Uses the large-E_J/E_C expansion for the bare frequency and exact
numerical diagonalization for dressed chi values.

```python
def from_measured(
    omega_01: float, alpha: float, chi: float, g: float,
    *, omega_r: float | None = None, detuning_branch: str = "positive",
    resonator_dim: int = 5, transmon_dim: int = 6,
) -> HamiltonianParams
```

Inverts measured qubit parameters into approximate circuit parameters. Solves
the dispersive equation to find the detuning, then calls `from_transmon_params`.

**Detuning branch options:** `"positive"`, `"negative"`, `"largest-magnitude"`, or
auto-select from closest root if `omega_r` is provided.

---

## 9A. Floquet Analysis (`cqed_sim.floquet`)

The Floquet module analyzes strictly periodic closed-system Hamiltonians of the form

$$
H(t + T) = H(t),
\qquad
\Omega = \frac{2\pi}{T}.
$$

It wraps QuTiP's `FloquetBasis` for the primary propagator-based route and adds cQED-specific helpers for drive-target resolution, bare-state overlap labeling, multiphoton resonance detection, harmonic-space Sambe construction, and overlap-based branch tracking over parameter sweeps.

### Core dataclasses

```python
@dataclass(frozen=True)
class PeriodicFourierComponent:
    harmonic: int
    amplitude: complex

@dataclass(frozen=True)
class PeriodicDriveTerm:
    operator: qt.Qobj | None = None
    target: str | TransmonTransitionDriveSpec | SidebandDriveSpec | None = None
    quadrature: str = "x"
    amplitude: complex = 1.0
    frequency: float = 0.0
    phase: float = 0.0
    waveform: str | Callable[[ndarray], ndarray] = "cos"
    fourier_components: Sequence[PeriodicFourierComponent] = ()
    label: str | None = None

@dataclass(frozen=True)
class FloquetProblem:
    period: float
    periodic_terms: Sequence[PeriodicDriveTerm] = ()
    static_hamiltonian: qt.Qobj | None = None
    model: Any | None = None
    frame: FrameSpec = FrameSpec()
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FloquetConfig:
    n_time_samples: int = 401
    atol: float = 1e-8
    rtol: float = 1e-7
    max_step: float | None = None
    sort: bool = True
    sparse: bool = False
    zone_center: float = 0.0
    precompute_times: Sequence[float] | None = None
    overlap_reference_time: float = 0.0
    sambe_harmonic_cutoff: int | None = None
    sambe_n_time_samples: int | None = None
```

`PeriodicDriveTerm` supports two complementary styles:

- explicit `operator=...` for parameter modulation or arbitrary periodic perturbations,
- model-aware `target=...` that reuses the same target semantics as the pulse runtime (`"qubit"`, `"storage"`, `SidebandDriveSpec(...)`, and so on).

Target-based Floquet terms default to the Hermitian in-phase quadrature constructed from the target's raising and lowering operators. This keeps the Floquet drive layer consistent with the repository's operator and frame conventions without forcing users to build those combinations manually.

### Main solver

```python
def solve_floquet(problem: FloquetProblem, config: FloquetConfig | None = None) -> FloquetResult
```

`FloquetResult` contains:

- folded quasienergies,
- one-period propagator,
- Floquet modes at `t = 0`,
- the underlying QuTiP `FloquetBasis`,
- overlaps with the static Hamiltonian eigenbasis,
- dominant bare-state labels,
- an effective static Floquet Hamiltonian,
- optional Sambe Hamiltonian and harmonic norms,
- truncation warnings when Floquet modes accumulate weight on the Hilbert-space boundary.

### Main helper functions

| Function | Description |
|---|---|
| `build_floquet_hamiltonian(problem)` | Build the `QobjEvo` Hamiltonian used by the Floquet solver |
| `compute_period_propagator(...)` | Return the one-period propagator |
| `compute_quasienergies(...)` | Return folded quasienergies |
| `compute_floquet_modes(...)` | Evaluate Floquet modes at arbitrary time |
| `compute_bare_state_overlaps(...)` | Floquet-mode overlaps with the static eigenbasis |
| `compute_floquet_transition_strengths(...)` | Harmonic-resolved transition matrix elements for a probe operator |
| `identify_multiphoton_resonances(...)` | Near-`Delta E ~= n Omega` resonance finder |
| `build_effective_floquet_hamiltonian(...)` | Effective static Hamiltonian in the Floquet eigenbasis |
| `build_sambe_hamiltonian(...)` | Truncated harmonic-space Floquet Hamiltonian |
| `extract_sambe_quasienergies(...)` | Cluster folded Sambe eigenvalues into physical quasienergy branches |
| `run_floquet_sweep(...)` | Solve a parameter sweep and track quasienergy branches |
| `track_floquet_branches(...)` | Overlap-based branch matching and zone unwrapping |

### cQED-specific modulation builders

| Function | Description |
|---|---|
| `build_target_drive_term(...)` | Build a periodic drive from an existing model target |
| `build_transmon_frequency_modulation_term(...)` | Modulate `n_q` for transmon-frequency modulation |
| `build_mode_frequency_modulation_term(...)` | Modulate a bosonic number operator |
| `build_dispersive_modulation_term(...)` | Modulate `n_mode * n_q` directly |

### Notes and caveats

- Floquet analysis assumes exact periodicity.
- Multi-tone drives are only strictly Floquet when commensurate with the supplied common period.
- Quasienergies are defined modulo the drive angular frequency.
- The current public API is closed-system. Open-system Floquet-Markov support is a future extension.
- For strongly driven transmons, increasing `n_tr` is often more important than increasing the Floquet time grid.

---

## 10. Backends

**Module:** `cqed_sim.backends`

Optional dense piecewise-constant solver backends for small systems and parity checks.

### BaseBackend (ABC)

Abstract interface with methods: `asarray`, `to_numpy`, `eye`, `zeros`, `reshape`,
`dagger`, `matmul`, `kron`, `expm`, `trace`, `expectation`, `lindbladian`.

### NumPyBackend

```python
class NumPyBackend(BaseBackend):
    name = "numpy"
```

All operations use `np.complex128`. Matrix exponential via `scipy.linalg.expm`.
Lindbladian constructs full superoperator for density-matrix propagation.
For Hilbert-space dimensions ≥ 20, the Liouvillian is assembled using
`scipy.sparse.kron` to reduce peak memory usage and construction time,
then converted to dense for `expm`.

### JaxBackend

```python
class JaxBackend(BaseBackend):
    name = "jax"
    def __init__(self, device: str | None = None)
```

Requires JAX. Uses `jnp.complex128` with 64-bit precision enabled. Optional device
targeting (`"cpu"`, `"gpu"`, `"tpu"`). Matrix exponential via
`jax.scipy.linalg.expm`.

**Import guard:** `JaxBackend` is `None` if JAX is not installed.

**Usage:**

```python
from cqed_sim.sim import SimulationConfig
from cqed_sim.backends import NumPyBackend

config = SimulationConfig(backend=NumPyBackend())
```

---

## 11. SQR Calibration

**Module:** `cqed_sim.calibration`

**Module path:** `cqed_sim.calibration.sqr`

Calibrates Selective Qubit Rotation (SQR) gates by optimizing per-manifold correction
parameters.

### SQRCalibrationResult

```python
@dataclass
class SQRCalibrationResult:
    sqr_name: str
    max_n: int
    d_lambda: list[float]        # Amplitude corrections per Fock level
    d_alpha: list[float]         # Phase corrections per Fock level
    d_omega_rad_s: list[float]   # Frequency corrections per Fock level (rad/s)
    theta_target: list[float]
    phi_target: list[float]
    initial_loss: list[float]
    optimized_loss: list[float]
    levels: list[SQRLevelCalibration] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

| Property / Method | Description |
|---|---|
| `d_omega_hz` | Frequency corrections in Hz |
| `correction_for_n(n)` | `(d_lambda_n, d_alpha_n, d_omega_n)` for Fock level n |
| `improvement_summary()` | Dict with mean/max improvement metrics |
| `to_dict()` / `from_dict(payload)` | Serialization |

### SQRLevelCalibration

Per-manifold calibration record stored in `SQRCalibrationResult.levels`.

```python
@dataclass
class SQRLevelCalibration:
    n: int                                       # Fock level index
    theta_target: float                          # Target polar angle
    phi_target: float                            # Target azimuthal angle
    skipped: bool                                # Whether this level was skipped
    initial_params: tuple[float, float, float]   # (d_lambda, d_alpha, d_omega) before optimization
    optimized_params: tuple[float, float, float] # (d_lambda, d_alpha, d_omega) after optimization
    initial_loss: float                          # Loss before optimization
    optimized_loss: float                        # Loss after optimization
    process_fidelity: float                      # Final conditional process fidelity
    success_stage1: bool = False                 # Stage-1 (Powell) converged
    success_stage2: bool = False                 # Stage-2 (L-BFGS-B) converged
    message_stage1: str = ""                     # Optimizer message, stage 1
    message_stage2: str = ""                     # Optimizer message, stage 2
```

### Core Calibration Functions

| Function | Signature | Description |
|---|---|---|
| `calibrate_sqr_gate(gate, config)` | `(SQRGate, Mapping) -> SQRCalibrationResult` | Two-stage optimization (Powell → L-BFGS-B) per manifold |
| `load_or_calibrate_sqr_gate(gate, config, cache_dir)` | `-> SQRCalibrationResult` | Cache-aware calibration with config hash matching |
| `calibrate_all_sqr_gates(gates, config, cache_dir)` | `-> dict[str, SQRCalibrationResult]` | Calibrate all SQR gates in a sequence |
| `export_calibration_result(result, path, config)` | `-> Path` | Write JSON |
| `load_calibration_result(path)` | `-> SQRCalibrationResult` | Read JSON |

### Evaluation and Benchmarking

| Function | Signature | Description |
|---|---|---|
| `evaluate_sqr_gate_levels(gate, config, corrections)` | `-> list[dict]` | Per-Fock-level fidelity evaluation |
| `extract_effective_qubit_unitary(n, theta, phi, config, d_lambda, d_alpha, d_omega)` | `-> (ndarray, dict)` | 2×2 qubit unitary from time-dependent Hamiltonian |
| `extract_multitone_effective_qubit_unitary(manifold_n, target, config, corrections)` | `(int, RandomSQRTarget, Mapping, Mapping\|None) -> (ndarray, dict)` | 2×2 qubit unitary for a multitone drive evaluated at a single manifold |
| `evaluate_guarded_sqr_target(target, config, corrections, lambda_guard, weight_mode, poisson_alpha)` | `(RandomSQRTarget, Mapping, ...) -> dict` | Evaluate a guarded target with leakage penalty; returns per-manifold metrics |
| `target_qubit_unitary(theta, phi)` | `-> ndarray` | Ideal SQR target unitary |
| `conditional_process_fidelity(target, simulated)` | `-> float` | Process fidelity, clipped to [0, 1] |
| `conditional_loss(params, n, theta, phi, config)` | `-> float` | Optimization objective: 1 − fidelity + regularization |

### Conditioned Multitone Validation

**Module path:** `cqed_sim.calibration.conditioned_multitone`

This reduced-objective API answers a narrower question than full SQR gate calibration:
whether a single common multitone qubit drive can place the qubit, conditioned on
each cavity Fock sector, at the requested target Bloch angles \\(\theta_n, \phi_n\\)
under the dispersive Hamiltonian. It deliberately does **not** enforce full joint-unitary
or cavity-branch correctness.

```python
@dataclass(frozen=True)
class ConditionedQubitTargets:
    theta: tuple[float, ...]
    phi: tuple[float, ...]
    weights: tuple[float, ...] = ()

@dataclass(frozen=True)
class ConditionedMultitoneCorrections:
    d_lambda: tuple[float, ...] = ()
    d_alpha: tuple[float, ...] = ()
    d_omega_rad_s: tuple[float, ...] = ()

@dataclass(frozen=True)
class ConditionedMultitoneRunConfig:
    frame: FrameSpec = FrameSpec()
    duration_s: float = 1.0e-6
    dt_s: float = 4.0e-9
    sigma_fraction: float = 1.0 / 6.0
    tone_cutoff: float = 1.0e-10
    include_all_levels: bool = False
    max_step_s: float | None = None
    fock_fqs_hz: tuple[float, ...] | None = None

@dataclass(frozen=True)
class ConditionedOptimizationConfig:
    active_levels: tuple[int, ...] = ()
    parameters: tuple[str, ...] = ("d_lambda", "d_alpha", "d_omega")
    method_stage1: str = "Powell"
    method_stage2: str | None = "L-BFGS-B"
    maxiter_stage1: int = 40
    maxiter_stage2: int = 60
```

| Function / Type | Description |
|---|---|
| `ConditionedQubitTargets.from_spec(targets, n_levels=None, weights=None)` | Accepts list/array/dict target specifications and normalizes sector weights |
| `qubit_state_from_angles(theta, phi)` / `qubit_density_matrix_from_angles(theta, phi)` | Pure-state target helpers for a single conditioned qubit sector |
| `build_conditioned_multitone_tones(model, targets, run_config, corrections)` | Builds per-tone amplitudes, phases, and carriers using the library’s additive-amplitude and carrier-sign conventions |
| `build_conditioned_multitone_waveform(tone_specs, run_config, ...)` | Converts tone specs into a reusable common multitone `Pulse` |
| `sample_conditioned_multitone_waveform(waveform, run_config, ...)` | Returns sampled complex envelope plus IQ traces for plotting |
| `evaluate_conditioned_multitone(model, targets, waveform, run_config, ...)` | Computes per-sector conditioned qubit fidelities, Bloch vectors, angle errors, and weighted aggregate cost |
| `run_conditioned_multitone_validation(model, targets, run_config, ...)` | Convenience wrapper that builds tones, builds the waveform, and evaluates the result |
| `optimize_conditioned_multitone(model, targets, run_config, ...)` | Tunes `d_lambda`, `d_alpha`, and/or `d_omega` against the reduced conditioned-qubit cost |

The validation path supports two simulation modes:

- `simulation_mode="reduced"`: exact sector-by-sector two-level evolution under the dispersive Hamiltonian decomposition.
- `simulation_mode="full"`: full qubit-cavity simulation followed by conditioned qubit-state extraction as a consistency check.

Use the reduced layer when the question is, "Can one common waveform place the qubit at the requested conditioned Bloch targets?" Move to the targeted-subspace layer when the question is, "Does that same waveform implement the intended coherent operator across a chosen logical qubit-cavity block structure?"

Practical entry points:

- `examples/conditioned_multitone_reduced_demo.py` shows the reduced workflow, including reduced-vs-full agreement and a small detuning-only optimization.
- `examples/logical_block_phase_targeted_subspace_demo.py` shows the full targeted-subspace workflow, including best-fit logical block phases and the ideal cavity-only correction layer.

### Targeted-Subspace Logical Block Phase

**Module path:** `cqed_sim.calibration.targeted_subspace_multitone`

This layer extends conditioned multitone validation to the full logical qubit-cavity subspace and can append an explicit ideal cavity-only logical block-phase layer after the simulated waveform operator.

```python
@dataclass(frozen=True)
class LogicalBlockPhaseCorrection:
    logical_levels: tuple[int, ...] = ()
    phases_rad: tuple[float, ...] = ()

@dataclass(frozen=True)
class TargetedSubspaceOptimizationConfig:
    conditioned: ConditionedOptimizationConfig = ConditionedOptimizationConfig()
    include_block_phase: bool = False
    block_phase_levels: tuple[int, ...] = ()
    block_phase_bounds_rad: tuple[float, float] = (-np.pi, np.pi)
    regularization_block_phase: float = 0.0
    block_phase_reference_level: int | None = None
```

| Function / Type | Description |
|---|---|
| `build_block_rotation_target_operator(targets, logical_levels=None)` | Ideal restricted target operator with 2×2 qubit blocks ordered as `|g,0>, |e,0>, |g,1>, |e,1>, ...` |
| `build_spanning_state_transfer_set(target_operator, include_pairwise_superpositions=True)` | Basis-plus-superposition transfer probes for restricted-state diagnostics |
| `analyze_targeted_subspace_operator(actual_full_operator, model, targets, ..., logical_block_phase=None)` | Evaluates restricted fidelity, state-transfer metrics, block populations, and logical block-phase diagnostics |
| `run_targeted_subspace_multitone_validation(model, targets, run_config, ..., logical_block_phase=None)` | Builds the common multitone waveform and evaluates the full targeted subspace |
| `optimize_targeted_subspace_multitone(model, targets, run_config, ..., initial_logical_block_phase=None, optimization_config=...)` | Two-stage targeted-subspace optimization over waveform corrections and, optionally, logical block-phase parameters |

`TargetedSubspaceValidationResult` records the applied logical block phase, the best-fit logical block phase extracted from the raw restricted operator, the corrected restricted-process fidelity, and a `LogicalBlockPhaseDiagnostics` payload.

Typical workflow order:

1. Start with `run_conditioned_multitone_validation(...)` to separate basic conditioned-qubit reachability from stronger logical-subspace failures.
2. If the reduced layer succeeds but the full logical action is still poor, switch to `run_targeted_subspace_multitone_validation(...)` and inspect restricted-process fidelity, block-preservation metrics, leakage, and block-phase diagnostics.
3. If the dominant defect is coherent logical block phase, either supply `logical_block_phase=` directly or enable `TargetedSubspaceOptimizationConfig(include_block_phase=True)` when you want the optimizer to fit that ideal post-layer.

### Random Target Benchmarking

```python
@dataclass(frozen=True)
class RandomSQRTarget:
    target_id: str
    target_class: str              # "iid", "smooth", "hard", "sparse"
    logical_n: int
    guard_levels: int
    theta: tuple[float, ...]
    phi: tuple[float, ...]

@dataclass
class GuardedBenchmarkResult:
    # target_id, target_class, duration_s, logical_n, guard_levels,
    # lambda_guard, weight_mode, poisson_alpha, logical_fidelity,
    # epsilon_guard, loss_total, success, converged, iterations,
    # objective_evaluations, calibration, per_manifold, convergence_trace, metadata
```

| Function | Description |
|---|---|
| `generate_random_sqr_targets(logical_n, guard_levels, n_targets_per_class, seed, ...)` | Generate random SQR targets across classes |
| `calibrate_guarded_sqr_target(target, config, ...)` | Optimize corrections with guard-level leakage penalty |
| `benchmark_random_sqr_targets_vs_duration(config, durations, targets, ...)` | Sweep durations × targets |
| `benchmark_results_table(results)` | Convert to record dicts |
| `summarize_duration_benchmark(results)` | Group by duration with statistics |

---

## 12. Calibration Targets

**Module:** `cqed_sim.calibration_targets`

Lightweight surrogate-model calibration sweeps that return fitted parameters.

### CalibrationResult

```python
@dataclass
class CalibrationResult:
    fitted_parameters: dict[str, float]
    uncertainties: dict[str, float]
    raw_data: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Calibration Sweep Functions

All functions take a `model` object (with `omega_q` and/or `alpha` attributes) and
return a `CalibrationResult`.

#### `run_spectroscopy`

```python
def run_spectroscopy(
    model,
    drive_frequencies: np.ndarray,
    *,
    linewidth: float | None = None,       # Defaults to |alpha|/10
    excited_state_fraction: float = 0.0,  # Thermal population fraction
) -> CalibrationResult
```

**Fitted parameters:** `omega_01`, `omega_12`.
Synthesizes Lorentzian response and locates peaks via quadratic interpolation.

#### `run_rabi`

```python
def run_rabi(
    model,
    amplitudes: np.ndarray,
    *,
    duration: float = 40e-9,
    omega_scale: float = 1.0,
) -> CalibrationResult
```

**Fitted parameters:** `omega_scale`, `duration`.
Fits sin²(Ω·A·T/2) via `curve_fit`.

#### `run_ramsey`

```python
def run_ramsey(
    model,
    delays: np.ndarray,
    *,
    detuning: float,
    t2_star: float = 20e-6,
) -> CalibrationResult
```

**Fitted parameters:** `delta_omega`, `t2_star`.
Fits Ramsey fringe 0.5(1 + e^{−t/T₂*} cos(Δω·t)).

#### `run_t1`

```python
def run_t1(
    model,
    delays: np.ndarray,
    *,
    t1: float = 30e-6,
) -> CalibrationResult
```

**Fitted parameters:** `t1`.

#### `run_t2_echo`

```python
def run_t2_echo(
    model,
    delays: np.ndarray,
    *,
    t2_echo: float = 40e-6,
) -> CalibrationResult
```

**Fitted parameters:** `t2_echo`.

#### `run_drag_tuning`

```python
def run_drag_tuning(
    model,
    drag_values: np.ndarray,
    *,
    optimal_drag: float | None = None,   # Defaults to −1/alpha
    baseline_leakage: float = 1e-3,
    curvature: float = 0.25,
) -> CalibrationResult
```

**Fitted parameters:** `drag_optimal`.
Synthesizes quadratic leakage curve and fits vertex.

---

## 13. Tomography

**Module:** `cqed_sim.tomo`

### DeviceParameters

**Module path:** `cqed_sim.tomo.device.DeviceParameters`

```python
@dataclass(frozen=True)
class DeviceParameters:
    ro_fq: float = 8596222556.078796       # Readout frequency (Hz)
    qb_fq: float = 6150369694.524461       # Qubit frequency (Hz)
    st_fq: float = 5240932800.0            # Storage frequency (Hz)
    ro_kappa: float = 4156000.0            # Readout linewidth (Hz)
    ro_chi: float = -913148.5              # Readout chi (Hz)
    anharmonicity: float = -255669694.5    # Alpha (Hz)
    st_chi: float = -2840421.354           # Storage chi (Hz)
    st_chi2: float = -21912.638            # Storage chi2 (Hz)
    st_chi3: float = -327.379              # Storage chi3 (Hz)
    st_K: float = -28844.0                 # Storage Kerr (Hz)
    st_K2: float = 1406.0                  # Storage Kerr2 (Hz)
    ro_therm_clks: float = 1000.0          # Readout thermal clocks
    qb_therm_clks: float = 19625.0         # Qubit T1 in clock cycles
    st_therm_clks: float = 200000.0        # Storage thermal clocks
    qb_t1_relax_ns: float = 9812.87        # Qubit T1 (ns)
    qb_t2_ramsey_ns: float = 6324.73       # Qubit T2* (ns)
    qb_t2_echo_ns: float = 8381.0          # Qubit T2_echo (ns)
```

| Method | Description |
|---|---|
| `hz_to_rad_per_ns(f_hz)` | Convert Hz to rad/ns: 2π·f·10⁻⁹ |
| `to_model(n_cav=12, n_tr=3)` | Build `DispersiveTransmonCavityModel` with all params in rad/ns |

**Note:** `DeviceParameters.to_model()` uses rad/ns units internally, which differs
from the library's standard rad/s convention. This is specific to the tomography
device-parameter workflow.

### Tomography Protocol

**Module path:** `cqed_sim.tomo.protocol`

#### QubitPulseCal

```python
@dataclass
class QubitPulseCal:
    amp90: float                       # Amplitude for π/2 rotation
    y_phase: float = np.pi / 2        # Relative Y-gate phase
    drag: float = 0.0
    detuning: float = 0.0
    duration_ns: float = 16.0

    @staticmethod
    def nominal() -> QubitPulseCal     # Default calibration
    def amp(self, label: str) -> float # "x90", "y90", "x180", "y180", "i"
    def phase(self, label: str) -> float
```

#### All-XY Calibration

```python
ALL_XY_21: list[tuple[str, str]]  # 21 standard gate pairs

def run_all_xy(
    model, cal: QubitPulseCal, dt_ns=0.2, frame=None, noise=None,
) -> dict[str, np.ndarray]  # {"measured_z", "expected_z", "rms_error"}

def autocalibrate_all_xy(
    model, initial_cal, dt_ns=0.2, max_iter=12, target_rms=0.08,
) -> tuple[QubitPulseCal, dict]
```

#### Fock-Resolved Tomography

```python
def run_fock_resolved_tomo(
    model, state_prep: Callable, n_max: int, cal: QubitPulseCal,
    tag_duration_ns=1000.0, tag_amp=0.0015, dt_ns=1.0,
    noise=None, ideal_tag=False, pre_rotation_mode="pulse",
    leakage_cal=None, unmix_lambda=1e-2,
) -> FockTomographyResult
```

| Field | Type | Description |
|---|---|---|
| `v_hat` | `dict[str, ndarray]` | `"x"`, `"y"`, `"z"` → raw Bloch components per Fock level |
| `p_n` | `ndarray` | Fock populations |
| `conditioned_bloch` | `dict[int, ndarray]` | n → [x, y, z] per Fock level |
| `v_rec` | `dict[str, ndarray] \| None` | Leakage-corrected vectors |

```python
def selective_pi_pulse(n, t0_ns, duration_ns, amp, model, drag=0.0) -> Pulse
```

Creates Gaussian π-pulse targeting Fock manifold n. Carrier frequency:
`−n·chi` (selective detuning).

```python
def true_fock_resolved_vectors(state, n_max) -> dict[str, ndarray]
```

Exact Bloch vectors by projecting onto each Fock manifold.

```python
def calibrate_leakage_matrix(
    model, n_max, alphas, bloch_states, cal, ...
) -> tuple[ndarray, dict[str, ndarray], float]
```

Returns (W matrix, bias dict, condition number) for the linear model
v_hat = W·v_true + b.

---

## 14. Observables

**Module:** `cqed_sim.observables`

### Bloch (`observables.bloch`)

Re-exports from `sim.extractors`: `reduced_qubit_state`, `reduced_cavity_state`,
`bloch_xyz_from_joint`, `cavity_moments`.

### Fock-Resolved Diagnostics (`observables.fock`)

| Function | Signature | Description |
|---|---|---|
| `fock_resolved_bloch_diagnostics(track, max_n, probability_threshold)` | `-> dict` | Bloch vectors conditioned on Fock n across snapshot sequence |
| `relative_phase_family_diagnostics(track, max_n, probability_threshold, unwrap, coherence_threshold)` | `-> dict` | Ground and excited phase families with optional unwrapping |
| `conditional_phase_diagnostics(track, max_n, ...)` | `-> dict` | Excited-family phase only (shortcut) |
| `relative_phase_debug_values(state, max_n, ...)` | `-> dict` | Phase analysis for a single state |
| `wrapped_phase_error(simulated, ideal)` | `-> dict \| ndarray` | Wrapped phase difference |

### Phase Diagnostics (`observables.phases`)

```python
def relative_phase_diagnostics(track, max_n, threshold, unwrap=False) -> dict
```

Flattened phase traces with labels like `"|g,0>"`, `"|e,1>"`.
Returns `{"labels", "traces", "amplitudes", "phase_mode", ...}`.

### Trajectories (`observables.trajectories`)

```python
def bloch_trajectory_from_states(
    states: list, conditioned_n_levels=None, probability_threshold=1e-8,
) -> dict
```

Returns `{"x", "y", "z", "conditioned": {n: {"x", "y", "z", "probability", "valid"}}}`.

### Weakness Metrics (`observables.weakness`)

| Function | Description |
|---|---|
| `attach_weakness_metrics(reference_track, track)` | Add `wigner_negativity`, `fidelity_weakness_vs_a` arrays to track |
| `comparison_metrics(track_a, track_b)` | `{"x_rmse", "y_rmse", "z_rmse", "n_rmse", "final_fidelity"}` |

### Wigner (`observables.wigner`)

| Function | Description |
|---|---|
| `cavity_wigner(rho_c, ...)` | Re-export from extractors |
| `selected_wigner_snapshots(track, stride)` | Subsample Wigner snapshots (always includes first and last) |
| `wigner_negativity(snapshot)` | max(0.5(∫\|W\| − 1), 0). Returns NaN if wigner is None. |

---

## 15. Operators

**Module:** `cqed_sim.operators`

### Basic Operators (`operators.basic`)

| Function | Returns | Description |
|---|---|---|
| `sigma_x()` | `Qobj` | Pauli X |
| `sigma_y()` | `Qobj` | Pauli Y |
| `sigma_z()` | `Qobj` | Pauli Z |
| `tensor_qubit_cavity(op_q, op_c)` | `Qobj` | qt.tensor(op_q, op_c) |
| `embed_qubit_op(op_q, n_cav)` | `Qobj` | op_q ⊗ I_cav |
| `embed_cavity_op(op_c, n_tr=2)` | `Qobj` | I_qubit ⊗ op_c |
| `build_qubit_state(label)` | `Qobj` | `"g"`, `"e"`, `"+x"`, `"-x"`, `"+y"`, `"-y"` → ket |
| `joint_basis_state(n_cav_dim, qubit_label, n)` | `Qobj` | \|qubit_label⟩ ⊗ \|n⟩ |
| `as_dm(state)` | `Qobj` | State → density matrix |
| `purity(state)` | `float` | Tr(ρ²) |

### Cavity Operators (`operators.cavity`)

| Function | Returns | Description |
|---|---|---|
| `destroy_cavity(n_cav_dim)` | `Qobj` | Lowering operator a |
| `create_cavity(n_cav_dim)` | `Qobj` | Raising operator a† |
| `number_operator(n_cav_dim)` | `Qobj` | a†a |
| `fock_projector(n_cav_dim, n)` | `Qobj` | \|n⟩⟨n\| |

**Performance note:** All Pauli and cavity operator factories use `@lru_cache`,
so repeated calls with the same dimensions return the same cached `Qobj`
rather than re-creating the object each time.

---

## 16. Plotting

**Module:** `cqed_sim.plotting`

All plotting functions return matplotlib `Figure` objects and are designed for
diagnostic visualization.

### Bloch Track (`plotting.bloch_plots`)

| Function | Description |
|---|---|
| `plot_bloch_track(track, title, label_stride)` | Bloch (X,Y,Z) evolution vs gate index with gate-type background shading |
| `add_gate_type_axis(ax, track, label_stride)` | Add top axis with gate type labels |

**Constant:** `GATE_COLORS = {"INIT": "black", "Displacement": "tab:blue", "Rotation": "tab:orange", "SQR": "tab:green"}`

### Calibration (`plotting.calibration_plots`)

| Function | Description |
|---|---|
| `plot_sqr_calibration_result(result)` | 4-panel: d_lambda, d_alpha, d_omega_hz, loss vs Fock level |

### Energy Levels (`plotting.energy_levels`)

| Function | Description |
|---|---|
| `plot_energy_levels(spectrum, max_levels=None, energy_scale=1.0, energy_unit_label="rad/s", annotate=True, title=None, ax=None)` | Ladder-style plot of vacuum-referenced energy levels |

### Gate Diagnostics (`plotting.gate_diagnostics`)

| Function | Description |
|---|---|
| `plot_fock_resolved_bloch_overlay(simulated, ideal, track, component, ...)` | Heatmap overlay of Fock-resolved Bloch component |
| `plot_fock_resolved_bloch_grouped_bars(...)` | Grouped bar chart per Fock level |
| `plot_phase_heatmap_overlay(...)` | Phase heatmap comparison |
| `plot_phase_overlay_lines(...)` | Line plot of relative phases |
| `plot_phase_error_heatmap(...)` | Phase error heatmap |
| `plot_phase_error_track(...)` | Wrapped phase error vs gate index |
| `plot_gate_bloch_trajectory_overlay(simulated, ideal)` | 3-panel time-domain Bloch within a gate |
| `plot_gate_bloch_trajectory_error(simulated, ideal)` | Error trajectory within a gate |
| `plot_combined_gate_diagnostics(...)` | Combined 4×3 layout with all diagnostic panels |
| `save_figure(fig, output_dir, filename, dpi=160)` | Save figure to disk |

### Phase Plots (`plotting.phase_plots`)

| Function | Description |
|---|---|
| `plot_relative_phase_track(track, max_n, threshold, unwrap=False, label_stride=1)` | Phase line plot with Fock coloring |

### Weakness Plots (`plotting.weakness_plots`)

| Function | Description |
|---|---|
| `plot_component_comparison(a, b, c, d=None, label_stride=1)` | 3-panel Bloch comparison |
| `plot_cavity_population_comparison(a, b, c, d=None, label_stride=1)` | Cavity ⟨n⟩ comparison |
| `plot_weakness(b, c, reference, d=None, label_stride=1)` | Wigner negativity + fidelity weakness |
| `print_mapping_rows(track)` | Print track metadata |

### Wigner Grids (`plotting.wigner_grids`)

| Function | Description |
|---|---|
| `plot_wigner_grid(track, title, stride, max_cols=None, show_colorbar=False)` | Grid of Wigner snapshots at gate indices |

---

## 17. Unitary Synthesis

**Module:** `cqed_sim.unitary_synthesis`

Flexible gate-sequence synthesis for matrix-defined primitives, model-backed
waveform primitives, unitary targets, state-mapping targets, and Phase 2
constraint-aware/robust optimization.

> **Convention note:** The synthesis drift-phase layer matches the runtime
> dispersive/Kerr convention. Model-backed waveform primitives reuse the same
> `Pulse`, `SequenceCompiler`, and `cqed_sim.sim` runtime stack, including the
> waveform sign convention `Pulse.carrier = -omega_transition(frame)`.

---

### 17.1 Subspace

**Module path:** `cqed_sim.unitary_synthesis.subspace.Subspace`

Frozen dataclass that selects a subspace of the full qubit-cavity Hilbert space.
All synthesis targets, metrics, and leakage computations operate relative to a
`Subspace` instance.

```python
@dataclass(frozen=True)
class Subspace:
    full_dim: int
    indices: tuple[int, ...]
    labels: tuple[str, ...]
    kind: str = "custom"
    metadata: dict[str, int | str] | None = None
```

#### Constructors

| Class method | Signature | Description |
|---|---|---|
| `Subspace.qubit_cavity_block(n_match, n_cav=None)` | `(int, int\|None) -> Subspace` | Selects the block spanned by ground/excited states for Fock levels 0…n_match. `n_cav` defaults to `n_match+1`. |
| `Subspace.cavity_only(n_match, qubit="g", n_cav=None)` | `(int, str, int\|None) -> Subspace` | Selects cavity Fock levels 0…n_match in the ground (`"g"`) or excited (`"e"`) qubit sector. |
| `Subspace.custom(full_dim, indices, labels=None)` | `(int, Iterable[int], Iterable[str]\|None) -> Subspace` | Arbitrary subspace by explicit index list. |

#### Properties and methods

| Name | Returns | Description |
|---|---|---|
| `dim` | `int` | Number of basis vectors in the subspace |
| `projector()` | `ndarray` | Full-space projector matrix (full_dim × full_dim) |
| `embed(vec_sub)` | `ndarray` | Embed a subspace vector into the full Hilbert space |
| `extract(vec_full)` | `ndarray` | Extract subspace components from a full-space vector |
| `restrict_operator(op_full)` | `ndarray` | Project a full-space operator onto the subspace |
| `per_fock_blocks()` | `list[slice]` | Block slices for `qubit_cavity_block` subspaces (one 2-element slice per Fock level, spanning the ground and excited entries for that level). Raises if `kind != "qubit_cavity_block"`. |

**Kind values:** `"qubit_cavity_block"`, `"cavity_only"`, `"custom"`.

---

### 17.2 System Backends

**Module path:** `cqed_sim.unitary_synthesis.systems`

```python
class QuantumSystem:
    """Abstract backend interface used by UnitarySynthesizer."""
    def hilbert_dimension(*, sequence, primitive, subspace, target) -> int | None
    def subsystem_dimensions(*, sequence, full_dim, subspace) -> tuple[int, ...]
    def infer_n_cav(*, sequence, full_dim, subspace) -> int | None
    def configure_sequence(sequence, *, subspace) -> GateSequence
    def build_sequence_from_gateset(gateset, *, subspace, ...) -> GateSequence
    def simulate_unitary(sequence, *, backend, **settings) -> np.ndarray
    def simulate_state(sequence, psi0, *, backend, **settings) -> qt.Qobj
    def simulate_states(sequence, states, *, backend, **settings) -> list[qt.Qobj]
    def simulate_sequence(sequence, subspace, *, backend, ...) -> SimulationResult
    def simulate_primitive_unitary(primitive, *, settings) -> np.ndarray
    def simulate_primitive_states(primitive, states, *, settings) -> list[qt.Qobj]
    def runtime_model() -> Any | None
    def with_model(model) -> QuantumSystem
    def to_record() -> dict[str, Any]

@dataclass(frozen=True, kw_only=True)
class CQEDSystemAdapter(QuantumSystem):
    model: Any   # any cQED model with subsystem_dims, operators(), etc.

@dataclass(frozen=True)
class GenericQuantumSystem(QuantumSystem):
    dimension: int | None = None
```

**Key points:**

- `UnitarySynthesizer` talks to a `QuantumSystem` backend rather than directly
  to a raw cQED model.
- `CQEDSystemAdapter` wraps any cQED model and provides waveform primitive simulation
  via the standard `Pulse`/`SequenceCompiler`/`cqed_sim.sim` runtime stack.
  `simulate_primitive_unitary` caches per-parameter operator results.
- `model=...` in `UnitarySynthesizer` is a compatibility shortcut auto-wrapped to
  `CQEDSystemAdapter(model=...)`.
- `GenericQuantumSystem` supports synthesis over any square-matrix gate set
  without a physical cQED model.

**`resolve_quantum_system` helper:**

```python
def resolve_quantum_system(
    *,
    system: QuantumSystem | None = None,
    model: Any | None = None,
    subspace: Subspace | None = None,
    primitives: Sequence[Any] | None = None,
    gateset: Sequence[str] | None = None,
) -> QuantumSystem
```

Selects the right backend automatically: `CQEDSystemAdapter` if `model` is provided,
a cQED-aware ideal system if cQED gate types are detected in `primitives`/`gateset`,
or `GenericQuantumSystem` otherwise.

---

### 17.3 Core Targets

```python
@dataclass(frozen=True)
class TargetUnitary:
    matrix: np.ndarray
    ignore_global_phase: bool = False
    allow_diagonal_phase: bool = False
    phase_blocks: tuple[tuple[int, ...], ...] | None = None
    probe_states: tuple[qt.Qobj | np.ndarray, ...] = ()
    open_system_probe_strategy: str = "basis_plus_uniform"
```

```python
class TargetStateMapping:
    def __init__(
        self,
        *,
        initial_states=None,
        target_states=None,
        initial_state=None,
        target_state=None,
        weights=None,
    ): ...
```

```python
class TargetReducedStateMapping:
    def __init__(
        self,
        *,
        initial_states,
        target_states,
        retained_subsystems,
        subsystem_dims=None,
        weights=None,
    ): ...
```

```python
@dataclass(frozen=True)
class TargetIsometry:
    matrix: np.ndarray
    input_states: tuple[qt.Qobj | np.ndarray, ...] = ()
    weights: tuple[float, ...] = ()
```

```python
class TargetChannel:
    def __init__(
        self,
        *,
        choi=None,
        superoperator=None,
        kraus_operators=None,
        unitary=None,
        retained_subsystems=None,
        subsystem_dims=None,
        environment_state=None,
        enforce_cptp=False,
    ): ...
```

```python
class ObservableTarget:
    def __init__(
        self,
        *,
        initial_states=None,
        observables=None,
        target_expectations=None,
        initial_state=None,
        observable=None,
        target_expectation=None,
        state_weights=None,
        observable_weights=None,
    ): ...
```

```python
@dataclass(frozen=True)
class TrajectoryCheckpoint:
    step: int
    target_states: tuple[qt.Qobj | np.ndarray, ...] = ()
    observables: tuple[qt.Qobj | np.ndarray, ...] = ()
    target_expectations: np.ndarray | Sequence = ()
    weight: float = 1.0
    state_weights: tuple[float, ...] = ()
    observable_weights: tuple[float, ...] = ()
    label: str | None = None


class TrajectoryTarget:
    def __init__(self, *, initial_states, checkpoints, state_weights=None): ...
```

**`TargetUnitary` notes:**

- Validates unitarity on construction; raises `ValueError` if the matrix is not unitary within `atol=1e-8`.
- Phase handling supports `ignore_global_phase`, `allow_diagonal_phase`, and per-index `phase_blocks`.
- Open-system targets use propagated probe states (strategy from `open_system_probe_strategy`).

**`TargetUnitary` methods:**

| Method | Signature | Description |
|---|---|---|
| `dim` | property → `int` | Matrix dimension |
| `resolved_gauge(fallback="global")` | `(str) -> str` | Returns `"block"`, `"diagonal"`, `"global"`, or `fallback` based on flags |
| `resolved_blocks(subspace=None, fallback="global")` | `-> tuple[tuple[int,...], ...] \| None` | Phase blocks for the block gauge, auto-derived from subspace if not set explicitly |
| `resolved_probe_pairs(*, full_dim, subspace=None)` | `-> (list[Qobj], list[Qobj])` | Returns `(initial_states, target_states)` for fidelity evaluation |

**`TargetStateMapping` notes:**

- Accepts plural (`initial_states`, `target_states`) or singular (`initial_state`, `target_state`) arguments; not both.
- Optional `weights` per state pair; uniform if omitted.
- `TargetReducedStateMapping` compares only retained subsystems after partial trace, allowing spectator or environment dynamics to be ignored deliberately.
- `TargetIsometry` matches only the relevant logical columns of a larger operation, which is useful for encoding and injection workflows.
- `TargetChannel` supports process-style matching from a unitary, Kraus list, superoperator, or Choi matrix and can reconstruct reduced subsystem channels from a fixed environment state.

**Additional task targets:**

- `ObservableTarget` matches weighted observable expectations on a relevant state ensemble.
- `TrajectoryTarget` compares intermediate protocol checkpoints using state targets, observable targets, or both.

**`TargetStateMapping` methods:**

| Method | Signature | Description |
|---|---|---|
| `resolved_pairs(*, full_dim, subspace=None)` | `-> (list[Qobj], list[Qobj], ndarray)` | Returns `(initial, targets, weights)` with weights normalized to sum to 1 |

**Convenience target builders** (`cqed_sim.unitary_synthesis.targets`):

| Function | Signature | Description |
|---|---|---|
| `make_target(name, n_match, variant="analytic", **kwargs)` | `(str, int, str, ...) -> ndarray` | Build a named target matrix. `name`: `"easy"`, `"ghz"`, `"cluster"`. `variant="mps"` tries to load from reference files first. |
| `make_easy_target(n_match)` | `(int) -> ndarray` | Easily realizable SQR+SNAP product unitary for dimension `2*(n_match+1)`. |
| `make_mps_like_target(kind, n_match, **kwargs)` | `(str, int) -> ndarray` | Analytic GHZ or cluster-state transfer matrix for the qubit-cavity block. |
| `coerce_target(target)` | `(ndarray\|Qobj\|SynthesisTarget) -> SynthesisTarget` | Wrap a raw matrix in `TargetUnitary`. |

For `"cluster"` targets, `kwargs["which"]` selects `"u1"` (default) or `"u2"` (includes a qubit Ry rotation).

---

### 17.4 Gate Primitives and Sequences

**Module path:** `cqed_sim.unitary_synthesis.sequence`

#### PrimitiveGate

```python
@dataclass
class PrimitiveGate:
    name: str
    duration: float
    optimize_time: bool = True
    time_bounds: tuple[float, float] = (1e-9, 1e-6)
    duration_ref: float = ...        # reference duration for time-group scaling
    time_group: Any | None = None    # gates sharing a group share one time parameter
    time_policy_locked: bool = False
    hilbert_dim: int | None = None
    matrix: np.ndarray | Callable | None = None
    waveform: Callable | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_bounds: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Supported simulation modes:

- **Fixed matrix:** `matrix` is a constant `ndarray`.
- **Parameterized matrix:** `matrix` is `(params: dict, model: Any) -> ndarray`.
- **Waveform-backed:** `waveform` is `(params: dict, model: Any) -> pulses_or_tuple`.
  May return `list[Pulse]`, `(pulses, drive_ops)`, `(pulses, drive_ops, meta)`, or a dict with those keys.

#### cQED Sequence Gate Types

All gate types share the common timing fields (`name`, `duration`, `optimize_time`,
`time_bounds`, `duration_ref`, `time_group`, `time_policy_locked`) plus gate-specific params:

| Class | Gate-specific fields | Physics |
|---|---|---|
| `QubitRotation` | `theta: float`, `phi: float` | Qubit rotation R(theta, phi) |
| `Displacement` | `alpha: complex` | Cavity displacement D(alpha) |
| `ConditionalDisplacement` | `alpha: complex` or `alpha_g: complex`, `alpha_e: complex` | ECD-style ideal conditional displacement; symmetric or asymmetric qubit-conditioned cavity translation |
| `SQR` | `theta_n`, `phi_n`, `tones`, `tone_freqs`, `include_conditional_phase`, `drift_model` | Selective qubit rotation (multi-tone Gaussian) |
| `CavityBlockPhase` | `phases: list[float]`, `fock_levels: tuple[int, ...]` | Ideal cavity-only logical block-phase gate acting identically on both qubit states |
| `SNAP` | `phases: list[float]` | Number-selective phase gate |
| `JaynesCummingsExchange` | `coupling: float`, `phase: float` | Native red-sideband / SWAP-like exchange between `|e,n>` and `|g,n+1>` |
| `BlueSidebandExchange` | `coupling: float`, `phase: float` | Native blue-sideband exchange between `|g,n>` and `|e,n+1>` |
| `ConditionalPhaseSQR` | `phases_n: list[float]`, `drift_model: DriftPhaseModel` | Conditional phase via free-evolution SQR block |
| `FreeEvolveCondPhase` | `drift_model: DriftPhaseModel` | Free-evolution conditional phase with no explicit drive pulse |

#### GateSequence

```python
@dataclass
class GateSequence:
    gates: list[GateBase]
    n_cav: int | None = None    # number of cavity Fock levels
    full_dim: int | None = None # full Hilbert-space dimension (= 2 * n_cav)
```

Key methods: `serialize()`, `get_parameter_vector()`, `get_time_raw_vector(active_only)`,
`sync_time_params_from_gates()`, `unitary(backend, backend_settings)`,
`propagate_states(states, backend, backend_settings)`.

#### GateTimeParam

```python
@dataclass
class GateTimeParam:
    group: Any         # time-group identifier
    t_raw: float       # unconstrained variable (mapped via sigmoid to [t_min, t_max])
    t_min: float
    t_max: float
    locked: bool = False
```

#### DriftPhaseModel and drift helpers

```python
@dataclass(frozen=True)
class DriftPhaseModel:
    chi: float = 0.0
    kerr: float = 0.0
    chi_higher: tuple[float, ...] = ()
    kerr_higher: tuple[float, ...] = ()
```

| Function | Signature | Description |
|---|---|---|
| `drift_phase_table(model, n_cav)` | `(DriftPhaseModel, int) -> ndarray` | Per-Fock-level phase accumulated per unit time |
| `drift_phase_from_hamiltonian(H, n_cav, dt)` | `(Qobj, int, float) -> ndarray` | Extract drift phases numerically |
| `drift_phase_unitary(phases_per_second, duration, n_cav)` | `(ndarray, float, int) -> ndarray` | Full-space drift unitary matrix |
| `drift_hamiltonian_qobj(model, n_cav)` | `(DriftPhaseModel, int) -> Qobj` | Static Hamiltonian as QuTiP Qobj |

---

### 17.5 Waveform Bridge

**Module path:** `cqed_sim.unitary_synthesis.waveform_bridge`

Bridges the synthesis gate representation (`QubitRotation`, `Displacement`, `SQR`)
to waveform-backed `PrimitiveGate` objects. This connects the synthesis optimizer
to the full `Pulse`/`SequenceCompiler`/`cqed_sim.sim` runtime stack so that
`CQEDSystemAdapter` can run real-waveform simulations during optimization.

```python
def waveform_primitive_from_gate(
    gate: QubitRotation | Displacement | SQR,
    *,
    index: int = 0,
    frame: FrameSpec | None = None,
    calibration: Any | None = None,
    config: Mapping[str, Any] | None = None,
    hilbert_dim: int | None = None,
) -> PrimitiveGate
```

| Parameter | Description |
|---|---|
| `gate` | Source gate. `QubitRotation`, `Displacement`, or `SQR` only; other types raise `TypeError`. |
| `index` | Position in the sequence; forwarded to the underlying `io.gates` constructor. |
| `frame` | `FrameSpec` used for carrier frequency computation; stored in gate metadata. |
| `calibration` | Optional SQR calibration result forwarded to `build_sqr_multitone_pulse`. |
| `config` | Optional config dict merged over the following defaults. |
| `hilbert_dim` | Overrides the inferred Hilbert-space dimension. |

**Default config keys:**

| Key | Default | Description |
|---|---|---|
| `rotation_sigma_fraction` | `0.18` | Gaussian sigma as fraction of duration for rotation pulses |
| `sqr_sigma_fraction` | `0.18` | Same for SQR multi-tone pulses |
| `sqr_theta_cutoff` | `1e-8` | Amplitude threshold below which SQR tones are dropped |
| `use_rotating_frame` | `False` | Rotating-frame carrier correction for SQR tones |
| `fock_fqs_hz` | `None` | Empirical per-Fock-level frequencies (Hz) for SQR |

The returned `PrimitiveGate` has `parameters` (current angles/amplitude + duration),
`parameter_bounds` (angle range ±2π, duration within `time_bounds`), and
`metadata["waveform_family"]` (`"rotation_gaussian"`, `"displacement_square"`, or
`"sqr_multitone_gaussian"`).

**Time bounds:** If the source gate has a `time_bounds` attribute it is used directly;
otherwise bounds default to `[max(T × 0.25, 1 ns), max(T × 4, lower + 1 ns)]` where
`T` is `gate.duration`.

```python
def waveform_sequence_from_gates(
    sequence: GateSequence,
    *,
    frame: FrameSpec | None = None,
    calibration: Any | None = None,
    config: Mapping[str, Any] | None = None,
    hilbert_dim: int | None = None,
) -> GateSequence
```

Calls `waveform_primitive_from_gate` on each gate and returns a new `GateSequence`
with the same `n_cav` and `full_dim`. `hilbert_dim` defaults to `sequence.full_dim`.

**Usage:**

```python
from cqed_sim.unitary_synthesis import (
    GateSequence, QubitRotation, SQR, CQEDSystemAdapter, Subspace,
)
from cqed_sim.unitary_synthesis.backends import simulate_sequence
from cqed_sim.unitary_synthesis.waveform_bridge import waveform_sequence_from_gates

seq = GateSequence(gates=[...], n_cav=3)
wf_seq = waveform_sequence_from_gates(seq, frame=frame)
result = simulate_sequence(
    wf_seq, subspace,
    backend="pulse",
    system=CQEDSystemAdapter(model=model),
    state_inputs=[psi0],
    dt=4e-9,
    frame=frame,
    noise=NoiseSpec(t1=60e-6, tphi=80e-6),
)
```

---

### 17.6 Phase 2 Configuration Objects

```python
@dataclass(frozen=True)
class SynthesisConstraints:
    max_amplitude: float | None = None
    max_duration: float | None = None
    max_primitives: int | None = None
    allowed_primitive_counts: tuple[int, ...] = ()
    smoothness_penalty: bool = False
    smoothness_weight: float = 1.0       # penalty weight for smoothness term
    max_bandwidth: float | None = None
    bandwidth_weight: float = 1.0        # penalty weight for bandwidth term
    forbidden_parameter_ranges: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    forbidden_range_weight: float = 1.0  # penalty weight for forbidden-range violations
    duration_mode: str = "penalty"       # "penalty" or "hard"
```

```python
@dataclass(frozen=True)
class LeakagePenalty:
    weight: float = 0.0
    allowed_subspace: Subspace | Sequence[int] | None = None
    metric: str = "worst"
    checkpoint_weight: float = 0.0
    checkpoints: tuple[int, ...] = ()
```

```python
@dataclass(frozen=True)
class MultiObjective:
    fidelity_weight: float = 1.0
    task_weight: float | None = None
    leakage_weight: float = 0.0
    duration_weight: float = 0.0
    gate_count_weight: float = 0.0
    pulse_power_weight: float = 0.0
    robustness_weight: float = 0.0
    smoothness_weight: float = 0.0
    hardware_penalty_weight: float = 1.0
    mode: str = "weighted_sum"   # currently only "weighted_sum" is supported
```

```python
@dataclass(frozen=True)
class ExecutionOptions:
    engine: str = "auto"
    fallback_engine: str = "legacy"
    device: str = "auto"
    use_fast_path: bool = True
    jit: bool = True
    vectorized_candidates: bool = True
    candidate_batch_size: int = 0
    cache_fast_path: bool = True
```

```python
ParameterDistribution(
    sample_count=4,
    aggregate="mean",
    chi=Normal(-2.8e6, 0.05e6),
    kerr=Uniform(-3200.0, -2800.0),
)
```

### 17.7 UnitarySynthesizer

```python
class UnitarySynthesizer:
    def __init__(
        self,
        ...,
        model: Any | None = None,
        system: QuantumSystem | None = None,
        synthesis_constraints: SynthesisConstraints | Mapping | None = None,
        leakage_penalty: LeakagePenalty | Mapping | None = None,
        objectives: MultiObjective | Mapping | None = None,
        parameter_distribution: ParameterDistribution | None = None,
        execution: ExecutionOptions | Mapping | None = None,
        warm_start: str | Path | Mapping | SynthesisResult | None = None,
        ...,
    )
```

```python
def fit(target=None, init_guess="heuristic", multistart=1, maxiter=300) -> SynthesisResult
```

```python
def explore_pareto(weight_sets, *, target=None, init_guess="heuristic", multistart=1, maxiter=300) -> ParetoFrontResult
```

Important behavior:

- Closed-system unitary targets still use direct subspace-unitary fidelity.
- Open-system unitary targets automatically switch to probe-state fidelity.
- `ObservableTarget`, `TrajectoryTarget`, `TargetReducedStateMapping`, `TargetIsometry`, and `TargetChannel` let the task be defined directly on relevant outputs, reduced states, logical columns, checkpointed trajectories, or full process actions.
- `system=...` is the preferred future-facing entry point for backend integration.
- `model=...` remains supported for cQED workflows and is auto-wrapped.
- `parameter_distribution` samples model variants and folds them into the synthesis objective.
- `execution=ExecutionOptions(...)` selects the legacy evaluator or the accelerated ideal closed-system evaluator. The fast path currently covers closed-system `backend="ideal"` unitary, state-mapping, isometry, observable, and trajectory objectives. Reduced-state and channel objectives fall back automatically with a recorded reason.
- `warm_start` accepts a saved payload, mapping, or previous `SynthesisResult`.
- `optimizer` supports `auto`, `nelder_mead`, `powell`, `bfgs`, `l_bfgs_b`, `differential_evolution`, and `cma_es`.

### 17.8 Results and Metrics

#### Synthesis SimulationResult

**Module path:** `cqed_sim.unitary_synthesis.backends`

> **Not the same class** as `cqed_sim.sim.runner.SimulationResult` documented
> in [§6.3](#63-simulationresult). This dataclass holds the propagated operator
> and subspace projection from a synthesis simulation backend.

```python
@dataclass
class SimulationResult:
    full_operator: np.ndarray | None           # Full-Hilbert-space unitary
    subspace_operator: np.ndarray | None       # Projected subspace unitary
    state_outputs: list[qt.Qobj] | None = None # Propagated output states
    metrics: dict[str, float] = field(default_factory=dict)
    backend: str = "ideal"                     # Backend that produced the result
    settings: dict[str, Any] = field(default_factory=dict)
```

#### simulate_sequence (synthesis)

**Module path:** `cqed_sim.unitary_synthesis.backends`

```python
def simulate_sequence(
    sequence: GateSequence,
    subspace: Subspace | None,
    backend: str = "ideal",
    target_subspace: np.ndarray | None = None,
    leakage_weight: float = 0.0,
    gauge: str = "global",
    block_slices: Sequence[slice | Sequence[int] | np.ndarray] | None = None,
    state_inputs: Sequence[qt.Qobj | np.ndarray] | None = None,
    need_operator: bool = True,
    system: Any | None = None,
    **backend_settings: Any,
) -> SimulationResult
```

Runs a gate sequence through the specified backend and returns the propagated
operator and metrics. The `"ideal"` backend multiplies out the ideal gate matrices;
the `"cqed"` backend delegates to the full cQED pulse simulator via a
`CQEDSystemAdapter`.

#### make_run_report

**Module path:** `cqed_sim.unitary_synthesis.reporting`

```python
def make_run_report(
    base_report: dict[str, Any],
    subspace_operator: np.ndarray,
) -> dict[str, Any]
```

Augments a base report dict with unitarity error, norm, and dimension fields
derived from the subspace operator.

#### SynthesisResult

```python
@dataclass
class SynthesisResult:
    success: bool
    objective: float
    sequence: GateSequence
    simulation: SimulationResult
    report: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    history_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    progress_schema_version: int = PROGRESS_SCHEMA_VERSION
```

| Method | Signature | Description |
|---|---|---|
| `save(path, *, include_history=True)` | `(str\|Path, ...) -> Path` | Serialize full result to JSON |
| `save_history(path)` | `(str\|Path) -> Path` | Save optimization history as JSON |
| `save_history_csv(path)` | `(str\|Path) -> Path` | Save optimization history as CSV |
| `to_payload(*, include_history=True)` | `-> dict` | Full JSON-serializable payload (schema version 2) |
| `diagnostics()` | `-> dict` | Compact diagnostic dict (sequence, report, history) |
| `plot_convergence(what, fidelity_what)` | `-> Figure` | 2-panel convergence plot (objective + fidelity) |

#### ParetoFrontResult

```python
@dataclass
class ParetoFrontResult:
    results: list[SynthesisResult]
    nondominated_indices: list[int]

    def nondominated(self) -> list[SynthesisResult]
```

#### Metric Functions

**Module path:** `cqed_sim.unitary_synthesis.metrics`

```python
@dataclass(frozen=True)
class LeakageMetrics:
    average: float
    worst: float
    per_probe: tuple[float, ...]

@dataclass(frozen=True)
class LogicalBlockPhaseDiagnostics:
    block_phases_rad: tuple[float, ...]
    relative_block_phases_rad: tuple[float, ...]
    applied_correction_phases_rad: tuple[float, ...]
    best_fit_correction_phases_rad: tuple[float, ...]
    corrected_block_phases_rad: tuple[float, ...]
    corrected_relative_block_phases_rad: tuple[float, ...]
    residual_block_phases_rad: tuple[float, ...]
    rms_block_phase_error_rad: float
    block_gauge_fidelity: float
    corrected_block_gauge_fidelity: float
    best_fit_block_gauge_fidelity: float
```

| Function | Signature | Description |
|---|---|---|
| `subspace_unitary_fidelity(actual, target, gauge="global", block_slices=None)` | `(ndarray, ndarray, str, ...) -> float` | Subspace fidelity with gauge `"none"`, `"global"`, `"diagonal"`, or `"block"` |
| `leakage_metrics(full_operator, subspace, probes=None, n_jobs=1)` | `(ndarray, Subspace, ...) -> LeakageMetrics` | Compute average/worst leakage from subspace basis vectors |
| `logical_block_phase_diagnostics(actual, target, *, block_slices, applied_correction_phases=None)` | `(ndarray, ndarray, ...) -> LogicalBlockPhaseDiagnostics` | Gauge-fixed block-overlap phases, best-fit block-phase correction, and residual RMS after an optional applied correction |
| `state_leakage_metrics(states, subspace)` | `(Sequence, Subspace) -> LeakageMetrics` | Leakage from already-propagated output states |
| `state_mapping_metrics(outputs, targets, *, weights=None)` | `(Sequence, Sequence, ...) -> dict` | Weighted state fidelity and error metrics |
| `channel_action_metrics(actual_superoperator, actual_choi, *, target_choi, target_superoperator=None, trace_values=None)` | `-> dict` | Channel/process error, overlap, trace-preservation, and CP diagnostics |
| `objective_breakdown(actual_sub, target_sub, full_op, subspace, ...)` | `-> dict` | Full breakdown: fidelity, leakage, objective |
| `truncation_sanity_metrics(states, subspace)` | `(Sequence, Subspace) -> dict` | Retained-edge and outside-tail population diagnostics from propagated states |
| `operator_truncation_sanity_metrics(full_operator, subspace)` | `(ndarray, Subspace) -> dict` | Truncation diagnostics derived from logical basis columns of a full operator |
| `unitarity_error(op)` | `(ndarray) -> float` | Frobenius norm of U†U − I; 0 for exact unitaries |

`state_mapping_metrics` returns: `state_error_mean`, `state_error_max`,
`state_fidelity_mean`, `state_fidelity_min`, `weighted_state_error`,
`weighted_state_infidelity`, `objective`.

Channel-style targets additionally report `channel_overlap`, `channel_choi_error`,
`channel_superoperator_error`, `trace_preservation_error_mean`,
`trace_preservation_error_max`, `trace_loss_mean`, `trace_loss_worst`,
`choi_hermiticity_error`, `choi_min_eig`, and
`complete_positivity_violation`. All synthesis reports now also include
`truncation.retained_edge_population_*` and `truncation.outside_tail_population_*`
for cutoff sanity checks.

#### Progress Reporting

**Module path:** `cqed_sim.unitary_synthesis.progress`

| Class / Function | Description |
|---|---|
| `ProgressEvent` | Frozen dataclass: one event per optimizer iteration with `run_id`, `iteration`, `objective_total`, `objective_terms`, `metrics`, `best_so_far`, `params_summary` |
| `ProgressReporter` | Base class with `on_start`, `on_event`, `on_end` hooks |
| `NullReporter` | No-op reporter |
| `HistoryReporter` | Accumulates all events in memory; access via `.events`, `.starts`, `.ends` |
| `JupyterLiveReporter` | Displays live progress in Jupyter notebooks |
| `history_to_dataframe(history)` | Convert history list to pandas DataFrame |
| `plot_history(history_or_dict, what, ax, title)` | Line plot of a named history field |

`PROGRESS_SCHEMA_VERSION = 1` (current event schema version).

---

## 17A. Holographic Quantum Algorithms (`cqed_sim.quantum_algorithms.holographic_sim`)

Generic bond-space holographic estimators inspired by the channel / MPS picture
in `paper_summary/holographic_quantum_algorithms.pdf`.

> **Convention note:** This package is intentionally system-agnostic. It uses a
> `physical register ⊗ bond register` factorization, explicit observable
> insertion schedules, and repeated-channel burn-in. It is not hardcoded to a
> cQED tensor structure even though it lives inside the `cqed_sim` repository.

### Core Objects

```python
HolographicChannel.from_unitary(U, physical_dim=2, bond_dim=4)
HolographicChannel.from_kraus(kraus_ops)
HolographicChannel.from_right_canonical_mps(tensor)
HolographicChannel.from_mps_state(psi, site=0)
right_canonical_tensor_to_stinespring_unitary(tensor)
BondNoiseChannel.dephasing(bond_dim=4, probability=0.05)
BondNoiseChannel.amplitude_damping(bond_dim=4, probability=0.10)
BondNoiseChannel.depolarizing(bond_dim=4, probability=0.02)
```

```python
ObservableSchedule([
    {"step": 10, "operator": Z},
    {"step": 14, "operator": X},
], total_steps=20)
```

```python
sampler = HolographicSampler(channel, burn_in=BurnInConfig(steps=50), bond_noise=noise)
result = sampler.sample_correlator(schedule, shots=5000)
exact = sampler.enumerate_correlator(schedule)
```

Highlights:

- `HolographicChannel` is the main transfer-channel abstraction.
- `BondNoiseChannel` is the optional bond-only CPTP layer for dephasing, amplitude damping, depolarizing noise, or imported QuTiP superoperators.
- `PurifiedChannelStep` formalizes the prepare-apply-measure-reset primitive.
- `ObservableSchedule` and `ObservableInsertion` make measurement locations explicit.
- `HolographicSampler` supports Monte Carlo sampling with uncertainty estimates.
- `HolographicSampler.enumerate_correlator(...)` performs exact branch enumeration for small problems.
- `MatrixProductState` plus `HolographicChannel.from_right_canonical_mps(...)` and `HolographicChannel.from_mps_state(...)` connect the estimator path to right-canonical MPS tensors.
- `right_canonical_tensor_to_stinespring_unitary(...)` exposes the dense Stinespring completion used by the legacy finite-sequence API.

Built-in `BondNoiseChannel` constructors:

- `BondNoiseChannel.dephasing(...)` preserves bond-basis populations and damps off-diagonal coherences.
- `BondNoiseChannel.amplitude_damping(...)` relaxes computational-basis weight toward a designated target basis state; for `bond_dim=2` and `target_index=0` it matches the standard qubit amplitude-damping channel.
- `BondNoiseChannel.depolarizing(...)` implements `rho -> (1 - p) rho + p I / bond_dim` using a Weyl-operator Kraus representation.
- `BondNoiseChannel.from_qutip_super(...)` wraps existing QuTiP superoperators without creating a separate channel stack.

### Diagnostics and Results

- `channel_diagnostics(...)` reports unitarity/completeness/trace-preservation checks.
- `CorrelatorEstimate`, `ExactCorrelatorResult`, and `BurnInSummary` are serializable result objects.
- `branch_probability_error(...)`, `validate_trace_preservation(...)`, and `fixed_point_residual(...)` expose low-level validation hooks.

### Future-Facing Scaffolding

```python
HoloVQEObjective([...])
HoloQUADSProgram([...])
```

- `HoloVQEObjective` provides a minimal energy-objective wrapper built from correlator schedules.
- `HoloQUADSProgram` and `TimeSlice` provide a lightweight time-sliced interface for future holographic dynamics workflows.
- Example constructors in `cqed_sim.quantum_algorithms.holographic_sim.models` provide spin-inspired and partial-swap channels without making them mandatory.

---

## 17B. RL Control And System Identification (`cqed_sim.rl_control`, `cqed_sim.system_id`)

The RL-facing package adds a measurement-aware control layer on top of the existing cQED runtime rather than introducing a second simulator stack.

### Core Environment Objects

```python
HybridCQEDEnv(config)

HybridEnvConfig(
    system=HybridSystemConfig(...),
    task=coherent_state_preparation_task(),
    action_space=PrimitiveActionSpace(...),
    observation_model=build_observation_model("measurement_iq"),
    reward_model=build_reward_model("state"),
)
```

```python
@dataclass
class HybridSystemConfig:
    regime: Literal["reduced_dispersive", "full_pulse"]
    reduced_model: ReducedDispersiveModelConfig | None = None
    full_model: FullPulseModelConfig | None = None
    frame: FrameSpec = FrameSpec()
    use_model_rotating_frame: bool = True
    noise: NoiseSpec | None = None
    hardware: dict[str, HardwareConfig] = field(default_factory=dict)
    crosstalk_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    dt: float = 4.0e-9
    max_step: float | None = None
```

Highlights:

- `HybridCQEDEnv.reset(...)` supports deterministic seeding, task selection, and episode-level parameter randomization.
- `HybridCQEDEnv.step(action)` parses the configured action space, generates pulses, compiles distortions, propagates the system, optionally measures, and returns the next observation plus reward.
- `render_diagnostics()` exposes simulator-side debugging state such as reduced states, ancilla populations, Wigner diagnostics, compiled channels, segment metadata, pulse summaries, and the resolved frame/regime metadata.
- `estimate_metrics(...)` evaluates a baseline sequence or user-supplied policy/actions over multiple seeded rollouts and reports distribution summaries.

### Action, Observation, and Reward Layers

```python
ParametricPulseActionSpace(family="hybrid_block")
PrimitiveActionSpace(primitives=("qubit_gaussian", "cavity_displacement", "sideband", "wait", "measure", "reset"))
WaveformActionSpace(segments=16, channels=("qubit", "storage"))
```

```python
build_observation_model("ideal_summary")
build_observation_model("measurement_iq", mode="iq_mean")
build_observation_model("measurement_classifier_logits")
build_observation_model("measurement_outcome")
build_observation_model("gate_metrics")
```

```python
build_reward_model("state")
build_reward_model("gate")
build_reward_model("cat")
build_reward_model("measurement_proxy")
```

Highlights:

- Parametric actions are the default low-dimensional interface for RL studies.
- Primitive actions provide a more hierarchical interface aligned with validated bosonic/ancilla control blocks.
- Waveform actions are included as an explicit scaffold for future higher-bandwidth control studies.
- Observation builders support ideal simulator summaries, reduced-density views, measurement-like IQ summaries, counts, classifier logits, confusion-noisy outcome labels, and history stacking.
- Reward builders combine fidelity-like objectives with leakage, ancilla-return, control-cost penalties, and an explicit measurement-assignment reward for partially observed proxy objectives.

### Benchmark Tasks, Randomization, and Calibration Hooks

```python
benchmark_task_suite()
vacuum_preservation_task()
coherent_state_preparation_task()
fock_state_preparation_task()
storage_superposition_task()
even_cat_preparation_task()
odd_cat_preparation_task()
ancilla_storage_bell_task()
conditional_phase_gate_task()
```

```python
DomainRandomizer(
    model_priors_train={"chi": NormalPrior(...)}
)

CalibrationEvidence(...)
randomizer_from_calibration(evidence)
```

Highlights:

- The benchmark ladder currently covers vacuum preservation, coherent-state preparation, Fock-state preparation, storage-basis superpositions, even/odd cat preparation, ancilla-storage entanglement, and a reduced conditional-phase gate task.
- `DomainRandomizer` separates train and eval priors and records the sampled per-episode metadata.
- `cqed_sim.system_id` intentionally provides lightweight fit-then-randomize scaffolding rather than a full inference engine.

Approximation boundary:

- The reduced regime is intended for faster RL iteration with dispersive/Kerr structure.
- The full regime uses the generalized model stack for a richer multilevel pulse path.
- Measurement-conditioned stochastic state updates and SME trajectories are not yet implemented in this first pass.

---

## 17C. Optimal Control (`cqed_sim.optimal_control`)

The optimal-control package adds a solver-agnostic direct-control layer on top of the existing model, pulse, and simulator stack. The current backend is a dense closed-system GRAPE solver on a piecewise-constant propagation grid, with support for both plain piecewise-constant schedules and hardware-aware held-sample command parameterizations.

The intended workflow is:

1. Build a `ControlProblem` directly from dense operators or from an existing `cqed_sim` model.
2. Define state-transfer and/or unitary objectives plus penalties.
3. Solve with `GrapeSolver`.
4. Export the optimized `ControlSchedule` back into standard `Pulse` objects.
5. Replay through `SequenceCompiler` and `simulate_sequence(...)`.

### Core Problem Objects

```python
ControlTerm(...)
ControlSystem(...)
ControlProblem(...)
ModelControlChannelSpec(...)
ModelEnsembleMember(...)
```

Highlights:

- `ControlTerm` stores one Hermitian control operator, amplitude bounds, export metadata, and quadrature labeling.
- `ControlSystem` holds the drift Hamiltonian plus the ordered control operators for one system member.
- `ControlProblem` packages the parameterization, one or more `ControlSystem` members, objectives, penalties, and the ensemble aggregation mode.
- `ModelControlChannelSpec` describes how model-level targets such as `"qubit"`, `"storage"`, `"readout"`, `"sideband"`, `TransmonTransitionDriveSpec(...)`, or `SidebandDriveSpec(...)` are converted into control terms.
- `ModelEnsembleMember` supports robust optimization over multiple parameter-shifted systems.

### Parameterization and Hardware Pipeline

```python
PiecewiseConstantTimeGrid.uniform(steps=16, dt_s=4.0e-9)
PiecewiseConstantParameterization(...)
HeldSampleParameterization(...)
ControlSchedule(...)
HardwareModel(...)
```

Key behaviors:

- the propagation grid is always `PiecewiseConstantTimeGrid`
- parameter-space values can be distinct from the command waveform seen on that propagation grid
- `PiecewiseConstantParameterization` is the identity map from parameters to command waveform
- `HeldSampleParameterization` stores coarse AWG-like samples and applies sample-and-hold onto the propagation grid
- hard control bounds are tracked in the parameterization
- schedules can be flattened/unflattened for optimizers
- schedules can be exported into standard repository `Pulse` objects using either command or physical waveforms

The explicit control pipeline is:

\[
	heta \rightarrow u_{\mathrm{cmd}}(\theta) \rightarrow u_{\mathrm{phys}} = \mathcal{H}[u_{\mathrm{cmd}}] \rightarrow H(t; u_{\mathrm{phys}}).
\]

The helper

```python
resolve_control_schedule(...)
```

returns the parameter values, command waveform, physical waveform, and hardware diagnostics for a concrete schedule.

The first hardware-aware building blocks are:

- `HardwareModel(...)`
- `FirstOrderLowPassHardwareMap(...)`
- `BoundaryWindowHardwareMap(...)`
- `SmoothIQRadiusLimitHardwareMap(...)`

For exported rotating-frame pulses, the complex baseband coefficient follows the same runtime convention as the rest of `cqed_sim`:

\[
c(t) = I(t) - i Q(t).
\]

This is the compatibility bridge between the real-valued Hermitian quadrature Hamiltonians used inside the optimizer and the complex envelope convention used by the runtime pulse stack.

### Objectives

```python
StateTransferPair(...)
StateTransferObjective(...)
UnitaryObjective(...)

state_preparation_objective(...)
multi_state_transfer_objective(...)
objective_from_unitary_synthesis_target(...)
```

Supported task styles:

- single-state preparation
- multi-state transfer
- retained-subspace gate synthesis
- full truncated-space unitary synthesis
- phase-tolerant subspace objectives compatible with the logical-gauge ideas already used by `cqed_sim.unitary_synthesis`

`UnitaryObjective` evaluates unitary targets through weighted probe-state transfer pairs so the same machinery can represent direct full-space targets and restricted logical-subspace targets.

### Penalties

```python
AmplitudePenalty(weight=..., reference=...)
SlewRatePenalty(weight=...)
BoundPenalty(weight=..., lower_bound=..., upper_bound=...)
BoundaryConditionPenalty(weight=..., ramp_slices=...)
IQRadiusPenalty(amplitude_max=..., weight=...)
LeakagePenalty(subspace=..., weight=..., metric="average")
```

The current penalty layer supports:

- amplitude regularization,
- finite-difference slew regularization across adjacent slices,
- explicit soft bound penalties,
- zero-start / zero-end boundary penalties,
- radial I/Q envelope penalties,
- final-time leakage penalties outside a retained subspace.

For hardware-aware problems, penalties can be applied to one of three domains:

- parameter-space values,
- command waveforms,
- physical waveforms after the attached hardware model.

### Model Builders

```python
build_control_terms_from_model(...)
build_control_system_from_model(...)
build_control_problem_from_model(...)
```

These helpers reuse the existing model-layer drive operators and tensor-ordering conventions instead of introducing a separate Hamiltonian-construction path.

`build_control_problem_from_model(...)` also accepts:

- `parameterization_cls=...` plus `parameterization_kwargs={...}` for structured command parameterizations such as `HeldSampleParameterization`,
- `hardware_model=HardwareModel(...)` to attach a command-to-physical waveform transform directly to the problem.

### GRAPE Solver

```python
GrapeConfig(...)
GrapeSolver(...)
solve_grape(...)
```

Solver behavior:

- dense closed-system propagation with exact matrix exponentials,
- exact slice derivatives via `scipy.linalg.expm_frechet` (NumPy engine) or JAX automatic differentiation (JAX engine),
- ensemble aggregation with `"mean"` or `"worst"`,
- optional hardware-aware forward propagation through `GrapeConfig(apply_hardware_in_forward_model=True)`,
- support for explicit initial schedules or built-in zero/random initialization,
- optional JAX-accelerated engine with JIT compilation and GPU support via `GrapeConfig(engine="jax")`.

**Engine selection:**

- `GrapeConfig(engine="numpy")` (default): NumPy + SciPy propagator with manual `expm_frechet` gradients.
- `GrapeConfig(engine="jax")`: JAX-accelerated propagator with `jax.value_and_grad` automatic differentiation.  Supports GPU via `GrapeConfig(engine="jax", jax_device="gpu")`.

Implementation note:

- the solver internally rescales the physical control amplitudes into a dimensionless optimization vector using the configured bounds before calling SciPy. This preserves physical `rad/s` units in the public API while avoiding false optimizer stagnation caused by numerically tiny raw gradients on short time slices.

### Results and Runtime Interoperability

```python
ControlResult(...)
GrapeResult(...)
GrapeIterationRecord(...)

ControlEvaluationCase(...)
ControlEvaluationResult(...)
evaluate_control_with_simulator(...)
```

`ControlResult` is the common result surface for optimized direct-control runs. `GrapeResult` is the current concrete result type returned by the GRAPE backend.

Shared result data includes:

- the optimized `ControlSchedule`,
- resolved command and physical waveforms on the propagation grid,
- scalar objective / fidelity summaries,
- per-system metrics,
- iteration history,
- optimizer status text,
- the nominal final unitary when available.

When a hardware model is present, the result also reports command-vs-physical fidelity summaries so users can see whether a numerically good command waveform remains good after the attached hardware transform.

`ControlResult.to_pulses()` exports the schedule into standard `Pulse` objects plus the corresponding `drive_ops` mapping so the optimized control can be replayed through the normal `SequenceCompiler` and `simulate_sequence(...)` path.

### Simulator-Backed Replay and Noisy Evaluation

```python
evaluation = result.evaluate_with_simulator(
    problem,
    cases=(
        ControlEvaluationCase(
            model=model,
            frame=frame,
            noise=NoiseSpec(t1=2.0e-6, tphi=1.0e-6),
            label="noisy",
        ),
    ),
    waveform_mode="physical",
)
```

The replay path is explicitly evaluation-only. It does not change the closed-system GRAPE optimizer. Instead it:

- exports the optimized schedule into standard runtime `Pulse` objects,
- replays those pulses through `SequenceCompiler` and `simulate_sequence(...)`,
- evaluates the original objective probe states under nominal or noisy Lindblad dynamics,
- reports weighted replay fidelities and, for retained-subspace unitary objectives, replay leakage metrics.

Available objects:

- `ControlEvaluationCase` for one model/frame/noise replay case,
- `ControlEvaluationResult` for aggregated replay metrics,
- `evaluate_control_with_simulator(...)` as the function form,
- `ControlResult.evaluate_with_simulator(...)` as the convenience method.

Replay can target either the command waveform or the physical waveform through `waveform_mode="command" | "physical" | "problem_default"`.

This path is the supported way to answer: "the optimizer says the pulse is good, but how does it behave when replayed through the simulator with noise?"

### Benchmark Harness

The repository now includes a dedicated benchmark and validation harness for larger GRAPE runs:

- `benchmarks/run_optimal_control_benchmarks.py`

The harness supports configurable slice count, duration, target style, model regime, penalty weights, optional robust ensemble shifts, optimizer backend selection, and noisy replay reporting.

### Minimal model-backed example

```python
import numpy as np

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    UnitaryObjective,
    build_control_problem_from_model,
)
from cqed_sim.unitary_synthesis import Subspace

model = DispersiveTransmonCavityModel(
    omega_c=2.0 * np.pi * 5.0e9,
    omega_q=2.0 * np.pi * 6.0e9,
    alpha=0.0,
    chi=0.0,
    kerr=0.0,
    n_cav=2,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
subspace = Subspace.custom(full_dim=4, indices=(0, 1), labels=("|g,0>", "|g,1>"))

problem = build_control_problem_from_model(
    model,
    frame=frame,
    time_grid=PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9),
    channel_specs=(
        ModelControlChannelSpec(
            name="storage_q",
            target="storage",
            quadratures=("Q",),
            amplitude_bounds=(-1.0e8, 1.0e8),
        ),
    ),
    objectives=(
        UnitaryObjective(
            target_operator=np.array(
                [
                    [np.cos(np.pi / 4.0), -np.sin(np.pi / 4.0)],
                    [np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
                ],
                dtype=np.complex128,
            ),
            subspace=subspace,
            ignore_global_phase=True,
        ),
    ),
)

result = GrapeSolver(GrapeConfig(maxiter=80, seed=7)).solve(problem)
pulses, drive_ops, meta = result.to_pulses()
```

### Multi-start GRAPE with parallelism

```python
from cqed_sim import GrapeMultistartConfig, solve_grape_multistart

# Thread-based parallelism (default, zero overhead)
results = solve_grape_multistart(
    problem,
    config=GrapeConfig(maxiter=200, seed=0),
    multistart_config=GrapeMultistartConfig(
        n_restarts=8, max_workers=4, mp_context="thread",
    ),
)
best = results[0]  # sorted best-first
```

**Parallelism strategies** (`GrapeMultistartConfig.mp_context`):

- `"thread"` (default): Thread-based via `ThreadPoolExecutor`.  Zero startup overhead.  Works well because NumPy/SciPy releases the GIL during linear algebra, and the JAX engine runs entirely in XLA (also GIL-free).
- `"loky"`: Reusable process pool (requires `loky` package).  Near-zero per-restart overhead.
- `"spawn"` / `"fork"`: Standard `multiprocessing` contexts.  `"spawn"` adds ~4-5 s startup per worker on Windows.

Primary reference implementations:

- `examples/grape_storage_subspace_gate_demo.py`
- `examples/hardware_constrained_grape_demo.py`
- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- `benchmarks/run_optimal_control_benchmarks.py`

---

## 18. Simulation Workflows

### Workflow A: One-off simulation

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import square_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9, omega_q=2*np.pi*6e9,
    alpha=2*np.pi*(-200e6), chi=2*np.pi*(-2.84e6),
    n_cav=8, n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

pulse = Pulse("q", 0.0, 100e-9, square_envelope, amp=np.pi/4)
compiled = SequenceCompiler(dt=2e-9).compile([pulse], t_end=102e-9)

result = simulate_sequence(
    model, compiled, model.basis_state(0, 0), {"q": "qubit"},
    config=SimulationConfig(frame=frame),
)
```

### Workflow B: High-throughput with session reuse

```python
from cqed_sim.sim import prepare_simulation, simulate_batch

session = prepare_simulation(model, compiled, {"q": "qubit"},
    config=SimulationConfig(frame=frame), e_ops={})

results = simulate_batch(session,
    [model.basis_state(0, n) for n in range(4)], max_workers=1)
```

### Workflow C: Direct prepare-compile-simulate-measure path

```python
from cqed_sim.core import StatePreparationSpec, fock_state, prepare_state, qubit_state
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit

initial = prepare_state(
    model,
    StatePreparationSpec(qubit=qubit_state("g"), storage=fock_state(0)),
)
result = simulate_sequence(
    model, compiled, initial, {"q": "qubit"},
    config=SimulationConfig(frame=frame),
)
measurement = measure_qubit(result.final_state, QubitMeasurementSpec(shots=2048, seed=42))
```

### Workflow D: Unitary synthesis

```python
from cqed_sim.unitary_synthesis import Subspace, make_target, UnitarySynthesizer

sub = Subspace.qubit_cavity_block(n_match=3)
target = make_target("easy", n_match=3)
synth = UnitarySynthesizer(sub, backend="ideal", leakage_weight=0.01)
result = synth.fit(target, multistart=4, maxiter=300)
```

---

## 19. Physics-Facing API and Conventions

This section summarizes conventions that are critical for correct use of physics-facing
APIs. The canonical reference is `physics_and_conventions/physics_conventions_report.tex`.

### Units

| Quantity | Unit |
|---|---|
| Hamiltonian coefficients | rad/s |
| Frequencies (ω_c, ω_q, chi, kerr, alpha) | rad/s |
| Times (t0, duration, dt, delays) | seconds |
| Noise T₁, T_φ | seconds |
| Noise κ | 1/s |
| Pulse carrier | rad/s |
| Pulse phase | radians |

### Tensor Product Ordering

- **Two-mode:** qubit ⊗ cavity: |q, n⟩ = |q⟩ ⊗ |n⟩
- **Three-mode:** qubit ⊗ storage ⊗ readout: |q, n_s, n_r⟩

### Computational Basis

- |g⟩ = |0⟩ (ground), |e⟩ = |1⟩ (excited)
- σ_z|g⟩ = +|g⟩, σ_z|e⟩ = −|e⟩ (QuTiP standard)
- Transmon excitation operator: n_q = b†b (NOT σ_z/2)

### Dispersive Shift (χ)

- **Runtime convention:** positive χ means the qubit transition moves upward with photon number
- ω_ge(n) = ω_ge(0) + χ·n
- In Hamiltonian: +χ·n_c·n_q
- Negative χ lowers the qubit transition frequency with photon number

### Complex Envelope Sign

- **Convention:** exp(+i(ω·t + φ)) throughout (positive exponent)
- A transition at rotating-frame frequency `ω_transition` is resonantly addressed by `Pulse.carrier = -ω_transition`

### Energy Spectrum Reference

- `compute_energy_spectrum(...)` and `model.energy_spectrum(...)` always subtract the bare vacuum-state energy before reporting `EnergySpectrum.energies`.
- The vacuum basis state means all subsystems are in level `0`: `|g,0>` for the two-mode model, `|g,0,0>` for the three-mode model, and `|0>` for cavity-only models.
- The diagonalized Hamiltonian is still the model's static Hamiltonian in the selected frame. Only the reported zero of energy is shifted.
- For interpretable absolute ladder plots, use the lab frame `FrameSpec()`. Rotating-frame spectra can contain negative energies even though the vacuum remains the zero-reference state.

### Confusion Matrix

- **Ordering:** column = latent state (g, e), row = observed state (g, e)
- p_obs = M @ p_latent

### Higher-Order Coefficients

- chi_higher[i] and kerr_higher[i] use **falling-factorial** form
- chi_higher[0] = chi_2 multiplies n_c(n_c−1), NOT n_c²

---

## 20. Notes on Internal Utilities

The following symbols are semi-internal or implementation-level. They are not
intended as primary user entry points but are accessible.

| Symbol | Location | Notes |
|---|---|---|
| `_operators_cache`, `_static_h_cache` | Model dataclasses | Internal caching; cleared on deepcopy |
| `_legacy_drive_couplings(model)` | `sim.runner` | Fallback for models without `drive_coupling_operators()` |
| `DenseSolverResult` | `sim.solver` | Return type for dense backend path |
| `solve_with_backend(...)` | `sim.solver` | Low-level dense piecewise-constant solver |
| `collapse_operators(model, noise)` | `sim.noise` | Generate Lindblad jump operators from NoiseSpec |
| `coupling_term_key(...)` | `core.hamiltonian` | Hashable key for cache lookup |
| `resolve_operator(operators, label)` | `core.hamiltonian` | Operator lookup by mode name |
| `_argmax_peak(x, y)` | `calibration_targets.common` | Quadratic-interpolation peak finder |
| `PHASE_FAMILY_SPECS` | `observables.fock` | Constant defining ground/excited phase families |

---

## 21. Ambiguities / Gaps / Known Mismatches

### Runtime/Synthesis Convention Alignment

The runtime Hamiltonian layer and the unitary-synthesis drift-phase model now use
the same projector-based dispersive convention:

| Convention | Runtime path | Synthesis path |
|---|---|---|
| Qubit basis | \|g⟩ = \|0⟩, \|e⟩ = \|1⟩ | \|g⟩ = \|0⟩, \|e⟩ = \|1⟩ |
| χ meaning | +χ n_c n_q in Hamiltonian | +χ n \|e⟩⟨e\| in drift model |
| χ mapping | χ_runtime | χ_synth = χ_runtime |
| Kerr mapping | kerr_runtime | kerr_synth = kerr_runtime |

The remaining sign convention users must track is the pulse waveform convention:
`Pulse.carrier = -omega_transition(frame)`.

### MODERATE: FrameSpec Legacy Field Names

The storage-mode frame frequency is stored as `omega_c_frame` (a legacy name from
the original two-mode API). An alias `omega_s_frame` exists as a property but
does not rename the underlying field. This can cause confusion in three-mode
workflows.

### MODERATE: DeviceParameters Uses rad/ns

`cqed_sim.tomo.device.DeviceParameters.to_model()` produces a model using rad/ns
units for all frequencies, which differs from the library's standard rad/s convention.
This is a deliberate choice for the tomography workflow (where nanosecond timescales
are natural) but requires care when mixing with the standard library path.

### ~~LOW: Higher-Order Coefficients Lack Isolated Tests~~ (RESOLVED)

**Resolved:** Dedicated tests exist in `tests/test_46_higher_order_coefficients.py`.

### LOW: Synthetic I/Q Is Not Calibrated

The convenience I/Q sampling layer (when no `ReadoutChain` is provided) uses a
simple Gaussian cluster model. It is **not** a calibrated hardware-response model
and should not be treated as such for quantitative readout studies.

### LOW: Dense Backend Path Limitations

The `NumPyBackend` and `JaxBackend` implement a dense piecewise-constant solver
intended for small-system checks and backend parity validation. It is not a
drop-in replacement for QuTiP's adaptive ODE solver on large systems.
GPU/JAX GRAPE integration has been deferred; the current JAX backend provides
only forward simulation via `jax.scipy.linalg.expm`.
See inline documentation in `cqed_sim/backends/base_backend.py`.

### LOW: Waveform Bridge Gate Type Coverage

`waveform_bridge` (`waveform_primitive_from_gate` / `waveform_sequence_from_gates`)
supports only `QubitRotation`, `Displacement`, and `SQR`. The synthesis sequence gate
types `SNAP`, `ConditionalPhaseSQR`, and `FreeEvolveCondPhase` are not currently
bridged to the waveform path. Passing them raises `TypeError`. Use the ideal or
symbolic backends for sequences that include these gate types.
This limitation is fundamental: these gate types have no pulse builders in
`pulses/builders.py` and no IO gate representations in `io/gates.py`.
See inline documentation in `cqed_sim/unitary_synthesis/waveform_bridge.py`.

### ~~LOW: `targets.py` Contains User-Specific Hardcoded Paths~~ (RESOLVED)

**Resolved:** Hardcoded paths were removed in the 2026-03-17 cleanup.

---

*Generated from codebase inspection. Last updated: 2026-06-12.*
