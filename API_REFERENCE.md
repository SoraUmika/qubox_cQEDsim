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
   - 3.1 [DispersiveTransmonCavityModel](#31-dispersivetransmoncavitymodel)
   - 3.2 [DispersiveReadoutTransmonStorageModel](#32-dispersivereadouttransmonstoragemodel)
   - 3.3 [FrameSpec](#33-framespec)
   - 3.4 [Coupling Specifications](#34-coupling-specifications)
   - 3.5 [Basis Conventions](#35-basis-conventions)
   - 3.6 [Frequency Helpers](#36-frequency-helpers)
   - 3.7 [Ideal Gate Operators](#37-ideal-gate-operators)
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
7. [Experiment Layer (`cqed_sim.experiment`)](#7-experiment-layer)
   - 7.1 [State Preparation](#71-state-preparation)
   - 7.2 [Qubit Measurement](#72-qubit-measurement)
   - 7.3 [SimulationExperiment](#73-simulationexperiment)
   - 7.4 [Readout Chain](#74-readout-chain)
   - 7.5 [Kerr Free Evolution](#75-kerr-free-evolution)
8. [Gate I/O (`cqed_sim.io`)](#8-gate-io)
9. [Analysis (`cqed_sim.analysis`)](#9-analysis)
10. [Backends (`cqed_sim.backends`)](#10-backends)
11. [SQR Calibration (`cqed_sim.calibration`)](#11-sqr-calibration)
12. [Calibration Targets (`cqed_sim.calibration_targets`)](#12-calibration-targets)
13. [Tomography (`cqed_sim.tomo`)](#13-tomography)
14. [Observables (`cqed_sim.observables`)](#14-observables)
15. [Operators (`cqed_sim.operators`)](#15-operators)
16. [Plotting (`cqed_sim.plotting`)](#16-plotting)
17. [Unitary Synthesis (`cqed_sim.unitary_synthesis`)](#17-unitary-synthesis)
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

**Dependencies:** NumPy ≥ 1.24, SciPy ≥ 1.10, QuTiP ≥ 5.0. Optional: JAX for the
dense-matrix backend path.

**Internal units:** Hamiltonian coefficients and rotating-frame frequencies are in
**rad/s**; times are in **seconds**. All user-facing constructors accept these units
unless suffixed otherwise (e.g., `_hz`, `_ns`).

---

## 2. Package Architecture

```
cqed_sim/
├── core/            # Hilbert-space conventions, models, frames, ideal gates
├── pulses/          # Pulse dataclass, envelopes, builders, calibration formulas, hardware
├── sequence/        # SequenceCompiler, compiled-channel timeline
├── sim/             # Hamiltonian assembly, solver, noise, extractors, couplings
├── experiment/      # State preparation, measurement, readout chain, protocols
├── analysis/        # Parameter translation (bare → dressed)
├── backends/        # Dense NumPy/JAX solver backends
├── calibration/     # SQR gate calibration
├── calibration_targets/  # Spectroscopy, Rabi, Ramsey, T1, T2 echo, DRAG tuning
├── io/              # Gate sequence JSON I/O
├── observables/     # Bloch, Fock-resolved, phase, trajectory, Wigner diagnostics
├── operators/       # Pauli, cavity ladder, embedding helpers
├── plotting/        # Bloch tracks, calibration, gate diagnostics, Wigner grids
├── tomo/            # Fock-resolved tomography, all-XY, leakage calibration
└── unitary_synthesis/  # Subspace targeting, gate sequences, optimization, constraints
```

**Main simulation path for a typical user:**

1. Build a model (`DispersiveTransmonCavityModel` or `DispersiveReadoutTransmonStorageModel`).
2. Define a `FrameSpec`.
3. Construct `Pulse` objects (directly or via builders).
4. Compile with `SequenceCompiler`.
5. Simulate with `simulate_sequence(...)`.
6. Extract results with extractors or `measure_qubit(...)`.

Or use the convenience wrapper `SimulationExperiment` which combines steps 1–6.

---

## 3. Core Abstractions

**Module:** `cqed_sim.core`

### 3.1 DispersiveTransmonCavityModel

**Module path:** `cqed_sim.core.model.DispersiveTransmonCavityModel`

Two-mode (qubit + cavity/storage) dispersive system model.

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
| `mode_operators(mode="storage")` | `(str) -> tuple[qt.Qobj, qt.Qobj]` | Retrieve `(creation, annihilation)` operators for the named bosonic mode |
| `sideband_drive_operators(mode="storage", lower_level=0, upper_level=1, sideband="red")` | `(str, int, int, str) -> tuple[qt.Qobj, qt.Qobj]` | Effective multilevel sideband coupling operators |
| `static_hamiltonian(frame)` | `(frame: FrameSpec \| None = None) -> qt.Qobj` | Full static Hamiltonian including all coupling terms. Cached by frame parameters. |
| `basis_state(q_level, cavity_level)` | `(int, int) -> qt.Qobj` | Returns \|q⟩⊗\|n⟩ ket |
| `basis_energy(q_level, cavity_level, frame)` | `(int, int, FrameSpec \| None) -> float` | Eigenvalue of \|q,n⟩ under static Hamiltonian |
| `coherent_qubit_superposition(n_cav)` | `(int) -> qt.Qobj` | (&#124;g,n⟩ + &#124;e,n⟩) / √2 |
| `manifold_transition_frequency(n, frame)` | `(int, FrameSpec \| None) -> float` | &#124;g,n⟩ ↔ &#124;e,n⟩ transition in given frame |
| `transmon_transition_frequency(cavity_level=0, lower_level=0, upper_level=1, frame=None)` | `(int, int, int, FrameSpec \| None) -> float` | General transmon transition frequency at fixed cavity Fock number |
| `sideband_transition_frequency(cavity_level=0, lower_level=0, upper_level=1, sideband="red", frame=None)` | `(int, int, int, str, FrameSpec \| None) -> float` | Rotating-frame frequency addressed by the effective sideband transition |

---

### 3.2 DispersiveReadoutTransmonStorageModel

**Module path:** `cqed_sim.core.readout_model.DispersiveReadoutTransmonStorageModel`

Three-mode (qubit + storage + readout) dispersive system.

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
| `mode_operators(mode="storage")` | `(str) -> tuple[qt.Qobj, qt.Qobj]` | Retrieve `(creation, annihilation)` operators for `"storage"` or `"readout"` |
| `sideband_drive_operators(mode="storage", lower_level=0, upper_level=1, sideband="red")` | `(str, int, int, str) -> tuple[qt.Qobj, qt.Qobj]` | Effective multilevel sideband coupling operators on the selected bosonic mode |
| `static_hamiltonian(frame)` | `(FrameSpec \| None) -> qt.Qobj` | Full three-mode static Hamiltonian |
| `basis_state(q, ns, nr)` | `(int, int, int) -> qt.Qobj` | \|q⟩⊗\|n_s⟩⊗\|n_r⟩ |
| `basis_energy(q, ns, nr, frame)` | `(int, int, int, FrameSpec \| None) -> float` | Eigenvalue for given basis state |
| `qubit_transition_frequency(ns, nr, q, frame)` | `(...) -> float` | Qubit transition at given storage/readout occupation |
| `storage_transition_frequency(q, ns, nr, frame)` | `(...) -> float` | Storage transition frequency |
| `readout_transition_frequency(q, ns, nr, frame)` | `(...) -> float` | Readout transition frequency |
| `transmon_transition_frequency(storage_level=0, readout_level=0, lower_level=0, upper_level=1, frame=None)` | `(...) -> float` | General transmon transition frequency in the three-mode model |
| `sideband_transition_frequency(mode="storage", storage_level=0, readout_level=0, lower_level=0, upper_level=1, sideband="red", frame=None)` | `(...) -> float` | Rotating-frame frequency addressed by the effective sideband transition |

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
| `snap_op(phases)` | `(array) -> qt.Qobj` | diag(e^{iφ_0}, e^{iφ_1}, ...). Dimension = len(phases). |
| `sqr_op(thetas, phis)` | `(array, array) -> qt.Qobj` | Σ_n \|n⟩⟨n\| ⊗ R(θ_n, φ_n). Two-mode unitary. |
| `embed_qubit_op(op_q, n_cav)` | `(qt.Qobj, int) -> qt.Qobj` | op_q ⊗ I_cav |
| `embed_cavity_op(op_c, n_tr)` | `(qt.Qobj, int) -> qt.Qobj` | I_qubit ⊗ op_c. Default `n_tr=2`. |
| `beamsplitter_unitary(n_a, n_b, theta)` | `(int, int, float) -> qt.Qobj` | exp[−iθ(a b† + a† b)]. Full two-mode unitary. |

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

```python
def collapse_operators(model, noise: NoiseSpec | None) -> list[qt.Qobj]
```

Returns Lindblad jump operators:
- √γ₁ · b (legacy aggregate transmon relaxation when `transmon_t1` is not supplied)
- `sqrt(1/T1_j) * |j-1><j|` for each explicit transmon ladder transition listed in `transmon_t1`
- √γ_φ · σ_z (dephasing for 2-level) or √γ_φ · n_q (multi-level)
- √(κ(n_th+1)) · a and √(κ·n_th) · a† per bosonic mode

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

---

### 6.8 Diagnostics

**Module path:** `cqed_sim.sim.diagnostics`

| Function | Signature | Description |
|---|---|---|
| `channel_norms(compiled)` | `(CompiledSequence) -> dict[str, float]` | L2 norm of distorted waveform per channel |
| `instantaneous_phase_frequency(signal, dt)` | `(ndarray, float) -> (phase, omega)` | Unwrapped phase and instantaneous frequency from complex signal |

---

## 7. Experiment Layer

**Module:** `cqed_sim.experiment`

### 7.1 State Preparation

**Module path:** `cqed_sim.experiment.state_prep`

#### SubsystemStateSpec

```python
@dataclass(frozen=True)
class SubsystemStateSpec:
    kind: str                           # "qubit", "vacuum", "fock", "coherent", "amplitudes", "density_matrix"
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
    readout: SubsystemStateSpec | None = None  # For 3-mode models
```

#### State Constructor Helpers

| Function | Signature | Description |
|---|---|---|
| `qubit_state(label)` | `(str) -> SubsystemStateSpec` | Labels: `"g"`, `"e"`, `"+x"`, `"-x"`, `"+y"`, `"-y"` |
| `qubit_level(level)` | `(int) -> SubsystemStateSpec` | Qubit by Fock level |
| `vacuum_state()` | `() -> SubsystemStateSpec` | Cavity ground state \|0⟩ |
| `fock_state(level)` | `(int) -> SubsystemStateSpec` | Cavity Fock state \|n⟩ |
| `coherent_state(alpha)` | `(complex) -> SubsystemStateSpec` | Coherent state \|α⟩ |
| `amplitude_state(amplitudes)` | `(Any) -> SubsystemStateSpec` | Arbitrary state from amplitudes |
| `density_matrix_state(rho)` | `(Any) -> SubsystemStateSpec` | State from density matrix |

#### Preparation Functions

```python
def prepare_state(model, spec: StatePreparationSpec | None = None) -> qt.Qobj
```

Builds the tensor-product initial state. Supports 2-mode and 3-mode models
automatically. Defaults to \|g⟩\|0⟩ (or \|g⟩\|0⟩\|0⟩) if spec is None.

```python
def prepare_ground_state(model) -> qt.Qobj
```

Convenience: prepare \|g,0⟩ (or \|g,0,0⟩ for 3-mode).

---

### 7.2 Qubit Measurement

**Module path:** `cqed_sim.experiment.measurement`

#### QubitMeasurementSpec

```python
@dataclass(frozen=True)
class QubitMeasurementSpec:
    shots: int | None = None                      # None = exact probabilities only
    confusion_matrix: np.ndarray | None = None    # Shape (2,2), ordering (g, e)
    iq_sigma: float | None = None                 # IQ noise std deviation
    seed: int | None = None                       # RNG seed for reproducibility
    lump_other_into: str = "e"                    # "g" or "e" for higher levels
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
    probabilities: dict[str, float]              # {"g": p_g, "e": p_e}
    observed_probabilities: dict[str, float]     # After confusion matrix
    expectation_z: float                         # p_g_obs − p_e_obs
    counts: dict[str, int] | None = None         # Shot counts
    samples: np.ndarray | None = None            # (shots,) of 0/1
    iq_samples: np.ndarray | None = None         # (shots, 2)
    post_measurement_state: qt.Qobj | None = None
    readout_centers: dict[str, np.ndarray] | None = None
    readout_metadata: dict[str, float] | None = None
```

#### measure_qubit

```python
def measure_qubit(state: qt.Qobj, spec: QubitMeasurementSpec | None = None) -> QubitMeasurementResult
```

**Pipeline:**

1. Extract reduced qubit state via `reduced_qubit_state()`.
2. Compute latent probabilities (p_g, p_e), lumping higher levels.
3. Apply confusion matrix: p_obs = M @ p_latent.
4. If readout chain is attached: apply backaction (dephasing, Purcell), generate IQ.
5. If shots requested: sample from p_obs (or classify from IQ).
6. Return comprehensive result.

**Confusion matrix convention:** column ordering is (g, e).

---

### 7.3 SimulationExperiment

**Module path:** `cqed_sim.experiment.protocol.SimulationExperiment`

Convenience wrapper that combines state preparation, compilation, simulation, and
measurement in a single object.

```python
@dataclass
class SimulationExperiment:
    model: Any
    pulses: list[Pulse]
    drive_ops: dict[str, str]
    dt: float
    t_end: float | None = None
    frame: FrameSpec = FrameSpec()
    initial_state: qt.Qobj | None = None
    state_prep: StatePreparationSpec | None = None
    noise: NoiseSpec | None = None
    e_ops: dict[str, qt.Qobj] | None = None
    measurement: QubitMeasurementSpec | None = None
    hardware: dict[str, HardwareConfig] | None = None
    crosstalk_matrix: dict[str, dict[str, float]] | None = None
    metadata: ExperimentMetadata = ExperimentMetadata()
```

| Method | Signature | Description |
|---|---|---|
| `resolve_initial_state()` | `-> qt.Qobj` | Returns `initial_state` or prepares from `state_prep` |
| `compile()` | `-> CompiledSequence` | Creates SequenceCompiler and compiles pulses |
| `run()` | `-> ExperimentResult` | Full pipeline: prepare → compile → simulate → measure |

```python
@dataclass
class ExperimentResult:
    initial_state: qt.Qobj
    compiled: CompiledSequence
    simulation: SimulationResult
    measurement: QubitMeasurementResult | None
    metadata: ExperimentMetadata

@dataclass
class ExperimentMetadata:
    label: str = ""
    target_state: qt.Qobj | None = None
    target_unitary: qt.Qobj | None = None
    notes: dict[str, Any] = field(default_factory=dict)
```

---

### 7.4 Readout Chain

**Module path:** `cqed_sim.experiment.readout_chain`

Physical readout model with resonator, Purcell filter, and amplifier chain.

#### ReadoutResonator

```python
@dataclass(frozen=True)
class ReadoutResonator:
    omega_r: float        # Resonator frequency (rad/s)
    kappa: float          # Linewidth (rad/s)
    g: float              # Coupling to qubit (rad/s)
    epsilon: complex      # Drive amplitude (rad/s)
    chi: float = 0.0      # Dispersive shift (rad/s)
    drive_frequency: float | None = None
```

**Key methods:**

| Method | Returns | Description |
|---|---|---|
| `dispersive_shift(qubit_state, chi)` | `float` | 0 for "g", +χ for "e" |
| `resonant_frequency(qubit_state, chi)` | `float` | ω_r + dispersive_shift |
| `steady_state_amplitude(qubit_state, ...)` | `complex` | α_ss = −iε / (κ_eff/2 + i(ω_r,q − ω_d)) |
| `gamma_meas(...)` | `float` | Measurement-induced dephasing rate |
| `purcell_rate(omega_q, ...)` | `float` | Purcell decay rate at qubit frequency |
| `purcell_limited_t1(omega_q, ...)` | `float` | 1 / purcell_rate |
| `response_trace(qubit_state, *, duration, dt, ...)` | `(tlist, field)` | Time-domain cavity response |

#### PurcellFilter

```python
@dataclass(frozen=True)
class PurcellFilter:
    omega_f: float | None = None           # Center frequency
    bandwidth: float | None = None         # Filter bandwidth (rad/s)
    quality_factor: float | None = None    # Q = center / bandwidth
    coupling_rate: float | None = None     # 0.5·√(κ·BW)
```

Implements frequency-dependent linewidth suppression:

$$\kappa_\text{eff}(\omega) = \frac{4J^2 \kappa_f}{\kappa_f^2 + 4(\omega - \omega_f)^2}$$

#### AmplifierChain

```python
@dataclass(frozen=True)
class AmplifierChain:
    noise_temperature: float = 0.0   # Kelvin
    gain: float = 1.0
    impedance_ohm: float = 50.0
    mixer_phase: float = 0.0         # Downconversion phase
```

| Method | Description |
|---|---|
| `noise_std(dt, n_samples)` | RMS noise per quadrature |
| `amplify(trace, *, dt, seed)` | Apply gain and thermal noise |
| `mix_down(trace)` | Multiply by exp(−i·phase) |
| `iq_sample(trace)` | Integrated I/Q: [mean_I, mean_Q] |

#### ReadoutChain

```python
@dataclass
class ReadoutChain:
    resonator: ReadoutResonator
    amplifier: AmplifierChain = AmplifierChain()
    purcell_filter: PurcellFilter | None = None
    integration_time: float = 1e-6    # s
    dt: float = 4e-9                  # s
```

| Method | Description |
|---|---|
| `simulate_trace(qubit_state, ...)` | Full readout trace with noise → `ReadoutTrace` |
| `iq_centers(...)` | Noiseless IQ centers: `{"g": [I,Q], "e": [I,Q]}` |
| `sample_iq(latent_states, ...)` | Noisy IQ samples (n × 2) |
| `classify_iq(iq_samples, ...)` | Nearest-neighbor classification → 0/1 |
| `apply_backaction(rho_q, *, omega_q, ...)` | Measurement dephasing ± Purcell relaxation on qubit ρ |

```python
@dataclass
class ReadoutTrace:
    tlist: np.ndarray          # (n_steps,)
    cavity_field: np.ndarray   # (n_steps,) complex
    output_field: np.ndarray   # √κ · cavity_field
    voltage_trace: np.ndarray  # Amplified + noise
    iq_sample: np.ndarray      # [I, Q]
```

---

### 7.5 Kerr Free Evolution

**Module path:** `cqed_sim.experiment.kerr_free_evolution`

Specialized helpers for simulating cavity free evolution under Kerr nonlinearity.

#### KerrParameterSet

```python
@dataclass(frozen=True)
class KerrParameterSet:
    name: str
    omega_q_hz: float
    omega_c_hz: float
    omega_ro_hz: float
    alpha_q_hz: float
    kerr_hz: float
    kerr2_hz: float = 0.0
    chi_hz: float = 0.0
    chi2_hz: float = 0.0
    chi3_hz: float = 0.0
```

Has a `build_model(*, n_cav=28, n_tr=3)` method that converts Hz → rad/s and
constructs a `DispersiveTransmonCavityModel`.

**Predefined sets:** `KERR_FREE_EVOLUTION_PARAMETER_SETS` is a dict with keys
`"phase_evolution"` and `"value_2"`.

#### Main Functions

| Function | Signature | Description |
|---|---|---|
| `run_kerr_free_evolution(times_s, *, parameter_set="phase_evolution", cavity_state=None, qubit=None, state_prep=None, n_cav=28, n_tr=3, use_rotating_frame=True, wigner_times_s=None, ...)` | `-> KerrFreeEvolutionResult` | Full free-evolution simulator with optional Wigner snapshots |
| `build_kerr_free_evolution_model(parameter_set, ...)` | `-> DispersiveTransmonCavityModel` | Build model from preset |
| `build_kerr_free_evolution_frame(model, *, use_rotating_frame)` | `-> FrameSpec` | Rotating-frame FrameSpec |
| `times_us_to_seconds(times_us)` | `-> ndarray` | Convert µs to seconds |
| `available_kerr_parameter_sets()` | `-> tuple[str, ...]` | Available preset names |
| `plot_kerr_wigner_snapshots(result, ...)` | `-> fig` | Grid of Wigner function snapshots |
| `verify_kerr_sign(...)` | `-> KerrSignVerificationResult` | Compare the documented Kerr sign against a flipped-sign control run for notebook-scale diagnostics |

#### Result Dataclasses

```python
@dataclass
class KerrEvolutionSnapshot:
    time_s: float
    time_us: float
    joint_state: qt.Qobj
    cavity_state: qt.Qobj
    cavity_mean: complex         # ⟨a⟩
    cavity_photon_number: float  # ⟨a†a⟩
    wigner: dict | None          # {"xvec", "yvec", "w"} if computed

@dataclass
class KerrFreeEvolutionResult:
    parameter_set: KerrParameterSet
    model: DispersiveTransmonCavityModel
    frame: FrameSpec
    initial_state: qt.Qobj
    state_prep: StatePreparationSpec
    snapshots: list[KerrEvolutionSnapshot]
    metadata: dict[str, Any]
    # Properties: times_s, times_us → ndarray

@dataclass(frozen=True)
class KerrSignVerificationResult:
    documented_kerr_hz: float
    flipped_kerr_hz: float
    cavity_mean_documented: complex
    cavity_mean_flipped: complex
    documented_phase_rad: float
    flipped_phase_rad: float
    matches_documented_sign: bool
```

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
| `target_qubit_unitary(theta, phi)` | `-> ndarray` | Ideal SQR target unitary |
| `conditional_process_fidelity(target, simulated)` | `-> float` | Process fidelity, clipped to [0, 1] |
| `conditional_loss(params, n, theta, phi, config)` | `-> float` | Optimization objective: 1 − fidelity + regularization |

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

Gradient-free optimization of pulse/gate sequences to implement target unitaries
within a qubit–cavity subspace.

> **Convention note:** The synthesis drift-phase layer now matches the runtime
> dispersive/Kerr convention. The remaining sign distinction is the waveform
> convention `Pulse.carrier = -omega_transition(frame)`.

### Subspace

**Module path:** `cqed_sim.unitary_synthesis.subspace.Subspace`

```python
@dataclass(frozen=True)
class Subspace:
    full_dim: int                      # Full Hilbert-space dimension (2·n_cav)
    indices: tuple[int, ...]           # Subspace state indices in full space
    labels: tuple[str, ...]            # Human-readable basis labels
    kind: str = "custom"               # "custom", "qubit_cavity_block", "cavity_only"
    metadata: dict | None = None
```

| Factory Method | Description |
|---|---|
| `Subspace.qubit_cavity_block(n_match, n_cav=None)` | Qubit–cavity block: levels 0..n_match. Basis: \|g,0⟩, \|e,0⟩, \|g,1⟩, \|e,1⟩, ... |
| `Subspace.cavity_only(n_match, qubit="g", n_cav=None)` | Cavity subspace for fixed qubit state |
| `Subspace.custom(full_dim, indices, labels=None)` | Arbitrary index selection |

| Method | Description |
|---|---|
| `projector()` | Full-space projection operator |
| `embed(vec_sub)` | Subspace → full-space embedding |
| `extract(vec_full)` | Full-space → subspace extraction |
| `restrict_operator(op_full)` | Full-space operator → subspace restriction |
| `per_fock_blocks()` | Block slices for qubit_cavity_block kind |

### Target Construction

**Module path:** `cqed_sim.unitary_synthesis.targets`

```python
def make_target(
    name: str,               # "easy", "ghz", "cluster"
    n_match: int,
    variant: str = "analytic",  # "analytic" | "mps"
    **kwargs,
) -> np.ndarray
```

Returns a (full_dim × full_dim) or (2*(n_match+1) × 2*(n_match+1)) target unitary.

### Gate Sequence Primitives

**Module path:** `cqed_sim.unitary_synthesis.sequence`

#### Gate Types

| Gate Class | Parameters | Description |
|---|---|---|
| `QubitRotation` | `theta`, `phi` | Qubit rotation ⊗ I_cav |
| `SQR` | `theta_n[]`, `phi_n[]` | Selective qubit rotation per Fock level |
| `SNAP` | `phases[]` | Diagonal cavity phase gate |
| `Displacement` | `alpha = re + i·im` | D(α) ⊗ I_qubit |
| `ConditionalPhaseSQR` | `phases_n[]` | Conditional phase: diag(e^{−iφ/2}, e^{+iφ/2}) per n |
| `FreeEvolveCondPhase` | (none—uses drift_model) | Idle evolution under drift Hamiltonian |

All gates inherit from `GateBase` which provides:
- `parameter_names(n_cav)`, `get_parameters(n_cav)`, `set_parameters(params, n_cav)`
- `ideal_unitary(n_cav)`, `pulse_unitary(n_cav)`
- `duration`, `optimize_time`, `time_bounds`

#### DriftPhaseModel

```python
@dataclass(frozen=True)
class DriftPhaseModel:
    chi: float = 0.0        # χ_synth (rad/s)
    chi2: float = 0.0       # χ₂_synth (rad/s)
    kerr: float = 0.0       # K_synth (rad/s)
    kerr2: float = 0.0      # K₂_synth (rad/s)
    delta_c: float = 0.0
    delta_q: float = 0.0
    frame: str = "rotating_omega_c_omega_q"
```

**Drift energies (shared runtime/synthesis convention):**

$$E_{g,n} = \Delta_c n + K(n)$$
$$E_{e,n} = \Delta_c n + \Delta_q + (\chi n + \chi_2 n(n-1)) + K(n)$$

| Function | Description |
|---|---|
| `drift_phase_table(n_cav, duration, model)` | Precompute drift phases |
| `drift_phase_unitary(n_cav, duration, model)` | exp(−iH₀t) as Qobj |
| `drift_hamiltonian_qobj(n_cav, model)` | Diagonal H₀ Hamiltonian |

#### GateSequence

```python
@dataclass
class GateSequence:
    gates: list[GatePrimitive]
    n_cav: int
```

Manages parameter vectors and time parameters across all gates. Supports per-instance,
per-type, and hybrid time-parameter sharing via `configure_time_parameters()`.

| Method | Description |
|---|---|
| `unitary(backend="ideal")` | Full sequence unitary |
| `total_duration()` | Sum of gate durations |
| `get_parameter_vector()` / `set_parameter_vector(v)` | Gate-angle parameters |
| `get_time_vector()` / `set_time_vector(t)` | Duration parameters |
| `serialize()` | List of gate records |

### UnitarySynthesizer

**Module path:** `cqed_sim.unitary_synthesis.optim.UnitarySynthesizer`

```python
class UnitarySynthesizer:
    def __init__(
        self,
        subspace: Subspace,
        backend: str = "ideal",          # "ideal" | "pulse"
        gateset: list[str] | None = None,
        optimize_times: bool = True,
        time_bounds: dict | None = None,
        time_policy: dict | None = None,
        time_mode: str = "per-instance",
        time_groups: dict | None = None,
        leakage_weight: float = 0.0,
        time_reg_weight: float = 0.0,
        time_smooth_weight: float = 0.0,
        gauge: str = "global",           # "global" | "block"
        drift_config: Mapping | None = None,
        include_conditional_phase_in_sqr: bool = False,
        hardware_limits: Mapping | None = None,
        time_grid: Mapping | None = None,
        constraints: Mapping | None = None,
        parallel: Mapping | None = None,
        progress: Mapping | None = None,
        seed: int = 0,
    )
```

```python
def fit(
    self,
    target: np.ndarray,
    init_guess: str = "heuristic",   # "heuristic" | "random"
    multistart: int = 1,
    maxiter: int = 300,
) -> SynthesisResult
```

**Optimization objective:**

$$L = (1 - F_\text{subspace}) + \lambda_L \cdot \text{leakage}_\text{worst} + \lambda_t \cdot \text{time\_reg} + \text{constraint\_penalties}$$

```python
@dataclass
class SynthesisResult:
    success: bool
    objective: float
    sequence: GateSequence
    simulation: SimulationResult    # From unitary_synthesis.backends
    report: dict[str, Any]
    history: list[dict]
    history_by_run: dict[str, list[dict]]
```

### Metrics

**Module path:** `cqed_sim.unitary_synthesis.metrics`

| Function | Signature | Description |
|---|---|---|
| `subspace_unitary_fidelity(actual, target, gauge, block_slices)` | `-> float` | Phase-invariant fidelity in [0, 1] |
| `leakage_metrics(full_operator, subspace, probes, n_jobs)` | `-> LeakageMetrics` | average, worst, per_probe leakage |
| `objective_breakdown(actual_sub, target_sub, full_op, subspace, ...)` | `-> dict` | Full objective decomposition |
| `unitarity_error(op)` | `-> float` | ‖U†U − I‖_F |

### Constraints

**Module path:** `cqed_sim.unitary_synthesis.constraints`

| Function | Returns | Description |
|---|---|---|
| `snap_times_to_grid(times, dt, mode)` | `TimeGridResult` | Quantize to nearest dt |
| `piecewise_constant_samples(amplitudes, durations, dt)` | `ndarray` | Sample piecewise-constant waveform |
| `enforce_slew_limit(samples, dt, s_max, mode)` | `SlewConstraintResult` | Slew-rate violation check and optional projection |
| `evaluate_tone_spacing(freqs, domega_min, ntones_max, forbidden_bands, project)` | `ToneSpacingResult` | Tone-spacing constraint |
| `project_tone_frequencies(freqs, domega_min, forbidden_bands)` | `ndarray` | Projected frequencies |

### Progress Reporting

**Module path:** `cqed_sim.unitary_synthesis.progress`

| Class | Description |
|---|---|
| `ProgressReporter` | Base class with `on_start`, `on_event`, `on_end` hooks |
| `NullReporter` | No-op |
| `HistoryReporter` | Accumulate events in memory. Has `to_dataframe()`. |
| `JupyterLiveReporter` | Live Jupyter progress plots |

| Function | Description |
|---|---|
| `history_to_dataframe(history)` | Convert events to pandas DataFrame |
| `plot_history(history, what="objective_total", ...)` | Plot optimization history |

### Reporting

```python
def make_run_report(base_report, subspace_operator) -> dict
```

Adds per-Fock-block breakdown to the synthesis result report.

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

### Workflow C: SimulationExperiment wrapper

```python
from cqed_sim.experiment import (
    SimulationExperiment, StatePreparationSpec,
    QubitMeasurementSpec, qubit_state, fock_state,
)

experiment = SimulationExperiment(
    model=model, pulses=[pulse], drive_ops={"q": "qubit"},
    dt=2e-9, frame=frame,
    state_prep=StatePreparationSpec(qubit=qubit_state("g"), storage=fock_state(0)),
    measurement=QubitMeasurementSpec(shots=2048, seed=42),
)
result = experiment.run()
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

### LOW: Higher-Order Coefficients Lack Isolated Tests

The falling-factorial form for `chi_higher` and `kerr_higher` is verified through
numerical integration tests but lacks dedicated isolated sign-and-factor analytic
tests. Trust in the implementation relies on agreement with downstream simulations
rather than purely symbolic verification.

### LOW: Synthetic I/Q Is Not Calibrated

The convenience I/Q sampling layer (when no `ReadoutChain` is provided) uses a
simple Gaussian cluster model. It is **not** a calibrated hardware-response model
and should not be treated as such for quantitative readout studies.

### LOW: Dense Backend Path Limitations

The `NumPyBackend` and `JaxBackend` implement a dense piecewise-constant solver
intended for small-system checks and backend parity validation. It is not a
drop-in replacement for QuTiP's adaptive ODE solver on large systems.

---

*Generated from codebase inspection. Last updated: 2026-03-10.*
