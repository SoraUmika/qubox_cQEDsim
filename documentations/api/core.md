# API Reference — Core (`cqed_sim.core`)

The core module defines Hilbert-space conventions, system models, rotating frames, coupling specifications, basis helpers, frequency utilities, and ideal gate operators.

---

## UniversalCQEDModel and Subsystem Specs

`UniversalCQEDModel` is the generalized model-layer abstraction used by the wrapper models. It centralizes operator construction, basis helpers, static Hamiltonian assembly, multilevel ancilla transitions, and bosonic-mode transition helpers.

### Subsystem Spec Dataclasses

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
```

### UniversalCQEDModel

```python
@dataclass
class UniversalCQEDModel:
    transmon: TransmonModeSpec | None = None
    bosonic_modes: Sequence[BosonicModeSpec] = ()
    dispersive_couplings: Sequence[DispersiveCouplingSpec] = ()
    cross_kerr_terms: Sequence[CrossKerrSpec] = ()
    self_kerr_terms: Sequence[SelfKerrSpec] = ()
    exchange_terms: Sequence[ExchangeSpec] = ()
```

**Notes:**

- Supports transmon-only, cavity-only, and multilevel transmon-plus-multimode systems
- Tensor ordering is transmon first when present, followed by bosonic modes in declaration order
- `DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel` are compatibility frontends that delegate to this shared core

#### Common Methods

| Method | Signature | Description |
|---|---|---|
| `operators()` | `-> dict[str, qt.Qobj]` | Cached full-space ladder and number operators |
| `hamiltonian(frame=None)` | `-> qt.Qobj` | Alias of `static_hamiltonian(frame)` |
| `basis_state(*levels)` | `-> qt.Qobj` | Basis ket using subsystem declaration order |
| `basis_energy(*levels, frame=None)` | `-> float` | Diagonal basis energy under the static Hamiltonian |
| `transmon_lowering()` | `-> qt.Qobj` | Full-space transmon lowering operator |
| `mode_annihilation(mode)` | `(str) -> qt.Qobj` | Full-space bosonic lowering operator for the named mode |
| `transmon_transition_frequency(...)` | `-> float` | Multilevel transmon transition at fixed bosonic occupations |
| `mode_transition_frequency(...)` | `-> float` | Bosonic ladder spacing at fixed occupations |
| `sideband_transition_frequency(...)` | `-> float` | Effective red/blue sideband manifold transition |

---

## DispersiveTransmonCavityModel

**Module path:** `cqed_sim.core.model.DispersiveTransmonCavityModel`

Two-mode (qubit + cavity/storage) dispersive system model. Thin compatibility wrapper around `UniversalCQEDModel`.

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

### Physics Convention

Positive `chi` means the qubit |g⟩→|e⟩ transition frequency **increases** with increasing cavity photon number:

$$\omega_{ge}(n) = \omega_{ge}(0) + \chi \cdot n + \chi_2 \cdot n(n-1) + \cdots$$

The static Hamiltonian in the rotating frame:

$$H_0/\hbar = \delta_c\, n_c + \delta_q\, n_q + \frac{\alpha}{2}\, b^{\dagger 2} b^2 + \frac{K}{2}\, n_c(n_c - 1) + \chi\, n_c\, n_q + \chi_2\, n_c(n_c-1)\, n_q + \cdots$$

where $\delta_c = \omega_c - \omega_c^{\text{frame}}$, $\delta_q = \omega_q - \omega_q^{\text{frame}}$.

Higher-order coefficients use **falling-factorial** form: `chi_higher[i]` multiplies $n_c(n_c-1)\cdots(n_c-i)$, **not** $n_c^{i+1}$.

### Properties

| Property | Type | Description |
|---|---|---|
| `subsystem_dims` | `tuple[int, ...]` | `(n_tr, n_cav)` |

### Methods

| Method | Signature | Description |
|---|---|---|
| `operators()` | `-> dict[str, qt.Qobj]` | `a`, `adag`, `b`, `bdag`, `n_c`, `n_q` |
| `drive_coupling_operators()` | `-> dict[str, tuple[qt.Qobj, qt.Qobj]]` | `(raising, lowering)` pairs keyed by `"cavity"`, `"storage"`, `"qubit"`, `"sideband"` |
| `transmon_level_projector(level)` | `(int) -> qt.Qobj` | Projector onto a transmon level |
| `transmon_transition_operators(lower, upper)` | `(int, int) -> tuple[qt.Qobj, qt.Qobj]` | Full-space raising/lowering operators |
| `mode_operators(mode="storage")` | `(str) -> tuple[qt.Qobj, qt.Qobj]` | `(annihilation, creation)` operators |
| `sideband_drive_operators(mode, lower, upper, sideband)` | `-> tuple[qt.Qobj, qt.Qobj]` | Effective multilevel sideband coupling operators |
| `static_hamiltonian(frame)` | `(FrameSpec \| None) -> qt.Qobj` | Full static Hamiltonian. Cached by frame. |
| `basis_state(q_level, cavity_level)` | `(int, int) -> qt.Qobj` | \|q⟩⊗\|n⟩ ket |
| `basis_energy(q, n, frame)` | `(int, int, FrameSpec \| None) -> float` | Eigenvalue of \|q,n⟩ |
| `manifold_transition_frequency(n, frame)` | `(int, FrameSpec \| None) -> float` | \|g,n⟩ ↔ \|e,n⟩ transition |
| `transmon_transition_frequency(cavity_level, lower, upper, frame)` | `-> float` | General transmon transition |
| `sideband_transition_frequency(cavity_level, lower, upper, sideband, frame)` | `-> float` | Effective sideband transition |
| `hamiltonian(frame=None)` | `(FrameSpec \| None) -> qt.Qobj` | Alias of `static_hamiltonian(frame)` |
| `as_universal_model()` | `-> UniversalCQEDModel` | Return delegated universal model |

---

## DispersiveReadoutTransmonStorageModel

**Module path:** `cqed_sim.core.readout_model.DispersiveReadoutTransmonStorageModel`

Three-mode (qubit + storage + readout) dispersive system. Thin compatibility wrapper around `UniversalCQEDModel`.

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

### Static Hamiltonian

$$H_0/\hbar = \delta_s n_s + \delta_r n_r + \delta_q n_q + \frac{\alpha}{2} b^{\dagger 2}b^2 + \chi_s n_s n_q + \chi_r n_r n_q + \chi_{sr} n_s n_r + \frac{K_s}{2} n_s(n_s-1) + \frac{K_r}{2} n_r(n_r-1)$$

### Methods

| Method | Signature | Description |
|---|---|---|
| `operators()` | `-> dict[str, qt.Qobj]` | `b`, `bdag`, `a_s`, `adag_s`, `a_r`, `adag_r`, `n_q`, `n_s`, `n_r` |
| `drive_coupling_operators()` | `-> dict[str, tuple]` | Keys: `"storage"`, `"cavity"`, `"qubit"`, `"transmon"`, `"readout"` |
| `static_hamiltonian(frame)` | `(FrameSpec \| None) -> qt.Qobj` | Full three-mode static Hamiltonian |
| `basis_state(q, ns, nr)` | `(int, int, int) -> qt.Qobj` | \|q⟩⊗\|n_s⟩⊗\|n_r⟩ |
| `as_universal_model()` | `-> UniversalCQEDModel` | Return delegated universal model |

---

## FrameSpec

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

`FrameSpec(0, 0, 0)` is the lab frame. Setting frame frequencies equal to the model frequencies removes bare rotations.

---

## Coupling Specifications

**Module path:** `cqed_sim.core.hamiltonian`

```python
@dataclass(frozen=True)
class CrossKerrSpec:
    left: str; right: str; chi: float       # χ · a†a · b†b

@dataclass(frozen=True)
class SelfKerrSpec:
    mode: str; kerr: float                  # (K/2) · a†²a²

@dataclass(frozen=True)
class ExchangeSpec:
    left: str; right: str; coupling: float | complex  # J · (a†b + ab†)
```

### Helper Functions

| Function | Signature | Description |
|---|---|---|
| `additional_coupling_terms(operators, *, ...)` | `-> list[qt.Qobj]` | Convert specs to Hamiltonian operator terms |
| `assemble_static_hamiltonian(base, operators, *, ...)` | `-> qt.Qobj` | Add coupling terms to a base Hamiltonian |

### Drive Target Specs

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

---

## Basis Conventions

**Module path:** `cqed_sim.core.conventions`

**Two-mode tensor ordering:** qubit ⊗ cavity → |q, n⟩ = |q⟩⊗|n⟩

**Three-mode tensor ordering:** qubit ⊗ storage ⊗ readout → |q, n_s, n_r⟩

| Function | Signature | Description |
|---|---|---|
| `qubit_cavity_dims(n_qubit, n_cav)` | `-> list[list[int]]` | QuTiP dims for 2-mode |
| `qubit_cavity_index(n_cav, qubit_level, cavity_level)` | `-> int` | Flat index = `q * n_cav + n` |
| `qubit_cavity_block_indices(n_cav, cavity_level)` | `-> tuple[int, int]` | `(index_g, index_e)` for Fock level n |
| `qubit_storage_readout_dims(n_q, n_s, n_r)` | `-> list[list[int]]` | QuTiP dims for 3-mode |
| `qubit_storage_readout_index(n_s, n_r, q, ns, nr)` | `-> int` | Flat index for 3-mode |

**Computational basis:** |g⟩ = |0⟩, |e⟩ = |1⟩. σ_z|g⟩ = +|g⟩, σ_z|e⟩ = −|e⟩ (QuTiP standard).

---

## Frequency Helpers

**Module path:** `cqed_sim.core.frequencies`

| Function | Signature | Description |
|---|---|---|
| `manifold_transition_frequency(model, n, frame)` | `-> float` | \|g,n⟩↔\|e,n⟩ transition with all chi_higher terms |
| `transmon_transition_frequency(model, ..., frame)` | `-> float` | General multilevel transmon transition |
| `sideband_transition_frequency(model, ..., frame)` | `-> float` | Effective sideband manifold transition |
| `effective_sideband_rabi_frequency(coupling, detuning)` | `-> float` | $\sqrt{(2g)^2 + \delta^2}$ |
| `falling_factorial_scalar(n, order)` | `-> float` | $n(n-1)(n-2)\cdots(n-\text{order}+1)$ |
| `carrier_for_transition_frequency(transition_frequency)` | `-> float` | Rotating-frame transition → resonant `Pulse.carrier` |
| `transition_frequency_from_carrier(carrier)` | `-> float` | `Pulse.carrier` → rotating-frame transition |

---

## Ideal Gate Operators

**Module path:** `cqed_sim.core.ideal_gates`

All functions return QuTiP `Qobj` unitaries.

| Function | Signature | Description |
|---|---|---|
| `qubit_rotation_xy(theta, phi)` | `(float, float) -> qt.Qobj` | $\exp(-i\theta/2 [\cos\phi\,\sigma_x + \sin\phi\,\sigma_y])$. 2×2. |
| `qubit_rotation_axis(theta, axis)` | `(float, str) -> qt.Qobj` | Rotation around `"x"`, `"y"`, or `"z"` |
| `displacement_op(n_cav, alpha)` | `(int, complex) -> qt.Qobj` | $D(\alpha) = \exp(\alpha a^\dagger - \alpha^* a)$ |
| `snap_op(phases)` | `(array) -> qt.Qobj` | $\text{diag}(e^{i\phi_0}, e^{i\phi_1}, \ldots)$ |
| `sqr_op(thetas, phis)` | `(array, array) -> qt.Qobj` | $\sum_n |n\rangle\langle n| \otimes R(\theta_n, \phi_n)$ |
| `embed_qubit_op(op_q, n_cav)` | `-> qt.Qobj` | $\text{op}_q \otimes I_{\text{cav}}$ |
| `embed_cavity_op(op_c, n_tr)` | `-> qt.Qobj` | $I_{\text{qubit}} \otimes \text{op}_c$ |
| `beamsplitter_unitary(n_a, n_b, theta)` | `-> qt.Qobj` | $\exp[-i\theta(a b^\dagger + a^\dagger b)]$ |
