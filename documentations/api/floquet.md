# API Reference — Floquet Analysis (`cqed_sim.floquet`)

The Floquet module analyzes strictly periodic closed-system Hamiltonians using the same model, frame, and drive-target conventions as the rest of `cqed_sim`.

---

## Core Types

### `PeriodicDriveTerm`

```python
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
```

- Supply either `operator` or a model-aware `target`.
- `quadrature="x"` builds the Hermitian in-phase combination of the target's raising and lowering operators.
- `frequency` is the angular frequency of the periodic scalar coefficient.
- `waveform` may be a named waveform (`"cos"`, `"sin"`, `"exp"`, `"constant"`, `"square"`) or a periodic callable of the phase angle.

Useful methods:

| Method | Returns | Description |
| --- | --- | --- |
| `coefficient(t)` | `complex or ndarray` | Scalar periodic coefficient at time `t` |
| `exact_fourier_components(period)` | `dict[int, complex] or None` | Exact harmonics when available analytically |

---

### `PeriodicFourierComponent`

```python
@dataclass(frozen=True)
class PeriodicFourierComponent:
    harmonic: int
    amplitude: complex
```

Represents a Fourier coefficient multiplying `exp(i * harmonic * Omega * t)`.

---

### `FloquetProblem`

```python
@dataclass(frozen=True)
class FloquetProblem:
    period: float
    periodic_terms: Sequence[PeriodicDriveTerm] = ()
    static_hamiltonian: qt.Qobj | None = None
    model: Any | None = None
    frame: FrameSpec = FrameSpec()
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Provide either:

- `static_hamiltonian`, or
- `model`, in which case `model.static_hamiltonian(frame=...)` is used.

`period` is mandatory because the Floquet module treats commensurability explicitly rather than inferring it from approximate floating-point tone ratios.

---

### `FloquetConfig`

```python
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

`zone_center` controls how quasienergies are folded into the Floquet Brillouin zone.

---

### `FloquetResult`

```python
@dataclass
class FloquetResult:
    problem: FloquetProblem
    config: FloquetConfig
    qutip_hamiltonian: qt.Qobj | qt.QobjEvo
    floquet_basis: qt.FloquetBasis
    period_propagator: qt.Qobj
    eigenphases: ndarray
    quasienergies: ndarray
    floquet_modes_0: tuple[qt.Qobj, ...]
    bare_hamiltonian_eigenenergies: ndarray
    bare_hamiltonian_eigenstates: tuple[qt.Qobj, ...]
    bare_state_overlaps: ndarray
    dominant_bare_state_indices: ndarray
    effective_hamiltonian: qt.Qobj | None
    sambe_hamiltonian: qt.Qobj | None
    warnings: tuple[str, ...]
```

Useful methods:

| Method | Signature | Description |
| --- | --- | --- |
| `modes(t)` | `(float) -> tuple[Qobj, ...]` | Floquet modes at time `t` |
| `states(t)` | `(float) -> tuple[Qobj, ...]` | Floquet states at time `t` |
| `mode(index, t)` | `(int, float) -> Qobj` | Single mode lookup |
| `state(index, t)` | `(int, float) -> Qobj` | Single state lookup |

---

## Main Solver

### `solve_floquet`

```python
def solve_floquet(problem: FloquetProblem, config: FloquetConfig | None = None) -> FloquetResult
```

Primary closed-system Floquet solve. Internally wraps QuTiP's `FloquetBasis` and returns quasienergies, Floquet modes, the one-period propagator, overlap labels against the static Hamiltonian, and optional Sambe-space data.

---

## Builders

| Function | Purpose |
| --- | --- |
| `build_floquet_hamiltonian(problem)` | Build the periodic `QobjEvo` Hamiltonian |
| `build_target_drive_term(...)` | Construct a target-based periodic drive using model target semantics |
| `build_transmon_frequency_modulation_term(...)` | Qubit-frequency modulation via `n_q` |
| `build_mode_frequency_modulation_term(...)` | Mode-frequency modulation via `n_mode` |
| `build_dispersive_modulation_term(...)` | Dispersive modulation via `n_mode * n_q` |
| `compute_hamiltonian_fourier_components(...)` | Numerical or exact Fourier components of the periodic Hamiltonian |

---

## Analysis Helpers

| Function | Purpose |
| --- | --- |
| `compute_period_propagator(...)` | Return the one-period propagator |
| `compute_quasienergies(...)` | Return folded quasienergies |
| `compute_floquet_modes(...)` | Evaluate Floquet modes at time `t` |
| `compute_bare_state_overlaps(...)` | Overlaps with static-Hamiltonian eigenstates |
| `compute_floquet_transition_strengths(...)` | Harmonic-resolved transition matrix elements under a probe operator |
| `identify_multiphoton_resonances(...)` | Detect near-`Delta E ~= n Omega` conditions |
| `run_floquet_sweep(...)` | Solve a parameter sweep and track branches |
| `track_floquet_branches(...)` | Overlap-based branch matching and zone unwrapping |

---

## Effective Models and Sambe Helpers

| Function | Purpose |
| --- | --- |
| `build_effective_floquet_hamiltonian(...)` | Effective static Hamiltonian in the Floquet eigenbasis |
| `build_sambe_hamiltonian(...)` | Truncated harmonic-space Floquet Hamiltonian |
| `extract_sambe_quasienergies(...)` | Cluster folded Sambe eigenvalues into physical quasienergy branches |

---

## Conventions and limitations

- Floquet analysis assumes strict periodicity.
- Multi-tone drives are only supported when a common period exists.
- Quasienergies are defined modulo the drive angular frequency.
- The public API is currently closed-system; Floquet-Markov hooks are a future extension.
