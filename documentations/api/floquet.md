# API Reference — Floquet Analysis (`cqed_sim.floquet`)

The Floquet module analyzes strictly periodic Hamiltonians using the same model, frame, and drive-target conventions as the rest of `cqed_sim`, with both closed-system spectral analysis and a Floquet-Markov open-system wrapper.

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

`overlap_reference_time` is the default mode-evaluation time used by `run_floquet_sweep(...)`
when no explicit `reference_time` is supplied.

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

### `FloquetMarkovBath`

```python
@dataclass(frozen=True)
class FloquetMarkovBath:
    operator: qt.Qobj
    spectrum: Callable[[ndarray], ndarray] | None = None
    label: str | None = None
```

One bath coupling operator and its spectral-density callback for Floquet-Markov evolution.

### `FloquetMarkovConfig`

```python
@dataclass(frozen=True)
class FloquetMarkovConfig:
    floquet: FloquetConfig = FloquetConfig()
    kmax: int = 5
    nT: int | None = None
    w_th: float = 0.0
    store_states: bool | None = None
    store_final_state: bool = True
    store_floquet_states: bool = False
    normalize_output: bool = True
```

Wraps Floquet basis settings plus QuTiP Floquet-Markov solver controls.

### `FloquetMarkovResult`

```python
@dataclass
class FloquetMarkovResult:
    floquet_result: FloquetResult
    config: FloquetMarkovConfig
    baths: tuple[FloquetMarkovBath, ...]
    tlist: ndarray
    solver_result: Any
```

Exposes convenience accessors for `states`, `expect`, `times`, `final_state`, and `floquet_states`.

---

## Main Solver

### `solve_floquet`

```python
def solve_floquet(problem: FloquetProblem, config: FloquetConfig | None = None) -> FloquetResult
```

Primary closed-system Floquet solve. Internally wraps QuTiP's `FloquetBasis` and returns quasienergies, Floquet modes, the one-period propagator, overlap labels against the static Hamiltonian, and optional Sambe-space data.

### `solve_floquet_markov`

```python
def solve_floquet_markov(problem, initial_state, tlist, *, baths=None, noise=None, e_ops=None, args=None, config=None)
```

Runs dissipative Floquet-Markov evolution while reusing the Floquet basis generated from `solve_floquet(...)`.

Provide either explicit `baths=[FloquetMarkovBath(...), ...]` or `noise=NoiseSpec(...)` with `FloquetProblem(model=...)` to use the convenience bridge.

---

## Builders

| Function | Purpose |
| --- | --- |
| `build_floquet_hamiltonian(problem)` | Build the periodic `QobjEvo` Hamiltonian |
| `build_target_drive_term(...)` | Construct a target-based periodic drive using model target semantics |
| `build_transmon_frequency_modulation_term(...)` | Qubit-frequency modulation via `n_q` |
| `build_mode_frequency_modulation_term(...)` | Mode-frequency modulation via `n_mode` |
| `build_dispersive_modulation_term(...)` | Dispersive modulation via `n_mode * n_q` |
| `build_floquet_markov_baths(problem, noise, spectrum=...)` | Convert a repository `NoiseSpec` into Floquet-Markov bath operators |
| `flat_markov_spectrum(scale=1.0)` | Convenience white-spectrum callback |
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
| `run_floquet_sweep(...)` | Solve a parameter sweep and track branches, defaulting to `config.overlap_reference_time` for overlap evaluation |
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
- `solve_floquet(...)` remains the closed-system spectral-analysis path.
- `solve_floquet_markov(...)` is the current Markovian open-system Floquet path.
- The `NoiseSpec` bridge is a convenience wrapper; use explicit `FloquetMarkovBath(...)` inputs for custom spectra and coupling models.
