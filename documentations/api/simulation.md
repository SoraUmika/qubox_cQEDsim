# API Reference — Simulation Engine (`cqed_sim.sim`)

The simulation module assembles time-dependent Hamiltonians, dispatches to QuTiP or dense-backend solvers, and provides state extractors and diagnostic tools.

---

## simulate_sequence

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

High-level entry point. Creates a `SimulationSession` and runs a single trajectory.

| Parameter | Type | Description |
|---|---|---|
| `model` | any model | Must have `operators()`, `subsystem_dims`, `static_hamiltonian()`, `drive_coupling_operators()` |
| `compiled` | `CompiledSequence` | Timeline from `SequenceCompiler.compile()` |
| `initial_state` | `qt.Qobj` | Initial ket or density matrix |
| `drive_ops` | `dict[str, ...]` | Maps pulse channel names to string targets or structured multilevel targets |
| `config` | `SimulationConfig \| None` | Solver configuration |
| `c_ops` | `Sequence[qt.Qobj] \| None` | Additional collapse operators |
| `noise` | `NoiseSpec \| None` | Lindblad noise specification |
| `e_ops` | `dict[str, qt.Qobj] \| None` | Custom observables; defaults to `default_observables(model)` |

**Solver selection:** If `config.backend` is set, uses the dense piecewise-constant solver. Otherwise, uses QuTiP's `sesolve` (pure state) or `mesolve` (density matrix / open system).

---

## SimulationConfig

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

## SimulationResult

```python
@dataclass
class SimulationResult:
    final_state: qt.Qobj                     # State at end of simulation
    states: list[qt.Qobj] | None             # Trajectory if store_states=True
    expectations: dict[str, np.ndarray]       # Time series per observable name
    solver_result: Any                        # Raw QuTiP solver result
```

---

## SimulationSession and Prepared Simulation

For high-throughput workloads (parameter sweeps, multiple initial states), prepare the session once and reuse it.

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
    # Computed:
    hamiltonian: list
    effective_c_ops: tuple
    observables: dict
    solver_options: dict
```

| Method | Signature | Description |
|---|---|---|
| `run(initial_state)` | `(qt.Qobj) -> SimulationResult` | Single-trajectory simulation |
| `run_many(initial_states, *, max_workers=1)` | `-> list[SimulationResult]` | Parallel execution via ProcessPoolExecutor |

```python
def simulate_batch(
    session: SimulationSession,
    initial_states: Iterable[qt.Qobj],
    *, max_workers: int = 1, mp_context: str = "spawn",
) -> list[SimulationResult]
```

### Supporting Functions

| Function | Signature | Description |
|---|---|---|
| `default_observables(model)` | `-> dict[str, qt.Qobj]` | P_e projector, mode quadratures & photon numbers |
| `hamiltonian_time_slices(model, compiled, drive_ops, frame)` | `-> list` | QuTiP format: `[H_0, [O_1⁺, coeff_1], ...]` |

---

## NoiseSpec

**Module path:** `cqed_sim.sim.noise.NoiseSpec`

```python
@dataclass(frozen=True)
class NoiseSpec:
    t1: float | None = None
    transmon_t1: tuple[float | None, ...] | None = None
    tphi: float | None = None
    kappa: float | None = None
    nth: float = 0.0
    kappa_storage: float | None = None
    kappa_readout: float | None = None
    nth_storage: float | None = None
    nth_readout: float | None = None
```

| Property | Returns | Description |
|---|---|---|
| `gamma1` | `float` | 1/t1 if set, else 0.0 |
| `gamma_phi` | `float` | 1/(2·tphi) if set, else 0.0 |

```python
def collapse_operators(model, noise: NoiseSpec | None) -> list[qt.Qobj]
```

Returns Lindblad jump operators:

- $\sqrt{\gamma_1} \cdot b$ (aggregate transmon relaxation when `transmon_t1` is not supplied)
- $\sqrt{1/T_{1,j}} \cdot |j-1\rangle\langle j|$ for each explicit ladder transition
- $\sqrt{\gamma_\phi} \cdot \sigma_z$ (2-level) or $\sqrt{\gamma_\phi} \cdot n_q$ (multilevel)
- $\sqrt{\kappa(n_{\text{th}}+1)} \cdot a$ and $\sqrt{\kappa \cdot n_{\text{th}}} \cdot a^\dagger$ per bosonic mode

---

## Coupling Helpers

**Module path:** `cqed_sim.sim.couplings`

| Function / Class | Signature | Description |
|---|---|---|
| `cross_kerr(a, b, chi)` | `-> Qobj` | $\chi \cdot a^\dagger a \cdot b^\dagger b$ |
| `self_kerr(a, kerr)` | `-> Qobj` | $(K/2) \cdot a^{\dagger 2} a^2$ |
| `exchange(a, b, coupling)` | `-> Qobj` | $J \cdot (a^\dagger b + a b^\dagger)$ |
| `TunableCoupler` | dataclass | Flux-tunable coupler: `j_max`, `flux_period`, `phase_offset`, `dc_offset` |
| `TunableCoupler.exchange_rate(flux)` | `-> float` | dc_offset + j_max·cos(...) |
| `TunableCoupler.operator(a, b, flux)` | `-> Qobj` | Exchange Hamiltonian at given flux |

---

## State Extractors

**Module path:** `cqed_sim.sim.extractors`

### Partial-Trace Extractors

| Function | Signature | Description |
|---|---|---|
| `reduced_qubit_state(state)` | `(Qobj) -> Qobj` | Trace out everything except qubit |
| `reduced_transmon_state(state)` | `(Qobj) -> Qobj` | Alias for `reduced_qubit_state` |
| `reduced_cavity_state(state)` | `(Qobj) -> Qobj` | Trace out qubit (2-mode only) |
| `reduced_storage_state(state)` | `(Qobj) -> Qobj` | Storage subsystem (index 1) |
| `reduced_readout_state(state)` | `(Qobj) -> Qobj` | Readout subsystem (index 2) |
| `reduced_subsystem_state(state, subsystem)` | `(Qobj, int\|str) -> Qobj` | General partial trace |

### Bloch Coordinates

| Function | Returns | Description |
|---|---|---|
| `bloch_xyz_from_joint(state)` | `(x, y, z)` | Bloch vector from joint state |
| `conditioned_bloch_xyz(state, n, fallback)` | `(x, y, z, p_n, valid)` | Bloch vector conditioned on Fock level n |
| `conditioned_qubit_state(state, n, fallback)` | `(rho_q, p_n, valid)` | Qubit state conditioned on Fock n |
| `conditioned_population(state, n)` | `float` | Probability of Fock state n |

### Multilevel Population Helpers

| Function | Returns | Description |
|---|---|---|
| `subsystem_level_population(state, subsystem, level)` | `float` | Population in a chosen level |
| `transmon_level_populations(state)` | `dict[int, float]` | All transmon level populations |
| `compute_shelving_leakage(initial, final, shelved_level, subsystem)` | `float` | Population change of shelved level |

### Mode Moments and Photon Numbers

| Function | Returns | Description |
|---|---|---|
| `mode_moments(state, subsystem, dim)` | `dict` | `{"a": ⟨a⟩, "adag_a": ⟨a†a⟩, "n": ⟨n⟩}` |
| `cavity_moments(state, n_cav)` | `dict` | Cavity mode moments |
| `storage_moments(state, n_storage)` | `dict` | Storage mode moments |
| `readout_moments(state, n_readout)` | `dict` | Readout mode moments |
| `storage_photon_number(state)` | `float` | $\langle n_s \rangle$ |
| `readout_photon_number(state)` | `float` | $\langle n_r \rangle$ |
| `joint_expectation(state, operator)` | `complex` | $\text{Tr}[\rho \cdot O]$ |

### Qubit-Conditioned Extractors

| Function | Returns | Description |
|---|---|---|
| `qubit_conditioned_subsystem_state(state, subsystem, qubit_level, fallback)` | `(rho, p, valid)` | Subsystem state conditioned on qubit |
| `qubit_conditioned_mode_moments(state, subsystem, qubit_level)` | `dict` | Moments conditioned on qubit state |
| `readout_response_by_qubit_state(state)` | `dict[int, dict]` | `{0: moments_g, 1: moments_e}` |

### Wigner Function

```python
def cavity_wigner(
    rho_c: qt.Qobj,
    xvec=None, yvec=None,
    n_points: int = 41,
    extent: float = 4.0,
    coordinate: str = "quadrature",  # or "alpha"/"coherent"
) -> tuple[ndarray, ndarray, ndarray]
```

Returns `(xvec, yvec, W)`. `"quadrature"` uses natural units; `"alpha"` scales by $\sqrt{2}$.

---

## Diagnostics

**Module path:** `cqed_sim.sim.diagnostics`

| Function | Signature | Description |
|---|---|---|
| `channel_norms(compiled)` | `-> dict[str, float]` | L2 norm of distorted waveform per channel |
| `instantaneous_phase_frequency(signal, dt)` | `-> (phase, omega)` | Unwrapped phase and instantaneous frequency |
