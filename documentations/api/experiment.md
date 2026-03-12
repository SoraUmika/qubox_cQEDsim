# API Reference — Experiment Layer (`cqed_sim.experiment`)

The experiment module provides state preparation, qubit measurement, the `SimulationExperiment` convenience wrapper, the physical readout chain, and Kerr free-evolution protocols.

---

## State Preparation

**Module path:** `cqed_sim.experiment.state_prep`

### SubsystemStateSpec

```python
@dataclass(frozen=True)
class SubsystemStateSpec:
    kind: str       # "qubit", "vacuum", "fock", "coherent", "amplitudes", "density_matrix"
    label: str | None = None
    level: int | None = None
    alpha: complex | None = None
    amplitudes: Any | None = None
    density_matrix: Any | None = None
```

### StatePreparationSpec

```python
@dataclass(frozen=True)
class StatePreparationSpec:
    qubit: SubsystemStateSpec = qubit_state("g")
    storage: SubsystemStateSpec = vacuum_state()
    readout: SubsystemStateSpec | None = None  # For 3-mode models
```

### State Constructor Helpers

| Function | Signature | Description |
|---|---|---|
| `qubit_state(label)` | `(str) -> SubsystemStateSpec` | Labels: `"g"`, `"e"`, `"+x"`, `"-x"`, `"+y"`, `"-y"` |
| `qubit_level(level)` | `(int) -> SubsystemStateSpec` | Qubit by Fock level |
| `vacuum_state()` | `() -> SubsystemStateSpec` | Cavity ground state \|0⟩ |
| `fock_state(level)` | `(int) -> SubsystemStateSpec` | Cavity Fock state \|n⟩ |
| `coherent_state(alpha)` | `(complex) -> SubsystemStateSpec` | Coherent state \|α⟩ |
| `amplitude_state(amplitudes)` | `(Any) -> SubsystemStateSpec` | Arbitrary state from amplitudes |
| `density_matrix_state(rho)` | `(Any) -> SubsystemStateSpec` | State from density matrix |

### Preparation Functions

```python
def prepare_state(model, spec: StatePreparationSpec | None = None) -> qt.Qobj
def prepare_ground_state(model) -> qt.Qobj
```

`prepare_state` builds the tensor-product initial state. Supports 2-mode and 3-mode models automatically. Defaults to |g⟩|0⟩ (or |g⟩|0⟩|0⟩) if spec is None.

---

## Qubit Measurement

**Module path:** `cqed_sim.experiment.measurement`

### QubitMeasurementSpec

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

### QubitMeasurementResult

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

### measure_qubit

```python
def measure_qubit(state: qt.Qobj, spec: QubitMeasurementSpec | None = None) -> QubitMeasurementResult
```

**Pipeline:**

1. Extract reduced qubit state via `reduced_qubit_state()`
2. Compute latent probabilities (p_g, p_e), lumping higher levels
3. Apply confusion matrix: $p_{\text{obs}} = M \cdot p_{\text{latent}}$
4. If readout chain attached: apply backaction (dephasing, Purcell), generate IQ
5. If shots requested: sample from p_obs (or classify from IQ)

**Confusion matrix convention:** column ordering is (g, e).

---

## SimulationExperiment

**Module path:** `cqed_sim.experiment.protocol.SimulationExperiment`

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

### ExperimentResult

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

## Readout Chain

**Module path:** `cqed_sim.experiment.readout_chain`

Physical readout model with resonator, Purcell filter, and amplifier chain.

### ReadoutResonator

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

| Method | Returns | Description |
|---|---|---|
| `dispersive_shift(qubit_state, chi)` | `float` | 0 for "g", +χ for "e" |
| `steady_state_amplitude(qubit_state, ...)` | `complex` | $\alpha_{ss} = -i\varepsilon / (\kappa_{\text{eff}}/2 + i(\omega_{r,q} - \omega_d))$ |
| `gamma_meas(...)` | `float` | Measurement-induced dephasing rate |
| `purcell_rate(omega_q, ...)` | `float` | Purcell decay rate |
| `response_trace(qubit_state, *, duration, dt, ...)` | `(tlist, field)` | Time-domain cavity response |

### PurcellFilter

```python
@dataclass(frozen=True)
class PurcellFilter:
    omega_f: float | None = None
    bandwidth: float | None = None
    quality_factor: float | None = None
    coupling_rate: float | None = None
```

Implements: $\kappa_{\text{eff}}(\omega) = \frac{4J^2 \kappa_f}{\kappa_f^2 + 4(\omega - \omega_f)^2}$

### AmplifierChain

```python
@dataclass(frozen=True)
class AmplifierChain:
    noise_temperature: float = 0.0
    gain: float = 1.0
    impedance_ohm: float = 50.0
    mixer_phase: float = 0.0
```

| Method | Description |
|---|---|
| `noise_std(dt, n_samples)` | RMS noise per quadrature |
| `amplify(trace, *, dt, seed)` | Apply gain and thermal noise |
| `mix_down(trace)` | Multiply by exp(−i·phase) |
| `iq_sample(trace)` | Integrated I/Q: [mean_I, mean_Q] |

### ReadoutChain

```python
@dataclass
class ReadoutChain:
    resonator: ReadoutResonator
    amplifier: AmplifierChain = AmplifierChain()
    purcell_filter: PurcellFilter | None = None
    integration_time: float = 1e-6
    dt: float = 4e-9
```

| Method | Description |
|---|---|
| `simulate_trace(qubit_state, ...)` | Full readout trace → `ReadoutTrace` |
| `iq_centers(...)` | Noiseless IQ centers |
| `sample_iq(latent_states, ...)` | Noisy IQ samples (n × 2) |
| `classify_iq(iq_samples, ...)` | Nearest-neighbor classification → 0/1 |
| `apply_backaction(rho_q, *, omega_q, ...)` | Measurement dephasing ± Purcell relaxation |

---

## Kerr Free Evolution

**Module path:** `cqed_sim.experiment.kerr_free_evolution`

### KerrParameterSet

```python
@dataclass(frozen=True)
class KerrParameterSet:
    name: str
    omega_q_hz: float; omega_c_hz: float; omega_ro_hz: float
    alpha_q_hz: float; kerr_hz: float; kerr2_hz: float = 0.0
    chi_hz: float = 0.0; chi2_hz: float = 0.0; chi3_hz: float = 0.0
```

Has `build_model(*, n_cav=28, n_tr=3)` converting Hz → rad/s.

Predefined sets: `KERR_FREE_EVOLUTION_PARAMETER_SETS` with keys `"phase_evolution"` and `"value_2"`.

### Main Functions

| Function | Description |
|---|---|
| `run_kerr_free_evolution(times_s, *, ...)` | Full free-evolution with optional Wigner snapshots → `KerrFreeEvolutionResult` |
| `build_kerr_free_evolution_model(parameter_set, ...)` | Build model from preset |
| `build_kerr_free_evolution_frame(model, *, use_rotating_frame)` | Build FrameSpec |
| `verify_kerr_sign(...)` | Compare documented Kerr sign against flipped-sign control |
| `plot_kerr_wigner_snapshots(result, ...)` | Grid of Wigner function snapshots |

### Result Dataclasses

```python
@dataclass
class KerrEvolutionSnapshot:
    time_s: float; time_us: float; joint_state: qt.Qobj
    cavity_state: qt.Qobj; cavity_mean: complex
    cavity_photon_number: float; wigner: dict | None

@dataclass
class KerrFreeEvolutionResult:
    parameter_set: KerrParameterSet
    model: DispersiveTransmonCavityModel
    frame: FrameSpec; initial_state: qt.Qobj
    state_prep: StatePreparationSpec
    snapshots: list[KerrEvolutionSnapshot]; metadata: dict

@dataclass(frozen=True)
class KerrSignVerificationResult:
    documented_kerr_hz: float; flipped_kerr_hz: float
    cavity_mean_documented: complex; cavity_mean_flipped: complex
    documented_phase_rad: float; flipped_phase_rad: float
    matches_documented_sign: bool
```
