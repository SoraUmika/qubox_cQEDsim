# API Reference — State Preparation & Measurement

This page documents the reusable state-preparation and measurement primitives that remain part of the `cqed_sim` library. High-level protocol recipes now live under `examples/`, not under `cqed_sim`.

---

## State Preparation

**Module path:** `cqed_sim.core.state_prep`

### SubsystemStateSpec

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

### StatePreparationSpec

```python
@dataclass(frozen=True)
class StatePreparationSpec:
    qubit: SubsystemStateSpec = qubit_state("g")
    storage: SubsystemStateSpec = vacuum_state()
    readout: SubsystemStateSpec | None = None
```

### Helpers

| Function | Description |
|---|---|
| `qubit_state(label)` | Named qubit states: `"g"`, `"e"`, `"+x"`, `"-x"`, `"+y"`, `"-y"` |
| `qubit_level(level)` | Qubit state by level index |
| `vacuum_state()` | Bosonic vacuum |
| `fock_state(level)` | Bosonic Fock state |
| `coherent_state(alpha)` | Bosonic coherent state |
| `amplitude_state(amplitudes)` | State from amplitudes |
| `density_matrix_state(rho)` | Mixed state from a density matrix |
| `prepare_state(model, spec=None)` | Build a tensor-product state matching model subsystem ordering |
| `prepare_ground_state(model)` | Convenience ground-state constructor |

---

## Qubit Measurement

**Module path:** `cqed_sim.measurement.qubit`

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

### Measurement Entry Point

| Function | Description |
|---|---|
| `measure_qubit(state, spec=None)` | Exact probabilities, optional confusion matrix, optional sampling, optional readout-chain backaction/IQ |

The confusion matrix convention remains `p_observed = M @ p_latent` with `(g, e)` ordering.

---

## Readout Chain

**Module path:** `cqed_sim.measurement.readout_chain`

### Core Dataclasses

| Dataclass | Description |
|---|---|
| `ReadoutResonator` | Single-pole dispersive readout resonator model |
| `PurcellFilter` | Frequency-selective linewidth suppression model |
| `AmplifierChain` | Linear gain plus additive thermal noise |
| `ReadoutChain` | Resonator + optional filter + amplifier + integration settings |
| `ReadoutTrace` | Time-domain cavity/output/voltage/IQ record |

### Common Methods

| API | Description |
|---|---|
| `ReadoutChain.simulate_trace(...)` | Time-domain resonator response and downconverted trace |
| `ReadoutChain.iq_centers(...)` | Noiseless I/Q centers for `g` and `e` |
| `ReadoutChain.sample_iq(...)` | Noisy I/Q sampling from latent labels |
| `ReadoutChain.classify_iq(...)` | Nearest-center classification |
| `ReadoutChain.apply_backaction(...)` | Measurement dephasing and optional Purcell relaxation |
| `ReadoutChain.gamma_meas(...)` | Measurement-induced dephasing rate |
| `ReadoutChain.purcell_rate(...)` | Purcell decay rate |
| `ReadoutChain.purcell_limited_t1(...)` | Purcell-limited `T1` |

---

## Workflow Boundary

Protocol-style orchestration is intentionally example-side now:

- `examples/protocol_style_simulation.py`
- `examples/kerr_free_evolution.py`
- `examples/kerr_sign_verification.py`
- `examples/sequential_sideband_reset.py`

Use `cqed_sim.core`, `cqed_sim.sequence`, `cqed_sim.sim`, and `cqed_sim.measurement` for reusable library code. Use `examples/` for end-to-end workflow recipes.
