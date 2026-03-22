# API Reference - State Preparation & Measurement

This page documents the reusable state-preparation and measurement primitives that remain part of the `cqed_sim` library. Guided notebook tutorials live under `tutorials/`, while standalone protocol recipes live under `examples/`.

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

Confusion-matrix convention: `p_observed = M @ p_latent` with `(g, e)` ordering.

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
| `ReadoutChain.simulate_waveform(...)` | Time-domain replay for an arbitrary complex drive waveform |
| `ReadoutChain.iq_centers(...)` | Noiseless I/Q centers for `g` and `e` |
| `ReadoutChain.sample_iq(...)` | Noisy I/Q sampling from latent labels |
| `ReadoutChain.classify_iq(...)` | Nearest-center classification |
| `ReadoutChain.apply_backaction(...)` | Measurement dephasing and optional Purcell relaxation |
| `ReadoutChain.gamma_meas(...)` | Measurement-induced dephasing rate |
| `ReadoutChain.purcell_rate(...)` | Purcell decay rate |
| `ReadoutChain.purcell_limited_t1(...)` | Purcell-limited `T1` |

---

## Continuous Readout Replay

**Module path:** `cqed_sim.measurement.stochastic`

### Core Dataclasses

| Dataclass | Description |
|---|---|
| `ContinuousReadoutSpec` | SME replay options: frame, monitored subsystem, number of trajectories, storage policy |
| `ContinuousReadoutTrajectory` | One trajectory's measurement record, final state, optional states, and expectations |
| `ContinuousReadoutResult` | Aggregate replay result with average expectations and all trajectories |

### Common APIs

| API | Description |
|---|---|
| `simulate_continuous_readout(...)` | QuTiP `smesolve(...)` wrapper using `cqed_sim` drive/noise conventions |
| `integrate_measurement_record(...)` | Integrate a homodyne or heterodyne record over its final time axis |

The monitored path is constructed from `cqed_sim.sim.split_collapse_operators(...)`: one selected bosonic emission channel is promoted to the stochastic measurement path, while relaxation, thermal excitation, and dephasing remain ordinary Lindblad terms.

---

## Strong-Readout Disturbance Helpers

**Module path:** `cqed_sim.measurement.strong_readout`

### Core Dataclasses

| Dataclass | Description |
|---|---|
| `StrongReadoutMixingSpec` | Occupancy- and slew-activated phenomenological strong-readout model |
| `StrongReadoutDisturbance` | Returned envelopes, activation profile, and occupancy estimate |

### Common APIs

| API | Description |
|---|---|
| `build_strong_readout_disturbance(...)` | Build auxiliary `g-e` / `e-f` disturbance envelopes from a readout waveform |
| `strong_readout_drive_targets(...)` | Matching `TransmonTransitionDriveSpec` mapping for those channels |
| `infer_dispersive_coupling(...)` | Infer `g` from dispersive parameters |
| `estimate_dispersive_critical_photon_number(...)` | Estimate `n_crit = (Delta / 2g)^2` |

This helper is intentionally operational rather than microscopic: it is designed for calibrated threshold studies where large readout occupancy and waveform slew act as proxies for non-QND disturbance.

For higher-level continuation studies, `StrongReadoutMixingSpec` accepts
`higher_ladder_scales`, `higher_ladder_start_level`, and `higher_channel_prefix`.
Those fields let one promote the calibrated `e-f` disturbance envelope onto additional
adjacent transmon transitions. `strong_readout_drive_targets(...)` can then return
channels such as `mix_high_2_3`, `mix_high_3_4`, ... up to an optional
`max_transmon_level`, and `build_strong_readout_disturbance(...)` returns the matching
envelopes in `StrongReadoutDisturbance.higher_envelopes`.

---

## Workflow Boundary

Protocol-style orchestration is intentionally repository-side:

- guided walkthroughs live under `tutorials/`
- standalone protocol recipes live under `examples/`
- the library layer provides reusable building blocks, not canned experimental workflows

Representative repo-side entry points:

- `tutorials/README.md`
- `tutorials/00_tutorial_index.ipynb`
- `tutorials/03_cavity_displacement_basics.ipynb`
- `tutorials/17_readout_resonator_response.ipynb`
- `examples/protocol_style_simulation.py`
- `examples/continuous_readout_replay_demo.py`
- `examples/sequential_sideband_reset.py`
