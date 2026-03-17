# API Reference: Unitary Synthesis (`cqed_sim.unitary_synthesis`)

Flexible gate-sequence synthesis for matrix-defined primitives, model-backed waveform primitives, unitary targets, state-mapping targets, reduced-state targets, isometry targets, channel/process targets, observable targets, trajectory/checkpoint targets, and relevance-aware multi-objective optimization.

!!! note "Convention"
    The synthesis drift-phase layer matches the runtime dispersive/Kerr convention. Model-backed waveform primitives use the same `Pulse`, `SequenceCompiler`, and `cqed_sim.sim` stack as the rest of the library, including the waveform sign convention `Pulse.carrier = -omega_transition(frame)`.

---

## System Interface

```python
class QuantumSystem:
    def hilbert_dimension(...): ...
    def simulate_sequence(...): ...
    def simulate_unitary(...): ...
    def simulate_state(...): ...
```

```python
CQEDSystemAdapter(model=my_cqed_model)
```

Highlights:

- `UnitarySynthesizer` now depends on a `QuantumSystem` backend rather than directly on a raw cQED model.
- `CQEDSystemAdapter` preserves the current cQED workflow by wrapping existing `cqed_sim.core` model objects.
- `model=...` is still accepted by `UnitarySynthesizer` and is automatically wrapped into `CQEDSystemAdapter(...)` for backward compatibility.
- The synthesizer is now architecturally system-agnostic even though only cQED adapters are implemented today.

---

## Core Targets

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

Highlights:

- `TargetUnitary` still validates unitarity on construction.
- `ignore_global_phase`, `allow_diagonal_phase`, and `phase_blocks` define phase-equivalence classes for the closed-system fidelity metric.
- When noisy/open-system synthesis is requested for a unitary target, the target is evaluated through probe-state propagation instead of direct unitary extraction.
- `TargetStateMapping` accepts either plural state lists or a single `initial_state` / `target_state` pair.
- `TargetReducedStateMapping` compares only retained subsystems after partial trace, so the task can ignore irrelevant spectator motion or ancillary entanglement.
- `TargetIsometry` matches only the logical columns that matter for a state-injection or encoding map rather than forcing a full square unitary target.
- `TargetChannel` supports process-style matching from a unitary, Kraus list, superoperator, or Choi matrix and can reconstruct reduced subsystem channels from a fixed environment state.
- `ObservableTarget` lets the task be defined directly in terms of expectation values on a weighted ensemble of relevant states.
- `TrajectoryTarget` evaluates the protocol at explicit checkpoints rather than only at the final operator or final state.

---

## Primitive Gates

```python
PrimitiveGate(
    name="ry",
    duration=40e-9,
    matrix=lambda params, model: build_unitary(params),
    parameters={"theta": 0.2, "duration": 40e-9},
    parameter_bounds={"theta": (-np.pi, np.pi), "duration": (10e-9, 100e-9)},
)
```

```python
PrimitiveGate(
    name="drive",
    duration=80e-9,
    waveform=waveform_function,  # waveform_function(params, model)
    parameters={"amp": 0.1, "phase": 0.0, "duration": 80e-9},
    parameter_bounds={"amp": (-1.0, 1.0), "phase": (-np.pi, np.pi), "duration": (20e-9, 200e-9)},
    hilbert_dim=full_dim,
)
```

Waveform primitives may return:

- `list[Pulse]`
- `(pulses, drive_ops)`
- `(pulses, drive_ops, meta)`
- `{"pulses": ..., "drive_ops": ..., "meta": ...}`

Common cQED gate dataclasses exposed by `cqed_sim.unitary_synthesis.sequence`:

| Class | Gate-specific fields | Physics |
|---|---|---|
| `QubitRotation` | `theta: float`, `phi: float` | Qubit rotation $R(\theta, \phi)$ |
| `Displacement` | `alpha: complex` | Cavity displacement $D(\alpha)$ |
| `SQR` | `theta_n`, `phi_n`, `tones`, `tone_freqs`, `include_conditional_phase`, `drift_model` | Selective qubit rotation |
| `CavityBlockPhase` | `phases: list[float]`, `fock_levels: tuple[int, ...]` | Ideal cavity-only logical block-phase gate acting identically on both qubit states |
| `SNAP` | `phases: list[float]` | Number-selective contiguous cavity phase gate |
| `ConditionalPhaseSQR` | `phases_n: list[float]`, `drift_model: DriftPhaseModel` | Conditional phase via free-evolution SQR block |
| `FreeEvolveCondPhase` | `drift_model: DriftPhaseModel` | Free-evolution conditional phase with no explicit drive pulse |

---

## Phase 2 Configuration Objects

```python
@dataclass(frozen=True)
class SynthesisConstraints:
    max_amplitude: float | None = None
    max_duration: float | None = None
    max_primitives: int | None = None
    allowed_primitive_counts: tuple[int, ...] = ()
    smoothness_penalty: bool = False
    smoothness_weight: float = 1.0
    max_bandwidth: float | None = None
    bandwidth_weight: float = 1.0
    forbidden_parameter_ranges: dict[str, tuple[tuple[float, float], ...]] = ...
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
```

```python
@dataclass(frozen=True)
class ExecutionOptions:
    engine: str = "auto"          # auto | legacy | numpy | jax
    fallback_engine: str = "legacy"
    device: str = "auto"          # auto | cpu | gpu
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

These objects lift the older low-level `hardware_limits`, `constraints`, and leakage knobs into a notebook-facing API. `LeakagePenalty.checkpoint_weight` adds a soft intermediate-leakage cost, `MultiObjective.gate_count_weight` measures effective active-gate count for fixed-structure sequences, and `ExecutionOptions` selects the legacy versus accelerated evaluation backend.

---

## UnitarySynthesizer

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
        ...
    )
```

```python
def fit(target=None, init_guess="heuristic", multistart=1, maxiter=300) -> SynthesisResult
```

```python
def explore_pareto(weight_sets, *, target=None, init_guess="heuristic", multistart=1, maxiter=300) -> ParetoFrontResult
```

Important behavior:

- Closed-system unitary targets still use direct unitary fidelity on the selected subspace.
- Noisy/open-system unitary targets now use probe-state fidelity automatically.
- `ObservableTarget`, `TrajectoryTarget`, `TargetReducedStateMapping`, `TargetIsometry`, and `TargetChannel` are all first-class task objectives and participate in the same leakage, duration, compactness, and robustness framework.
- `system=...` is the preferred architecture-facing entry point for future backends.
- `model=...` remains supported for cQED usage and is auto-wrapped into `CQEDSystemAdapter`.
- `synthesis_constraints.max_amplitude` is compiled into the existing hardware-amplitude limits.
- `synthesis_constraints.max_duration` can act as a hard projection or a penalty term.
- `parameter_distribution` averages or worst-cases the simulation objective across sampled model variants.
- `execution=ExecutionOptions(...)` selects the legacy evaluator or the accelerated ideal closed-system path. The current fast path covers `backend="ideal"` unitary, state-mapping, isometry, observable, and trajectory problems and falls back automatically for waveform-backed, noisy, reduced-state, or channel objectives.
- `warm_start` accepts a saved JSON payload, a mapping, or a previous `SynthesisResult`.

---

## Results and Diagnostics

```python
result = synth.fit(...)
result.save("optimized_gate.json")
result.plot_convergence()
payload = result.to_payload()
```

`SynthesisResult.report` now records:

- target/gauge metadata
- objective-component breakdown
- execution-backend selection and fallback reason
- residual and checkpoint leakage summaries
- reduced-state, channel, observable, and trajectory task metrics when applicable
- truncation-edge and outside-tail population diagnostics for Hilbert-cutoff sanity checks
- sequence parameters and time-grid details
- optimizer and multistart history metadata

`ParetoFrontResult` stores the full set of weighted runs plus the nondominated subset.

---

## Metrics

Key metrics in `cqed_sim.unitary_synthesis.metrics`:

- `subspace_unitary_fidelity(...)`
- `leakage_metrics(...)`
- `logical_block_phase_diagnostics(...)`
- `state_leakage_metrics(...)`
- `state_mapping_metrics(...)`
- `channel_action_metrics(...)`
- `observable_expectation_metrics(...)`
- `objective_breakdown(...)`
- `truncation_sanity_metrics(...)`
- `operator_truncation_sanity_metrics(...)`
- `unitarity_error(...)`

`logical_block_phase_diagnostics(...)` extracts gauge-fixed block-overlap phases from a restricted operator, reports the best-fit ideal cavity-only block-phase correction, and returns the residual RMS after an optional applied correction.

Phase handling now supports `none`, `global`, `diagonal`, and `block` gauges.
