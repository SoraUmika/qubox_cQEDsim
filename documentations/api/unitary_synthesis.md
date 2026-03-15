# API Reference: Unitary Synthesis (`cqed_sim.unitary_synthesis`)

Flexible gate-sequence synthesis for matrix-defined primitives, model-backed waveform primitives, unitary targets, state-mapping targets, and Phase 2 constraint-aware optimization.

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

Highlights:

- `TargetUnitary` still validates unitarity on construction.
- `ignore_global_phase`, `allow_diagonal_phase`, and `phase_blocks` define phase-equivalence classes for the closed-system fidelity metric.
- When noisy/open-system synthesis is requested for a unitary target, the target is evaluated through probe-state propagation instead of direct unitary extraction.
- `TargetStateMapping` accepts either plural state lists or a single `initial_state` / `target_state` pair.

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
```

```python
@dataclass(frozen=True)
class MultiObjective:
    fidelity_weight: float = 1.0
    leakage_weight: float = 0.0
    duration_weight: float = 0.0
    pulse_power_weight: float = 0.0
    robustness_weight: float = 0.0
    smoothness_weight: float = 0.0
    hardware_penalty_weight: float = 1.0
```

```python
ParameterDistribution(
    sample_count=4,
    aggregate="mean",
    chi=Normal(-2.8e6, 0.05e6),
    kerr=Uniform(-3200.0, -2800.0),
)
```

These objects lift the older low-level `hardware_limits`, `constraints`, and leakage knobs into a notebook-facing API.

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
- `system=...` is the preferred architecture-facing entry point for future backends.
- `model=...` remains supported for cQED usage and is auto-wrapped into `CQEDSystemAdapter`.
- `synthesis_constraints.max_amplitude` is compiled into the existing hardware-amplitude limits.
- `synthesis_constraints.max_duration` can act as a hard projection or a penalty term.
- `parameter_distribution` averages or worst-cases the simulation objective across sampled model variants.
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
- leakage and robustness summaries
- sequence parameters and time-grid details
- optimizer and multistart history metadata

`ParetoFrontResult` stores the full set of weighted runs plus the nondominated subset.

---

## Metrics

Key metrics in `cqed_sim.unitary_synthesis.metrics`:

- `subspace_unitary_fidelity(...)`
- `leakage_metrics(...)`
- `state_leakage_metrics(...)`
- `state_mapping_metrics(...)`
- `objective_breakdown(...)`
- `unitarity_error(...)`

Phase handling now supports `none`, `global`, `diagonal`, and `block` gauges.
