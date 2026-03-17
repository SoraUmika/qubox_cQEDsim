# Phase 1 Design And Implementation Plan: RL-Ready Hybrid cQED Control

Date: March 17, 2026

## Design Goal

Add a serious first-pass RL control stack without creating a parallel simulator. The design keeps the existing pulse compilation, Hamiltonian construction, Lindblad evolution, and readout-chain code as the source of truth and adds an RL-facing layer that orchestrates those existing components.

## Chosen Architecture

### New package: `cqed_sim.rl_control`

The new RL-facing package contains:

- `configs.py`
  - system-regime and environment configuration dataclasses.
- `action_spaces.py`
  - continuous parametric pulse actions,
  - primitive / hierarchical actions,
  - waveform-level action scaffolding.
- `runtime.py`
  - pulse generation,
  - distortion / compilation wrapper,
  - model factory,
  - open-system engine,
  - measurement wrapper,
  - classical measurement post-processing.
- `observations.py`
  - ideal summary observations,
  - reduced-density observations,
  - measurement-like IQ / counts observations,
  - history stacking wrapper.
- `rewards.py`
  - shaped fidelity rewards,
  - process-fidelity rewards,
  - measurement-proxy rewards,
  - leakage / ancilla-return / control-cost penalties,
  - composable reward aggregation.
- `metrics.py`
  - state and process metrics,
  - leakage metrics,
  - ancilla-return metrics,
  - robustness sweep statistics.
- `tasks.py`
  - benchmark task dataclass,
  - state-preparation and gate-synthesis task factories,
  - scripted baseline action sequences.
- `diagnostics.py`
  - rollout diagnostics and physics-debug snapshots.
- `domain_randomization.py`
  - priors,
  - deterministic seeded sampling,
  - train/eval parameter distributions,
  - per-episode metadata export.
- `demonstrations.py`
  - scripted demonstration rollouts and export-friendly records.
- `env.py`
  - `HybridCQEDEnv`, the RL-facing environment.

### New package: `cqed_sim.system_id`

Add light scaffolding for future system-identification workflows:

- parameter-prior aliases,
- calibration-evidence containers,
- posterior-to-randomizer hook functions.

This is intentionally a scaffold, not a full inference engine.

## Physics Model Strategy

### Regime A: full multilevel pulse model

Implemented through `UniversalCQEDModel` with:

- multilevel transmon Duffing nonlinearity,
- bosonic self-Kerr,
- optional residual cross-Kerr,
- explicit transmon-storage exchange coupling through `ExchangeSpec`,
- pulse drives on transmon and bosonic channels,
- optional effective sideband drives through the existing structured sideband target path.

This is the slower but more physics-rich regime for pulse-level control studies.

### Regime B: reduced dispersive model

Implemented through `DispersiveTransmonCavityModel`.

This is the faster regime for higher-throughput RL iteration and conditional-phase style tasks.

### Coexistence strategy

Both regimes use the same environment wrapper and the same action / observation / reward interfaces. The difference is isolated in the runtime model factory and transition-frequency helpers.

## Separation Of Concerns

### Pulse generation

Action spaces parse raw user / policy actions into structured control requests. `PulseGenerator` in `runtime.py` converts those requests into actual `Pulse` objects plus the drive-operator map.

### Distortion model

`SequenceCompiler` and `HardwareConfig` remain the source of truth for channel distortions. `DistortionModel` in `runtime.py` is only a thin wrapper over the existing compiler.

### Hamiltonian model

`HamiltonianModelFactory` builds either the full multilevel model or the reduced dispersive model from a single environment config.

### Open-system engine

`OpenSystemEngine` wraps `prepare_simulation(...)`, `SimulationSession`, and `simulate_sequence(...)`.

### Measurement model

`MeasurementModel` wraps `measure_qubit(...)` and existing readout-chain simulation. Measurement updates are intentionally limited to:

- observation generation,
- optional approximate projective ancilla collapse.

No SME or continuous conditioned-state update is claimed in this first pass.

### Classical processing

Observation encoders and reward terms consume measurement results and simulator diagnostics without modifying the physics layer.

## Core Interfaces

### Environment

```python
env = HybridCQEDEnv(config)
obs, info = env.reset(seed=..., options=...)
obs, reward, terminated, truncated, info = env.step(action)
diagnostics = env.render_diagnostics()
metrics = env.estimate_metrics(policy_or_actions, ...)
```

### Benchmark task

Each task defines:

- task kind: state preparation or unitary synthesis,
- initial state,
- target state or target operator,
- optional tracked subspace,
- episode horizon,
- success threshold,
- default scripted baseline actions,
- expected diagnostics.

### Reward model

The reward path is composable. A `CompositeReward` sums named reward terms and returns a scalar reward plus a full breakdown.

### Observation model

Observation encoders are independent objects, so the same task can be evaluated under ideal, measurement-like, or history-aware observations.

## Measurement Realism

Measurement realism enters through the existing `QubitMeasurementSpec` and `ReadoutChain`. The first pass supports:

- assignment error via confusion matrices,
- additive IQ noise,
- repeated-shot summary statistics,
- classifier-based observations,
- optional projective ancilla collapse for branching experiments.

## Reproducibility

- `reset(seed=...)` seeds environment sampling deterministically.
- Domain randomization uses NumPy generators with explicit seed splitting.
- Episode metadata always includes the sampled hidden parameters and the randomization mode.

## System Identification Hook Path

The environment will accept a `DomainRandomizer` directly, but the design reserves a future `system_id` path that can build that randomizer from calibration evidence or posterior summaries. This makes it easy to add:

- fit-then-randomize workflows,
- posterior sampling,
- calibration-informed evaluation distributions.

## First-Pass Benchmark Suite

The initial benchmark ladder will include:

- vacuum preservation,
- coherent-state preparation,
- even-cat preparation,
- ancilla-storage Bell-state preparation,
- a reduced-model conditional-phase gate benchmark on a selected qubit-cavity subspace.

The regression tests will focus on deterministic baselines for:

- coherent-state preparation,
- reduced-model conditional-phase gate synthesis.

## Numerical Strategy

- keep Hilbert spaces small in the first-pass tests,
- reuse `SimulationSession` for repeated probe-state propagation,
- use opt-in diagnostics for Wigner calculations,
- keep waveform-level control as a scaffold rather than the default training mode.

## Non-Goals Of The First Pass

The first pass will not claim:

- full SME support,
- hardware-calibrated readout cavity ringdown memory effects,
- differentiable backpropagation through the solver,
- RL-library-specific wrappers,
- batched GPU trajectory execution.

Those are explicit future extensions, not hidden assumptions.