# `cqed_sim.rl_control`

The `rl_control` module implements a reinforcement learning (RL) control layer on top of the `cqed_sim` physics stack. It provides a Gym-compatible quantum control environment (`HybridCQEDEnv`), action/observation/reward abstractions, a benchmark task suite, and domain randomization support for training RL agents on cQED systems.

## Relevance in `cqed_sim`

RL-based control is complementary to GRAPE and unitary synthesis: rather than analytically computing an optimal control sequence, an RL agent learns a policy by interacting with a simulator environment. This is relevant for:

- tasks with partially observed state or measurement-dependent feedback,
- hierarchical control problems where some actions are high-level primitives and some are waveform-level,
- exploration of novel control strategies without a fixed gate target,
- and as a testbed for hybrid classical-quantum control architectures where the agent must act on measurement outcomes.

The environment wraps the `cqed_sim` physics models and simulators directly, so policies trained here are compatible with the physical conventions of the rest of the library.

## Main Capabilities

### Environment

- **`HybridCQEDEnv`**: The main environment class. Implements `reset(seed)`, `step(action)`, `render_diagnostics()`, and `estimate_metrics(...)`. Compatible with the standard Gym/Gymnasium interface.
- **`HybridEnvConfig`**: Configuration for the environment (task, action space, observation model, reward model, episode length, domain randomization).
- **`HybridSystemConfig`**: Selects between a fast reduced dispersive model (`ReducedDispersiveModelConfig`) and a fuller multilevel pulse model (`FullPulseModelConfig`).

### Action spaces

Three action-space families:

- **`ParametricPulseActionSpace`**: Actions parameterize a small set of pulse parameters (e.g. amplitude, phase, duration) for a fixed pulse family. Low-dimensional; suitable for continuous-control agents.
- **`PrimitiveActionSpace`**: Actions select from a discrete set of named gate primitives (`PrimitiveAction`, `QubitGaussianAction`, `CavityDisplacementAction`, `SidebandAction`, `WaitAction`, `MeasurementAction`, `ResetAction`, `HybridBlockAction`).
- **`WaveformActionSpace`**: Actions are full waveform arrays for each drive channel. High-dimensional; suitable for direct waveform optimization.

### Observation models

- **`build_observation_model(...)`**: Factory for assembling an observation encoder from a spec.
- Available encoders: `IdealSummaryObservation` (full density matrix summary), `ReducedDensityObservation` (partial trace), `MeasurementIQObservation` (noisy I/Q readout), `GateMetricObservation` (fidelity-like scalars).
- **`HistoryObservationWrapper`**: Wraps an observation model to include a history window of past observations.

### Reward models

- **`build_reward_model(...)`**: Factory for assembling a composite reward from a spec.
- Available reward terms: `StateFidelityReward`, `ProcessFidelityReward`, `ParityRewardTerm`, `WignerSampleRewardTerm`, `AncillaReturnPenaltyTerm`, `LeakagePenaltyTerm`, `ControlCostPenaltyTerm`, `MeasurementAssignmentRewardTerm`.
- **`CompositeReward`**: Weighted sum of multiple reward terms.

### Benchmark task suite

- **`benchmark_task_suite()`**: Returns a list of `HybridBenchmarkTask` objects spanning:
  - vacuum preservation,
  - coherent-state preparation,
  - Fock-state preparation,
  - storage-basis superpositions,
  - even/odd cat state preparation,
  - ancilla-storage entanglement,
  - and a reduced conditional-phase gate task.
- Individual task constructors: `coherent_state_preparation_task(...)`, `fock_state_preparation_task(...)`, `even_cat_preparation_task(...)`, `odd_cat_preparation_task(...)`, `ancilla_storage_bell_task(...)`, `conditional_phase_gate_task(...)`.

### Domain randomization

- **`DomainRandomizer`**: Applies structured perturbations to model parameters at reset time, simulating hardware variability.
- Prior distributions: `FixedPrior`, `UniformPrior`, `NormalPrior`, `ChoicePrior`.
- **`CalibrationEvidence`** (from `cqed_sim.system_id`): Encodes posterior knowledge from a calibration run; passed to `randomizer_from_calibration(...)` to build a calibration-informed randomizer.

### Diagnostics and demonstrations

- **`build_rollout_diagnostics(...)`: Returns a diagnostic bundle including compiled channels, segment metadata, pulse summaries, and frame/regime metadata.
- **`scripted_demonstration(env, actions)`**: Runs a scripted action sequence for debugging.
- **`rollout_records(...)`, `DemonstrationRollout`**: Record and replay rollouts.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `HybridCQEDEnv` | Main RL environment |
| `HybridEnvConfig` | Environment configuration |
| `HybridSystemConfig` | Physics model selection |
| `ReducedDispersiveModelConfig` | Fast reduced model |
| `FullPulseModelConfig` | Full multilevel pulse model |
| `ParametricPulseActionSpace` | Continuous parametric pulse actions |
| `PrimitiveActionSpace` | Discrete gate primitive actions |
| `WaveformActionSpace` | Full waveform actions |
| `build_observation_model(...)` | Build an observation encoder |
| `build_reward_model(...)` | Build a composite reward |
| `benchmark_task_suite()` | Standard benchmark task list |
| `DomainRandomizer` | Model parameter randomization |
| `randomizer_from_calibration(...)` | Build randomizer from calibration evidence |

## Usage Guidance

```python
from cqed_sim.rl_control import (
    HybridCQEDEnv, HybridEnvConfig, HybridSystemConfig,
    FullPulseModelConfig, ParametricPulseActionSpace,
    build_observation_model, build_reward_model,
    fock_state_preparation_task,
)

task = fock_state_preparation_task(target_n=2)
env = HybridCQEDEnv(
    config=HybridEnvConfig(
        task=task,
        system=HybridSystemConfig(model_config=FullPulseModelConfig(...)),
        action_space=ParametricPulseActionSpace(...),
        observation_model=build_observation_model("ideal_summary"),
        reward_model=build_reward_model("state_fidelity"),
    )
)

obs, info = env.reset(seed=0)
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

For a full notebook walkthrough: `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`
For a repo-side script template: `examples/rl_hybrid_control_rollout.py`

## Important Assumptions / Conventions

- The environment wraps `cqed_sim.sim` internally; the same model conventions (carrier sign, tensor ordering, frequency units) apply. Qubit and storage primitives expose detunings around positive physical drive frequencies at the wrapper boundary and then translate those values into the raw internal `Pulse.carrier` used by the runtime.
- The `ReducedDispersiveModelConfig` computes the dispersive Hamiltonian analytically without running the full QuTiP solver; it is fast enough for inner-loop RL but may miss higher-order effects.
- The `FullPulseModelConfig` calls `simulate_sequence(...)` at each step; it is physically accurate but slower.
- Reward signals are in `[−∞, 1]` by convention; fidelity-based rewards are bounded in `[0, 1]` unless penalties dominate.
- The environment is not thread-safe by default; use separate environment instances per parallel worker.

## Relationships to Other Modules

- **`cqed_sim.sim`**: `FullPulseModelConfig` uses `simulate_sequence(...)` internally at each step.
- **`cqed_sim.core`**: provides the model, frame, and state preparation used inside the environment.
- **`cqed_sim.pulses`** and **`cqed_sim.sequence`**: `ParametricPulseActionSpace` and `WaveformActionSpace` actions are compiled through the pulse/sequence stack.
- **`cqed_sim.system_id`**: `CalibrationEvidence` and domain randomization priors live there; `rl_control` re-exports the prior types for convenience.
- **`cqed_sim.observables`** and **`cqed_sim.measurement`**: observation and reward models use these for state diagnostics.

## Limitations / Non-Goals

- The module provides a training environment but does not include a specific RL algorithm implementation. Users supply their own agent (e.g. from Stable-Baselines3 or a custom implementation).
- Domain randomization currently supports independent per-parameter perturbations; correlated parameter uncertainty (e.g. from a full posterior distribution) is not directly supported.
- The benchmark task suite is designed for development and validation; it is not a standardized community benchmark.
- Hardware-in-the-loop operation is not supported; the environment is entirely simulator-backed.

## References

- Tutorial notebook: `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`
- Example script: `examples/rl_hybrid_control_rollout.py`
