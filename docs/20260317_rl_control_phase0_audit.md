# Phase 0 Audit Note: RL-Ready Hybrid cQED Control Extension

Date: March 17, 2026

## Summary

The repository already contains a strong reusable simulation core for a serious first-pass RL control stack. The right path is to extend the existing package with an RL-facing wrapper layer that reuses `UniversalCQEDModel`, `SequenceCompiler`, `SimulationSession`, `NoiseSpec`, the readout-chain measurement APIs, and the unitary-synthesis metrics and primitive abstractions.

## Reusable Infrastructure Already Present

### Physics and model layer

- `cqed_sim.core.UniversalCQEDModel` already centralizes multilevel transmon plus bosonic-mode Hamiltonian assembly.
- `DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel` are compatibility wrappers on top of the shared universal model layer.
- The model layer already supports:
  - multilevel transmon truncation,
  - bosonic self-Kerr,
  - higher-order dispersive terms in falling-factorial form,
  - additional cross-Kerr and exchange terms through `CrossKerrSpec`, `SelfKerrSpec`, and `ExchangeSpec`,
  - explicit basis-state helpers and transition-frequency helpers,
  - structured sideband drive targets.

### Pulse, distortion, and sequence infrastructure

- `cqed_sim.pulses.Pulse` already represents analytic or sampled complex envelopes.
- `cqed_sim.sequence.SequenceCompiler` already performs:
  - timing quantization,
  - zero-order hold,
  - first-order low-pass filtering,
  - IQ imbalance / skew / image leakage / detuning,
  - amplitude quantization,
  - linear crosstalk mixing.
- `HardwareConfig` already covers most of the required first-pass distortion hooks.

### Simulation and open-system evolution

- `cqed_sim.sim.simulate_sequence(...)` and `SimulationSession` already provide the core pulse-to-Hamiltonian-to-evolution path.
- `SimulationSession` is the cleanest extension point for RL rollouts because it caches the Hamiltonian, collapse operators, observables, and solver options.
- `NoiseSpec` already supports:
  - transmon relaxation,
  - multilevel transmon ladder-resolved T1 via `transmon_t1`,
  - qubit dephasing,
  - cavity / storage loss,
  - readout loss,
  - bosonic thermal occupancy hooks.

### Measurement and partial observability infrastructure

- `measure_qubit(...)` already supports exact probabilities, confusion matrices, repeated shots, synthetic IQ clusters, physical readout-chain simulation, and nearest-center IQ classification.
- `ReadoutResonator`, `PurcellFilter`, `AmplifierChain`, and `ReadoutChain` already provide measurement-aware observables that are usable as RL observations or proxy rewards.

### Diagnostics, tomography, and metrics

- `cqed_sim.sim.extractors` already provides reduced states, conditioned Bloch vectors, moments, photon numbers, and Wigner calculations.
- `cqed_sim.unitary_synthesis.metrics` already provides reusable state-fidelity, leakage, and restricted-unitary metrics.
- `cqed_sim.unitary_synthesis.sequence` already exposes interpretable primitives such as `Displacement`, `SNAP`, `SQR`, and `FreeEvolveCondPhase`.

### Existing optimization / control-adjacent infrastructure

- `cqed_sim.calibration.sqr`, `conditioned_multitone`, and `targeted_subspace_multitone` already contain:
  - fidelity-oriented objective functions,
  - process-style metrics on selected logical subspaces,
  - constrained waveform parameterizations,
  - optimization traces and result summaries.
- That existing code is the best source for RL baseline metrics and primitive-level task definitions.

## What Is Missing

The repository does not currently provide an RL-facing control stack. Missing pieces include:

- a Gym-like environment API,
- action-space abstractions tailored to pulse-level and primitive-level bosonic/ancilla control,
- observation encoders for partial observability and history stacking,
- modular reward composition,
- a formal episode-level domain-randomization system,
- a benchmark task registry for state preparation and gate synthesis,
- rollout diagnostics packaged for RL debugging,
- a demonstration / scripted-baseline export path,
- explicit system-identification hook points for future posterior-guided training.

## Recommended Extension Points

### Keep the physics stack where it is

The new work should sit above the existing stack:

- model construction stays in `cqed_sim.core`,
- pulse distortion stays in `cqed_sim.pulses` and `cqed_sim.sequence`,
- open-system propagation stays in `cqed_sim.sim`,
- measurement realism stays in `cqed_sim.measurement`.

### Add a new RL-oriented package layer

The cleanest extension is a new `cqed_sim.rl_control` package that contains:

- action spaces,
- observation encoders,
- reward terms,
- benchmark tasks,
- domain randomization,
- diagnostics,
- metrics,
- demonstrations,
- the environment wrapper itself.

### Add system-identification scaffolding separately

Future fit-then-randomize workflows are better isolated in a small `cqed_sim.system_id` package rather than buried inside the environment wrapper.

## Refactor vs New Code

### Prefer new modules

Most of the work should be new code, not a rewrite of existing solver code.

### Only minimal generalization is justified

The only justified generalizations are:

- thin wrappers that turn existing model/pulse/measurement primitives into reusable RL-facing interfaces,
- reuse of existing synthesis metrics and subspace abstractions in the new RL metrics layer,
- top-level API export updates so the new package surface is discoverable.

## Performance Bottlenecks Likely To Matter

- QuTiP solver cost is still the dominant runtime bottleneck for long-horizon RL rollouts.
- Rebuilding and recompiling pulse schedules every step will matter, especially for waveform-level action spaces.
- Full operator / Wigner diagnostics are too expensive for every step and should remain opt-in.
- Gate-synthesis tasks that propagate a basis of probe states are workable for small subspaces but should stay small in the first pass.

## Physics Assumptions The Extension Must Respect

- Internal Hamiltonian coefficients remain in rad/s and times remain in seconds.
- Tensor ordering remains transmon first, then bosonic modes.
- `Pulse.carrier` remains the negative of the rotating-frame transition frequency it addresses.
- Runtime dispersive semantics remain `+chi * n_boson * n_q`, with negative `chi` lowering the qubit transition frequency with photon number.
- The repository measurement confusion matrix convention remains latent `(g, e)` columns and reported `(g, e)` rows.
- The new RL layer must not introduce a second sign convention, alternate basis ordering, or alternate unit convention.

## Existing Code That Already Partially Supports The Requested Scope

### Pulse parameter optimization

- `cqed_sim.calibration.sqr`
- `cqed_sim.calibration.conditioned_multitone`
- `cqed_sim.calibration.targeted_subspace_multitone`
- `cqed_sim.unitary_synthesis`

### Bosonic control primitives

- `cqed_sim.core.ideal_gates`
- `cqed_sim.unitary_synthesis.sequence`
- pulse builders in `cqed_sim.pulses.builders`

### Measurement simulation

- `cqed_sim.measurement.qubit`
- `cqed_sim.measurement.readout_chain`

### Reduced dispersive models

- `DispersiveTransmonCavityModel`
- `DispersiveReadoutTransmonStorageModel`

### Fuller multilevel models

- `UniversalCQEDModel` with multilevel transmon truncation plus exchange / Kerr / cross-Kerr terms

### Tomography-based or simulator-side metrics

- reduced states and conditioned observables in `cqed_sim.sim.extractors`
- fidelity and leakage metrics in `cqed_sim.unitary_synthesis.metrics`
- targeted-subspace validation metrics in `cqed_sim.calibration.targeted_subspace_multitone`

## Conclusion

The repository already contains the hard physics and solver machinery. The missing work is an integrated control-and-evaluation layer for RL. A serious first pass should therefore focus on building a modular environment stack that reuses the existing physics, distortion, measurement, and metric infrastructure rather than replacing it.