# Phase 0 Audit: GRAPE Optimal Control in cqed_sim

Created: 2026-03-17

## Scope reviewed

Source-of-truth and convention documents reviewed:

- `README.md`
- `AGENTS.md`
- `API_REFERENCE.md`
- `physics_and_conventions/physics_conventions_report.tex`
- `physics_and_conventions/conventions.py`
- `physics_and_conventions/convention_cleanup_report.md`
- `physics_and_conventions/experimental_protocol_alignment_note.md`

Implementation areas reviewed:

- `cqed_sim/core`
- `cqed_sim/pulses`
- `cqed_sim/sequence`
- `cqed_sim/sim`
- `cqed_sim/calibration/conditioned_multitone.py`
- `cqed_sim/calibration/targeted_subspace_multitone.py`
- `cqed_sim/unitary_synthesis`
- `cqed_sim/rl_control/runtime.py`
- existing unitary-synthesis and targeted-subspace tests
- prior inconsistency audits touching unitary synthesis and waveform runtime scope

## Existing optimal-control-relevant infrastructure

### 1. Model and operator layer is already reusable

The strongest reusable foundation is the model layer:

- `UniversalCQEDModel` exposes:
  - `static_hamiltonian(frame=...)`
  - `operators()`
  - `subsystem_dims`
  - `drive_coupling_operators()`
  - `transmon_transition_operators(...)`
  - `sideband_drive_operators(...)`
  - basis-state and transition-frequency helpers
- `DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel` are thin wrappers over the same shared operator/Hamiltonian core.

This is a clean extension point for optimal control. A GRAPE backend can work directly with dense operators derived from `static_hamiltonian(frame)` and the structured drive operators without changing the model stack.

### 2. Pulse and waveform assembly already exist and should be reused for export

The pulse stack is already mature:

- `cqed_sim.pulses.Pulse` supports analytic and sampled complex envelopes
- `SequenceCompiler` turns pulses into uniform sampled channel waveforms
- `CompiledSequence` holds baseband, distorted, and RF traces per channel
- `cqed_sim.sim.runner.hamiltonian_time_slices(...)` already consumes compiled channel coefficients and repository drive-target conventions

This means the GRAPE layer does not need to invent a second pulse runtime. It should optimize in a compact control parameterization, then export controls back into `Pulse` objects for compatibility with the existing stack.

### 3. Runtime propagation stack already supports the needed Hamiltonian form

`cqed_sim.sim.runner` already assembles

`H(t) = H0 + sum_k c_k(t) H_k + c_k(t)^* H_k^dagger`

through `hamiltonian_time_slices(...)` and QuTiP solvers. That is the same control structure needed for rotating-frame piecewise-constant GRAPE when the control layer is expressed in baseband quadratures.

### 4. Existing synthesis code already solved several API problems

`cqed_sim.unitary_synthesis` already contains reusable ideas that should not be duplicated:

- generic target abstractions:
  - `TargetUnitary`
  - `TargetStateMapping`
  - `TargetReducedStateMapping`
  - `TargetIsometry`
  - `TargetChannel`
- `Subspace` selectors and block-phase-aware logical subspace handling
- fidelity and leakage metrics in `metrics.py`
- result reporting, progress reporting, and robust-parameter sampling config
- `QuantumSystem` and `CQEDSystemAdapter` abstractions that already separate system details from optimization flow

The new optimal-control layer should reuse these conventions where practical, especially for target semantics and subspace interpretation.

### 5. Calibration validation code already contains relevant physics metrics

`cqed_sim.calibration.targeted_subspace_multitone` already implements:

- pure-state fidelity evaluation
- restricted process fidelity on projected operators
- leakage and block-preservation diagnostics
- objective weighting and optimization reporting

That code is not a general optimal-control framework, but it confirms the repository already values projected-operator objectives, multi-state transfer sets, leakage accounting, and logical block phase handling.

### 6. RL runtime proves low-level control segments are architecturally acceptable

`cqed_sim.rl_control.runtime.PulseGenerator` already works below the gate-sequence layer. It turns control actions into explicit `Pulse` segments plus drive-operator maps. This is important evidence that a first-class low-level control layer belongs in the architecture and is not conceptually alien to the repository.

## What can be reused directly

- Model Hamiltonians and operator generation from `cqed_sim.core`
- Structured drive-target conventions from `drive_coupling_operators`, `TransmonTransitionDriveSpec`, and `SidebandDriveSpec`
- `Pulse`, `SequenceCompiler`, and `CompiledSequence` for waveform export and runtime verification
- `Subspace` and `subspace_unitary_fidelity(...)` from `cqed_sim.unitary_synthesis`
- `TargetUnitary` probe-state semantics for phase-tolerant unitary matching
- Robust-sampling ideas from `ParameterDistribution`
- Existing top-level documentation patterns, test layout, and report style

## What needs refactoring or bridging

### 1. The current optimizer is not a direct control optimizer

`cqed_sim.unitary_synthesis.optim.UnitarySynthesizer` is centered on gate sequences and primitive-gate parameters. It is flexible, but it is not a first-class time-slice control optimizer.

What is missing there for GRAPE:

- a generic control-problem abstraction independent of gate sequences
- direct piecewise-constant controls as the primary optimization variables
- gradient evaluation over time slices
- a solver backend interface where GRAPE is one implementation rather than the entire abstraction

### 2. Current target abstractions are reusable, but their evaluation path is sequence-centric

The target classes themselves are good. The missing piece is a target evaluator that accepts a low-level control problem and propagated state histories instead of a `GateSequence`.

### 3. Pulse parameterization is missing as a first-class reusable object

The repository has pulses and compiled sequences, but not a reusable control-parameter object representing:

- a time grid
- a fixed list of control Hamiltonian terms
- control bounds
- waveform export
- flatten/unflatten utilities for solvers

That needs to be added explicitly.

## What is currently missing

- No `cqed_sim.optimal_control` package
- No generic `ControlProblem` abstraction
- No first-class GRAPE backend
- No exact closed-system slice gradient implementation
- No reusable piecewise-constant control parameterization
- No ensemble aggregation at the direct-control layer
- No reusable penalty layer for direct controls beyond ad hoc optimization terms in other modules
- No warm-start object that cleanly exports direct-control schedules for later RL use

## Clean extension points

The cleanest extension is:

1. Add a new `cqed_sim.optimal_control` package.
2. Reuse `cqed_sim.core` for Hamiltonians and drive operators.
3. Reuse `cqed_sim.unitary_synthesis.subspace` and selected metrics/target semantics where appropriate.
4. Export optimized schedules back into `Pulse` objects so the existing compile/sim path remains the verification and interoperability path.

This avoids rewriting the current gate-sequence optimizer while still adding a serious first-class optimal-control layer.

## Adequacy of current Hamiltonian and pulse APIs for GRAPE

### Hamiltonian API

Adequate for a first implementation.

Why:

- static drift Hamiltonians are available directly
- drive operators are available in structured form
- frames are explicit and already documented
- subsystem dimensions and basis-state helpers are available for state and subspace targets

### Pulse API

Adequate for export, not sufficient by itself as the optimization abstraction.

Why:

- it can represent the optimized result
- it already integrates with compilation and runtime simulation
- but it does not itself represent solver-facing control variables, bounds, or gradients

So the GRAPE layer should optimize on a dedicated control parameterization, then translate to pulses.

## Likely numerical bottlenecks

The main bottlenecks will be:

1. Dense matrix exponentials for every time slice and every ensemble member.
2. Fr"echet derivative evaluations for every slice-control pair when computing exact gradients.
3. Large probe sets for projected unitary objectives.
4. Repeated dense propagation over larger truncated cavity spaces.

The implementation should therefore:

- keep the nominal backend closed-system and dense
- cache slice Hamiltonians and propagators within one objective evaluation
- keep target evaluation state-based where that simplifies gradients
- reserve full runtime pulse simulation for verification and exported-solution replay, not inner-loop optimization

## Conventions that must be preserved

- Internal Hamiltonians and frequencies remain in `rad/s`.
- Times remain in `s`.
- Tensor ordering remains transmon first, then bosonic modes in declaration order.
- Drive operators must remain aligned with the repository's structured target conventions.
- Exported pulses must respect the documented waveform sign convention.
- Rotating-frame choices must remain explicit through `FrameSpec`.
- Public docs must remain synchronized across `README.md`, `API_REFERENCE.md`, `documentations/`, and `physics_and_conventions/physics_conventions_report.tex`.

## Partial support already present today

### Target state fidelity evaluation

Already partially supported in:

- `cqed_sim.unitary_synthesis.metrics.state_mapping_metrics`
- `cqed_sim.calibration.targeted_subspace_multitone._state_fidelity`

### Target unitary fidelity evaluation

Already partially supported in:

- `cqed_sim.unitary_synthesis.metrics.subspace_unitary_fidelity`
- `cqed_sim.calibration.targeted_subspace_multitone._restricted_process_fidelity`

### Subspace fidelity evaluation

Already supported through:

- `Subspace`
- projected-operator restriction
- block-phase-aware gauges in `unitary_synthesis.metrics`

### Pulse-family parameterizations

Already partially supported in:

- pulse builders
- RL action-to-pulse translation
- synthesis primitives with waveform callables

But there is no reusable direct-control parameterization object yet.

### Time-grid abstractions

Already partially supported in:

- `SequenceCompiler(dt=...)`
- synthesis constraint helpers such as `piecewise_constant_samples(...)`

But there is no public optimal-control time-grid object.

### Waveform assembly

Already supported in:

- `Pulse`
- `SequenceCompiler`
- `CompiledSequence`
- runtime Hamiltonian assembly in `cqed_sim.sim.runner`

### Propagator caching

Only partial and ad hoc.

- `SequenceCompiler` has optional caching.
- The direct-control optimization path does not yet have reusable propagator caching.

### Gradient-friendly operator generation

Only partially supported.

- Dense operators are available.
- There is no reusable exact-gradient implementation for optimal control.

## Audit conclusion

`cqed_sim` already has the physics, operator, pulse, runtime, and target semantics needed for a reusable optimal-control layer. The missing piece is not low-level simulator capability. The missing piece is a solver-facing abstraction that treats direct controls as first-class objects and uses the existing simulator stack as its model/export/runtime substrate.

That argues strongly for adding a new `cqed_sim.optimal_control` package instead of trying to stretch the current gate-sequence optimizer into a direct-control framework.