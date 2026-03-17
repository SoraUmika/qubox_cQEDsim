# Phase 1 Design: Modular Optimal Control for cqed_sim

Created: 2026-03-17

## Design goal

Add a reusable optimal-control layer where:

- a generic control problem is the primary abstraction
- solver backends are pluggable
- GRAPE is the first solver backend
- the implementation reuses `cqed_sim` model, subspace, pulse, and runtime conventions instead of bypassing them

## Architectural decision

The design uses a solver-agnostic control problem abstraction and places GRAPE behind a dedicated solver API.

This is preferable to putting GRAPE logic directly into existing gate-sequence optimization code because:

- direct-control optimization variables are time-slice amplitudes, not primitive-gate parameters
- GRAPE needs propagator and gradient infrastructure that is orthogonal to gate-sequence optimization
- future CRAB, Krotov, or RL warm-start workflows can reuse the same control problem and result objects

## Package structure

The new package should be:

```text
cqed_sim/optimal_control/
  __init__.py
  problems.py
  parameterizations.py
  objectives.py
  penalties.py
  propagators.py
  grape.py
  initial_guesses.py
  result.py
  utils.py
```

## Core abstractions

### 1. `ControlTerm`

Represents one control Hamiltonian term:

- name
- dense operator matrix
- amplitude bounds
- export channel metadata
- structured drive target metadata for pulse/runtime export
- quadrature label (`I` or `Q`) when the term comes from a complex drive channel

This keeps the solver-facing object independent of any single pulse family.

### 2. `ControlSystem`

Represents one closed-system member of the optimization problem:

- dense drift Hamiltonian
- dense control operators aligned with the control-term ordering
- optional weight and label
- optional metadata describing the model variant

This object is the bridge between the model layer and the solver.

### 3. `PiecewiseConstantTimeGrid`

Represents the control time discretization:

- explicit slice durations in seconds
- total duration
- slice boundaries and midpoints

The implementation should allow arbitrary positive slice durations, not only a single uniform `dt`.

### 4. `PiecewiseConstantParameterization`

Represents the solver-facing control parameterization:

- ordered control terms
- time grid
- flatten/unflatten helpers
- bound generation for optimizers
- pulse export back into `Pulse` objects

This is the main future extension point for non-piecewise parameterizations.

### 5. `ControlSchedule`

Represents one concrete set of control values under a given parameterization.

Responsibilities:

- hold the control sample matrix with shape `(n_controls, n_slices)`
- flatten/unflatten for optimizers
- clip/project to term bounds
- export pulses and drive-operator maps for runtime replay

## Problem abstraction

### `ControlProblem`

The generic problem object should contain:

- a parameterization
- one or more `ControlSystem` objects
- one or more objectives
- optional penalties
- ensemble aggregation policy (`mean` or `worst`)
- optional metadata

This makes the control problem reusable independent of the chosen solver.

## Model integration

### `ModelControlChannelSpec`

This helper object maps a `cqed_sim` model drive target into control terms:

- target channel or structured drive target
- which quadratures to include
- amplitude bounds
- exported channel name

Builder functions should convert a `UniversalCQEDModel` or wrapper model into:

- dense drift Hamiltonian from `static_hamiltonian(frame)`
- dense control operators from `drive_coupling_operators(...)`, `transmon_transition_operators(...)`, or `sideband_drive_operators(...)`
- a `PiecewiseConstantParameterization`
- a nominal `ControlSystem`

This keeps optimal control aligned with the existing model and frame APIs.

## Objective design

### State objectives

Add direct support for:

- state preparation
- multi-state transfer

through a `StateTransferObjective` that stores weighted initial/target pure-state pairs.

### Unitary objectives

Add `UnitaryObjective` for:

- subspace gate synthesis
- full unitary synthesis on truncated spaces

Instead of relying on a separate operator-gradient path in the first implementation, the design evaluates unitary objectives through weighted probe-state transfer pairs derived from the target operator.

This has three advantages:

1. It reuses the same GRAPE state-gradient machinery.
2. It naturally supports global-phase, diagonal-phase, and block-phase equivalence classes.
3. It aligns with the existing `TargetUnitary` probe-state semantics already present in `cqed_sim.unitary_synthesis.targets`.

`UnitaryObjective` should therefore either:

- reuse `TargetUnitary.resolved_probe_pairs(...)`, or
- reproduce the same semantics exactly.

### Adapters to existing synthesis targets

Add a small adapter that converts existing unitary-synthesis target objects into optimal-control objectives where practical. This keeps the new layer connected to existing tooling without forcing everything through the same optimizer.

## Penalty design

The first implementation should support three direct-control penalties:

1. `AmplitudePenalty`
   - quadratic penalty on control amplitudes
2. `SlewRatePenalty`
   - quadratic penalty on adjacent control differences
3. `LeakagePenalty`
   - penalty on final population outside a retained subspace across the propagated probe states

These penalties cover the most immediate control-quality needs while staying compatible with exact slice gradients.

## Propagator design

### Closed-system dense backend first

The GRAPE backend should start with a closed-system dense propagator:

- slice Hamiltonian: `H_j = H0 + sum_k u_{k,j} H_k`
- slice propagator: `U_j = exp(-i H_j dt_j)`

### Exact slice gradients

Use `scipy.linalg.expm_frechet(...)` to evaluate exact derivatives of each slice propagator with respect to each control amplitude.

This keeps the implementation mathematically clean and avoids ad hoc finite differences.

### Caching within one objective evaluation

Each objective evaluation should cache:

- slice Hamiltonians
- slice propagators
- forward state histories
- backward costates

This is enough for a serious first implementation without prematurely building a more elaborate backend system.

## GRAPE solver design

### `GrapeConfig`

Configuration should include:

- optimizer method, defaulting to `L-BFGS-B`
- iteration limits and tolerances
- history recording control
- optional random seed for initial guesses
- SciPy option passthrough

### `GrapeSolver`

Responsibilities:

- accept a `ControlProblem`
- accept optional warm starts / initial schedules
- evaluate objective and exact gradient
- aggregate ensemble members by mean or worst case
- return a reusable result object with history and export helpers

## Result design

### `GrapeResult`

The result object should include:

- optimized `ControlSchedule`
- convergence history
- success flag and optimizer message
- nominal final unitary
- per-objective and per-system metrics
- penalty breakdown
- waveform export helpers
- pulse export helper

This should be sufficient for:

- test assertions
- example scripts
- future RL warm-start extraction
- benchmark reporting

## Why this design fits the repository

This design matches current repository architecture because:

- physics stays in `cqed_sim.core`
- waveform export stays in `cqed_sim.pulses` and `cqed_sim.sequence`
- runtime replay stays in `cqed_sim.sim`
- subspace semantics reuse existing synthesis conventions
- optimal control gains its own first-class package instead of being hidden inside notebooks or calibration scripts

## Scope of the first implementation

Implemented now:

- closed-system dense GRAPE
- piecewise-constant controls
- model-backed control-term builders
- state preparation and multi-state transfer
- subspace and full-space unitary objectives
- ensemble optimization with `mean` and `worst`
- leakage, amplitude, and slew penalties
- pulse export and runtime replay support

Deferred intentionally:

- open-system GRAPE gradients
- second-order optimizers beyond what SciPy already provides
- CRAB/Krotov backends
- automatic RL warm-start consumption
- hardware distortion in the inner optimization loop

The deferred items remain compatible with the same `ControlProblem` abstraction, which is the primary reason to establish it first.