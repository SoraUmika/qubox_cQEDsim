# Holographic Quantum Algorithms Implementation Plan

## Repository Audit Summary

- `cqed_sim` is organized as a reusable library with public APIs documented in `API_REFERENCE.md` and mirrored under `documentations/api/`.
- Public-facing examples are kept under `examples/`, while validation coverage lives under `tests/`.
- Existing reusable patterns worth following:
  - dataclass-based configuration and result objects with explicit validation and `to_record()` helpers
  - small public top-level entry points re-exported from `__init__.py`
  - plotting and persistence methods attached to result containers only when they are clearly user-facing
- The current holographic code under `cqed_sim/quantum_algorithms/holographic_sim/` is still prototype-level:
  - `holographicSim.py` contains useful ideal-case logic for unitary input, projective measurement, Monte Carlo sampling, exact branch enumeration, and optional progress bars
  - `mps.py` contains useful right-canonical MPS utilities, but it is not yet integrated into a package-level channel abstraction
  - the prototype hardcodes qubit-system assumptions (`SYS_DIM = 2`) into the main execution path and returns low-level tuples / arrays rather than structured API objects

## Proposed Architecture

The implementation follows the decomposition recommended by `paper_summary/holographic_quantum_algorithms.pdf`:

1. Physics / model helpers
   - lightweight spin-style model helpers and example channels
2. MPS / channel layer
   - MPS utilities
   - `HolographicChannel` built from unitary, Kraus, or right-canonical MPS data
   - `PurifiedChannelStep` for the prepare-apply-measure-reset primitive
3. Schedule / observable layer
   - explicit observable insertions, burn-in config, and optional right-boundary postselection
4. Execution / inference layer
   - Monte Carlo sampler
   - exact branch enumeration for small cases
   - running statistics and structured result objects
5. Diagnostics layer
   - CPTP / completeness / normalization checks
   - Monte Carlo vs exact comparison helpers
6. Future-extension scaffolding
   - holoVQE energy-objective layer
   - holoQUADS time-slice / causal-cone scaffolding

## Proposed Public API

Primary user-facing objects:

- `HolographicChannel`
- `PurifiedChannelStep`
- `PhysicalObservable`
- `ObservableInsertion`
- `ObservableSchedule`
- `BurnInConfig`
- `BoundaryCondition`
- `HolographicSampler`
- `HolographicMPSAlgorithm`
- `CorrelatorEstimate`
- `ExactCorrelatorResult`
- `ChannelDiagnostics`

Secondary helpers:

- `MatrixProductState`
- `complete_right_isometry(...)`
- `pauli_x()`, `pauli_y()`, `pauli_z()`
- `transverse_field_ising_terms(...)`, `xxz_chain_terms(...)`
- `HoloVQEObjective`
- `HoloQUADSPlan`

## Integration Points With Existing `cqed_sim`

Reused:

- repository documentation style and public API export conventions
- dataclass/result-object patterns from modules such as `unitary_synthesis`
- optional QuTiP object compatibility via `.full()` conversion

Intentionally kept self-contained:

- the holographic channel and MPS abstractions
- exact / Monte Carlo holographic estimators
- burn-in, schedule, and boundary-condition logic

Reasoning:

- the holographic module is conceptually generic and should not inherit cQED-only model or pulse abstractions
- `cqed_sim` integration is helpful at the packaging/documentation level, but the internal math should remain backend-agnostic and hardware-agnostic

## What Will Be Implemented Now

- a structured `cqed_sim.quantum_algorithms.holographic_sim` package
- reusable channel constructors from unitary, Kraus, and right-canonical MPS data
- purified step execution with projective measurements in observable eigenbases
- Monte Carlo correlator sampling
- exact branch enumeration for small schedules
- diagnostics and validation utilities
- result/config objects with serialization-friendly records
- compatibility wrappers for the old prototype entry points
- examples, tests, and API/tutorial docs
- lightweight holoVQE and holoQUADS scaffolding

## What Is Intentionally Deferred

- hardware-target circuit IR / backend compilation layers
- noisy hardware execution models and leakage-aware hardware mitigation
- a full optimizer stack for holoVQE
- a full time-evolution engine for holoQUADS
- advanced entropy estimation protocols such as full replica-SWAP workflows
- large-scale benchmark reproduction against the paper

Those are left as explicit future work, but the new package layout is designed to make them natural extensions.
