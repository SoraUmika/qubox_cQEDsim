# Floquet Analysis Design Note

Date: 2026-03-21

## Phase 0 survey summary

The current cqed_sim runtime already provides the core pieces needed for Floquet analysis:

- Model-layer Hamiltonian construction through `static_hamiltonian(frame=...)` on `UniversalCQEDModel` and the two convenience wrappers.
- Canonical drive-target resolution through `drive_coupling_operators()`, `transmon_transition_operators(...)`, and `sideband_drive_operators(...)`.
- A QuTiP-backed time-dependent path in `cqed_sim.sim.runner`, which builds `QobjEvo` objects from model operators and complex coefficients.
- Existing project conventions for tensor ordering, frame semantics, and drive sign documented in `README.md`, `API_REFERENCE.md`, and `physics_and_conventions/physics_conventions_report.tex`.

The installed QuTiP version is 5.2.3. It exposes `FloquetBasis`, `floquet_tensor`, and `FMESolver`. For the first implementation stage, the correct native closed-system wrapper boundary is `qutip.FloquetBasis`.

## Integration choice

Floquet support should be implemented as a new top-level package:

- `cqed_sim.floquet`

This is the correct integration point because Floquet analysis is a reusable scientific capability parallel to `cqed_sim.analysis`, `cqed_sim.sim`, and `cqed_sim.calibration`, not a one-off calibration helper.

The new package should reuse the model-layer Hamiltonian and operator infrastructure directly. It should not be built on top of the finite-pulse `Pulse` plus `SequenceCompiler` workflow because Floquet analysis assumes strict periodicity over an indefinitely repeated period, whereas the pulse compiler represents finite-duration schedules.

## Proposed architecture

- `cqed_sim/floquet/__init__.py`
- `cqed_sim/floquet/core.py`
- `cqed_sim/floquet/builders.py`
- `cqed_sim/floquet/analysis.py`
- `cqed_sim/floquet/effective_models.py`
- `cqed_sim/floquet/utils.py`
- `cqed_sim/floquet/README.md`

## Public API direction

The first implementation should expose:

- `PeriodicDriveTerm`
- `PeriodicFourierComponent`
- `FloquetProblem`
- `FloquetConfig`
- `FloquetResult`
- `solve_floquet(...)`
- `build_floquet_hamiltonian(...)`
- `compute_period_propagator(...)`
- `compute_quasienergies(...)`
- `compute_floquet_transition_strengths(...)`
- `compute_bare_state_overlaps(...)`
- `identify_multiphoton_resonances(...)`
- `track_floquet_branches(...)`
- `build_effective_floquet_hamiltonian(...)`

## Data-model decisions

### Periodic drives

Floquet drives should be represented independently of the pulse compiler.

`PeriodicDriveTerm` should support two equivalent ways to specify the driven operator:

- explicit `operator=Qobj`
- model-aware `target=...`, where `target` can be a string, `TransmonTransitionDriveSpec`, or `SidebandDriveSpec`

Each term should carry a periodic scalar coefficient defined relative to the repository's existing convention that the Hamiltonian contribution is built from physical operators and complex periodic coefficients. Supported coefficient specifications should include:

- cosine and sine waveforms with amplitude, angular frequency, and phase
- arbitrary periodic callable coefficients with a declared fundamental period
- optional explicit Fourier metadata for Sambe-space analysis

Multi-tone drives should be supported as multiple `PeriodicDriveTerm` entries. If the tones are not commensurate, the constructor or validator should reject them unless the user supplies an explicit common period.

### Floquet problem container

`FloquetProblem` should own:

- `static_hamiltonian`
- `periodic_terms`
- `period`
- optional `model`
- optional `frame`
- optional `hilbert_space_label`
- optional `metadata`

When `model` is supplied, helper builders can resolve named drive targets and compute bare-basis labeling data. When only `static_hamiltonian` is supplied, the solver should still work as a generic periodic-Hamiltonian wrapper.

### Solver/result containers

`FloquetConfig` should include explicit convergence controls:

- `n_time_samples`
- `atol`
- `rtol`
- `max_step`
- `sort`
- `sparse`
- `zone_center`
- `overlap_reference_time`
- `sambe_harmonic_cutoff`
- `sambe_n_time_samples`
- `precompute_times`

`FloquetResult` should include at least:

- `problem`
- `config`
- `quasienergies`
- `eigenphases`
- `period_propagator`
- `floquet_modes_0`
- `floquet_states_0`
- `floquet_basis`
- `bare_hamiltonian_eigenenergies`
- `bare_state_overlaps`
- `dominant_sidebands`
- `effective_hamiltonian`
- `sambe_hamiltonian`
- `metadata`
- `warnings`

## Numerical strategy

### Route A: primary implementation

The primary solver path should use `qutip.FloquetBasis` on a `QobjEvo` Hamiltonian assembled from:

- the static Hamiltonian
- the periodic drive terms

This provides:

- quasienergies from the one-period propagator eigenphases
- time-dependent Floquet modes and states
- a callable period propagator `U(T)` through the QuTiP propagator wrapper

This route matches the existing repo preference to reuse QuTiP-native capabilities wherever possible.

### Route B: Sambe-space extension

If the term coefficients are Fourier-resolvable, add a harmonic-space builder that samples each scalar coefficient over the period, extracts Fourier components numerically, and constructs a truncated Sambe Hamiltonian.

This should be implemented as an analysis/builder helper rather than the primary solve path. It is mainly for:

- dominant harmonic identification
- sideband interpretation
- cross-checks against the propagator route

## cQED-specific helpers

The module should add cQED-focused analysis on top of the generic Floquet solve:

- overlap of Floquet modes with bare static eigenstates
- dressed-state labeling from dominant bare-state support
- multiphoton resonance detection near `Delta E ~= n Omega`
- drive-frequency and drive-amplitude sweeps with branch tracking
- transition strengths under probe operators such as transmon charge-like or cavity quadrature-like operators
- an effective static Hamiltonian defined by `H_eff = (i / T) log U(T)` with configurable branch folding

## Validation plan

The first test set should cover:

1. Static limit: zero-amplitude drive recovers static energies modulo the Floquet zone.
2. Propagator consistency: quasienergies and the eigenphases of `U(T)` agree.
3. Weak-drive sanity: a driven two-level or weakly anharmonic transmon shows the expected small dressed-state splitting behavior.
4. cQED resonance sanity: transmon or transmon-cavity resonance indicators appear near expected multiphoton conditions.
5. Branch tracking continuity for a one-parameter frequency or amplitude sweep.
6. Sambe cross-checks if the truncated Fourier route is implemented.

## Documentation impact

Because this adds a new reusable feature area, the task must update:

- `cqed_sim/floquet/README.md`
- `API_REFERENCE.md`
- relevant pages under `documentations/`
- `physics_and_conventions/physics_conventions_report.tex`

The physics documentation needs a Floquet section because the new feature carries convention-sensitive meaning: periodicity assumptions, quasienergy zone folding, and how driven operators inherit the repository's frame and drive-target semantics.