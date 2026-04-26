# QuTiP Solver Option Visibility Gaps

Created: 2026-04-25 23:22:44 America/Chicago

## Confirmed Issues

- `cqed_sim.solvers.MasterEquationConfig` and `cqed_sim.sim.SimulationConfig` exposed tolerances and `max_step` but not QuTiP `nsteps` or a general solver-option escape hatch. Large-cutoff, long-duration Hamiltonian runs could therefore fail through package APIs even when the same Hamiltonian solved through a local helper with `options={"nsteps": ...}`.
- Several wrappers constructed fresh configs or native option dictionaries internally, including deterministic trajectory helpers, conditioned/targeted multitone validation, legacy SQR calibration, unitary/map synthesis propagator paths, optimal-control replay, readout-emptying Lindblad validation/refinement, RL full-pulse runtime, tomography helpers, Floquet propagation, and continuous-readout SME replay.
- The targeted-subspace multitone path hard-coded `nsteps=100000`, while other paths either omitted `nsteps` entirely or used ad hoc option dictionaries.

## Affected Components

- `cqed_sim.solvers`
- `cqed_sim.sim`
- `cqed_sim.measurement`
- `cqed_sim.floquet`
- `cqed_sim.calibration`
- `cqed_sim.unitary_synthesis` and `cqed_sim.map_synthesis`
- `cqed_sim.optimal_control`
- `cqed_sim.rl_control`
- `cqed_sim.tomo`

## Consequences

Users could lose control of native QuTiP integrator settings when moving from local feasibility helpers into supported package entry points. This especially affected large-cutoff full-cosine dynamics where `nsteps` is needed for stable long-duration integration. No-drive and weak-drive helper checks remained feasibility evidence only, not headline optimization evidence.

## Resolution Status

Fixed in this task:

- Added shared QuTiP option merging and conflict validation through `cqed_sim.solvers.options`.
- Added `nsteps` and/or `solver_options` to direct QuTiP-facing configs and propagated those controls through the wrapper surfaces listed above.
- Preserved legacy SQR `qutip_nsteps_sqr_calibration` behavior while giving explicit `nsteps` and `solver_options["nsteps"]` precedence.
- Added focused regression tests and a slow, environment-gated Nq=9, Nr=16 full-cosine stability check.

## Remaining Questions

- Headline optimization evidence still requires replaying the actual optimized full-duration waveform and reporting task metrics through the patched package APIs.
- The slow large-cutoff test is gated by `CQED_SIM_RUN_LARGE_CUTOFF_FULL_COSINE=1`; ordinary test runs only verify option propagation and compact solves.
