# Inconsistency Report: Prepared-session QobjEvo gap

**Date/time:** 2026-03-21 11:08:02

## Confirmed Issues

### 1. Prepared simulation sessions reused Hamiltonian slices but not QuTiP's compiled time-dependent Hamiltonian

**What:** `SimulationSession` cached the raw Hamiltonian slice list from `hamiltonian_time_slices(...)`, but each `session.run(...)` call still passed that raw list back into `qt.sesolve` / `qt.mesolve`.

**Where:** `cqed_sim/sim/runner.py`

**Affected components:** `SimulationSession`, `prepare_simulation(...)`, `simulate_batch(...)`, repeated calibration / sweep / rollout workloads that reuse the same pulse and Hamiltonian across many initial states.

**Why this is inconsistent:** The prepared-session API exists specifically to amortize repeated setup work across runs, but the QuTiP coefficient/interpolator layer was still rebuilt inside every repeated solve because the cached object stopped one layer too early.

**Consequences:** Repeated prepared-session runs left measurable performance on the table in the main simulator path. A direct local A/B benchmark on the 20-run prepared-session workload showed that reusing a prebuilt `QobjEvo` improved runtime by about `1.12x` on average and `1.22x` in the best run relative to the raw Hamiltonian-list fallback.

## Suspected / Follow-up Questions

1. The main remaining runtime ceiling is still QuTiP ODE integration itself. Larger gains will require either lower-level propagator paths or a solver/backend shift rather than more Python-side caching.
2. The existing Windows multiprocessing APIs remain coarse-grained only; `spawn` overhead still dominates small jobs even after the prepared-session fix.

## Status

Fixed in this task.

## Fix Record

- `cqed_sim/sim/runner.py`: prebuilds a `qt.QobjEvo` once per `SimulationSession` for QuTiP-backed runs, reuses it across `session.run(...)` calls, and rebuilds it locally after worker-process unpickling so the Windows spawn path remains functional.
- `cqed_sim/core/universal_model.py`: caches drive-coupling mappings so repeated session construction on the same model no longer rebuilds those operator pairs every time.
- `tests/test_29_performance_paths.py`: adds regression coverage for the drive-coupling cache and the prepared-session `QobjEvo` path.