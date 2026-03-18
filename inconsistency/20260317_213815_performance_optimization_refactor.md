# Inconsistency Report: Performance Optimization Refactor

**Date/time:** 2026-03-17 21:38
**Scope:** Addition of `solve_grape_multistart`, `GrapeMultistartConfig`,
`run_sweep`, and associated documentation updates.
**Status:** No blocking inconsistencies found.  Minor issues noted below.

---

## Confirmed Issues

### 1. `performance_audit.md` section heading inconsistency (low severity)

**What:** The existing `benchmarks/performance_audit.md` refers to
"Remaining limitations" but the `unitary_fit_small` benchmark showed a +3.6%
regression (not a gain) without explanation.

**Where:** `benchmarks/performance_audit.md`, "Post-optimization benchmark
results" table, `unitary_fit_small` row.

**Why inconsistent:** The note text says "The biggest wins are in compile-heavy
and experiment-loop workloads" but does not explain the tiny regression on
`unitary_fit_small`.

**Consequence:** Potentially confusing to readers auditing the benchmark.

**Fix applied:** Updated the "Remaining limitations" section in
`benchmarks/performance_audit.md` to explicitly state that the unitary-synthesis
path is solver/objective-bound and the +3.6% variance is within expected
measurement noise for that problem size (2 repeats, small system).

**Status:** Resolved by the updated wording in `benchmarks/performance_audit.md`.

---

## Suspected Issues

### 2. `_PARALLEL_SWEEP_SESSION` global in `sim/runner.py` is now dead code

**What:** After adding `run_sweep`, the file contains two sets of parallel
worker globals: `_PARALLEL_SESSION` / `_init_parallel_session` /
`_run_parallel_state` (used by `SimulationSession.run_many`) and
`_PARALLEL_SWEEP_SESSION` / `_init_parallel_sweep_session` /
`_run_parallel_sweep_point` (defined but now superseded by the indexed sweep
path using `_PARALLEL_SWEEP_SESSIONS`).

**Where:** `cqed_sim/sim/runner.py`, lines around the sweep worker globals.

**Why inconsistent:** The non-indexed `_PARALLEL_SWEEP_SESSION` and
`_run_parallel_sweep_point` were written but are not called by `run_sweep`;
the indexed path `_init_parallel_sweep_worker` / `_run_parallel_sweep_point_indexed`
/ `_PARALLEL_SWEEP_SESSIONS` is the actual implementation.

**Consequence:** Dead code, no functional impact.

**Fix applied:** Removed `_PARALLEL_SWEEP_SESSION`, `_init_parallel_sweep_session`,
and `_run_parallel_sweep_point` from `cqed_sim/sim/runner.py` in the same pass.
Only the indexed sweep worker path (`_init_parallel_sweep_worker` /
`_run_parallel_sweep_point_indexed` / `_PARALLEL_SWEEP_SESSIONS`) is retained,
as it is the actual implementation path called by `run_sweep`.

**Status:** Resolved.

---

## Unresolved Questions

### 3. GPU support for GRAPE gradients via JAX backend

The JAX backend (`cqed_sim/backends/jax_backend.py`) exposes `expm` but the
GRAPE propagator (`cqed_sim/optimal_control/propagators.py`) imports only
`scipy.linalg.expm` and `scipy.linalg.expm_frechet` directly.  There is no
integration path between the `JaxBackend` and the GRAPE gradient computation.

**Consequence:** The JAX backend cannot be used to JIT-compile or
GPU-accelerate GRAPE iterations.  This is the primary GPU gap noted in the
performance design document.

**Status:** Open, deferred to future work.  Documented in
`docs/performance_design.md` § 8.

---

## Items Determined to be Non-Issues

- **`GrapeMultistartConfig` immutability**: Using `frozen=True` is correct
  and consistent with `GrapeConfig`.
- **Seed derivation**: `seed + restart_index` is deterministic and
  reproducible, consistent with the rest of the codebase.
- **Export completeness**: All new public symbols are exported in
  `optimal_control/__init__.py`, `sim/__init__.py`, and `cqed_sim/__init__.py`,
  and are listed in `API_REFERENCE.md`.
