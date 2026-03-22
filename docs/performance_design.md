# `cqed_sim` Performance Architecture Design

**Date:** 2026-03-17
**Scope:** Built-in parallelization, performance optimization, and optional GPU acceleration
**Status:** Phases 0–5 complete; Phases 6–9 ongoing

---

## 1. Overview and Motivation

`cqed_sim` is a hardware-faithful cQED simulator built on QuTiP.  Its core
workflows — parameter sweeps, calibration loops, GRAPE optimization, unitary
synthesis, and RL episode rollouts — involve repeated expensive operations that
are tractable individually but become bottlenecks at realistic scale.

The goal of this design effort is to make the library meaningfully faster
through a structured, evidence-based sequence of improvements:

1. Eliminate unnecessary work (caching, structural deduplication).
2. Exploit mathematical structure (static/dynamic Hamiltonian decomposition, vectorization).
3. Cache reusable expensive objects (operators, Hamiltonians, default observables).
4. Vectorize repeated operations (pulse compilation, ZOH, lowpass).
5. Parallelize coarse-grained independent tasks (initial-state batches, parameter sweeps, GRAPE restarts).
6. Use GPU only for kernels that genuinely benefit.

---

## 2. Phase 0 — Bottleneck Classification

A `cProfile` pass over representative workloads identified four primary bottleneck classes:

| Bottleneck class | Affected code paths | Root cause |
|---|---|---|
| Compile-heavy Python loops | `SequenceCompiler.compile`, `Pulse._sample_analytic`, ZOH, lowpass | Per-sample Python overhead and full-grid sampling |
| Repeated solver setup | `simulate_sequence` in loops | Re-assembling Hamiltonian, observables, and collapse ops on every call |
| Calibration objective evaluation | SQR calibration, `conditional_loss` | Repeated solver calls with redundant pre-computation |
| Unitary synthesis inner loop | `UnitarySynthesizer.fit` | Repeated backend-simulation objective evaluations |

Baseline measurements (before optimization, from `benchmarks/performance_audit.md`):

| Workload | Baseline avg. |
|---|---:|
| `compile_heavy_20` | `0.031422 s` |
| `repeat_simulate_20` | `0.091461 s` |
| `run_all_xy` | `0.106114 s` |
| `unitary_fit_small` | `0.057405 s` |

---

## 3. Phase 1 — Design Principles

### 3.1 Algorithmic improvements first

Structural refactors were prioritized over brute-force parallelism:

- **Operator and Hamiltonian caching**: `UniversalCQEDModel._operators_cache` and
  `_static_h_cache` avoid repeated tensor-product construction and Hamiltonian
  assembly for the same frame.  Cache invalidation is keyed on the full
  structural signature of the model.
- **Default observables caching**: `default_observables(model)` stamps a cached
  result onto the model object so the same projectors are not rebuilt on every
  simulation call.
- **Static/dynamic Hamiltonian separation**: The assembled `hamiltonian_time_slices`
  structure separates the static part (assembled once) from the time-varying
  drive coefficients (sampled from the compiled channel).
- **Support-aware pulse compilation**: `SequenceCompiler.compile` samples each
  pulse only on its support interval rather than the full global grid.
- **ZOH vectorization**: `apply_zoh` vectorized via `numpy.repeat`.
- **Fast lowpass path**: `apply_first_order_lowpass` dispatched to `scipy.lfilter`
  when input is real or complex 1-D.

### 3.2 Session reuse for repeated simulations

`SimulationSession` (created by `prepare_simulation(...)`) pre-assembles the
Hamiltonian, collapse operators, observable operators, and solver options once
and reuses them across repeated `session.run(psi0)` calls.  This is the correct
pattern for calibration loops and RL episode batches where the same pulse is
applied to many initial states.

For QuTiP-backed runs, the session now also prebuilds the time-dependent
Hamiltonian as a `qt.QobjEvo` and reuses it across repeated solves.  This avoids
rebuilding QuTiP coefficient/interpolator objects on every `session.run(...)`
call while preserving the same solver path and physics behavior.

### 3.3 Coarse-grained CPU parallelism

CPU parallelism is exposed at two complementary levels:

#### `simulate_batch` — same Hamiltonian, many initial states

```
session = prepare_simulation(model, compiled, drive_ops)
results = simulate_batch(session, [psi_1, psi_2, ..., psi_N], max_workers=K)
```

#### `run_sweep` — different Hamiltonians, one state per point

```
sessions = [prepare_simulation(model_at_param(p), compiled, drive_ops) for p in params]
results = run_sweep(sessions, [psi0] * len(params), max_workers=K)
```

#### `solve_grape_multistart` — independent GRAPE restarts

```
results = solve_grape_multistart(
    problem,
    config=GrapeConfig(maxiter=200, seed=0),
    multistart_config=GrapeMultistartConfig(n_restarts=6, max_workers=1),
)
best = results[0]  # sorted best-first by objective value
```

All three parallel paths share the same design:
- `ProcessPoolExecutor` with the `spawn` context.
- Deterministic seed management (each restart or worker index derives its seed
  from a shared base seed).
- Serial execution (`max_workers=1`) is the default and the correct choice for
  small workloads.

### 3.4 Backend abstraction

`cqed_sim/backends/` provides a `BaseBackend` abstract interface with concrete
implementations:

- `NumPyBackend`: dense matrix-exponentiation solver (SciPy `expm`), CPU-only.
- `JaxBackend`: dense JAX-based solver; optional dependency; supports JIT and
  automatic differentiation.  Falls back to `None` if JAX is not installed.

Backends are selected via `SimulationConfig(backend=...)`.  The QuTiP solver
is the default and is not replaced; backends are opt-in for use cases where
dense-matrix propagation is preferable (small systems, gradient computation,
backend parity checks).

---

## 4. Phase 2 — Execution Backend Architecture

The backend hierarchy is:

```
BaseBackend (abstract)
├── NumPyBackend     — dense CPU, always available
└── JaxBackend       — dense CPU/GPU, optional (JAX must be installed)
```

Each backend implements:
- `asarray`, `to_numpy`, `eye`, `zeros`, `reshape`, `dagger`
- `matmul`, `kron`, `expm`, `trace`
- `expectation`, `lindbladian`

The `sim/solver.py` module provides `solve_with_backend(hamiltonian, tlist, psi0,
observables, collapse_ops, backend, store_states)` which uses the selected backend
for a piecewise-constant propagator integration.

---

## 5. Phase 5 — GPU Feasibility Assessment

GPU support was explicitly evaluated and **deferred**.

### Why GPU is not yet warranted

1. **QuTiP dominates runtime.** The main solver path is `qt.sesolve` /
   `qt.mesolve`, which uses SciPy ODE integrators internally.  These are not
   GPU-accelerated and cannot be trivially replaced.
2. **State spaces are small.** Typical system dimensions (2×12 = 24 for the
   reduced model, 3×10 = 30 for the full model) do not saturate GPU memory
   bandwidth.  Dense matrix operations at these sizes are faster on CPU.
3. **Transfer overhead dominates.** Moving QuTiP `Qobj` objects to GPU arrays
   (via CuPy or JAX/CUDA) at each simulation call would dominate the GPU compute
   time for current workloads.
4. **GRAPE propagators.** The GRAPE inner loop calls `scipy.linalg.expm` and
   `expm_frechet` per time slice.  Batching these across slices and using CUDA
   BLAS would require a fundamental restructure of the propagator path
   (`propagators.py`), which is not yet warranted.

### When GPU will become relevant

GPU support should be revisited when:
- Hilbert space dimensions grow to 50×50 or larger (e.g., multi-mode systems
  with higher truncation).
- A lower-level propagator path (bypassing QuTiP) is adopted for GRAPE or RL
  inner loops.
- Batched gradient evaluation over many independent GRAPE systems is needed.

If/when GPU support is added, the `BaseBackend` interface is designed to
accommodate a `CupyBackend` or `JaxGpuBackend` with no changes to the
simulation runner.

---

## 6. Phase 6 — Benchmark Results Summary

See `benchmarks/performance_audit.md` for full details.

| Workload | Before | After | Change |
|---|---:|---:|---:|
| `compile_heavy_20` | `0.031422 s` | `0.000466 s` | `-98.5%` |
| `repeat_simulate_20` | `0.091461 s` | `0.067807 s` | `-25.9%` |
| `repeat_simulate_20` (prepared session) | `0.091461 s` | `0.065020 s` | `-28.9%` |
| `run_all_xy` | `0.106114 s` | `0.055380 s` | `-47.8%` |

The compile-heavy path improved by ~67× due to support-aware sampling.
The simulation loops improved modestly (~25–30%) because QuTiP `sesolve` still
dominates once setup costs are eliminated.

A later follow-up on the prepared-session path showed that reusing a prebuilt
`QobjEvo` within `SimulationSession` still removed a measurable slice of the
remaining setup overhead: on the local 20-run prepared-session microbenchmark,
the average runtime dropped from `0.075338 s` to `0.067438 s` (`~11.7%`), with
best-case runtime improving from `0.073404 s` to `0.060390 s` (`~21.5%`).

---

## 7. Design Constraints and Quality Bar

### Preserved

- Public API stability: existing `simulate_sequence`, `prepare_simulation`,
  `solve_grape`, `GrapeConfig`, and all model/operator interfaces are unchanged.
- Physics correctness: all optimized paths are validated against the reference
  QuTiP solver in `tests/test_29_performance_paths.py` and
  `tests/test_45_performance_multistart_and_sweep.py`.
- Scientific conventions: no optimization changes Hamiltonian assembly logic,
  frame handling, or operator ordering.

### Not done (by design)

- No blanket multiprocessing sprinkled across the codebase.
- No duplicate codepaths for "fast" vs. "normal" execution beyond what the
  backend interface already provides.
- No GPU dependencies added without clear rationale.
- No JAX-JIT for GRAPE gradients yet (deferred to when propagator path is
  restructured).

---

## 8. Recommended Future Work

1. **JAX-JIT GRAPE propagator**: Implement `build_propagation_data` using JAX
   `vmap` + `jit` for batched time-step propagation.  This would enable
   automatic differentiation of the full GRAPE gradient and GPU execution.
2. **Stochastic trajectory solver**: Add a Monte Carlo trajectory backend to
   `sim/solver.py` for open-system simulations that do not need the full density
   matrix.
3. **Larger truncation stress tests**: Benchmark with N_cav=20–30 to determine
   at what dimension GPU-accelerated dense propagation becomes beneficial.
4. **Thread-based parallelism for RL episodes**: Evaluate whether `threading`
   (rather than `multiprocessing`) can avoid `spawn` overhead for RL rollout
   batches.
5. **Parallel SQR calibration**: Apply `run_sweep` to the SQR calibration loop
   which currently evaluates the same objective over many Fock sectors serially.
