# `cqed_sim` Performance Audit

This note summarizes the profiling pass and the optimization changes applied to the core simulator path.

## How to reproduce

Run the benchmark harness from the repository root:

```bash
python benchmarks/run_performance_benchmarks.py --repeat 2 --output benchmarks/latest_results.json
```

The committed machine-specific results from this pass are in `benchmarks/latest_results.json`.

## Baseline profiling summary

The initial `cProfile` pass on representative workloads showed four distinct bottleneck classes.

1. Compile-heavy paths were dominated by Python/NumPy overhead in `SequenceCompiler.compile`, `Pulse._sample_analytic`, `apply_first_order_lowpass`, and `apply_zoh`.
2. Repeated simulation loops were solver-bound, but a nontrivial fraction of time was still spent in QuTiP result bookkeeping because the runtime always stored state history and always requested default expectations.
3. Calibration loops were dominated by repeated solver and propagator calls, especially in SQR-related objective evaluations.
4. Unitary-synthesis fits were dominated by repeated backend simulation rather than by the surrounding optimizer logic.

Measured pre-optimization averages from that audit:

| Workload | Baseline avg. runtime |
| --- | ---: |
| `compile_heavy_20` | `0.031422 s` |
| `repeat_simulate_20` | `0.091461 s` |
| `run_all_xy` | `0.106114 s` |
| `small_calibration` | `0.274942 s` |
| `unitary_fit_small` | `0.057405 s` |

## Implemented optimizations

### Core runtime

- Added per-model caching of lifted operators and static Hamiltonians in `cqed_sim/core/model.py` and `cqed_sim/core/readout_model.py`.
- Added cached default observables in `cqed_sim/sim/runner.py`.
- Stopped forcing full state-history storage for every run. The solver now requests `store_final_state=True` and only stores the full trajectory when `SimulationConfig(store_states=True)` is requested.
- Added `SimulationSession`, `prepare_simulation(...)`, and `simulate_batch(...)` so repeated runs can reuse Hamiltonian assembly, collapse operators, observables, and solver options.
- Prepared QuTiP-backed simulation sessions now also prebuild and reuse a `qt.QobjEvo`, so repeated `session.run(...)` calls do not rebuild QuTiP's coefficient/interpolator objects on every solve.

### Compile path

- Changed `SequenceCompiler.compile(...)` to sample pulses only on their support interval instead of evaluating every pulse across the full global grid.
- Vectorized zero-order hold processing in `cqed_sim/pulses/hardware.py`.
- Added a SciPy `lfilter` fast path for the first-order low-pass filter.

### Calibration and tomography helpers

- Cached SQR calibration time grids and reused Pauli matrices in `cqed_sim/calibration/sqr.py`.
- Updated tomography helpers in `cqed_sim/tomo/protocol.py` to use the no-expectation fast path when they only consume the final state.

## Post-optimization benchmark results

Local results from `benchmarks/latest_results.json`:

| Workload | After avg. runtime | Notes |
| --- | ---: | --- |
| `compile_heavy_20` | `0.000466 s` | Compile-path microbenchmark with 20 pulses and hardware distortion enabled. |
| `repeat_simulate_20_raw` | `0.067807 s` | Repeated `simulate_sequence(...)` calls with `e_ops={}`. |
| `repeat_simulate_20_prepared` | `0.065020 s` | Same workload via `prepare_simulation(...)` + `session.run(...)`. |
| `run_all_xy` | `0.055380 s` | Benefits from disabling unused expectations in the tomography helper. |
| `run_fock_resolved_tomo_n2` | `0.076254 s` | Small protocol-style tomography run. |
| `build_sqr_multitone_pulse` | `0.000024 s` | Pulse-construction overhead is negligible relative to solve time. |
| `conditional_loss_12` | `0.025104 s` | Twelve SQR objective evaluations. |
| `unitary_fit_small` | `0.059487 s` | Tiny synthesis fit; still solver/objective bound. |

Representative before/after deltas on the audited workloads:

| Workload | Before avg. | After avg. | Change |
| --- | ---: | ---: | ---: |
| `compile_heavy_20` | `0.031422 s` | `0.000466 s` | `-98.5%` |
| `repeat_simulate_20` | `0.091461 s` | `0.067807 s` | `-25.9%` |
| `repeat_simulate_20` via prepared session | `0.091461 s` | `0.065020 s` | `-28.9%` |
| `run_all_xy` | `0.106114 s` | `0.055380 s` | `-47.8%` |
| `unitary_fit_small` | `0.057405 s` | `0.059487 s` | `+3.6%` |

Interpretation:

- The biggest wins are in compile-heavy and experiment-loop workloads where the old path paid repeated Python overhead or unnecessary expectation bookkeeping.
- The prepared-session API gives a real gain, but the improvement is modest because the underlying QuTiP solve still dominates once setup costs are removed.
- Tiny unitary-synthesis fits did not materially speed up. Their cost still lives in repeated backend simulation and objective evaluation.

### Prepared-session follow-up

A direct follow-up microbenchmark on the prepared-session path compared the old list-based QuTiP handoff against the new prebuilt `QobjEvo` reuse for the same 20-run `session.run(...)` workload.

| Path | Best runtime | Avg. runtime | Relative speed |
| --- | ---: | ---: | ---: |
| Prepared session with raw Hamiltonian list | `0.073404 s` | `0.075338 s` | `1.00x` |
| Prepared session with reused `QobjEvo` | `0.060390 s` | `0.067438 s` | `1.12x` avg, `1.22x` best |

This is a worthwhile follow-up optimization because it attacks the remaining repeated setup cost inside the main solver-backed simulator path without changing public APIs or the physics model.

## Parallel execution

The new batch API supports multiprocessing through `SimulationSession.run_many(...)` / `simulate_batch(...)`, but the local benchmark shows that Windows `spawn` overhead dominates small and medium jobs:

| Batch size | Serial | Parallel (`2` workers) | Speedup |
| --- | ---: | ---: | ---: |
| `4` | `0.006851 s` | `4.603528 s` | `0.0015x` |
| `8` | `0.012033 s` | `4.776998 s` | `0.0025x` |
| `16` | `0.023917 s` | `4.695389 s` | `0.0051x` |

Practical guidance:

- Use serial prepared sessions for inner calibration and sweep loops unless each task is much heavier than the benchmarks above.
- Use multiprocessing only for coarse-grained jobs where each worker runs many milliseconds to seconds of solver work.
- On Windows, benchmark from a real `.py` file. `spawn` cannot be measured correctly from an inline stdin script.
- On this Windows machine, spawned worker validation is also more reliable from a plain `.py` script than from `pytest`, because the `pytest` launcher itself is not an ideal parent process for spawn-based timing or hang diagnosis.

## GPU feasibility

No GPU path was added.

Reasoning:

- The main runtime is still QuTiP-solver dominated.
- This codebase does not currently have a clean GPU-backed solver path for `sesolve` / `mesolve`.
- The highest-value wins came from reducing Python overhead and avoiding redundant solver setup, not from array throughput alone.

For the current architecture, prepared sessions plus coarse CPU parallelism are the correct optimization direction. GPU work would only become compelling after moving substantial parts of the solver and operator stack away from the current QuTiP execution model.

## Multi-start GRAPE

`solve_grape_multistart(...)` runs N independent GRAPE restarts and returns all results sorted by objective value (best first).  Each restart uses a different seed derived from the base `GrapeConfig.seed`.

```python
from cqed_sim import GrapeConfig, GrapeMultistartConfig, solve_grape_multistart

results = solve_grape_multistart(
    problem,
    config=GrapeConfig(maxiter=200, seed=0),
    multistart_config=GrapeMultistartConfig(n_restarts=6, max_workers=1),
)
best = results[0]
```

Parallel execution (`max_workers > 1`) uses `ProcessPoolExecutor` with the `spawn` context.  The same Windows `spawn` overhead documented above applies: only profitable when each individual GRAPE run takes several seconds or more.

## Parameter sweep runner

`run_sweep(sessions, initial_states, max_workers)` handles the complementary case to `simulate_batch`: many independent sessions each with a potentially different Hamiltonian.  This is the natural API for parameter sweeps (detuning, chi, amplitude, …) where the model changes per sweep point.

```python
from cqed_sim import prepare_simulation, run_sweep

sessions = [prepare_simulation(model_at_chi(chi), compiled, drive_ops) for chi in chi_values]
results = run_sweep(sessions, [psi0] * len(chi_values))
```

Serial execution is the default.  Parallel execution carries the same `spawn`-overhead caveat as `simulate_batch`.

## Remaining limitations

- SQR calibration and unitary-synthesis optimization remain dominated by repeated solver/objective evaluations.
- `simulate_batch(..., max_workers>1)` and `run_sweep(..., max_workers>1)` are correct, but not a win for small tasks on this Windows environment.
- GRAPE gradient computation is already vectorized over the state dimension; the per-slice `expm_frechet` calls are inherently sequential and are not further vectorizable without a different propagator formulation.
- The benchmark results are machine-specific and should be treated as regression-reference numbers, not universal performance guarantees.
