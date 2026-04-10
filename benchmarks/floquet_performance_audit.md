# Floquet Performance Audit

This note summarizes the first dedicated Floquet benchmark pass.

## How to reproduce

Run the benchmark harness from the repository root:

```bash
python benchmarks/run_floquet_benchmarks.py --repeat 2 --output benchmarks/floquet_latest_results.json
```

The committed machine-specific results from this pass are in `benchmarks/floquet_latest_results.json`.

## Scope

The Floquet harness measures three workload families:

- single-point solves for a sinusoidally frequency-modulated transmon,
- a cQED red-sideband single-point solve,
- parameter sweeps using `run_floquet_sweep(...)`.

Each case records runtime and peak Python memory measured with `tracemalloc`.
This is useful for comparative operating envelopes but does not capture every
native allocation inside NumPy, SciPy, or QuTiP.

## Operating-envelope interpretation

- Small single-point Floquet solves should remain in the interactive regime.
- Medium single-point solves with Sambe enabled are acceptable for exploratory work but are noticeably heavier than the propagator-only path.
- Longer sweeps are the first cases that should be treated as batch-style workloads, especially when higher `n_time_samples` or larger Hilbert spaces are required.

## Practical guidance

- Prefer the propagator route for routine use; enable Sambe only when harmonic-space interpretation is needed.
- Keep `n_time_samples` modest for early scans and only raise it after locating the physically relevant region.
- Use short sweeps for interactive branch inspection and reserve dense sweeps for batch jobs or cached results.
- Treat the committed numbers as machine-specific regression references, not universal guarantees.