# GRAPE Optimal Control Tutorial

The optimal-control walkthrough lives in:

- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`

It demonstrates:

- building a `ControlProblem` from an existing `cqed_sim` model
- choosing a `PiecewiseConstantTimeGrid`
- defining a `UnitaryObjective` on a retained logical subspace
- solving the problem with `GrapeSolver`
- exporting the optimized schedule back into `Pulse` objects
- replaying the result through `SequenceCompiler` and `simulate_sequence(...)`
- comparing nominal replay and noisy replay through `evaluate_with_simulator(...)`
- launching the benchmark harness for larger GRAPE validation cases

The companion standalone script is:

- `examples/grape_storage_subspace_gate_demo.py`

The companion benchmark harness is:

- `benchmarks/run_optimal_control_benchmarks.py`

The script uses the same model-backed workflow outside the notebook and now also demonstrates noisy replay reporting.

This tutorial is the starting point for direct optimal-control workflows inside `cqed_sim` when the task is naturally expressed as piecewise-constant control amplitudes rather than a sequence of higher-level gate primitives.