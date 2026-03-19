# GRAPE Optimal Control Tutorial

The optimal-control walkthrough lives in:

- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`

It demonstrates:

- building a `ControlProblem` from an existing `cqed_sim` model
- choosing a `PiecewiseConstantTimeGrid`
- choosing either plain piecewise-constant or held-sample command parameterizations
- defining a `UnitaryObjective` on a retained logical subspace
- solving the problem with `GrapeSolver`
- attaching a `HardwareModel` to transform command waveforms into physical waveforms
- exporting the optimized schedule back into `Pulse` objects
- replaying either the command waveform or the physical waveform through `SequenceCompiler` and `simulate_sequence(...)`
- comparing nominal replay and noisy replay through `evaluate_with_simulator(...)`
- launching the benchmark harness for larger GRAPE validation cases

The companion standalone script is:

- `examples/grape_storage_subspace_gate_demo.py`
- `examples/hardware_constrained_grape_demo.py`

The companion benchmark harness is:

- `benchmarks/run_optimal_control_benchmarks.py`

The standalone scripts cover both the original model-backed GRAPE workflow and the new hardware-aware workflow with held-sample controls, low-pass filtering, IQ-radius limits, boundary windows, and command-vs-physical replay.

This tutorial is the starting point for direct optimal-control workflows inside `cqed_sim` when the task is naturally expressed as command waveforms on a propagation grid rather than a sequence of higher-level gate primitives.