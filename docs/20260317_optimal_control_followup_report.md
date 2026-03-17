# 2026-03-17 Optimal-Control Follow-up Report

## Summary

This follow-up completed three main tasks around the new optimal-control stack:

- broader validation of the surrounding simulator, tutorial, and unitary-synthesis surfaces,
- one architecture extension built on top of `ControlProblem`,
- a dedicated benchmark and validation harness for larger GRAPE cases.

The chosen extension was a simulator-backed evaluation layer rather than a second optimizer backend. That keeps `ControlProblem` central, preserves the current closed-system GRAPE implementation, and makes the optimize-versus-evaluate boundary explicit when Lindblad noise is introduced through the standard runtime.

## Architecture Extension

The result layer now exposes a backend-agnostic `ControlResult` while keeping `GrapeResult` as the GRAPE-specific subtype. A new simulator replay layer in `cqed_sim/optimal_control/evaluation.py` adds:

- `ControlEvaluationCase`
- `ControlObjectiveEvaluation`
- `ControlMemberEvaluation`
- `ControlEvaluationResult`
- `evaluate_control_with_simulator(...)`

This path reuses the repository's existing pulse compilation and simulation flow, including `SequenceCompiler`, `prepare_simulation(...)`, and `NoiseSpec`. It is evaluation-only: optimization remains closed-system, while runtime replay can now report nominal and noisy fidelities for the same optimized schedule.

## Benchmark Harness

`benchmarks/run_optimal_control_benchmarks.py` now provides a configurable benchmark script for state-transfer and retained-subspace cases. It supports configurable slice count, duration, penalty weights, backend selection, seed, robust frequency shifts, optional noisy replay, and JSON output.

Representative validation included:

- updated example replay: nominal runtime fidelity `0.999444`, noisy replay fidelity `0.997861`
- larger benchmark case after warm-start correction: nominal replay fidelity `0.9999913151104448`, noisy replay fidelity `0.9742297440642164`

## Validation

Focused post-change validation completed successfully:

- `tests/test_35_tutorial_api_conventions.py` and `tests/test_41_optimal_control_followup.py`: `6 passed in 3.66s`
- `tests/test_37_api_export_completeness.py`, `tests/test_40_optimal_control_grape.py`, and `tests/test_41_optimal_control_followup.py`: `47 passed in 4.22s`
- adjacent simulator/tutorial slice: `47 passed in 3.23s`
- neighboring unitary-synthesis helpers (`test_metrics.py`, `test_primitives_and_backends.py`, `test_subspace.py`, `test_synthesis_and_targets.py`): `27 passed in 36.81s`

## Documentation and Tutorial Updates

The public docs were synchronized across:

- `README.md`
- `API_REFERENCE.md`
- `documentations/api/optimal_control.md`
- `documentations/tutorials/optimal_control.md`
- `documentations/index.md`
- `tutorials/README.md`
- `physics_and_conventions/physics_conventions_report.tex`
- `documentations/physics_conventions.md`

The tutorial notebook `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb` was restored and extended to cover solving, pulse export, nominal replay, noisy replay, and benchmark usage.

## Remaining Limitations

- optimization backends remain GRAPE-only
- noisy replay is evaluation-only and does not yet provide noisy gradients
- broader benchmark suites still need more representative long-horizon and higher-dimensional coverage for performance tracking
