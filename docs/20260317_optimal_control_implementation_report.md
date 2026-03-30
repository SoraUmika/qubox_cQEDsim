# 2026-03-17 Optimal-Control Implementation Report

## Scope

This task added a first-class modular optimal-control layer to `cqed_sim` with GRAPE as the first solver backend.

Implemented deliverables:

- Phase 0 audit note: `docs/20260317_optimal_control_phase0_audit.md`
- Phase 1 design note: `docs/20260317_optimal_control_phase1_design.md`
- new package: `cqed_sim/optimal_control`
- regression coverage: `tests/test_40_optimal_control_grape.py`
- public API export updates in `cqed_sim/__init__.py`
- example script: `examples/grape_storage_subspace_gate_demo.py`
- tutorial notebook: `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- public documentation synchronization across `README.md`, `API_REFERENCE.md`, `documentations/`, and physics conventions material

## Architecture Summary

The implementation follows the requested design principle: define a generic control-problem abstraction first, then implement GRAPE as one solver backend.

Added layers:

- `problems.py`
  - solver-agnostic `ControlTerm`, `ControlSystem`, `ControlProblem`, model-backed builders, and ensemble-member support
- `parameterizations.py`
  - piecewise-constant time grids, schedules, flatten/unflatten helpers, and pulse export back into the standard runtime stack
- `objectives.py`
  - state-transfer and unitary objectives, including adapters for existing `unitary_synthesis` targets
- `penalties.py`
  - amplitude, slew-rate, and leakage penalties
- `propagators.py`
  - dense propagation and exact slice derivatives using Fr\'echet derivatives of the matrix exponential
- `grape.py`
  - GRAPE solver configuration, evaluation, aggregation, optimizer integration, and results assembly
- `result.py`
  - serializable result objects and pulse export helpers

## Conventions Preserved

The new package preserves existing repository conventions rather than introducing a parallel control stack.

- drift and control Hamiltonian coefficients remain in `rad/s`
- times remain in `s`
- tensor ordering remains transmon-first when present
- model-backed control problems reuse the existing drive-operator helpers from the model layer
- exported runtime pulses follow

$$
c(t) = I(t) + i Q(t)
$$

with model-backed `Q` terms constructed as `+i(raising - lowering)`, so the optimized quadrature controls replay through the existing pulse/runtime stack with the same effective Hamiltonian meaning used during optimization

## Important Numerical Fix

The main implementation issue during testing was not a wrong gradient; it was an ill-conditioned optimizer variable scale.

- raw gradients in physical `rad/s` units were extremely small because each derivative carries the short slice duration factor
- finite-difference checks confirmed that the analytic gradient was correct
- `GrapeSolver` was updated to optimize over a bound-scaled dimensionless vector and apply the gradient chain rule back to physical amplitudes

That change resolved the main convergence failure in the model-backed regression cases.

## Validation Performed

Focused regression tests:

- command:
  - `C:/Users/dazzl/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_37_api_export_completeness.py tests/test_40_optimal_control_grape.py`
- result:
  - `39 passed in 0.76s`

Validated behaviors covered by the new regression file:

- model-backed subspace unitary optimization
- state-preparation objective optimization
- ensemble worst-case improvement
- pulse export and runtime replay

Example verification:

- command:
  - `C:/Users/dazzl/AppData/Local/Programs/Python/Python312/python.exe examples/grape_storage_subspace_gate_demo.py`
- observed output:
  - optimizer converged successfully
  - nominal fidelity `1.000000`
  - exported pulse count `1`
  - runtime replay fidelity on `|g,0>`: `0.999444`

Physics documentation build:

- `latexmk` task failed because the local MiKTeX installation could not find `perl`
- fallback `pdflatex` task succeeded
- final artifact rebuilt successfully:
  - `physics_and_conventions/physics_conventions_report.pdf`

## Documentation Updates

Public documentation was synchronized across the canonical and website-facing surfaces.

- `README.md`
- `API_REFERENCE.md`
- `documentations/api/optimal_control.md`
- `documentations/tutorials/optimal_control.md`
- `documentations/api/overview.md`
- `documentations/examples.md`
- `documentations/architecture.md`
- `documentations/index.md`
- `documentations/tutorials/index.md`
- `tutorials/README.md`
- `mkdocs.yml`
- `physics_and_conventions/physics_conventions_report.tex`
- `documentations/physics_conventions.md`

## Current Scope Boundaries

This first pass intentionally focuses on:

- dense closed-system GRAPE
- piecewise-constant controls
- direct state-transfer and unitary objectives
- simple control penalties and ensemble aggregation

Not implemented in this pass:

- open-system GRAPE gradients
- stochastic measurement-conditioned control
- Krotov / CRAB / GOAT backends
- automatic hardware-distortion-aware gradient propagation

## Outcome

The repository now has an integrated optimal-control layer that fits the existing `cqed_sim` architecture, reuses current model and runtime conventions, and is covered by dedicated regression tests, a runnable example script, a notebook workflow, and synchronized public documentation.
