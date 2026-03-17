# 2026-03-17 13:36:45 Optimal-Control Follow-up Audit

## Confirmed Issues

### 1. Empty public tutorial notebook

- What:
  - `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb` is currently empty on disk.
- Where:
  - the notebook file itself, plus documentation surfaces that advertise it as the main optimal-control tutorial.
- Affected components:
  - tutorial workflow, tutorial indices, and the public documentation path for optimal control.
- Why this is inconsistent:
  - the repository docs describe the notebook as a supported workflow artifact, but the file contents do not currently match that claim.
- Consequences:
  - users following the documented tutorial path land on a blank notebook and cannot reproduce the advertised workflow.

### 2. Solver-agnostic problem abstraction paired with solver-specific result type

- What:
  - the architecture exposes a generic `ControlProblem`, but the result object is currently named and structured only as `GrapeResult`.
- Where:
  - `cqed_sim/optimal_control/result.py`, top-level exports, and API documentation.
- Affected components:
  - future backend extensibility, API naming consistency, and follow-up work that wants to add non-GRAPE evaluation or solver paths.
- Why this is inconsistent:
  - the problem side of the architecture is intentionally backend-agnostic, while the result side still encodes the first backend as the only public result abstraction.
- Consequences:
  - later backends or evaluation-only workflows either need awkward compatibility shims or must keep returning a GRAPE-branded result object.

## Suspected / Follow-up Questions

- The broader pytest slice around the new optimal-control layer appears to pass without direct test failures, but the terminal wrapper produced an intermittent `KeyboardInterrupt` artifact while capturing a large batch summary. This looks like an execution-wrapper issue rather than a repository failure, but it should be re-checked after the follow-up changes.

## Status

- Current status: fixed on 2026-03-17
- Resolution summary:
  - restored `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb` with a complete GRAPE solve, replay, noisy evaluation, and benchmark walkthrough,
  - introduced `ControlResult` in `cqed_sim/optimal_control/result.py` while preserving `GrapeResult` as the GRAPE-specific subtype,
  - added simulator-backed evaluation in `cqed_sim/optimal_control/evaluation.py` and exported it through `cqed_sim/optimal_control/__init__.py` and `cqed_sim/__init__.py`,
  - updated API, tutorial, README, and physics/conventions documentation to match the new public surface.

## Fix Record

### Empty public tutorial notebook

- Fixed by:
  - restoring `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
  - adding notebook validation coverage in `tests/test_41_optimal_control_followup.py`
- Remaining concerns:
  - none confirmed in this follow-up

### Solver-agnostic problem abstraction paired with solver-specific result type

- Fixed by:
  - adding `ControlResult` and preserving `GrapeResult` compatibility in `cqed_sim/optimal_control/result.py`
  - exporting the generalized result/evaluation API from `cqed_sim/optimal_control/__init__.py` and `cqed_sim/__init__.py`
  - documenting the generalized surface in `API_REFERENCE.md` and the website docs
- Remaining concerns:
  - optimization backends are still GRAPE-only, which is acceptable for the current architecture because the generalized result layer now cleanly supports evaluation-only workflows.

### Terminal summary artifact during broader pytest capture

- Re-check status:
  - no repository test failure was reproduced from the earlier `KeyboardInterrupt` capture artifact
  - post-change reruns completed successfully for the focused follow-up slice and adjacent simulator/tutorial slice
  - the neighboring unitary-synthesis subset was rerun separately and completed with `27 passed in 36.81s`
- Remaining concerns:
  - large pytest summaries can still be clipped by the terminal wrapper, but the affected neighboring subset was recovered with a narrower rerun.
