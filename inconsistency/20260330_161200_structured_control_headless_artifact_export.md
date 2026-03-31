# 2026-03-30 16:12:00 Structured-Control Headless Artifact Export

## Confirmed Issues

### 1. Structured-control artifact plotting depended on an interactive Matplotlib backend

- What:
  - `save_structured_control_artifacts(...)` created figures through `matplotlib.pyplot.subplots(...)`, which pulled in the default GUI backend during artifact export.
- Where:
  - `cqed_sim/optimal_control/structured.py`
  - surfaced through `tests/test_52_structured_optimal_control.py::test_structured_solver_runs_hardware_aware_model_workflow_and_saves_artifacts`
- Affected components:
  - structured-control artifact export
  - example and tutorial validation runs in headless environments
  - CI or workstation setups without a working Tk installation
- Why this is inconsistent:
  - artifact export is intended to be a reproducible non-interactive workflow, but the plotting path still assumed a locally available GUI toolkit.
- Consequences:
  - structured-control studies could optimize successfully and then fail while saving plots
  - pytest coverage for example- and artifact-facing workflows failed on systems without Tk

## Suspected / Follow-up Questions

- Other plotting-heavy artifact writers outside structured control may still rely on `pyplot` rather than an explicit non-interactive backend. They were not changed in this pass because the failing surface was isolated to structured-control artifact export.

## Status

Fixed on 2026-03-30.

Structured-control artifact export now builds figures through Agg-backed `Figure` objects, so plot saving no longer depends on a GUI backend or a working Tk installation.

## Fix Record

- Fixed by:
  - `cqed_sim/optimal_control/structured.py`
- Validated by:
  - `tests/test_52_structured_optimal_control.py::test_structured_solver_runs_hardware_aware_model_workflow_and_saves_artifacts`
  - example/tutorial validation slice run via pytest over `examples/` plus the tutorial/convention tests
