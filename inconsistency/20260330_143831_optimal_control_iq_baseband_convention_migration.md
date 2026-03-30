# 2026-03-30 14:38:31 Optimal-Control I/Q Baseband Convention Migration

## Confirmed Issues

### 1. Optimal-control export and documentation still used the legacy `I - iQ` baseband rule

- What:
  - `cqed_sim.optimal_control` still combined exported I/Q controls as `c(t) = I(t) - i Q(t)`.
  - Public documentation and internal implementation reports described the same legacy rule.
- Where:
  - `cqed_sim/optimal_control/parameterizations.py`
  - `cqed_sim/optimal_control/README.md`
  - `API_REFERENCE.md`
  - `documentations/api/optimal_control.md`
  - `documentations/physics_conventions.md`
  - `documentations/tutorials/optimal_control.md`
  - `physics_and_conventions/physics_conventions_report.tex`
  - `docs/20260317_optimal_control_implementation_report.md`
  - `docs/20260327_structured_optimal_control_implementation_report.md`
- Affected components:
  - model-backed pulse export
  - structured pulse export and artifact generation
  - user-facing optimal-control documentation
  - physics/convention documentation
- Why this is inconsistent:
  - The requested repo convention for the optimal-control/public I/Q layer is the standard positive-imaginary baseband `I + iQ`.
  - Keeping the legacy `I - iQ` rule left the optimal-control surface inconsistent with newer repo surfaces that already described or assumed `I + iQ`.
- Consequences:
  - exported pulse phases for pure-Q schedules were flipped relative to the intended public convention
  - public docs taught a different baseband rule than some newer examples and audits
  - users had to reason about an avoidable sign inversion at the optimizer/export boundary

### 2. Model-backed and structured Q-quadrature implementations were tied to the legacy export sign

- What:
  - Model-backed `Q` terms were built from `-i(raising - lowering)`.
  - Structured Fourier evaluation, structured command decomposition, pullback reconstruction, and channel export all encoded the same legacy sign.
- Where:
  - `cqed_sim/optimal_control/utils.py`
  - `cqed_sim/optimal_control/structured.py`
  - `cqed_sim/optimal_control/problems.py` via `quadrature_operators(...)`
- Affected components:
  - model-backed GRAPE problems
  - structured pulse families
  - structured gradient pullback
  - structured artifact spectra and waveform export
- Why this is inconsistent:
  - Switching the exported baseband to `I + iQ` without also changing the model-backed and structured `Q` definitions would silently change the physical Hamiltonian represented by the optimized controls.
- Consequences:
  - a partial sign flip would have broken replay consistency between optimizer-side control operators and runtime pulse export
  - structured parameter gradients and pure-Q Fourier envelopes would have been internally inconsistent

## Suspected / Follow-up Questions

- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb` still contains stored output text from the historical `I - iQ` mapping. The code path is now fixed, but the notebook output itself would need a notebook-specific rerun/update pass if that stored output is meant to remain a canonical rendered artifact.
- `cqed_sim.floquet.builders` uses a separate quadrature labeling surface (`x` / `y` rather than optimal-control `I` / `Q`). That sign choice was not changed here and should be treated as a separate convention audit if repo-wide quadrature semantics are later unified further.

## Status

Fixed on 2026-03-30 for the optimal-control stack.

The canonical optimal-control/public baseband convention is now `c(t) = I(t) + i Q(t)`. Model-backed `Q` controls are built as `+i(raising - lowering)`, and the structured backend now uses the same sign in waveform evaluation, command decomposition, pullback reconstruction, and export. Public docs, the physics report, and the related implementation reports were synchronized to the same rule.

## Fix Record

- Updated the model-backed quadrature helper in `cqed_sim/optimal_control/utils.py` so `Q = +i(raising - lowering)`.
- Updated optimal-control pulse export and metadata in `cqed_sim/optimal_control/parameterizations.py`.
- Updated structured Fourier envelopes, structured command decomposition, structured pullback, and structured export in `cqed_sim/optimal_control/structured.py`.
- Added regression coverage in:
  - `tests/test_40_optimal_control_grape.py`
  - `tests/test_52_structured_optimal_control.py`
- Updated public and convention documentation in:
  - `API_REFERENCE.md`
  - `documentations/api/optimal_control.md`
  - `documentations/physics_conventions.md`
  - `documentations/tutorials/optimal_control.md`
  - `cqed_sim/optimal_control/README.md`
  - `physics_and_conventions/physics_conventions_report.tex`
- Updated related implementation and migration reports in:
  - `docs/20260317_optimal_control_implementation_report.md`
  - `docs/20260327_structured_optimal_control_implementation_report.md`
  - `inconsistency/20260330_121748_positive_drive_frequency_carrier_api_migration.md`
- Updated stale benchmark metadata artifacts in:
  - `outputs/tutorial_grape_benchmark.json`
  - `outputs/optimal_control_larger_benchmark.json`
