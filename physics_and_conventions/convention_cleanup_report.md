# Convention Cleanup Report

Date: 2026-03-13

This cleanup pass treated `physics_and_conventions/physics_conventions_report.tex`
as the canonical source of truth for physics-facing behavior.

## Corrected mismatches

- Tutorials 23 and 25 were brought back into alignment with the public Ramsey
  calibration-target API:
  - use `ramsey.fitted_parameters["delta_omega"]`
  - no longer use the stale key `ramsey.fitted_parameters["detuning"]`
- Calibration-workflow tutorial labels now make the spectroscopy axis explicit as
  drive frequency relative to bare `omega_q`, which avoids mixing that lab-frame
  offset with the transition-detuning axes used in the spectroscopy tutorials.
- The canonical conventions report now reflects the current repo layout:
  - guided lessons live under `tutorials/`
  - repo-side workflow helpers and standalone scripts live under `examples/`
  - `tests/test_33_usage_examples_spectroscopy_sign.py` guards Tutorials 06 and 07
- Added the missing dedicated Kerr-sign regression file
  `tests/test_32_kerr_sign_notebook_regression.py` so the report's test inventory
  matches the actual automated coverage.
- Clarified the `pure_dephasing_time_from_t1_t2(...)` wording:
  - the inferred rate `1/T_phi` is clamped to be non-negative
  - if the inferred extra dephasing rate vanishes, the helper returns `None`
    to indicate that no additional pure-dephasing term should be added

## Files updated in this pass

- `cqed_sim/sim/noise.py`
- `tutorials/_generate_tutorials.py`
- regenerated `tutorials/23_analysis_fitting_and_result_extraction.ipynb`
- regenerated `tutorials/25_small_calibration_workflow_end_to_end.ipynb`
- `documentations/physics_conventions.md`
- `physics_and_conventions/physics_conventions_report.tex`
- `tests/test_14_dissipation.py`
- `tests/test_32_kerr_sign_notebook_regression.py`
- `tests/test_35_tutorial_api_conventions.py`

## Validation performed

- Regenerated all tutorial notebooks from `tutorials/_generate_tutorials.py`
- Rebuilt `physics_conventions_report.pdf`
- Parsed all 27 tutorial notebooks as JSON
- Ran focused direct-Python regression checks for:
  - Ramsey calibration targets
  - pure-dephasing helper semantics
  - Kerr-sign regression
  - spectroscopy sign/tutorial guards
  - tutorial API-convention guards

## Remaining future work

- The core Hamiltonian, frame, Kerr, cross-Kerr, and dispersive-chi implementations
  already matched the canonical report in the audited paths, so this pass did not
  require deeper solver or operator-sign changes.
- `pytest` is not installed in the current machine Python environment, so the
  verification above used direct Python invocation of the affected test functions
  rather than the `pytest` runner.
