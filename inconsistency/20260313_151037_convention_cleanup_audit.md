# Convention Cleanup Audit

Created: 2026-03-13 15:10:37 local time
Status: fixed

## Confirmed issues

### 1. Ramsey calibration-target notebooks used a stale fitted-parameter key

- What it is:
  - The public calibration-target helper `cqed_sim.calibration_targets.run_ramsey(...)` returns fitted parameters under the keys `delta_omega` and `t2_star`.
  - `tutorials/25_small_calibration_workflow_end_to_end.ipynb` still read `ramsey.fitted_parameters["detuning"]`.
- Where it appears:
  - `tutorials/_generate_tutorials.py`
  - generated `tutorials/25_small_calibration_workflow_end_to_end.ipynb`
- Affected components:
  - user-facing tutorial API usage
  - calibration workflow example truthfulness
- Why inconsistent:
  - `API_REFERENCE.md`, `documentations/api/calibration_targets.md`, and `tests/calibration_targets/test_targets.py` all document and validate the public Ramsey result key as `delta_omega`.
- Consequence:
  - users following the calibration workflow notebook would hit a stale public-API lookup even though the rest of the repo documents the correct interface.

### 2. The canonical physics report still referenced the pre-refactor tutorial layout

- What it is:
  - `physics_and_conventions/physics_conventions_report.tex` still said high-level recipes live under `examples/workflows/` and still described `tests/test_33_usage_examples_spectroscopy_sign.py` as guarding the deleted notebook `examples/workflows/cqed_sim_usage_examples.ipynb`.
- Where it appears:
  - `physics_and_conventions/physics_conventions_report.tex`
- Affected components:
  - canonical conventions documentation
  - tutorial/test discoverability
- Why inconsistent:
  - the repository's guided learning path now lives under top-level `tutorials/`, and `tests/test_33_usage_examples_spectroscopy_sign.py` was already updated to check Tutorials 06 and 07 instead of the deleted example notebook.
- Consequence:
  - the source-of-truth conventions report would incorrectly point users and future contributors at removed notebook paths.

### 3. The canonical physics report claimed a dedicated Kerr-sign regression file that did not exist

- What it is:
  - `physics_and_conventions/physics_conventions_report.tex` listed `tests/test_32_kerr_sign_notebook_regression.py` as a passed direct regression test, but no such file existed in `tests/`.
- Where it appears:
  - `physics_and_conventions/physics_conventions_report.tex`
  - `tests/`
- Affected components:
  - canonical test inventory
  - self-Kerr sign regression coverage
- Why inconsistent:
  - the repo already had Kerr-sign verification code in `examples/kerr_sign_verification.py` and `examples/workflows/kerr_free_evolution.py`, but the named automated regression file was missing.
- Consequence:
  - the report overstated the automated guardrails around notebook-scale Kerr-sign behavior.

## Suspected / wording-level issue

### A. The pure-dephasing helper wording conflated zero inferred dephasing rate with a returned dephasing time value

- What it is:
  - `pure_dephasing_time_from_t1_t2(...)` computes the inferred rate `1/T_phi = max(0, 1/T2 - 1/(2T1))`.
  - When the inferred rate is non-positive, the implementation returns `None`, which the rest of the code interprets as "no extra dephasing term".
  - The conventions report previously said only that the helper "clamps negative inferred `1/T_phi` to zero".
- Where it appears:
  - `cqed_sim/sim/noise.py`
  - `physics_and_conventions/physics_conventions_report.tex`
- Why this needed clarification:
  - the helper returns a dephasing time, not a rate, so a plain "clamped to zero" sentence is easy to misread as "returns `0.0`".

## Resolution update

Updated: 2026-03-13 local time

- Issue 1 fixed:
  - Updated the tutorial generator and regenerated the affected notebooks so the public Ramsey calibration-target API is accessed through `ramsey.fitted_parameters["delta_omega"]`.
  - Renamed the end-to-end workflow summary field to `ramsey_delta_omega_hz` to keep the user-facing label aligned with the public API name.
- Issue 2 fixed:
  - Updated the canonical physics report so the workflow boundary distinguishes `tutorials/` (guided curriculum) from `examples/` and so the `test_33` inventory entry points at the current spectroscopy tutorials.
- Issue 3 fixed:
  - Added `tests/test_32_kerr_sign_notebook_regression.py` so the report's dedicated self-Kerr-sign regression entry now corresponds to a real automated test file.
- Wording issue A fixed:
  - Clarified the report wording so it states that the helper clamps the inferred dephasing rate to zero and encodes that zero-extra-dephasing case by returning `None`.
