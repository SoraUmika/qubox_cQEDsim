# Tutorial Curriculum Refactor Audit

Created: 2026-03-13 14:30:30 local time
Status: fixed

## Confirmed issues

### 1. Guided notebook tutorials still live under `examples/` and the docs still describe `examples/` as the main educational entry point

- What it is:
  - The repository still keeps the primary guided notebooks in `examples/workflows/` and `examples/`, including:
    - `examples/workflows/cqed_sim_usage_examples.ipynb`
    - `examples/workflows/sequential_simulation.ipynb`
    - `examples/workflows/sqr_calibration_workflow.ipynb`
    - `examples/sideband_interactions.ipynb`
  - `README.md`, `documentations/getting_started.md`, `documentations/examples.md`, `documentations/architecture.md`, `documentations/tutorials/*.md`, and `mkdocs.yml` still point users to `examples/` for tutorial-style or worked-example material.
- Where it appears:
  - `README.md`
  - `documentations/*`
  - `mkdocs.yml`
  - `examples/`
- Affected components:
  - user onboarding
  - notebook discoverability
  - examples/tutorials folder boundary
- Why inconsistent:
  - The project now wants a dedicated top-level `tutorials/` curriculum for guided learning material, with `examples/` reserved for smaller standalone snippets or non-curriculum workflows.
- Consequence:
  - users are taught to look in the wrong place, and the repo boundary between tutorial material and examples remains blurry.

### 2. Some active user-facing example material still uses legacy or mixed unit presentation that conflicts with the current documented SI-first tutorial convention

- What it is:
  - `examples/displacement_qubit_spectroscopy.py` uses `ns` / `rad/ns` helpers and labels, while the current public docs and conventions report describe `cqed_sim` internal frequencies in `rad/s` and time in `s`.
- Where it appears:
  - `examples/displacement_qubit_spectroscopy.py`
  - nearby tutorial/documentation references that present the script as a tutorial-style example
- Affected components:
  - spectroscopy onboarding
  - unit/convention teaching
  - tutorial truthfulness
- Why inconsistent:
  - the repository has already standardized the public convention language around SI-style internal units, but at least one still-promoted educational example presents a different unit style.
- Consequence:
  - users can infer that multiple internal unit conventions are equally canonical, which is especially risky in spectroscopy tutorials where frame and carrier sign already require careful interpretation.

### 3. Tutorial-related regression coverage still points at the old example notebook path

- What it is:
  - `tests/test_33_usage_examples_spectroscopy_sign.py` asserts content inside `examples/workflows/cqed_sim_usage_examples.ipynb`.
- Where it appears:
  - `tests/test_33_usage_examples_spectroscopy_sign.py`
- Affected components:
  - regression coverage for spectroscopy-sign tutorial behavior
  - tutorial notebook relocation
- Why inconsistent:
  - moving the curriculum to `tutorials/` without updating the regression target would keep the test suite coupled to an obsolete notebook location.
- Consequence:
  - either the test breaks during the refactor or the old notebook must be left behind just to satisfy stale regression wiring.

## Resolution update

Updated: 2026-03-13 local time

- Issue 1 fixed:
  - Added a numbered top-level `tutorials/` curriculum (`00` through `26`) plus `tutorials/README.md`, `tutorials/tutorial_manifest.md`, and `tutorials/conventions_quick_reference.md`.
  - Updated `README.md`, `API_REFERENCE.md`, `documentations/*`, and `mkdocs.yml` so the guided-learning entry point now points to `tutorials/`.
  - Removed the duplicated guided notebooks from `examples/`:
    - `examples/workflows/cqed_sim_usage_examples.ipynb`
    - `examples/workflows/sequential_simulation.ipynb`
    - `examples/workflows/sqr_calibration_workflow.ipynb`
    - `examples/sideband_interactions.ipynb`
- Issue 2 fixed:
  - Converted `examples/displacement_qubit_spectroscopy.py` to the repository's SI-style internal unit convention (`s`, `rad/s`) and corrected the pulse-amplitude scaling so the script still produces a meaningful displaced-state spectroscopy result.
- Issue 3 fixed:
  - Updated `tests/test_33_usage_examples_spectroscopy_sign.py` to validate `tutorials/06_qubit_spectroscopy.ipynb` and `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`.
  - The tutorial regression checks were rerun directly through Python function calls because `pytest` is not available in the current machine Python environment.

## Suspected / follow-up items

### A. The advanced workflow notebooks may need to be split between `tutorials/` and `examples/` rather than moved one-for-one

- What it is:
  - `examples/workflows/sequential_simulation.ipynb` and `examples/workflows/sqr_calibration_workflow.ipynb` are educational, but they are also specialized orchestration notebooks with repo-specific JSON/config assumptions.
- Open question:
  - it may be clearer to replace them with smaller curriculum notebooks in `tutorials/` and leave only lightweight scripts or helpers in `examples/`.
- Resolution:
  - This was resolved during the refactor by replacing the old notebooks with smaller curriculum notebooks in `tutorials/` (`21` through `25`) and leaving only the narrower workflow helpers/scripts in `examples/`.
