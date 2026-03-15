# Example Tutorial Migration Audit

Created: 2026-03-15 14:35:48 local time  
Status: fixed

## Confirmed issues

### 1. The repository did not yet have a categorized notebook suite that rewrote the current representative example scripts as workflow tutorials

- What it is:
  - The top-level `examples/*.py` scripts already covered realistic workflows such as displacement spectroscopy, Kerr free evolution, sideband swap, sequential sideband reset, and synthesis studies.
  - `tutorials/` still primarily reflected the earlier flat numbered curriculum rather than a script-migration-oriented workflow taxonomy.
- Where it appears:
  - `examples/*.py`
  - `tutorials/`
  - discovery docs pointing users to tutorials without a workflow-script mapping
- Affected components:
  - user onboarding
  - tutorial discoverability
  - examples/tutorials boundary
- Why inconsistent:
  - The project wanted notebook-first educational artifacts for the representative example workflows, with a clear split between beginner workflows, advanced studies, and validation notebooks.
- Consequence:
  - The scripts remained more discoverable than their notebook counterparts for several practical workflows, and validation-oriented scripts were easy to misread as beginner examples.

### 2. `examples/run_snap_optimization_demo.py` was broken as a standalone script from the repository root

- What it is:
  - Running `python examples/run_snap_optimization_demo.py` raised `ModuleNotFoundError: No module named 'examples.studies'`.
- Where it appears:
  - `examples/run_snap_optimization_demo.py`
- Affected components:
  - advanced example usability
  - tutorial migration audit accuracy
- Why inconsistent:
  - Most of the other top-level example scripts already add the repository root to `sys.path` before importing repo-local modules.
- Consequence:
  - the script could not honestly be treated as a runnable example until the import path was repaired.

### 3. `examples/kerr_sign_verification.py` and `examples/sideband_swap.py` had roles that were easy to misinterpret from their filenames alone

- What it is:
  - `kerr_sign_verification.py` is primarily a convention-validation artifact, not a beginner usage example.
  - `sideband_swap.py` is a compatibility wrapper around `sideband_swap_demo.py`, not a richer separate workflow in its current form.
- Where it appears:
  - `examples/kerr_sign_verification.py`
  - `examples/sideband_swap.py`
- Affected components:
  - tutorial classification
  - example discoverability
- Why inconsistent:
  - The folder structure already distinguishes workflows, studies, and audits, but the old example-script presentation did not make those roles explicit enough.
- Consequence:
  - users can start in the wrong place or infer that duplicate scripts represent distinct maintained tutorial paths.

## Resolution update

Updated: 2026-03-15 local time

- Issue 1 fixed:
  - Added a categorized workflow tutorial suite under:
    - `tutorials/00_getting_started/`
    - `tutorials/10_core_workflows/`
    - `tutorials/20_bosonic_and_sideband/`
    - `tutorials/30_advanced_protocols/`
    - `tutorials/40_validation_and_conventions/`
  - Added `tutorials/TUTORIAL_MIGRATION_PLAN.md` to record the audit, classification, runnable status, and destination of each example-side Python script.
  - Updated `README.md`, `tutorials/README.md`, `documentations/getting_started.md`, `documentations/examples.md`, and the tutorial landing/topic pages to point users to the new notebook-first workflow suite.
- Issue 2 fixed:
  - Patched `examples/run_snap_optimization_demo.py` to add the repository root to `sys.path`, matching the other top-level example scripts.
  - Re-ran the script successfully from the repository root on 2026-03-15.
- Issue 3 fixed:
  - Migrated the Kerr-sign check into `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`.
  - Documented `examples/sideband_swap.py` as a retained compatibility wrapper while pointing the user-facing flow to `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb`.

## Remaining follow-up items

- `examples/ringdown_noise.py`
  - Retained as a compact script in this pass rather than converted into a workflow notebook.
  - This is intentional, not an unresolved breakage.
