# Workflow Tutorial Plot Alignment Report

Created: 2026-03-27 19:53:38 local time
Status: fixed

## Confirmed Issues

### 1. The workflow/tutorial displacement-spectroscopy presentation claimed selective Gaussian number splitting while the published image path was still driven by a mismatched spectroscopy setup

- What:
  - The workflow notebook/docs layer described a selective-Gaussian number-splitting experiment with resolved manifold peaks and Fock-weight recovery, but the published documentation image did not reflect that regime.
- Where:
  - `examples/displacement_qubit_spectroscopy.py`
  - `tools/generate_tutorial_plots.py`
  - `tutorials/_generate_workflow_tutorials.py`
  - `documentations/tutorials/displacement_spectroscopy.md`
- Affected components:
  - spectroscopy tutorial truthfulness
  - docs/tutorial image accuracy
  - workflow notebook interpretation
- Why this is inconsistent:
  - The tutorial text taught resolved Gaussian-like number splitting with peak heights tied to displaced-state Fock weights, but the old workflow/docs path still used a broader or stale plotting configuration that did not visibly support that interpretation.
- Consequences:
  - users could not reliably infer how probe bandwidth, displaced-state weights, and dispersive line positions were connected.

### 2. The workflow/tutorial Kerr Wigner image labeled quadrature-space data as if it were plotted in coherent-state alpha coordinates

- What:
  - The published Kerr Wigner image showed a coherent state prepared at `alpha = 2` peaking near `~2.8` on an axis labeled `Re(alpha)`, which is the quadrature-space value `sqrt(2) * alpha` rather than the coherent-state coordinate itself.
- Where:
  - `examples/workflows/kerr_free_evolution.py`
  - `tools/generate_tutorial_plots.py`
  - `tutorials/_generate_workflow_tutorials.py`
  - `documentations/tutorials/kerr_free_evolution.md`
- Affected components:
  - phase-space convention teaching
  - Kerr tutorial interpretation
  - user trust in tutorial visuals
- Why this is inconsistent:
  - The simulator already supports both `quadrature` and `alpha` Wigner coordinates, but the tutorial-facing plot path mixed the quadrature calculation with alpha-style labeling.
- Consequences:
  - users could misdiagnose a correct coherent-state preparation as a simulation or calibration error.

### 3. The Kerr docs image layout crowded the final panel and colorbar enough to obscure the intended `3T_K/4` interpretation

- What:
  - The old subplot/colorbar layout visibly crowded the last panel and label area.
- Where:
  - `examples/workflows/kerr_free_evolution.py`
  - `tools/generate_tutorial_plots.py`
- Affected components:
  - docs readability
  - tutorial figure quality
- Why this is inconsistent:
  - The tutorial image is meant to support qualitative phase-space interpretation across the full Kerr cycle; layout crowding undermines that purpose.
- Consequences:
  - the final snapshot was harder to read and looked less trustworthy than the underlying simulation warranted.

## Suspected / Follow-up Questions

- Additional workflow notebooks that make strong visual or theory claims should continue to be audited case by case, especially when a plot can be generated in multiple coordinate systems or bandwidth regimes.
- The docs/tutorial layer may benefit from a small future regression that checks the generated markdown/index entries for newly added workflow notebooks, but that was not required to resolve the current physics mismatch.

## Status

- Fixed in the same task.
- The spectroscopy workflow, docs assets, workflow notebooks, and topical docs now use a calibrated displacement plus selective Gaussian probe presentation that matches the intended physics.
- The Kerr workflow, docs assets, workflow notebooks, and topical docs now use alpha-coordinate Wigner plots for tutorial-facing figures and a cleaner figure layout.

## Fix Record

- Updated `tutorials/tutorial_support.py` with selective-spectroscopy helper functions for coherent-state photon weights and weak-drive Gaussian line-shape overlays.
- Reworked `examples/displacement_qubit_spectroscopy.py` to use calibrated displacement, a selective Gaussian probe, displaced-state/Fock-weight analysis, and docs-image export.
- Updated `examples/workflows/kerr_free_evolution.py` so Wigner snapshots default to `alpha` coordinates and the plotting helper uses a cleaner layout.
- Aligned `examples/kerr_free_evolution.py` with the tutorial narrative by using `alpha = 2.0`.
- Reworked `tools/generate_tutorial_plots.py` so the published displacement and Kerr images are regenerated from the corrected workflow paths.
- Updated `tutorials/_generate_workflow_tutorials.py` so the repaired workflows regenerate their notebooks and added two new companion notebooks:
  - `tutorials/10_core_workflows/03_phase_space_coordinates_and_wigner_conventions.ipynb`
  - `tutorials/10_core_workflows/04_selective_gaussian_number_splitting.ipynb`
- Updated tutorial landing/docs pages:
  - `tutorials/README.md`
  - `documentations/tutorials/index.md`
  - `documentations/tutorials/displacement_spectroscopy.md`
  - `documentations/tutorials/kerr_free_evolution.md`
- Added a regression guard in `examples/workflows/tests/test_kerr_free_evolution.py` that checks the tutorial-facing default Wigner coordinate and the expected `alpha = 2` peak location.