Status: partially fixed on 2026-03-11.

## Confirmed issues

### 1. Several active notebooks are stored at the repository root even though they are examples, studies, or literature checks

- Where:
  - `usage_examples.ipynb`
  - `sequential_simulation.ipynb`
  - `SQR_calibration.ipynb`
  - `SQR_optimization_demo.ipynb`
  - `SQR_speedlimit_multitone_gaussian.ipynb`
  - `SQR_three_gate_optimization.ipynb`
  - `one_tone_sqr_xyz_demo.ipynb`
  - `landgraf_reproduction.ipynb`
  - `chi_evolution_copy.ipynb`
- What:
  - These notebooks are not generic repo-root assets. They are specific user workflows, studies, audits, or literature reproductions.
- Why inconsistent:
  - The current project structure and README reserve purpose-specific material for `examples/`, `examples/audits`, `examples/studies`, `examples/workflows`, or `test_against_papers`.
- Consequence:
  - The repo root obscures purpose, makes ownership unclear, and encourages stale references to files that should live in purpose-matched folders.

### 2. Some notebook names no longer match their actual role

- Where:
  - `one_tone_sqr_xyz_demo.ipynb`
  - `chi_evolution_copy.ipynb`
  - `test_against_papers/bosonic_controls.ipynb`
- What:
  - The names are vague, legacy-like, or omit the actual paper/workflow/audit role.
- Why inconsistent:
  - Current standards call for names and locations that reflect whether a file is a workflow, study, audit, or paper check.
- Consequence:
  - Readers cannot infer purpose from the filename, and downstream references stay coupled to legacy naming.

### 3. Several root-level example scripts are stored outside the subfolders that already exist for their purpose

- Where:
  - `examples/fock_tomo_workflow.py`
  - `examples/simulate_fock_tomo_and_sqr_calibration.py`
  - `examples/sqr_convention_metric_audit.py`
  - `examples/sqr_block_phase_study.py`
  - `examples/sqr_block_phase_followup.py`
  - `examples/sqr_multitone_study.py`
  - `examples/sqr_route_b_enlarged_control.py`
  - `examples/make_figures_like_paper.py`
  - `examples/sanity_run.py`
  - `examples/unitary_synthesis_cluster_optimization.ipynb`
- What:
  - These files clearly act as workflows, audits, studies, smoke checks, or study notebooks but sit at `examples/` top level.
- Why inconsistent:
  - The repo already has purpose-specific directories under `examples/`.
- Consequence:
  - The top-level examples directory becomes a mixed catch-all rather than a clean public-demo layer.

### 4. Generator scripts and test metadata still point at legacy notebook locations

- Where:
  - `outputs/generate_sequential_simulation_notebook.py`
  - `outputs/generate_sqr_calibration_notebook.py`
  - `outputs/repair_chi_notebook.py`
  - `examples/workflows/tests/test_sqr_transfer_artifact.py`
  - notebook markdown that references other notebooks by old root paths
- What:
  - These files embed old destinations and names.
- Why inconsistent:
  - If notebooks are reorganized without updating these references, the repo will drift back to the old layout or report incorrect provenance.
- Consequence:
  - Broken links, stale metadata, and regeneration to the wrong location.

## Suspected / unresolved issues

- `outputs/chi_evolution_copy_HEAD.ipynb` appears to be an archival snapshot rather than a live workflow notebook. It should either be moved to a clearly labeled archive location or explicitly kept as an artifact with that status documented.
- The `examples/paper_reproductions/` subtree is clearly named, but it partially overlaps in purpose with `test_against_papers/`. This audit will prioritize the obvious root-level notebook misplacements first and only change that subtree if a minimal move is justified by the surrounding references.

## Resolution status

- Fixed:
  - root-level notebooks were relocated into `examples/workflows`, `examples/studies`, `examples/audits`, or `test_against_papers`
  - vague notebook names were replaced with purpose-specific names
  - root-level example scripts were moved into `examples/workflows`, `examples/studies`, `examples/audits`, or `examples/smoke_tests`
  - generator scripts, provenance strings, import paths, and notebook references were updated to the new locations
- Remaining open:
  - the purpose boundary between `examples/paper_reproductions/` and `test_against_papers/` is still broader than this minimal reorganization pass and should be normalized only with a dedicated migration plan
