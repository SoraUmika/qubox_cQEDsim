Status: fixed on 2026-03-13.

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

## Follow-up items re-audited on 2026-03-13

- `outputs/chi_evolution_copy_HEAD.ipynb`
  - This remains an output-side artifact rather than a maintained workflow notebook, so it is no longer treated as an active layout inconsistency.
- `examples/paper_reproductions/` vs `test_against_papers/`
  - The repository now documents this boundary explicitly:
    - `examples/paper_reproductions/` for maintained example-side reproduction code
    - `test_against_papers/` for notebook-style literature checks and validation workflows
  - With that documented separation in `README.md` and `documentations/examples.md`, this is no longer tracked as an unresolved inconsistency.

## Resolution status

- Fixed:
  - root-level notebooks were relocated into `examples/workflows`, `examples/studies`, `examples/audits`, or `test_against_papers`
  - vague notebook names were replaced with purpose-specific names
  - root-level example scripts were moved into `examples/workflows`, `examples/studies`, `examples/audits`, or `examples/smoke_tests`
  - generator scripts, provenance strings, import paths, and notebook references were updated to the new locations
  - maintained example scripts and workflow notebooks now resolve outputs from canonical `examples/outputs*` locations instead of depending on the current working directory
- Remaining open:
  - none from this audit after the 2026-03-13 re-audit
