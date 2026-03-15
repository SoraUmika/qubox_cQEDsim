# Tutorial Migration Plan

Created: 2026-03-15  
Scope: audit the current `examples/` Python scripts, classify them, record runnable status, and map the representative workflow scripts into the new notebook-first tutorial taxonomy under `tutorials/`.

## Summary

- Converted to tutorial notebooks in this pass:
  - `examples/protocol_style_simulation.py`
  - `examples/displacement_qubit_spectroscopy.py`
  - `examples/kerr_free_evolution.py`
  - `examples/sideband_swap_demo.py`
  - `examples/detuned_sideband_sync_demo.py`
  - `examples/sequential_sideband_reset.py`
  - `examples/shelving_isolation_demo.py`
  - `examples/multimode_crosskerr_demo.py`
  - `examples/open_system_sideband_degradation.py`
  - `examples/unitary_synthesis_demo.py`
  - `examples/run_snap_optimization_demo.py`
  - `examples/kerr_sign_verification.py`
- Retained as scripts on purpose:
  - `examples/ringdown_noise.py` as a compact decay/noise script
  - `examples/sideband_swap.py` as a compatibility wrapper around `sideband_swap_demo.py`
- Not migrated into user-facing tutorials:
  - audit scripts in `examples/audits/`
  - study scripts in `examples/studies/`
  - paper-reproduction modules in `examples/paper_reproductions/`
  - workflow helper modules in `examples/workflows/`

## Top-Level Standalone Scripts

| Script | Runnable status | Classification | Proposed destination | Notes |
|---|---|---|---|---|
| `examples/protocol_style_simulation.py` | Executed successfully on 2026-03-15 | User-facing / getting started | `tutorials/00_getting_started/01_protocol_style_simulation.ipynb` | Good first end-to-end API walkthrough. |
| `examples/displacement_qubit_spectroscopy.py` | Executed successfully on 2026-03-15 | User-facing core workflow | `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb` | Best practical displacement + spectroscopy example. |
| `examples/kerr_free_evolution.py` | Executed successfully on 2026-03-15 | User-facing core workflow | `tutorials/10_core_workflows/02_kerr_free_evolution.ipynb` | Built on the reusable workflow helper in `examples/workflows/kerr_free_evolution.py`. |
| `examples/kerr_sign_verification.py` | Executed successfully on 2026-03-15 | Validation / convention-oriented | `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb` | Better as a validation notebook than as a beginner example. |
| `examples/multimode_crosskerr_demo.py` | Executed successfully on 2026-03-15 | Advanced protocol study | `tutorials/30_advanced_protocols/01_multimode_crosskerr.ipynb` | Clear three-mode conditional-phase example. |
| `examples/open_system_sideband_degradation.py` | Executed successfully on 2026-03-15 | Advanced protocol study | `tutorials/30_advanced_protocols/02_open_system_sideband_degradation.ipynb` | Strong closed-vs-open comparison notebook candidate. |
| `examples/ringdown_noise.py` | Executed successfully on 2026-03-15 | Compact diagnostic script | Retain as script | Small enough to remain script-first; concept already overlaps with existing decay/dynamics tutorials. |
| `examples/run_snap_optimization_demo.py` | Fixed and executed successfully on 2026-03-15 | Advanced study workflow | `tutorials/30_advanced_protocols/04_snap_optimization_workflow.ipynb` | Was broken from repo root before this migration because it did not add the repo root to `sys.path`. |
| `examples/sequential_sideband_reset.py` | Executed successfully on 2026-03-15 | User-facing advanced workflow | `tutorials/20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb` | Uses the canonical helper module under `examples/workflows/`. |
| `examples/shelving_isolation_demo.py` | Executed successfully on 2026-03-15 | User-facing sideband workflow | `tutorials/20_bosonic_and_sideband/04_shelving_isolation.ipynb` | Good multilevel selectivity example. |
| `examples/sideband_swap_demo.py` | Executed successfully on 2026-03-15 | User-facing sideband workflow | `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb` | Strong canonical bosonic-transfer demo. |
| `examples/sideband_swap.py` | Executed successfully on 2026-03-15 | Compatibility wrapper | Retain as script wrapper | Thin alias to `sideband_swap_demo.py`; keep for compatibility, but notebook documentation points to the richer tutorial. |
| `examples/detuned_sideband_sync_demo.py` | Executed successfully on 2026-03-15 | User-facing sideband study | `tutorials/20_bosonic_and_sideband/02_detuned_sideband_synchronization.ipynb` | Benefits from notebook explanation more than a plain script. |
| `examples/unitary_synthesis_demo.py` | Executed successfully on 2026-03-15 | Advanced workflow | `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb` | Advanced but still user-facing enough for a guided notebook. |

## Workflow Helpers

| Script | Runnable status | Classification | Proposed destination | Notes |
|---|---|---|---|---|
| `examples/workflows/kerr_free_evolution.py` | Module helper, exercised indirectly by scripts/tests/notebooks | Workflow helper | Retain in `examples/workflows/` | Canonical helper backing both the script and the tutorial notebook. |
| `examples/workflows/sequential_sideband_reset.py` | Module helper, exercised indirectly by scripts/tests/notebooks | Workflow helper | Retain in `examples/workflows/` | Canonical helper backing the new sequential-reset notebook. |
| `examples/workflows/fock_tomo_workflow.py` | Standalone helper script with `main()`; not re-run in this pass | Workflow helper | Retain in `examples/workflows/` | Outside the scope of the current tutorial suite conversion. |
| `examples/workflows/simulate_fock_tomo_and_sqr_calibration.py` | Standalone helper script; not re-run in this pass | Workflow helper | Retain in `examples/workflows/` | Better treated as workflow glue than as a user-facing notebook tutorial. |
| `examples/workflows/sqr_transfer.py` | Module / workflow helper | Workflow helper | Retain in `examples/workflows/` | Specialized study artifact. |
| `examples/workflows/universal_cqed_model_demo.py` | Executed successfully on 2026-03-15 | Workflow helper / demo | Retain in `examples/workflows/` | Existing foundational tutorials already cover the public model API. |
| `examples/workflows/sequential/common.py` | Module helper | Workflow helper | Retain in `examples/workflows/sequential/` | Internal helper, not a tutorial target. |
| `examples/workflows/sequential/ideal.py` | Module helper | Workflow helper | Retain in `examples/workflows/sequential/` | Internal helper, not a tutorial target. |
| `examples/workflows/sequential/pulse_calibrated.py` | Module helper | Workflow helper | Retain in `examples/workflows/sequential/` | Internal helper, not a tutorial target. |
| `examples/workflows/sequential/pulse_open.py` | Module helper | Workflow helper | Retain in `examples/workflows/sequential/` | Internal helper, not a tutorial target. |
| `examples/workflows/sequential/pulse_unitary.py` | Module helper | Workflow helper | Retain in `examples/workflows/sequential/` | Internal helper, not a tutorial target. |
| `examples/workflows/sequential/trajectories.py` | Module helper | Workflow helper | Retain in `examples/workflows/sequential/` | Internal helper, not a tutorial target. |

## Audits

| Script | Runnable status | Classification | Proposed destination | Notes |
|---|---|---|---|---|
| `examples/audits/experiment_convention_audit.py` | Standalone audit script; not re-run in this pass | Validation / audit | Retain in `examples/audits/` | Not a user-facing tutorial. |
| `examples/audits/experiment_convention_audit_notebook.py` | Script-style notebook generator/helper | Validation / audit | Retain in `examples/audits/` | Repo-maintenance artifact, not a user tutorial. |
| `examples/audits/repair_chi_phase_evolution_notebook.py` | Standalone repair helper; not re-run in this pass | Validation / audit | Retain in `examples/audits/` | Repair tool, not tutorial content. |
| `examples/audits/sqr_convention_metric_audit.py` | Standalone audit script; not re-run in this pass | Validation / audit | Retain in `examples/audits/` | Specialized convention audit. |

## Studies

| Script | Runnable status | Classification | Proposed destination | Notes |
|---|---|---|---|---|
| `examples/studies/make_snap_opt_figures.py` | Standalone study script; not re-run in this pass | Study / figure-generation | Retain in `examples/studies/` | Output-generation helper, not tutorial content. |
| `examples/studies/sqr_block_phase_followup.py` | Standalone study script; not re-run in this pass | Study | Retain in `examples/studies/` | Too specialized for the new workflow suite. |
| `examples/studies/sqr_block_phase_study.py` | Standalone study script; not re-run in this pass | Study | Retain in `examples/studies/` | Too specialized for the new workflow suite. |
| `examples/studies/sqr_multitone_study.py` | Standalone study script; not re-run in this pass | Study | Retain in `examples/studies/` | Too specialized for the new workflow suite. |
| `examples/studies/sqr_route_b_enlarged_control.py` | Standalone study script; not re-run in this pass | Study | Retain in `examples/studies/` | Too specialized for the new workflow suite. |
| `examples/studies/sqr_speedlimit_multitone_gaussian.py` | Standalone study script; not re-run in this pass | Study | Retain in `examples/studies/` | Too specialized for the new workflow suite. |
| `examples/studies/snap_opt/experiments.py` | Module helper | Study helper | Retain in `examples/studies/snap_opt/` | Backing module for the new SNAP notebook. |
| `examples/studies/snap_opt/metrics.py` | Module helper | Study helper | Retain in `examples/studies/snap_opt/` | Backing module for the new SNAP notebook. |
| `examples/studies/snap_opt/model.py` | Module helper | Study helper | Retain in `examples/studies/snap_opt/` | Backing module for the new SNAP notebook. |
| `examples/studies/snap_opt/optimizer.py` | Module helper | Study helper | Retain in `examples/studies/snap_opt/` | Backing module for the new SNAP notebook. |
| `examples/studies/snap_opt/pulses.py` | Module helper | Study helper | Retain in `examples/studies/snap_opt/` | Backing module for the new SNAP notebook. |

## Paper Reproductions

| Script | Runnable status | Classification | Proposed destination | Notes |
|---|---|---|---|---|
| `examples/paper_reproductions/snap_prl133/errors.py` | Module helper | Paper reproduction | Retain in `examples/paper_reproductions/snap_prl133/` | Literature-specific code, not tutorial content. |
| `examples/paper_reproductions/snap_prl133/model.py` | Module helper | Paper reproduction | Retain in `examples/paper_reproductions/snap_prl133/` | Literature-specific code, not tutorial content. |
| `examples/paper_reproductions/snap_prl133/optimize.py` | Module helper | Paper reproduction | Retain in `examples/paper_reproductions/snap_prl133/` | Literature-specific code, not tutorial content. |
| `examples/paper_reproductions/snap_prl133/pulses.py` | Module helper | Paper reproduction | Retain in `examples/paper_reproductions/snap_prl133/` | Literature-specific code, not tutorial content. |
| `examples/paper_reproductions/snap_prl133/reproduce.py` | Standalone reproduction script with `main()`; not re-run in this pass | Paper reproduction | Retain in `examples/paper_reproductions/snap_prl133/` and `test_against_papers/` | Should stay literature-facing, not user-facing tutorial content. |

## Smoke / Utility Scripts

| Script | Runnable status | Classification | Proposed destination | Notes |
|---|---|---|---|---|
| `examples/smoke_tests/sanity_run.py` | Executed successfully on 2026-03-15 | Smoke / integration check | Retain in `examples/smoke_tests/` | Useful for quick repo sanity, not a teaching notebook. |

## API and Convention Issues Found During Migration

### Confirmed and fixed

- `examples/run_snap_optimization_demo.py` failed from the repository root with `ModuleNotFoundError: No module named 'examples.studies'`.
  - Fix applied: added repo-root insertion to `sys.path` consistent with the other top-level example scripts.

### Confirmed and handled by classification

- `examples/kerr_sign_verification.py` is not really a beginner usage example.
  - Resolution: migrated into `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`.
- `examples/sideband_swap.py` duplicates `examples/sideband_swap_demo.py` as a wrapper.
  - Resolution: kept as a compatibility wrapper, but the notebook migration points users to the richer sideband tutorial instead.

### Retained intentionally

- `examples/ringdown_noise.py` was not converted in this pass.
  - Reason: it is a compact decay diagnostic and already overlaps with the repository's broader decay / cavity-dynamics coverage.

## Validation Performed

- Executed the converted standalone scripts listed above with the current machine Python on 2026-03-15.
- Generated the new notebooks from `tutorials/_generate_workflow_tutorials.py`.
- Executed all 12 new notebooks top-to-bottom with a plain-Python JSON-cell harness because `nbformat` / `nbclient` are not installed in the current environment.
