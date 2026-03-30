# Tutorials

This directory now has two complementary notebook tracks:

- A notebook-first workflow suite under categorized folders such as `00_getting_started/`, `10_core_workflows/`, and `20_bosonic_and_sideband/`. These notebooks are the polished replacements for the most representative script-style examples in `examples/`.
- The earlier flat numbered curriculum (`00_tutorial_index.ipynb` through `26_frame_sanity_checks_and_common_failure_modes.ipynb`), which remains useful as a broader API and conventions primer.

## Start Here

If you want the real-world workflow path first, use this order:

1. `00_getting_started/01_protocol_style_simulation.ipynb`
2. `10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`
3. `10_core_workflows/02_kerr_free_evolution.ipynb`
4. `10_core_workflows/03_phase_space_coordinates_and_wigner_conventions.ipynb`
5. `10_core_workflows/04_selective_gaussian_number_splitting.ipynb`
6. `20_bosonic_and_sideband/01_sideband_swap.ipynb`
7. `20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb`
8. `30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`
9. `30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
10. `30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`
11. `31_system_identification_and_domain_randomization/01_calibration_targets_and_fitting.ipynb`
12. `31_system_identification_and_domain_randomization/02_evidence_to_randomizer_and_env.ipynb`
13. `50_floquet_driven_systems/01_sideband_quasienergy_scan.ipynb`
14. `50_floquet_driven_systems/02_branch_tracking_and_multiphoton_resonances.ipynb`
15. `40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`

If you want the older foundations-first path, start with:

1. `00_tutorial_index.ipynb`
2. `01_getting_started_minimal_dispersive_model.ipynb`
3. `02_units_frames_and_conventions.ipynb`
4. `06_qubit_spectroscopy.ipynb`

## Workflow Tutorial Taxonomy

### `00_getting_started`

- `01_protocol_style_simulation.ipynb`
  - Direct model -> frame -> pulse -> compile -> simulate -> measure example.

### `10_core_workflows`

- `01_displacement_then_qubit_spectroscopy.ipynb`
  - Calibrated displacement followed by selective Gaussian qubit spectroscopy and resolved number splitting.
- `02_kerr_free_evolution.ipynb`
  - Storage self-Kerr phase winding and alpha-coordinate Wigner-function distortion.
- `03_phase_space_coordinates_and_wigner_conventions.ipynb`
  - Compare quadrature and alpha-coordinate Wigner plots for the same coherent state.
- `04_selective_gaussian_number_splitting.ipynb`
  - Diagnose when resolved spectroscopy peak heights recover cavity photon-number weights.

### `20_bosonic_and_sideband`

- `01_sideband_swap.ipynb`
  - Basic `|f,0> <-> |g,1>` red-sideband transfer.
- `02_detuned_sideband_synchronization.ipynb`
  - Branch synchronization with a detuned sideband drive.
- `03_sequential_sideband_reset.ipynb`
  - Three-mode storage reset through storage/readout sidebands and ringdown.
- `04_shelving_isolation.ipynb`
  - Shelving one transmon branch while transferring another branch into the cavity.

### `30_advanced_protocols`

- `01_multimode_crosskerr.ipynb`
  - Storage-readout cross-Kerr phase accumulation in a three-mode model.
- `02_open_system_sideband_degradation.ipynb`
  - Closed- versus open-system sideband transfer.
- `03_unitary_synthesis_workflow.ipynb`
  - Target-unitary synthesis inside a qubit-cavity subspace.
- `06_grape_optimal_control_workflow.ipynb`
  - Model-backed GRAPE optimization with direct piecewise-constant controls, nominal/noisy replay, and the benchmark harness entry point.
- `04_snap_optimization_workflow.ipynb`
  - Repo-side SNAP optimization study helper with an honest scope note.
- `05_rl_hybrid_control_environment.ipynb`
  - Hybrid bosonic-ancilla RL environment construction, measurement-like observations, diagnostics, and domain-randomized evaluation.

### `31_system_identification_and_domain_randomization`

- `01_calibration_targets_and_fitting.ipynb`
  - Generate spectroscopy, Rabi, and T1 calibration targets and inspect the fitted parameter summaries they produce.
- `02_evidence_to_randomizer_and_env.ipynb`
  - Convert calibration evidence into domain-randomization priors and wire them into a hybrid control environment.

### `40_validation_and_conventions`

- `01_kerr_sign_and_frame_checks.ipynb`
  - Convention-validation notebook for Kerr sign and frame interpretation.

### `50_floquet_driven_systems`

- `01_sideband_quasienergy_scan.ipynb`
  - Sweep a periodic sideband drive and track the resulting quasienergy branches across detuning.
- `02_branch_tracking_and_multiphoton_resonances.ipynb`
  - Relate dressed Floquet structure to integer-order multiphoton resonance conditions.

## Migration Notes

- The workflow notebooks above are the primary replacements for the representative top-level example scripts under `examples/`.
- The flat numbered curriculum remains because it still teaches broader API surface area and convention background.
- `tutorials/TUTORIAL_MIGRATION_PLAN.md` records the audit, script classification, runnable status, destination decisions, and migration issues found during this conversion.

## Conventions

These notebooks follow the repository's documented conventions:

- internal frequencies are in `rad/s`
- times are in `s`
- tensor ordering is qubit first, then bosonic modes
- `chi` is the per-photon qubit-transition shift
- user-facing tone frequencies should use the positive drive-frequency helpers, while raw `Pulse.carrier` remains a low-level compatibility field

Use `physics_and_conventions/physics_conventions_report.tex` as the source of truth when a notebook touches sign, frame, or Hamiltonian interpretation.
