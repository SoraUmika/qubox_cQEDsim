# Tutorials

This directory now has two complementary notebook tracks:

- A notebook-first workflow suite under categorized folders such as `00_getting_started/`, `10_core_workflows/`, and `20_bosonic_and_sideband/`. These notebooks are the polished replacements for the most representative script-style examples in `examples/`.
- The earlier flat numbered curriculum (`00_tutorial_index.ipynb` through `26_frame_sanity_checks_and_common_failure_modes.ipynb`), which remains useful as a broader API and conventions primer.

## Start Here

If you want the real-world workflow path first, use this order:

1. `00_getting_started/01_protocol_style_simulation.ipynb`
2. `10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`
3. `10_core_workflows/02_kerr_free_evolution.ipynb`
4. `20_bosonic_and_sideband/01_sideband_swap.ipynb`
5. `20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb`
6. `30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`
7. `30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`
8. `40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`

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
  - Coherent displacement followed by qubit spectroscopy and number splitting.
- `02_kerr_free_evolution.ipynb`
  - Storage self-Kerr phase winding and Wigner-function distortion.

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
- `04_snap_optimization_workflow.ipynb`
  - Repo-side SNAP optimization study helper with an honest scope note.
- `05_rl_hybrid_control_environment.ipynb`
  - Hybrid bosonic-ancilla RL environment construction, measurement-like observations, diagnostics, and domain-randomized evaluation.

### `40_validation_and_conventions`

- `01_kerr_sign_and_frame_checks.ipynb`
  - Convention-validation notebook for Kerr sign and frame interpretation.

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
- `Pulse.carrier` follows the repository waveform sign convention

Use `physics_and_conventions/physics_conventions_report.tex` as the source of truth when a notebook touches sign, frame, or Hamiltonian interpretation.
