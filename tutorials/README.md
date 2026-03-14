# Tutorials

This directory is the primary guided-learning path for `cqed_sim`. The notebooks here are ordered from orientation and first simulations through spectroscopy, calibration-style workflows, bosonic dynamics, numerical-care topics, and advanced research-style examples.

Start with `00_tutorial_index.ipynb`, then follow the numbered notebooks in order unless you already know you want a narrower path.

## Recommended Order

### Tier 0 - Orientation

- `00_tutorial_index.ipynb` - landing page, learning paths, and notebook navigation
- `01_getting_started_minimal_dispersive_model.ipynb` - first dressed-spectrum and transition-frequency walkthrough
- `02_units_frames_and_conventions.ipynb` - units, frames, carrier sign, and dispersive-sign sanity checks

### Tier 1 - Basic Operations and Observables

- `03_cavity_displacement_basics.ipynb` - storage displacement and Wigner-function intuition
- `04_qubit_drive_and_basic_population_dynamics.ipynb` - resonant qubit driving and population transfer
- `05_observables_states_and_visualization.ipynb` - reduced states, conditioned observables, and visualization helpers

### Tier 2 - Canonical cQED Signatures

- `06_qubit_spectroscopy.ipynb` - bare qubit spectroscopy in the rotating-frame convention used by `cqed_sim`
- `07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb` - number splitting and manifold-resolved qubit lines
- `08_dispersive_shift_and_dressed_frequencies.ipynb` - dressed frequencies and dispersive peak interpretation

### Tier 3 - Calibration-Style Workflows

- `09_power_rabi.ipynb` - amplitude sweep and fitted pi-pulse calibration
- `10_time_rabi.ipynb` - duration sweep and fitted Rabi-rate extraction
- `11_qubit_T1_relaxation.ipynb` - open-system relaxation and exponential fitting
- `12_qubit_ramsey_T2star.ipynb` - Ramsey fringes, detuning, and `T2*`
- `13_spin_echo_and_dephasing_mitigation.ipynb` - echo-style coherence recovery

### Tier 4 - Bosonic / Cavity Physics

- `14_kerr_free_evolution.ipynb` - self-Kerr phase winding and non-Gaussian distortion
- `15_cross_kerr_and_conditional_phase_accumulation.ipynb` - conditional phase from cross-Kerr coupling
- `16_storage_cavity_coherent_state_dynamics.ipynb` - coherent-state evolution and decay
- `17_readout_resonator_response.ipynb` - readout-chain response and synthetic I/Q-style signals

### Tier 5 - Realism and Numerical Care

- `18_multilevel_transmon_effects.ipynb` - moving beyond the two-level transmon approximation
- `19_anharmonicity_and_leakage_under_strong_drive.ipynb` - leakage under strong resonant drive
- `20_truncation_convergence_checks.ipynb` - cavity cutoff convergence checks tied to observables

### Tier 6 - Workflow and Composition

- `21_building_sequences_from_gates_and_pulses.ipynb` - public gate objects, pulse builders, and schedule compilation
- `22_parameter_sweeps_and_batch_simulation.ipynb` - prepared sessions and batched execution
- `23_analysis_fitting_and_result_extraction.ipynb` - fitted calibration summaries from public convenience helpers

### Tier 7 - Advanced / Research-Style Topics

- `24_sideband_like_interactions.ipynb` - effective sideband driving through `SidebandDriveSpec`
- `25_small_calibration_workflow_end_to_end.ipynb` - compact spectroscopy/Rabi/relaxation/coherence workflow
- `26_frame_sanity_checks_and_common_failure_modes.ipynb` - frame, carrier, and truncation debugging checklist

## Prerequisites and Optional Functionality

- Tutorials `01` through `08` only assume the baseline model, frame, pulse, and simulation APIs.
- Tutorials `09` through `13` use the same low-level pulse path plus simple fitting helpers from `tutorials/tutorial_support.py`.
- Tutorial `17` depends on the public readout-chain API in `cqed_sim.measurement`.
- Tutorials `18` through `20` assume you are comfortable with multilevel transmons and bosonic truncation.
- Tutorials `23` and `25` use the public `cqed_sim.calibration_targets` convenience layer. They are intentionally lighter-weight than a full lab automation stack.
- Tutorial `24` uses the current effective sideband interface (`SidebandDriveSpec` and `build_sideband_pulse(...)`), not a microscopic coupler model.

## Conventions

These notebooks follow the repository's documented conventions:

- internal frequencies are in `rad/s`
- times are in `s`
- `Pulse.carrier` follows the repository waveform sign convention
- dispersive `chi` is the per-photon qubit-transition shift
- Kerr and truncation choices are interpreted exactly as documented by the runtime model

Read [`conventions_quick_reference.md`](conventions_quick_reference.md) for a short checklist and `physics_and_conventions/physics_conventions_report.tex` for the detailed source of truth.

## Tutorials vs Tests

`tutorials/` is for guided learning, explanation, and reproducible walkthroughs. `tests/` is for automated correctness, regression, and validation coverage. If a notebook demonstrates a phenomenon and a test verifies that same convention, the notebook should teach it and the test should guard it; they should not replace each other.

For a compact machine-readable curriculum summary, see [`tutorial_manifest.md`](tutorial_manifest.md).
