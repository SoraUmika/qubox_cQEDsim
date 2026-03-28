# Tutorial Physics Audit — 2026-03-27

## Summary

This audit focused on whether the numbered `tutorials/` curriculum presents simulation as a physics-verification workflow rather than as plot generation.

The highest-risk tutorials for convention drift and simulation-theory mismatch were the foundational drive/coherence/Kerr notebooks. Two genuine tutorial-level physics issues were confirmed and fixed:

1. Tutorial 04 used the wrong square-pulse `pi`-time convention for the repo's drive normalization.
2. Tutorial 13 claimed Hahn-echo refocusing of static detuning while actually simulating only Markovian dephasing and reading out a mismatched final basis.

In the same pass, explicit theory overlays or quantitative comparison checks were added to the surrounding calibration and phase-sensitive notebooks so the key physics claims are now stated in equations and compared numerically where the theory is simple enough to be exact.

## Confirmed Issues Fixed

### Tutorial 04 — resonant square-pulse convention

- Old state:
  - described `t_pi = pi / Omega`
  - simulated a pulse with that duration
- Correct state:
  - repo convention is `P_e(t) = sin^2(Omega t)`
  - corrected notebook now uses `t_pi = pi / (2 Omega)`
  - added direct simulation-versus-theory overlay

### Tutorial 13 — Hahn echo physics target

- Old state:
  - modeled only Lindblad `tphi`
  - used a final `x90` readout that made the echo population trace complementary to the Ramsey-like trace
  - claimed mitigation of static detuning without actually simulating static detuning disorder
- Correct state:
  - models quasi-static detuning as an ensemble of frame offsets
  - uses a final `-x90` readout for a like-for-like refocused population signal
  - compares the ensemble-averaged pulse-level traces to the closed-form Gaussian Ramsey envelope and ideal echo plateau

## Theory-Verification Upgrades Added

### Explicit simulation-versus-theory overlays

- Tutorial 04: resonant square-pulse `sin^2(Omega t)`
- Tutorial 09: power Rabi `sin^2(Omega T)`
- Tutorial 10: time Rabi `sin^2(Omega t)`
- Tutorial 11: `T1` exponential `exp(-t / T1)`
- Tutorial 12: Ramsey fringe `0.5 (1 + exp(-t/T2*) cos(Delta t))`
- Tutorial 13: quasi-static detuning Ramsey and Hahn-echo ensemble theory
- Tutorial 14: conserved photon number `⟨n(t)⟩ = |alpha|^2`
- Tutorial 15: conditional phase `phi_cond(t) = -chi_sr t`

### Existing comparison notebooks kept as validated anchors

- Tutorial 07: dispersive line positions via `manifold_transition_frequency(...)`
- Tutorial 08: dressed-spectrum versus transition-helper consistency
- Tutorial 20: truncation convergence against the known coherent-state photon number target
- Tutorial 26: carrier-sign and frame bookkeeping sanity checks

## Curriculum Classification

### Physics-verification notebooks after this pass

- `04_qubit_drive_and_basic_population_dynamics.ipynb`
- `06_qubit_spectroscopy.ipynb`
- `07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`
- `08_dispersive_shift_and_dressed_frequencies.ipynb`
- `09_power_rabi.ipynb`
- `10_time_rabi.ipynb`
- `11_qubit_T1_relaxation.ipynb`
- `12_qubit_ramsey_T2star.ipynb`
- `13_spin_echo_and_dephasing_mitigation.ipynb`
- `14_kerr_free_evolution.ipynb`
- `15_cross_kerr_and_conditional_phase_accumulation.ipynb`
- `20_truncation_convergence_checks.ipynb`
- `26_frame_sanity_checks_and_common_failure_modes.ipynb`

### API/concept notebooks that remain primarily instructional rather than closed-form validation notebooks

- `00_tutorial_index.ipynb`
- `01_getting_started_minimal_dispersive_model.ipynb`
- `02_units_frames_and_conventions.ipynb`
- `03_cavity_displacement_basics.ipynb`
- `05_observables_states_and_visualization.ipynb`
- `16_storage_cavity_coherent_state_dynamics.ipynb`
- `17_readout_resonator_response.ipynb`
- `18_multilevel_transmon_effects.ipynb`
- `19_anharmonicity_and_leakage_under_strong_drive.ipynb`
- `21_building_sequences_from_gates_and_pulses.ipynb`
- `22_parameter_sweeps_and_batch_simulation.ipynb`
- `23_analysis_fitting_and_result_extraction.ipynb`
- `24_sideband_like_interactions.ipynb`
- `25_small_calibration_workflow_end_to_end.ipynb`

These notebooks were still checked for obvious sign/frame misuse during the audit, but they are not all intended to be closed-form benchmark notebooks.

## Validation Artifacts Added

- Shared analytical helpers in `tutorials/tutorial_support.py`
- Focused regression coverage in `tests/test_56_tutorial_physics_validation.py`
- Inconsistency report in `inconsistency/20260327_173500_tutorial_physics_verification_audit.md`

## Practical Outcome

The tutorial layer now states the repo's drive, Ramsey, echo, Kerr, and cross-Kerr conventions more explicitly where users are most likely to build physical intuition from plots. The repaired notebooks no longer rely on ambiguous normalization, hidden sign assumptions, or a mismatched echo model to make their main claims.