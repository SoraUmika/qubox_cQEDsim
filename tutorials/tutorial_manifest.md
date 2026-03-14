# Tutorial Manifest

This manifest records what each notebook is meant to teach and whether it uses the baseline public API or a more specialized public interface.

| Notebook | Tier | Focus | Notes |
|---|---|---|---|
| `00_tutorial_index.ipynb` | 0 | Orientation | Markdown-first landing notebook |
| `01_getting_started_minimal_dispersive_model.ipynb` | 0 | First model | Core model/frame/spectrum API |
| `02_units_frames_and_conventions.ipynb` | 0 | Units and signs | Core frame/frequency helpers |
| `03_cavity_displacement_basics.ipynb` | 1 | Displacement | Public displacement builder |
| `04_qubit_drive_and_basic_population_dynamics.ipynb` | 1 | Qubit drive | Direct pulse path |
| `05_observables_states_and_visualization.ipynb` | 1 | Observables | Extractors and Wigner utilities |
| `06_qubit_spectroscopy.ipynb` | 2 | Spectroscopy | Carrier mapping plus Lorentzian fit |
| `07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb` | 2 | Number splitting | Uses `manifold_transition_frequency(...)` |
| `08_dispersive_shift_and_dressed_frequencies.ipynb` | 2 | Dressed lines | Energy-spectrum helpers |
| `09_power_rabi.ipynb` | 3 | Power Rabi | Public pulse path plus notebook fit |
| `10_time_rabi.ipynb` | 3 | Time Rabi | Public pulse path plus notebook fit |
| `11_qubit_T1_relaxation.ipynb` | 3 | `T1` | `NoiseSpec` open-system example |
| `12_qubit_ramsey_T2star.ipynb` | 3 | Ramsey | Detuning plus dephasing example |
| `13_spin_echo_and_dephasing_mitigation.ipynb` | 3 | Echo | Pulse-level coherence comparison |
| `14_kerr_free_evolution.ipynb` | 4 | Self-Kerr | Bosonic nonlinear phase evolution |
| `15_cross_kerr_and_conditional_phase_accumulation.ipynb` | 4 | Cross-Kerr | Conditional phase accumulation |
| `16_storage_cavity_coherent_state_dynamics.ipynb` | 4 | Storage dynamics | Coherent-state moments and decay |
| `17_readout_resonator_response.ipynb` | 4 | Readout response | Uses `cqed_sim.measurement` readout-chain API |
| `18_multilevel_transmon_effects.ipynb` | 5 | Multilevel transmon | `n_tr > 2` public model path |
| `19_anharmonicity_and_leakage_under_strong_drive.ipynb` | 5 | Leakage | Three-level strong-drive example |
| `20_truncation_convergence_checks.ipynb` | 5 | Numerical care | Observable-level cutoff study |
| `21_building_sequences_from_gates_and_pulses.ipynb` | 6 | Composition | Gate objects, builders, and compiler |
| `22_parameter_sweeps_and_batch_simulation.ipynb` | 6 | Batch execution | `prepare_simulation(...)` and `simulate_batch(...)` |
| `23_analysis_fitting_and_result_extraction.ipynb` | 6 | Analysis | Public `calibration_targets` convenience layer |
| `24_sideband_like_interactions.ipynb` | 7 | Sideband-like dynamics | Effective sideband interface, not microscopic coupler physics |
| `25_small_calibration_workflow_end_to_end.ipynb` | 7 | End-to-end workflow | Compact notebook-scale calibration summary |
| `26_frame_sanity_checks_and_common_failure_modes.ipynb` | 7 | Debugging | Frame/carrier/truncation sanity checks |

## Notes

- No notebook in this curriculum invents a private or nonexistent API.
- Tutorials `23`, `24`, and `25` are intentionally honest about the abstraction level they use.
- If a future notebook needs functionality the current public API does not support cleanly, it should be listed here as a planned addition rather than simulated with an ad hoc interface.
