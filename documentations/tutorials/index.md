# Tutorial Curriculum

The `cqed_sim` tutorial library contains **47 Jupyter notebooks** organized into two complementary tracks:

1. **Workflow tutorials** — categorized by topic, starting from a first simulation and building toward optimal control and RL
2. **Foundational curriculum** — numbered 01–26, covering individual API features and physics concepts in depth

Both tracks can be used independently. Start with the workflow tutorials for a guided learning path, and dip into the foundational curriculum for deeper treatments of specific topics.

!!! tip "Recommended starting point"
    If you are new to `cqed_sim`, begin with the [Quickstart](../quickstart.md) page, then open
    `tutorials/00_getting_started/01_protocol_style_simulation.ipynb`.

---

## Learning Path

```
Getting Started → Core Workflows → Bosonic & Sideband → Advanced Protocols → System ID & Randomization → Floquet → Validation
```

Each stage builds on the previous one. Within each stage, notebooks are intended to be read in order.

---

## Workflow Tutorials

### 00 — Getting Started

| Notebook | Description |
|---|---|
| `00_getting_started/01_protocol_style_simulation.ipynb` | End-to-end simulation: model → pulses → compile → simulate → measure |

### 10 — Core Workflows

| Notebook | Description |
|---|---|
| `10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb` | Calibrated displacement plus selective Gaussian qubit spectroscopy with resolved photon-number peaks |
| `10_core_workflows/02_kerr_free_evolution.ipynb` | Coherent-state evolution under self-Kerr in alpha-coordinate Wigner space |
| `10_core_workflows/03_phase_space_coordinates_and_wigner_conventions.ipynb` | Compare quadrature and alpha-coordinate Wigner plots and verify the expected sqrt(2) rescaling |
| `10_core_workflows/04_selective_gaussian_number_splitting.ipynb` | Interpret resolved peak heights as cavity photon-number weights and compare against coherent-state theory |

### 20 — Bosonic & Sideband

| Notebook | Description |
|---|---|
| `20_bosonic_and_sideband/01_sideband_swap.ipynb` | Red-sideband swap between transmon and storage cavity |
| `20_bosonic_and_sideband/02_detuned_sideband_synchronization.ipynb` | Off-resonance sideband dynamics and synchronization conditions |
| `20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb` | Multi-step reset protocol using sequential sideband pulses |
| `20_bosonic_and_sideband/04_shelving_isolation.ipynb` | Shelving isolation with multilevel sideband transitions |

### 30 — Advanced Protocols

| Notebook | Description |
|---|---|
| `30_advanced_protocols/01_multimode_crosskerr.ipynb` | Multi-mode cross-Kerr interaction and conditional phase accumulation |
| `30_advanced_protocols/02_open_system_sideband_degradation.ipynb` | Open-system degradation of sideband swap under T₁, T₂, and cavity loss |
| `30_advanced_protocols/03_unitary_synthesis_workflow.ipynb` | Subspace-targeted unitary synthesis with gate sequence optimization |
| `30_advanced_protocols/04_snap_optimization_workflow.ipynb` | SNAP gate optimization for bosonic state preparation |
| `30_advanced_protocols/05_rl_hybrid_control_environment.ipynb` | Gym-compatible RL environment for cQED control with measurement-like observations |
| `30_advanced_protocols/06_grape_optimal_control_workflow.ipynb` | GRAPE optimal control: problem setup, hardware maps, solve, replay, and benchmark |

### 31 — System ID & Domain Randomization

| Notebook | Description |
|---|---|
| `31_system_identification_and_domain_randomization/01_calibration_targets_and_fitting.ipynb` | Build calibration-target traces and inspect fitted spectroscopy, Rabi, and T1 parameters |
| `31_system_identification_and_domain_randomization/02_evidence_to_randomizer_and_env.ipynb` | Convert calibration evidence into domain-randomization priors and inspect environment reset metadata |

### 40 — Validation & Conventions

| Notebook | Description |
|---|---|
| `40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb` | Verify Kerr sign convention and rotating-frame consistency |

### 50 — Floquet Driven Systems

| Notebook | Description |
|---|---|
| `50_floquet_driven_systems/01_sideband_quasienergy_scan.ipynb` | Sweep a periodic sideband drive and track avoided crossings in the dressed quasienergy spectrum |
| `50_floquet_driven_systems/02_branch_tracking_and_multiphoton_resonances.ipynb` | Interpret Floquet quasienergies using integer-order multiphoton resonance conditions |

---

## Foundational Curriculum

These numbered notebooks cover individual physics topics and API features in detail.

| # | Notebook | Topic |
|---|---|---|
| 00 | `00_tutorial_index.ipynb` | Index and roadmap for the foundational series |
| 01 | `01_getting_started_minimal_dispersive_model.ipynb` | Minimal dispersive model construction |
| 02 | `02_units_frames_and_conventions.ipynb` | Units (rad/s, seconds), rotating frames, carrier sign convention |
| 03 | `03_cavity_displacement_basics.ipynb` | Cavity displacement: coherent states, Wigner functions |
| 04 | `04_qubit_drive_and_basic_population_dynamics.ipynb` | Qubit Rabi oscillation under a single drive pulse |
| 05 | `05_observables_states_and_visualization.ipynb` | Bloch vectors, photon number, purity, reduced states |
| 06 | `06_qubit_spectroscopy.ipynb` | Frequency-swept qubit spectroscopy |
| 07 | `07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb` | Number-splitting spectroscopy in dispersive regime |
| 08 | `08_dispersive_shift_and_dressed_frequencies.ipynb` | Computing χ, dressed levels, and photon-number-dependent shifts |
| 09 | `09_power_rabi.ipynb` | Power Rabi: amplitude vs excitation probability |
| 10 | `10_time_rabi.ipynb` | Time Rabi: duration vs excitation probability |
| 11 | `11_qubit_T1_relaxation.ipynb` | T₁ energy relaxation under Lindblad dynamics |
| 12 | `12_qubit_ramsey_T2star.ipynb` | Ramsey experiment, T₂* extraction |
| 13 | `13_spin_echo_and_dephasing_mitigation.ipynb` | Hahn echo, T₂ echo measurement |
| 14 | `14_kerr_free_evolution.ipynb` | Self-Kerr dynamics in cavity Fock space |
| 15 | `15_cross_kerr_and_conditional_phase_accumulation.ipynb` | Cross-Kerr-mediated conditional phases |
| 16 | `16_storage_cavity_coherent_state_dynamics.ipynb` | Long-time coherent state evolution in storage |
| 17 | `17_readout_resonator_response.ipynb` | Readout resonator dynamics and dispersive readout |
| 18 | `18_multilevel_transmon_effects.ipynb` | Higher transmon levels: leakage, frequency shifts |
| 19 | `19_anharmonicity_and_leakage_under_strong_drive.ipynb` | Strong-drive leakage and DRAG correction |
| 20 | `20_truncation_convergence_checks.ipynb` | Convergence with Fock truncation |
| 21 | `21_building_sequences_from_gates_and_pulses.ipynb` | Gate → pulse → sequence compilation workflow |
| 22 | `22_parameter_sweeps_and_batch_simulation.ipynb` | Parameter sweeps and batched simulation |
| 23 | `23_analysis_fitting_and_result_extraction.ipynb` | Fitting, signal extraction, calibration analysis |
| 24 | `24_sideband_like_interactions.ipynb` | Sideband-like interactions and coupling mechanisms |
| 25 | `25_small_calibration_workflow_end_to_end.ipynb` | End-to-end calibration workflow |
| 26 | `26_frame_sanity_checks_and_common_failure_modes.ipynb` | Common frame mistakes and how to diagnose them |

---

## Topical Guide Pages

These documentation pages provide physics background, plots, and code summaries for each tutorial topic:

### Getting Started
- [Protocol-Style Simulation](getting_started_simulation.md) — model → pulse → compile → simulate → measure

### Core Physics
- [Displacement & Spectroscopy](displacement_spectroscopy.md) — dispersive Hamiltonian, cavity displacement, number-splitting spectrum
- [Number Splitting](number_splitting.md) — selective Gaussian probe, photon-number weight extraction, Poisson distribution comparison
- [Kerr Free Evolution](kerr_free_evolution.md) — self-Kerr origin, Wigner collapse/revival, Schrödinger cat states
- [Phase Space Conventions](phase_space_conventions.md) — quadrature vs. alpha coordinates, √2 rescaling, Wigner function background
- [Cross-Kerr Interaction](cross_kerr.md) — conditional phase accumulation, three-mode system, controlled-phase gates

### Sideband Protocols
- [Sideband Swap & Bosonic Control](sideband_swap.md) — red-sideband energy matching, |e,0⟩↔|g,1⟩ Rabi oscillation, detuned/sequential/shelving variants

### Advanced Control
- [Unitary Synthesis](unitary_synthesis.md) — gate sequence optimization: constraint, leakage, robust, Pareto, trajectory, flexible targets
- [GRAPE Optimal Control](optimal_control.md) — GRAPE algorithm, hardware-aware mode, convergence plots, replay verification
- [RL Hybrid Control](rl_hybrid_control.md) — Gym environment, measurement-conditioned control, domain randomization
- [Hardware-Aware Control](hardware_context.md) — signal chain: filtering, DAC, IQ limits, gradient-through-hardware GRAPE

### System Identification
- [Calibration & Domain Randomization](system_identification.md) — spectroscopy/Rabi/T1 fitting, calibration posteriors, RL training priors

### Floquet & Driven Systems
- [Floquet Driven Systems](floquet_driven_systems.md) — Floquet theorem, quasienergies, avoided crossings, multiphoton resonances

### Advanced Topics
- [Holographic Quantum Algorithms](holographic_quantum_algorithms.md) — holographic channel estimation, MPS-inspired correlator sampling
- [Generalized Holographic Workflow](holographic_generalized_unitary_workflow.md) — finite per-step unitary sequences, MPS lifting, and exact-vs-sampled validation

---

## Conventions

All tutorials follow the runtime conventions documented in:

- [Physics & Conventions](../physics_conventions.md) — full reference
- `tutorials/conventions_quick_reference.md` — quick lookup card

Key conventions used throughout:

| Convention | Value |
|---|---|
| Hamiltonian frequencies | rad/s |
| Times | seconds |
| Tensor order | qubit ⊗ cavity |
| Dispersive term | $+\chi \hat{n}_c \hat{n}_q$ ($\chi < 0$ typical) |
| Carrier sign | $\text{carrier} = -\omega_{\text{transition}}$ in rotating frame |

---

## Tutorials vs Examples

**Tutorials** (`tutorials/`) are structured, step-by-step learning material meant to be read in order.

**Examples** (`examples/`) are standalone scripts, studies, and workflow helpers. They are task-focused and concise, not pedagogical. See the [Examples](../examples.md) page for the full index.
