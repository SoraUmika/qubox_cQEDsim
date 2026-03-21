# Examples

This page indexes the repository-side assets that remain in `examples/`.

The primary guided learning path no longer lives here. Use the top-level `tutorials/` curriculum for numbered notebook walkthroughs, starting with:

- `tutorials/README.md`
- `tutorials/00_getting_started/01_protocol_style_simulation.ipynb`
- `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`

---

## Repo Automation

This repository also includes repo-side workflow automation outside the reusable `cqed_sim` package surface:

- `agent_workflow/`
- `README_AGENT_WORKFLOW.md`
- `tools/run_agent_workflow.py`

This automation layer is intended for task orchestration, prompt management, persistent run-state tracking, and resumable validation workflows. It is not part of the installed simulation library API.

---

## Top-Level Standalone Scripts

| Script | Description | Notebook companion |
|---|---|---|
| `protocol_style_simulation.py` | Direct prepare -> compile -> simulate -> measure workflow using stable library primitives | `tutorials/00_getting_started/01_protocol_style_simulation.ipynb` |
| `kerr_free_evolution.py` | Standalone Kerr free-evolution script built on low-level `cqed_sim` primitives | `tutorials/10_core_workflows/02_kerr_free_evolution.ipynb` |
| `kerr_sign_verification.py` | Kerr-sign diagnostic companion to the Kerr workflow | `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb` |
| `sequential_sideband_reset.py` | Sequential sideband-reset recipe for the explicit three-mode model | `tutorials/20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb` |
| `displacement_qubit_spectroscopy.py` | Standalone displacement-plus-spectroscopy script using the current SI-style runtime convention | `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb` |
| `sideband_swap_demo.py` | Basic sideband swap between transmon and storage | `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb` |
| `sideband_swap.py` | Compatibility wrapper around the sideband swap demo | `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb` |
| `detuned_sideband_sync_demo.py` | Detuned sideband synchronization | `tutorials/20_bosonic_and_sideband/02_detuned_sideband_synchronization.ipynb` |
| `shelving_isolation_demo.py` | Shelving isolation with multilevel sideband | `tutorials/20_bosonic_and_sideband/04_shelving_isolation.ipynb` |
| `open_system_sideband_degradation.py` | Sideband performance under open-system noise | `tutorials/30_advanced_protocols/02_open_system_sideband_degradation.ipynb` |
| `multimode_crosskerr_demo.py` | Multi-mode cross-Kerr interaction demo | `tutorials/30_advanced_protocols/01_multimode_crosskerr.ipynb` |
| `run_snap_optimization_demo.py` | SNAP optimization demo using repo-side study helpers | `tutorials/30_advanced_protocols/04_snap_optimization_workflow.ipynb` |
| `conditioned_multitone_reduced_demo.py` | Reduced conditioned-multitone workflow showing reduced-vs-full agreement and detuning-only correction optimization | none |
| `logical_block_phase_targeted_subspace_demo.py` | Targeted-subspace multitone demo showing gauge-fixed logical block phases and the ideal cavity-only correction layer | none |
| `unitary_synthesis_demo.py` | Target-unitary synthesis inside a qubit-cavity subspace | `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb` |
| `unitary_synthesis_relevance_aware_optimizer.py` | Relevance-aware synthesis with observable, state-ensemble, and trajectory objectives plus accelerated ideal evaluation | `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb` |
| `unitary_synthesis_flexible_target_actions.py` | Channel, reduced-state, and isometry target examples with truncation-aware diagnostics | `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb` |
| `grape_storage_subspace_gate_demo.py` | Model-backed GRAPE optimization of a storage logical-subspace gate with pulse export and runtime replay | `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb` |
| `hardware_constrained_grape_demo.py` | Hardware-aware GRAPE comparison showing held-sample controls, low-pass filtering, boundary windows, IQ-radius limits, and command-vs-physical replay | `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb` |
| `rl_hybrid_control_rollout.py` | Hybrid RL environment rollout with measurement-like observations, diagnostics, and domain-randomized evaluation | `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb` |
| `ringdown_noise.py` | Cavity ringdown with noise | retained as a compact standalone diagnostic script |

---

## Quantum Algorithms Examples

Under `examples/quantum_algorithms/`:

| Script | Description |
|---|---|
| `holographic_minimal_correlator.py` | Small ideal correlator example with Monte Carlo and exact branch comparison |
| `holographic_burn_in_translation_invariant.py` | Translation-invariant channel with explicit burn-in before bulk estimation |
| `holographic_spin_model_example.py` | Spin-inspired transfer-channel example motivated by the holographic report |

---

## Workflow Helpers

Under `examples/workflows/`:

| File | Description |
|---|---|
| `kerr_free_evolution.py` | Workflow helper backing the Kerr scripts and workflow tests |
| `sequential_sideband_reset.py` | Workflow helper backing the sideband-reset script |
| `fock_tomo_workflow.py` | Fock-resolved tomography workflow |
| `sqr_transfer.py` | SQR transfer artifact generation |
| `simulate_fock_tomo_and_sqr_calibration.py` | Combined Fock tomography and SQR calibration |
| `universal_cqed_model_demo.py` | Generalized-model workflow example |

### Sequential Workflow Modules

Under `examples/workflows/sequential/`:

| Module | Description |
|---|---|
| `common.py` | Shared sequential simulation setup |
| `ideal.py` | Ideal gate-level sequential simulation |
| `pulse_calibrated.py` | Pulse-level sequential simulation with calibrated corrections |
| `pulse_open.py` | Pulse-level sequential simulation with dissipation |
| `pulse_unitary.py` | Pulse-level unitary simulation |
| `trajectories.py` | Trajectory extraction helpers |

---

## Audits, Studies, and Reproductions

- `examples/audits/` contains convention and consistency audits.
- `examples/studies/` contains optimization and parameter studies.
- `examples/paper_reproductions/` contains paper-specific reproduction code.
- `test_against_papers/` contains notebook-style literature checks.

---

## Examples vs Tutorials

- Use `tutorials/` for structured, user-facing notebook lessons.
- Use `examples/` for standalone scripts, advanced study helpers, repo-side workflow utilities, and non-curriculum artifacts.
- Use `tests/` for automated correctness and regression coverage.

---

## Example-Side Tests

Workflow validation that is specific to repo-side examples still lives next to the example code:

- `examples/workflows/tests/`
- `examples/audits/tests/`
- `examples/studies/tests/`
- `examples/smoke_tests/tests/`
