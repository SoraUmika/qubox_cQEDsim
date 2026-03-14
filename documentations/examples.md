# Examples

This page indexes the repository-side assets that remain in `examples/`.

The primary guided learning path no longer lives here. Use the top-level `tutorials/` curriculum for numbered notebook walkthroughs, starting with:

- `tutorials/README.md`
- `tutorials/00_tutorial_index.ipynb`

---

## Top-Level Standalone Scripts

| Script | Description |
|---|---|
| `protocol_style_simulation.py` | Direct prepare -> compile -> simulate -> measure workflow using stable library primitives |
| `kerr_free_evolution.py` | Standalone Kerr free-evolution script built on low-level `cqed_sim` primitives |
| `kerr_sign_verification.py` | Kerr-sign diagnostic companion to the Kerr workflow |
| `sequential_sideband_reset.py` | Sequential sideband-reset recipe for the explicit three-mode model |
| `displacement_qubit_spectroscopy.py` | Standalone displacement-plus-spectroscopy script using the current SI-style runtime convention |
| `sideband_swap_demo.py` | Basic sideband swap between transmon and storage |
| `sideband_swap.py` | Extended sideband swap workflow |
| `detuned_sideband_sync_demo.py` | Detuned sideband synchronization |
| `shelving_isolation_demo.py` | Shelving isolation with multilevel sideband |
| `open_system_sideband_degradation.py` | Sideband performance under open-system noise |
| `multimode_crosskerr_demo.py` | Multi-mode cross-Kerr interaction demo |
| `ringdown_noise.py` | Cavity ringdown with noise |

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
