# Examples

This page indexes the example scripts, notebooks, and workflow helpers in the repository's `examples/` directory.

---

## Top-Level Example Scripts

| Script | Description |
|---|---|
| `protocol_style_simulation.py` | Direct prepare -> compile -> simulate -> measure workflow using stable library primitives |
| `kerr_free_evolution.py` | Kerr free-evolution recipe built on low-level `cqed_sim` primitives |
| `kerr_sign_verification.py` | Kerr-sign diagnostic companion to the Kerr workflow |
| `sequential_sideband_reset.py` | Sequential sideband-reset recipe for the explicit three-mode model |
| `displacement_qubit_spectroscopy.py` | Cavity displacement followed by qubit spectroscopy |
| `sideband_swap_demo.py` | Basic sideband swap between transmon and storage |
| `sideband_swap.py` | Extended sideband swap workflow |
| `detuned_sideband_sync_demo.py` | Detuned sideband synchronization |
| `shelving_isolation_demo.py` | Shelving isolation with multilevel sideband |
| `open_system_sideband_degradation.py` | Sideband performance under open-system noise |
| `multimode_crosskerr_demo.py` | Multi-mode cross-Kerr interaction demo |
| `ringdown_noise.py` | Cavity ringdown with noise |

---

## Workflow Modules and Notebooks

Under `examples/workflows/`:

| File | Description |
|---|---|
| `kerr_free_evolution.py` | Workflow helper backing the Kerr example scripts and workflow tests |
| `sequential_sideband_reset.py` | Workflow helper backing the sideband-reset example script and notebook |
| `cqed_sim_usage_examples.ipynb` | Interactive usage notebook using stable low-level imports |
| `sequential_simulation.ipynb` | Sequential gate-by-gate simulation |
| `sqr_calibration_workflow.ipynb` | SQR calibration workflow notebook |
| `fock_tomo_workflow.py` | Fock-resolved tomography workflow |
| `sqr_transfer.py` | SQR transfer artifact generation |
| `simulate_fock_tomo_and_sqr_calibration.py` | Combined Fock tomography and SQR calibration |

### Sequential Simulation Helpers

Under `examples/workflows/sequential/`:

| Module | Description |
|---|---|
| `common.py` | Shared sequential simulation setup |
| `ideal.py` | Ideal gate-level sequential simulation |
| `pulse_calibrated.py` | Pulse-level with calibrated corrections |
| `pulse_open.py` | Pulse-level with open-system dynamics |
| `pulse_unitary.py` | Pulse-level unitary simulation |
| `trajectories.py` | Trajectory extraction helpers |

---

## Audits, Studies, and Reproductions

- `examples/audits/` contains convention and consistency audits.
- `examples/studies/` contains optimization and parameter studies.
- `examples/paper_reproductions/` contains paper-specific reproduction code.
- `test_against_papers/` contains notebook-style literature checks.

---

## Example-Side Tests

Workflow validation now lives next to the example code when it tests example-only orchestration rather than reusable library primitives:

- `examples/workflows/tests/`
- `examples/audits/tests/`
- `examples/studies/tests/`
- `examples/smoke_tests/tests/`
