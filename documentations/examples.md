# Examples

This page indexes the example scripts, notebooks, and workflows in the repository's `examples/` directory.

---

## Top-Level Example Scripts

These standalone scripts demonstrate specific `cqed_sim` capabilities:

| Script | Description |
|---|---|
| `displacement_qubit_spectroscopy.py` | Cavity displacement followed by qubit spectroscopy |
| `kerr_free_evolution.py` | Cavity free evolution under Kerr nonlinearity |
| `kerr_sign_verification.py` | Diagnostic verifying the Kerr sign convention |
| `sideband_swap_demo.py` | Basic sideband swap between transmon and storage |
| `sideband_swap.py` | Extended sideband swap workflow |
| `detuned_sideband_sync_demo.py` | Detuned sideband synchronization |
| `shelving_isolation_demo.py` | Shelving isolation with multilevel sideband |
| `open_system_sideband_degradation.py` | Sideband performance under open-system noise |
| `multimode_crosskerr_demo.py` | Multi-mode cross-Kerr interaction demo |
| `ringdown_noise.py` | Cavity ringdown with noise |
| `unitary_synthesis_demo.py` | Gate-sequence optimization for target unitaries |
| `run_snap_optimization_demo.py` | SNAP gate optimization |

---

## Workflow Scripts and Notebooks

Under `examples/workflows/`:

| File | Description |
|---|---|
| `universal_cqed_model_demo.py` | UniversalCQEDModel usage demonstration |
| `cqed_sim_usage_examples.ipynb` | Interactive usage examples notebook |
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

## Audits

Under `examples/audits/`:

| File | Description |
|---|---|
| `chi_phase_evolution_audit.ipynb` | Chi-dependent phase evolution audit |
| `experiment_convention_audit.py` | Convention consistency checks |
| `sqr_convention_metric_audit.py` | SQR convention and metric audit |

---

## Studies

Under `examples/studies/`:

| File | Description |
|---|---|
| `sqr_speedlimit_multitone_gaussian.py` | SQR speed-limit analysis |
| `sqr_block_phase_study.py` | SQR block-phase behavior |
| `sqr_multitone_study.py` | SQR multitone pulse analysis |
| `sqr_route_b_enlarged_control.py` | SQR Route B enlarged control study |

### SNAP Optimization

Under `examples/studies/snap_opt/`:

A complete SNAP pulse optimization package with model, experiments, metrics, optimizer, and pulse modules.

---

## Paper Reproductions

Under `examples/paper_reproductions/snap_prl133/`:

Reproduction of SNAP gate results from PRL 133, including independent model, optimization, and metric validation.

---

## Smoke Tests

Under `examples/smoke_tests/`:

| File | Description |
|---|---|
| `sanity_run.py` | Basic sanity check |
| `tests/test_sanity.py` | Sanity test runner |
| `tests/test_sqr_calibration.py` | SQR calibration smoke test |

---

## Missing Example Coverage

The following workflows would benefit from dedicated example scripts:

1. **Three-mode readout simulation** — using `DispersiveReadoutTransmonStorageModel` with `ReadoutChain`
2. **Fock-resolved tomography** — end-to-end tomography protocol usage
3. **All-XY calibration** — qubit pulse calibration workflow
4. **Calibration targets** — spectroscopy, Rabi, Ramsey, T₁, T₂ echo usage
5. **Parameter translation** — converting bare transmon params to dispersive params
