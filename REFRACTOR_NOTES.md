# Sequential Simulation Refactor Notes

## What moved where
- JSON loading, typo-tolerant path handling, validation, and typed gate objects:
  - `cqed_sim/io/gates.py`
- SQR numerical calibration, cache handling, and per-manifold conditional-unitary extraction:
  - `cqed_sim/calibration/sqr.py`
- Shared state/operator helpers:
  - `cqed_sim/operators/basic.py`
  - `cqed_sim/operators/cavity.py`
- Pulse envelopes and analytic amplitude mappings:
  - `cqed_sim/pulses/envelopes.py`
  - `cqed_sim/pulses/calibration.py`
- Sequential simulation engines:
  - `cqed_sim/simulators/ideal.py`
  - `cqed_sim/simulators/pulse_unitary.py`
  - `cqed_sim/simulators/pulse_open.py`
  - `cqed_sim/simulators/pulse_calibrated.py`
  - shared helpers in `cqed_sim/simulators/common.py`
- Observables and metrics:
  - `cqed_sim/observables/bloch.py`
  - `cqed_sim/observables/fock.py`
  - `cqed_sim/observables/wigner.py`
  - `cqed_sim/observables/phases.py`
  - `cqed_sim/observables/trajectories.py`
  - `cqed_sim/observables/weakness.py`
- Plotting helpers:
  - `cqed_sim/plotting/bloch_plots.py`
  - `cqed_sim/plotting/calibration_plots.py`
  - `cqed_sim/plotting/gate_diagnostics.py`
  - `cqed_sim/plotting/wigner_grids.py`
  - `cqed_sim/plotting/phase_plots.py`
  - `cqed_sim/plotting/weakness_plots.py`
- Notebook-facing sanity helpers:
  - `cqed_sim/tests/test_sanity.py`
  - `cqed_sim/tests/test_sqr_calibration.py`

## New API entry points
- Gate IO:
  - `from cqed_sim.io.gates import load_gate_sequence, render_gate_table`
- Calibration:
  - `from cqed_sim.calibration.sqr import calibrate_sqr_gate, load_or_calibrate_sqr_gate, calibrate_all_sqr_gates`
- Simulators:
  - `from cqed_sim.simulators.ideal import run_case_a`
  - `from cqed_sim.simulators.pulse_unitary import run_case_b`
  - `from cqed_sim.simulators.pulse_open import run_case_c`
  - `from cqed_sim.simulators.pulse_calibrated import run_case_d`
  - `from cqed_sim.simulators.trajectories import simulate_gate_bloch_trajectory`
- Metrics:
  - `from cqed_sim.observables.weakness import attach_weakness_metrics, comparison_metrics`
  - `from cqed_sim.observables.fock import fock_resolved_bloch_diagnostics, conditional_phase_diagnostics`
- Plots:
  - `from cqed_sim.plotting.bloch_plots import plot_bloch_track`
  - `from cqed_sim.plotting.calibration_plots import plot_sqr_calibration_result`
  - `from cqed_sim.plotting.wigner_grids import plot_wigner_grid`
  - `from cqed_sim.plotting.phase_plots import plot_relative_phase_track`
  - `from cqed_sim.plotting.gate_diagnostics import plot_fock_resolved_bloch_heatmaps, plot_relative_phase_heatmap, plot_gate_bloch_trajectory`
  - `from cqed_sim.plotting.weakness_plots import plot_component_comparison, plot_cavity_population_comparison, plot_weakness`
- Sanity checks:
  - `from cqed_sim.tests.test_sanity import baseline_vs_refactor_sanity, run_notebook_sanity_suite`
  - `from cqed_sim.tests.test_sqr_calibration import run_sqr_calibration_sanity_suite`

## How to run
- Regenerate the notebook:
- Regenerate the notebooks:
  - `python outputs/generate_sequential_simulation_notebook.py`
  - `python outputs/generate_sqr_calibration_notebook.py`
- Run the notebook:
  - open `sequential_simulation.ipynb` in a kernel with `qutip`, `matplotlib`, and the editable package installed
- Run the calibration notebook:
  - open `SQR_calibration.ipynb` in a kernel with `qutip`, `matplotlib`, `scipy`, and the editable package installed
- Run the notebook sanity tests:
  - `pytest -q cqed_sim/tests/test_sanity.py`
  - `pytest -q cqed_sim/tests/test_sqr_calibration.py`

## Notes
- The refactored notebook is now an orchestrator: configuration, gate loading, simulator calls, plots, and summary tables.
- `SQR_calibration.ipynb` is also package-backed and exports reusable calibration JSON for Case D caching.
- Case A Bloch plots keep gate labels only on the top x-axis.
- Wigner tomography uses compact batched subplot grids with fixed per-case color scaling and minimal axis labels.
- Section Y of `sequential_simulation.ipynb` now saves gate-indexed Fock/Bloch and conditional-phase heatmaps plus a selected-gate pulse trajectory under `outputs/figures/`.
- The baseline-vs-refactor sanity check compares the refactored Case A implementation against an independent direct-unitary reference path.
