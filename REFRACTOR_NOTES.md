# Sequential Simulation Refactor Notes

## What moved where
- JSON loading, typo-tolerant path handling, validation, and typed gate objects:
  - `cqed_sim/io/gates.py`
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
  - shared helpers in `cqed_sim/simulators/common.py`
- Observables and metrics:
  - `cqed_sim/observables/bloch.py`
  - `cqed_sim/observables/wigner.py`
  - `cqed_sim/observables/phases.py`
  - `cqed_sim/observables/weakness.py`
- Plotting helpers:
  - `cqed_sim/plotting/bloch_plots.py`
  - `cqed_sim/plotting/wigner_grids.py`
  - `cqed_sim/plotting/phase_plots.py`
  - `cqed_sim/plotting/weakness_plots.py`
- Notebook-facing sanity helpers:
  - `cqed_sim/tests/test_sanity.py`

## New API entry points
- Gate IO:
  - `from cqed_sim.io.gates import load_gate_sequence, render_gate_table`
- Simulators:
  - `from cqed_sim.simulators.ideal import run_case_a`
  - `from cqed_sim.simulators.pulse_unitary import run_case_b`
  - `from cqed_sim.simulators.pulse_open import run_case_c`
- Metrics:
  - `from cqed_sim.observables.weakness import attach_weakness_metrics, comparison_metrics`
- Plots:
  - `from cqed_sim.plotting.bloch_plots import plot_bloch_track`
  - `from cqed_sim.plotting.wigner_grids import plot_wigner_grid`
  - `from cqed_sim.plotting.phase_plots import plot_relative_phase_track`
  - `from cqed_sim.plotting.weakness_plots import plot_component_comparison, plot_cavity_population_comparison, plot_weakness`
- Sanity checks:
  - `from cqed_sim.tests.test_sanity import baseline_vs_refactor_sanity, run_notebook_sanity_suite`

## How to run
- Regenerate the notebook:
  - `python outputs/generate_sequential_simulation_notebook.py`
- Run the notebook:
  - open `sequential_simulation.ipynb` in a kernel with `qutip`, `matplotlib`, and the editable package installed
- Run the notebook sanity tests:
  - `pytest -q cqed_sim/tests/test_sanity.py`

## Notes
- The refactored notebook is now an orchestrator: configuration, gate loading, simulator calls, plots, and summary tables.
- Case A Bloch plots keep gate labels only on the top x-axis.
- Wigner tomography uses compact batched subplot grids with fixed per-case color scaling and minimal axis labels.
- The baseline-vs-refactor sanity check compares the refactored Case A implementation against an independent direct-unitary reference path.
