# `cqed_sim.plotting`

The `plotting` module provides a collection of cQED-specific visualization functions for simulation results, gate diagnostics, phase errors, Wigner functions, energy level diagrams, and calibration outputs. All functions return Matplotlib figures or axes objects.

## Relevance in `cqed_sim`

Visualization is essential for understanding simulation results and diagnosing gate performance. This module provides plot functions that are grounded in the specific observables and data structures used throughout `cqed_sim`, rather than generic Matplotlib utilities.

## Main Capabilities

### Energy levels (`energy_levels`)

- **`plot_energy_levels(spectrum, max_levels, energy_scale, energy_unit_label)`**: Plots the dressed energy level ladder from a `compute_energy_spectrum(...)` result.

### Gate diagnostics (`gate_diagnostics`)

- **`plot_fock_resolved_bloch_overlay(tomo_result, model)`**: Overlays per-sector Bloch vectors from Fock-resolved tomography.
- **`plot_fock_resolved_bloch_grouped_bars(tomo_result, model)`**: Bar chart of per-sector fidelities.
- **`plot_gate_bloch_trajectory_overlay(states, model)`**: Bloch-sphere trajectory overlays for a set of states.
- **`plot_gate_bloch_trajectory_error(states, target_states, model)`**: Error in the Bloch trajectory relative to an ideal target.
- **`plot_phase_error_track(phase_errors, sectors)`**: Phase error as a function of Fock sector.
- **`plot_phase_error_heatmap(phase_error_matrix)`**: 2D heatmap of phase errors.
- **`plot_phase_overlay_lines(phase_data, sectors)`**: Overlay of phase values across multiple runs.
- **`plot_phase_heatmap_overlay(phase_matrix, model)`**: Combined phase heatmap with model annotations.
- **`plot_combined_gate_diagnostics(...)`**: Composite figure combining Bloch, phase, and fidelity diagnostics.
- **`save_figure(fig, path)`**: Save a Matplotlib figure to a file.

### Bloch plots (`bloch_plots`)

- **`plot_bloch_track(states, times, model)`**: Qubit Bloch vector as a function of time.
- **`add_gate_type_axis(fig, ax, gate_types)`**: Adds a color-coded gate-type axis to an existing figure.
- **`GATE_COLORS`**: Color mapping for gate types used across diagnostic plots.

### Phase plots (`phase_plots`)

- **`plot_relative_phase_track(diagnostics, model)`**: Relative conditional phase between Fock sectors over time.

### Calibration plots (`calibration_plots`)

- **`plot_sqr_calibration_result(result)`**: Fidelity and convergence plot for a `SQRCalibrationResult`.

### Wigner function grids (`wigner_grids`)

- **`plot_wigner_grid(wigner_snapshots, xvec, yvec, times)`**: Grid of Wigner function snapshots at selected times.

### Weakness / comparison plots (`weakness_plots`)

- **`plot_weakness(comparison, model)`**: Visualizes gate weakness metrics across Fock sectors.
- **`plot_component_comparison(result_a, result_b, model)`**: Compares individual observable components between two simulation runs.
- **`plot_cavity_population_comparison(result_a, result_b, model)`**: Compares cavity population distributions.
- **`print_mapping_rows(comparison)`**: Prints a formatted table of the comparison metrics.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `plot_energy_levels(spectrum, ...)` | Dressed energy level diagram |
| `plot_fock_resolved_bloch_overlay(...)` | Per-sector Bloch vectors |
| `plot_combined_gate_diagnostics(...)` | Composite gate diagnostic figure |
| `plot_phase_error_track(...)` | Phase error vs. Fock sector |
| `plot_wigner_grid(...)` | Wigner function snapshots |
| `plot_bloch_track(states, times, model)` | Bloch vector trajectory |
| `plot_sqr_calibration_result(result)` | SQR calibration convergence |
| `plot_relative_phase_track(diagnostics, model)` | Relative conditional phase |
| `save_figure(fig, path)` | Save figure to file |

## Usage Guidance

```python
import numpy as np
from cqed_sim import compute_energy_spectrum, FrameSpec
from cqed_sim.plotting import plot_energy_levels

spectrum = compute_energy_spectrum(model, frame=FrameSpec(), levels=12)
fig = plot_energy_levels(
    spectrum,
    max_levels=12,
    energy_scale=1.0 / (2*np.pi*1e6),
    energy_unit_label="MHz",
)
fig.savefig("energy_levels.png", dpi=150)
```

```python
from cqed_sim.observables import selected_wigner_snapshots
from cqed_sim.plotting import plot_wigner_grid

xvec = np.linspace(-4, 4, 101)
snapshots = selected_wigner_snapshots(result.states, result.times, model, xvec, xvec)
fig = plot_wigner_grid(snapshots, xvec, xvec)
```

## Important Assumptions / Conventions

- All plot functions expect inputs in the data structures and units produced by `cqed_sim` (e.g. `FockTomographyResult`, `SQRCalibrationResult`, `EnergySpectrum`). They are not general-purpose plotting utilities.
- Energy level plots use lab-frame energies shifted to vacuum = 0 by default; use `FrameSpec()` (no rotation) when calling `compute_energy_spectrum(...)` for physically interpretable diagrams.
- All functions return Matplotlib `Figure` or `Axes` objects; display or saving is the caller's responsibility unless `save_figure(...)` is used.

## Relationships to Other Modules

- **`cqed_sim.observables`**: Wigner and phase diagnostic data consumed by the plot functions is computed there.
- **`cqed_sim.tomo`**: `FockTomographyResult` from `run_fock_resolved_tomo(...)` is the input to Bloch-resolved plot functions.
- **`cqed_sim.calibration`**: `SQRCalibrationResult` is the input to `plot_sqr_calibration_result(...)`.
- **`cqed_sim.core`**: `EnergySpectrum` from `compute_energy_spectrum(...)` is the input to `plot_energy_levels(...)`.

## Limitations / Non-Goals

- Does not provide a general-purpose plotting framework; functions are tightly coupled to `cqed_sim` data structures.
- Publication-quality figure styling is not guaranteed; callers should adjust Matplotlib `rcParams` and figure dimensions for final figures.
- Interactive or animated visualizations are not included.
