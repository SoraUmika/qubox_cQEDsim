# API Reference — Plotting (`cqed_sim.plotting`)

All plotting functions return matplotlib `Figure` objects and are designed for diagnostic visualization.

---

## Bloch Track (`plotting.bloch_plots`)

| Function | Description |
|---|---|
| `plot_bloch_track(track, title, label_stride)` | Bloch (X,Y,Z) evolution vs gate index with gate-type background shading |
| `add_gate_type_axis(ax, track, label_stride)` | Add top axis with gate type labels |

**Constant:** `GATE_COLORS = {"INIT": "black", "Displacement": "tab:blue", "Rotation": "tab:orange", "SQR": "tab:green"}`

---

## Calibration (`plotting.calibration_plots`)

| Function | Description |
|---|---|
| `plot_sqr_calibration_result(result)` | 4-panel: d_lambda, d_alpha, d_omega_hz, loss vs Fock level |
| `plot_energy_levels(spectrum, max_levels=None, energy_scale=1.0, energy_unit_label="rad/s", annotate=True, title=None, ax=None)` | Ladder-style plot of vacuum-referenced energy levels |

---

## Gate Diagnostics (`plotting.gate_diagnostics`)

| Function | Description |
|---|---|
| `plot_fock_resolved_bloch_overlay(simulated, ideal, track, component, ...)` | Heatmap overlay of Fock-resolved Bloch component |
| `plot_fock_resolved_bloch_grouped_bars(...)` | Grouped bar chart per Fock level |
| `plot_phase_heatmap_overlay(...)` | Phase heatmap comparison |
| `plot_phase_overlay_lines(...)` | Line plot of relative phases |
| `plot_phase_error_heatmap(...)` | Phase error heatmap |
| `plot_phase_error_track(...)` | Wrapped phase error vs gate index |
| `plot_gate_bloch_trajectory_overlay(simulated, ideal)` | 3-panel time-domain Bloch within a gate |
| `plot_gate_bloch_trajectory_error(simulated, ideal)` | Error trajectory within a gate |
| `plot_combined_gate_diagnostics(...)` | Combined 4×3 layout with all diagnostic panels |
| `save_figure(fig, output_dir, filename, dpi=160)` | Save figure to disk |

---

## Phase Plots (`plotting.phase_plots`)

| Function | Description |
|---|---|
| `plot_relative_phase_track(track, max_n, threshold, unwrap, label_stride)` | Phase line plot with Fock coloring |

---

## Weakness Plots (`plotting.weakness_plots`)

| Function | Description |
|---|---|
| `plot_component_comparison(a, b, c, d=None, label_stride=1)` | 3-panel Bloch comparison |
| `plot_cavity_population_comparison(a, b, c, d=None, label_stride=1)` | Cavity ⟨n⟩ comparison |
| `plot_weakness(b, c, reference, d=None, label_stride=1)` | Wigner negativity + fidelity weakness |
| `print_mapping_rows(track)` | Print track metadata |

---

## Wigner Grids (`plotting.wigner_grids`)

| Function | Description |
|---|---|
| `plot_wigner_grid(track, title, stride, max_cols=None, show_colorbar=False)` | Grid of Wigner snapshots at gate indices |
