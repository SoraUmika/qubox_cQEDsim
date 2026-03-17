# `cqed_sim.observables`

The `observables` module provides higher-level observable extractors, phase diagnostics, Wigner function utilities, and weakness/comparison metrics for analyzing `cqed_sim` simulation states. It is the diagnostic and analysis layer for post-processing simulation results.

## Relevance in `cqed_sim`

After running a simulation with `cqed_sim.sim`, the raw result is a QuTiP density matrix or ket. This module provides structured tools for extracting physically meaningful quantities from those states, including:

- reduced subsystem states and Bloch vectors,
- conditional phase and Fock-resolved phase diagnostics,
- cavity Wigner function computation and negativity,
- Bloch-vector trajectory analysis across a time series of states,
- and weakness/comparison metrics for benchmarking gate approximations.

## Main Capabilities

### State reduction and Bloch vectors (`bloch`)

- **`reduced_qubit_state(state)`**: Partial trace to the qubit subsystem.
- **`reduced_cavity_state(state)`**: Partial trace to the cavity subsystem.
- **`bloch_xyz_from_joint(state, n)`**: Bloch vector of the qubit conditioned on Fock sector `n`.
- **`cavity_moments(state)`**: Mean photon number and second moment of the cavity.

### Fock-resolved phase diagnostics (`fock`)

- **`fock_resolved_bloch_diagnostics(state, model)`**: Per-sector Bloch vectors and angles for the full qubit-cavity state.
- **`conditional_phase_diagnostics(state, model)`**: Conditional phase accumulated on the qubit per Fock sector.
- **`relative_phase_family_diagnostics(states, model)`**: Phase diagnostics across a family of states (e.g. from a sweep).
- **`relative_phase_debug_values(state, model)`**: Raw phase values for debugging phase extraction.
- **`wrapped_phase_error(phi_actual, phi_target)`**: Phase error with 2π wrapping.

### Phase diagnostics (`phases`)

- **`relative_phase_diagnostics(states, model)`**: Relative phase between Fock sectors across a time series.

### Wigner function (`wigner`)

- **`cavity_wigner(state, xvec, yvec)`**: Wigner function of the reduced cavity state on a grid.
- **`selected_wigner_snapshots(states, times, model, xvec, yvec)`**: Wigner snapshots at selected time points.
- **`wigner_negativity(W, xvec, yvec)`**: Integral of the negative part of the Wigner function (non-classicality measure).

### Bloch trajectory (`trajectories`)

- **`bloch_trajectory_from_states(states, model)`**: Extracts the qubit Bloch vector trajectory from a time series of joint states.

### Weakness and comparison metrics (`weakness`)

- **`attach_weakness_metrics(result, target_result, model)`**: Attaches a weakness diagnostic bundle to a simulation result, comparing it against a target (e.g. ideal) result.
- **`comparison_metrics(result, target_result, model)`**: Returns a structured comparison of Bloch vectors, photon numbers, and phase errors between two results.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `reduced_qubit_state(state)` | Qubit partial trace |
| `reduced_cavity_state(state)` | Cavity partial trace |
| `bloch_xyz_from_joint(state, n)` | Qubit Bloch vector conditioned on Fock-n |
| `cavity_moments(state)` | Cavity photon-number moments |
| `fock_resolved_bloch_diagnostics(state, model)` | Per-sector Bloch vectors |
| `conditional_phase_diagnostics(state, model)` | Conditional qubit phase per Fock sector |
| `cavity_wigner(state, xvec, yvec)` | Wigner function |
| `wigner_negativity(W, xvec, yvec)` | Wigner negativity |
| `bloch_trajectory_from_states(states, model)` | Bloch-vector time trajectory |
| `comparison_metrics(result, target, model)` | Weakness/comparison diagnostics |

## Usage Guidance

```python
import numpy as np
from cqed_sim.observables import (
    reduced_qubit_state, cavity_wigner, wigner_negativity,
    fock_resolved_bloch_diagnostics, bloch_trajectory_from_states,
)

# Cavity Wigner function
xvec = np.linspace(-4, 4, 101)
W = cavity_wigner(result.final_state, xvec, xvec)
neg = wigner_negativity(W, xvec, xvec)

# Fock-resolved Bloch diagnostics
diagnostics = fock_resolved_bloch_diagnostics(result.final_state, model)

# Bloch trajectory from stored states
trajectory = bloch_trajectory_from_states(result.states, model)
```

## Important Assumptions / Conventions

- All functions assume the standard tensor ordering from `cqed_sim.core`: qubit first, then cavity/storage modes.
- `bloch_xyz_from_joint(state, n)` returns the unnormalized Bloch vector (not divided by the sector probability); the sector population must be accounted for separately if needed.
- Wigner function computation wraps the QuTiP `wigner(...)` function; the `xvec`/`yvec` arguments specify the phase-space grid in dimensionless quadrature units.
- Phase diagnostics use `wrapped_phase_error(...)` which wraps to `[-π, π]`.

## Relationships to Other Modules

- **`cqed_sim.sim`**: many functions here wrap extractor functions from `cqed_sim.sim.extractors` with higher-level diagnostic structure.
- **`cqed_sim.plotting`**: functions like `plot_wigner_grid(...)`, `plot_fock_resolved_bloch_overlay(...)`, `plot_bloch_track(...)`, and `plot_relative_phase_track(...)` consume outputs from this module.
- **`cqed_sim.tomo`**: `true_fock_resolved_vectors(...)` delegates to `bloch_xyz_from_joint(...)` here.

## Limitations / Non-Goals

- Wigner negativity is computed by numerical integration on a grid; accuracy depends on grid resolution and the extent of the phase-space window.
- Phase diagnostics assume the joint state has non-negligible population in each Fock sector of interest; they may produce misleading values if a sector has near-zero population.
- This module does not perform state tomography or parameter estimation; it computes observables from known simulation states.
