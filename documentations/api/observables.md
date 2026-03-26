# API Reference — Observables (`cqed_sim.observables`)

The observables module provides post-simulation diagnostics: Bloch coordinates, Fock-resolved analysis, phase diagnostics, trajectory extraction, weakness metrics, and Wigner functions.

---

## Bloch (`observables.bloch`)

Re-exports from `sim.extractors`:

| Function | Description |
|---|---|
| `reduced_qubit_state(state)` | Partial trace to qubit |
| `reduced_cavity_state(state)` | Partial trace to cavity |
| `bloch_xyz_from_joint(state)` | Bloch vector from joint state |
| `cavity_moments(state, n_cav)` | Cavity mode moments |

---

## Fock-Resolved Diagnostics (`observables.fock`)

| Function | Signature | Description |
|---|---|---|
| `fock_resolved_bloch_diagnostics(track, max_n, probability_threshold)` | `-> dict` | Bloch vectors conditioned on Fock n across snapshot sequence |
| `relative_phase_family_diagnostics(track, max_n, probability_threshold, unwrap, coherence_threshold)` | `-> dict` | Ground and excited phase families with optional unwrapping |
| `conditional_phase_diagnostics(track, max_n, ...)` | `-> dict` | Excited-family phase only |
| `relative_phase_debug_values(state, max_n, ...)` | `-> dict` | Phase analysis for a single state |
| `wrapped_phase_error(simulated, ideal)` | `-> dict \| ndarray` | Wrapped phase difference |

---

## Phase Diagnostics (`observables.phases`)

```python
def relative_phase_diagnostics(track, max_n, threshold, unwrap=False) -> dict
```

Returns `{"labels", "traces", "amplitudes", "phase_mode", ...}` with flattened phase traces and labels like `"|g,0>"`, `"|e,1>"`.

---

## Trajectories (`observables.trajectories`)

```python
def bloch_trajectory_from_states(
    states: list, conditioned_n_levels=None, probability_threshold=1e-8,
) -> dict
```

Returns `{"x", "y", "z", "conditioned": {n: {"x", "y", "z", "probability", "valid"}}}`.

---

## Usage

```python
from cqed_sim.observables import (
    reduced_qubit_state,
    bloch_xyz_from_joint,
    fock_resolved_bloch_diagnostics,
    bloch_trajectory_from_states,
)

# Extract qubit Bloch vector from a joint qubit ⊗ cavity state
rho_q = reduced_qubit_state(result.final_state)
x, y, z = bloch_xyz_from_joint(result.final_state)

# Fock-resolved diagnostics over a gate track
diag = fock_resolved_bloch_diagnostics(track, max_n=5, probability_threshold=1e-6)

# Full trajectory from stored states
traj = bloch_trajectory_from_states(result.states, conditioned_n_levels=4)
```
```

Returns `{"x", "y", "z", "conditioned": {n: {"x", "y", "z", "probability", "valid"}}}`.

---

## Weakness Metrics (`observables.weakness`)

| Function | Description |
|---|---|
| `attach_weakness_metrics(reference_track, track)` | Add `wigner_negativity`, `fidelity_weakness_vs_a` arrays to track |
| `comparison_metrics(track_a, track_b)` | `{"x_rmse", "y_rmse", "z_rmse", "n_rmse", "final_fidelity"}` |

---

## Wigner (`observables.wigner`)

| Function | Description |
|---|---|
| `cavity_wigner(rho_c, ...)` | Re-export from extractors |
| `selected_wigner_snapshots(track, stride)` | Subsample Wigner snapshots (always includes first and last) |
| `wigner_negativity(snapshot)` | $\max(0.5(\int|W| - 1), 0)$. Returns NaN if wigner is None. |
