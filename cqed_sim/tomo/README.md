# `cqed_sim.tomo`

The `tomo` module provides tomography routines for cQED simulations. It implements AllXY pulse sequences for qubit calibration diagnostics, Fock-resolved qubit state tomography, leakage-matrix calibration, and pulse-calibration helpers used as building blocks for multi-protocol tomographic workflows.

## Relevance in `cqed_sim`

Tomography is essential for diagnosing and benchmarking quantum gate operations in simulation. This module is relevant when:

- diagnosing qubit calibration quality via AllXY sequence simulation,
- measuring the qubit Bloch vector conditioned on each Fock sector (Fock-resolved tomography),
- quantifying leakage out of the qubit computational subspace,
- or extracting tomographic state vectors for gate-error analysis.

## Main Capabilities

- **`run_all_xy(model, cal, device)`**: Runs the standard 21-point AllXY gate sequence on the simulated qubit and returns the per-sequence X/Y expectation values. AllXY is a coarse calibration diagnostic that indicates over/under-rotation and phase errors.
- **`autocalibrate_all_xy(model, device)`**: Automatically calibrates a `QubitPulseCal` from the model and runs AllXY.
- **`selective_qubit_drive_frequency(model, n)`**: Returns the positive physical qubit drive frequency for the Fock-sector-selective tag tone.
- **`selective_pi_pulse(model, cal, n, device)`**: Constructs a π-pulse selective to Fock sector `n`, translating that positive drive frequency into the raw internal carrier expected by the runtime.
- **`run_fock_resolved_tomo(model, cal, device, n_sectors)`**: Runs Fock-resolved qubit tomography: measures the qubit Bloch vector conditioned on each Fock number sector up to `n_sectors`. Returns a `FockTomographyResult`.
- **`simulation_config`**: Optional runtime config accepted by the tomography helpers for QuTiP controls such as `nsteps` and `solver_options`.
- **`true_fock_resolved_vectors(state, model)`**: Computes the exact conditioned Bloch vectors from a simulation state, without running additional simulations.
- **`calibrate_leakage_matrix(model, cal, device)`**: Calibrates the leakage matrix relating population in higher transmon levels to readout outcomes.
- **`ALL_XY_21`**: The canonical list of 21 AllXY gate pairs.
- **`DeviceParameters`**: Device-level parameter container used across tomography routines.
- **`QubitPulseCal`**: Qubit pulse calibration dataclass (amplitude, frequency, duration, DRAG coefficient).
- **`FockTomographyResult`**: Result of Fock-resolved tomography: per-sector Bloch vectors and fidelity data.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `run_all_xy(model, cal, device)` | AllXY calibration diagnostic |
| `autocalibrate_all_xy(model, device)` | Auto-calibrated AllXY |
| `run_fock_resolved_tomo(model, cal, device, n_sectors)` | Fock-resolved Bloch tomography |
| `true_fock_resolved_vectors(state, model)` | Exact conditioned Bloch vectors from state |
| `selective_qubit_drive_frequency(model, n)` | Positive physical tag-tone frequency for Fock sector `n` |
| `selective_pi_pulse(model, cal, n, device)` | Fock-sector-selective π-pulse |
| `calibrate_leakage_matrix(model, cal, device)` | Leakage matrix calibration |
| `FockTomographyResult` | Result type for Fock-resolved tomo |
| `QubitPulseCal` | Qubit pulse calibration parameters |
| `DeviceParameters` | Device parameter container |

## Usage Guidance

### AllXY diagnostic

```python
from cqed_sim.tomo import autocalibrate_all_xy, DeviceParameters

device = DeviceParameters(model=model, frame=frame)
xy_result = autocalibrate_all_xy(model, device)
# xy_result contains per-sequence X and Y expectation values
```

### Fock-resolved tomography

```python
from cqed_sim.tomo import run_fock_resolved_tomo, QubitPulseCal, DeviceParameters

cal = QubitPulseCal(
    amplitude=2*np.pi*1.0e6,
    frequency=model.omega_q,
    duration=100e-9,
    drag=0.0,
)
device = DeviceParameters(model=model, frame=frame)
tomo_result = run_fock_resolved_tomo(model, cal, device, n_sectors=8)
for n, bloch in enumerate(tomo_result.bloch_vectors):
    print(f"Fock sector {n}: {bloch}")
```

### Exact conditioned Bloch vectors from a simulation state

```python
from cqed_sim.tomo import true_fock_resolved_vectors

bloch_vectors = true_fock_resolved_vectors(result.final_state, model)
```

## Important Assumptions / Conventions

- AllXY sequences use the standard 21-pair protocol from the cQED gate calibration literature.
- Fock-resolved tomography measures the qubit state conditioned on each Fock sector using selective pulses; it requires that the selective π-pulses are well-calibrated to the `chi` shift of the model.
- Leakage-matrix calibration assumes a multilevel transmon and uses level-selective operations; it is not meaningful for a strictly two-level qubit model.
- The `DeviceParameters` helper in `cqed_sim.tomo.device` stores frequencies in **Hz** and converts to **rad/ns** via `to_model()`. This is a helper-specific convenience path for tomography workflows that use nanosecond time scales. The underlying `cqed_sim` model layer remains unit-coherent, so this does not mean the core simulator requires nanoseconds globally. When using `DeviceParameters.to_model()`, keep the surrounding tomography workflow internally consistent by expressing pulse durations, `t_end`, and `dt` in nanoseconds as well.

## Relationships to Other Modules

- **`cqed_sim.sim`**: `run_all_xy`, `run_fock_resolved_tomo`, and `calibrate_leakage_matrix` call `simulate_sequence(...)` internally.
- **`cqed_sim.core`**: uses `manifold_transition_frequency(...)` for sector-selective frequency targeting.
- **`cqed_sim.pulses`**: uses `build_rotation_pulse(...)` for qubit drives and selective pulses.
- **`cqed_sim.observables`**: `true_fock_resolved_vectors(...)` delegates to `conditioned_bloch_xyz(...)` from the observables/extractors layer.
- **`cqed_sim.plotting`**: `plot_fock_resolved_bloch_overlay(...)` and related functions can visualize `FockTomographyResult` outputs.

## Limitations / Non-Goals

- Full quantum state tomography (reconstructing a density matrix from operator measurements) is not implemented; this module focuses on qubit-subsystem and Fock-resolved diagnostics.
- Does not implement process tomography (gate set tomography).
- The AllXY protocol is a calibration diagnostic, not a fidelity estimator; it does not quantify the full gate error rate.
- Fock-resolved tomography assumes the qubit and cavity are dispersively coupled; it is not designed for strongly coupled or resonant systems.
