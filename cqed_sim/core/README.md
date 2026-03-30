# `cqed_sim.core`

The `core` module is the physical model foundation of the `cqed_sim` library. It defines the Hilbert-space structure, system Hamiltonians, rotating-frame conventions, state preparation helpers, transition-frequency utilities, and ideal gate operators that the rest of the library builds on.

## Relevance in `cqed_sim`

Every other module depends on `core` either directly or through the simulation runner. The module establishes:

- how modes are dimensioned and ordered in the joint Hilbert space,
- what the Hamiltonian looks like in a given rotating frame,
- how to prepare joint-system states,
- how to compute transition frequencies and carrier frequencies from model parameters.

## Main Capabilities

- **System models**: `DispersiveTransmonCavityModel` (qubit + cavity), `DispersiveReadoutTransmonStorageModel` (qubit + storage + readout), and `UniversalCQEDModel` (generalized N-mode system with arbitrary bosonic modes and dispersive couplings).
- **Frame handling**: `FrameSpec` specifies the rotating frame for each mode channel, and the models generate frame-dressed Hamiltonians accordingly.
- **Transition frequencies**: helpers for qubit manifold frequencies, sideband frequencies, Rabi rates, and carrier-to-frequency conversion.
- **State preparation**: `prepare_state(...)`, `prepare_ground_state(...)`, and a suite of single-subsystem constructors (`fock_state`, `coherent_state`, `qubit_state`, `qubit_level`, `amplitude_state`, `density_matrix_state`, `vacuum_state`).
- **Energy spectra**: `compute_energy_spectrum(...)` and `plot_energy_levels(...)` for dressed-level diagnostics.
- **Ideal gates**: `qubit_rotation_xy`, `qubit_rotation_axis`, `displacement_op`, `snap_op`, `sqr_op`, and tensor embedding helpers.
- **Hilbert-space index utilities**: `qubit_cavity_dims`, `qubit_cavity_index`, `qubit_cavity_block_indices`, and three-mode equivalents.
- **Optional coupling terms**: `CrossKerrSpec`, `SelfKerrSpec`, `ExchangeSpec` for adding bosonic cross-Kerr, self-Kerr, or exchange interactions.
- **Multilevel drive specs**: `TransmonTransitionDriveSpec` and `SidebandDriveSpec` for targeting specific multilevel transitions in the pulse layer.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `UniversalCQEDModel` | Generalized N-mode model; preferred entry point for new code |
| `DispersiveTransmonCavityModel` | Two-mode convenience wrapper |
| `DispersiveReadoutTransmonStorageModel` | Three-mode convenience wrapper |
| `FrameSpec` | Specifies rotating-frame frequencies per channel |
| `prepare_state(model, spec)` | Constructs a joint initial state from per-subsystem specs |
| `StatePreparationSpec` | Declarative spec for `prepare_state` |
| `compute_energy_spectrum(model, frame, levels)` | Returns dressed energy levels |
| `manifold_transition_frequency(model, n, frame)` | n-th Fock-sector qubit frequency |
| `carrier_for_transition_frequency(...)` | Converts a rotating-frame transition frequency to the raw low-level `Pulse.carrier` value |
| `drive_frequency_for_transition_frequency(...)` | Converts a rotating-frame transition frequency plus a chosen frame frequency into a positive physical drive tone |
| `internal_carrier_from_drive_frequency(...)` | Converts a positive physical drive tone into the raw low-level `Pulse.carrier` |
| `TransmonModeSpec`, `BosonicModeSpec`, `DispersiveCouplingSpec` | Components for building `UniversalCQEDModel` |

## Usage Guidance

### Build a two-mode model

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec

model = DispersiveTransmonCavityModel(
    omega_c=2.0 * np.pi * 5.0e9,
    omega_q=2.0 * np.pi * 6.0e9,
    alpha=2.0 * np.pi * (-200.0e6),
    chi=2.0 * np.pi * (-2.84e6),
    kerr=2.0 * np.pi * (-2.0e3),
    n_cav=8,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
H_static = model.static_hamiltonian(frame=frame)
```

### Build a generalized multilevel model

```python
from cqed_sim.core import UniversalCQEDModel, TransmonModeSpec, BosonicModeSpec, DispersiveCouplingSpec

model = UniversalCQEDModel(
    transmon=TransmonModeSpec(omega=2*np.pi*6e9, dim=5, alpha=2*np.pi*(-200e6), label="qubit"),
    bosonic_modes=(
        BosonicModeSpec(label="storage", omega=2*np.pi*5e9, dim=12, kerr=2*np.pi*(-2e3)),
    ),
    dispersive_couplings=(DispersiveCouplingSpec(mode="storage", chi=2*np.pi*(-2.84e6)),),
)
```

### Prepare an initial state

```python
from cqed_sim.core import StatePreparationSpec, fock_state, qubit_state, prepare_state

psi0 = prepare_state(model, StatePreparationSpec(qubit=qubit_state("g"), storage=fock_state(3)))
```

## Important Assumptions / Conventions

- **Tensor ordering**: qubit first, storage/cavity second, readout third.
- **Basis**: `|g> = |0>`, `|e> = |1>`.
- **Frequencies**: all internal frequencies are in `rad/s`; times in `s`.
- **Drive waveform sign**: `exp(+i*(omega*t + phase))`, so the raw low-level `Pulse.carrier` satisfies `carrier = -omega_transition(frame)`. For user-facing code, prefer the positive physical tone-frequency helpers instead of carrying the raw sign yourself.
- **chi convention**: `chi` is the per-photon shift of the qubit transition frequency. Negative `chi` lowers the frequency with photon number.
- **`n_q` projector**: the transmon excitation projector is `b†b`; for a two-level qubit this equals `|e><e|`.
- **Energy spectrum**: `compute_energy_spectrum` always shifts energies so the vacuum state has energy 0. Use `frame=FrameSpec()` (lab frame) for physically interpretable ladder diagrams.

## Relationships to Other Modules

- `pulses` uses carrier/frequency helpers from `core` to build physically calibrated pulses.
- `sim` calls `model.static_hamiltonian(frame=...)` to assemble the time-dependent Hamiltonian.
- `measurement` and `tomo` take model parameters directly.
- `unitary_synthesis` wraps models through `CQEDSystemAdapter`.
- `calibration` and `calibration_targets` build on the models and frequency helpers.

## Limitations / Non-Goals

- Does not perform time evolution itself — that lives in `cqed_sim.sim`.
- Does not include pulse-level drive terms — those are assembled in `cqed_sim.sim.runner`.
- The dispersive model is a fixed-point approximation; it does not derive coupling from the microscopic transmon Hamiltonian. Use `cqed_sim.analysis.parameter_translation` to translate from bare parameters to dispersive parameters if needed.
- `UniversalCQEDModel` does not currently support non-dispersive (g-style) coupling or multimode exchange terms directly; use `CrossKerrSpec` / `ExchangeSpec` in `cqed_sim.sim` for those.

## References

- Physics conventions and sign derivations: `physics_and_conventions/physics_conventions_report.tex`
- Root `README.md` — contains worked examples for all three model constructors.
