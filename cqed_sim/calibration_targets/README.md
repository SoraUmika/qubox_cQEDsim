# `cqed_sim.calibration_targets`

The `calibration_targets` module provides simulator-backed implementations of standard qubit calibration protocols: spectroscopy, Rabi oscillations, Ramsey interferometry, T1 decay, T2 echo, and DRAG pulse tuning. Each routine runs a calibration sweep through the `cqed_sim` simulator and returns fitted parameters along with raw simulated data.

## Relevance in `cqed_sim`

Standard qubit calibration protocols are the first step in preparing a working quantum processor. This module provides simulator-backed versions of these protocols so that:

- model parameters can be verified against expected calibration signatures,
- calibration workflows can be prototyped and tested in simulation before hardware runs,
- and calibration-derived parameters can be used to initialize other library components.

This module is distinct from `cqed_sim.calibration`, which provides SQR gate calibration and multitone-drive validation routines.

## Main Capabilities

- **`run_spectroscopy(model, config)`**: Sweeps a probe frequency and returns the qubit transition frequency from the simulated dispersive response.
- **`run_rabi(model, config)`**: Sweeps drive amplitude or duration and returns the Rabi frequency and fitted amplitude.
- **`run_ramsey(model, config)`**: Ramsey interferometry sweep returning the qubit frequency detuning and T2* dephasing time.
- **`run_t1(model, config)`**: Exponential decay fit returning the qubit T1 relaxation time.
- **`run_t2_echo(model, config)`**: Hahn echo sweep returning the T2 coherence time.
- **`run_drag_tuning(model, config)`**: DRAG coefficient sweep returning the optimal DRAG parameter for minimizing leakage and phase errors.
- **`CalibrationResult`**: Shared result dataclass holding fitted parameters, uncertainties, and raw simulated data arrays.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `run_spectroscopy(model, config)` | Qubit spectroscopy sweep |
| `run_rabi(model, config)` | Rabi oscillation calibration |
| `run_ramsey(model, config)` | Ramsey frequency and T2* calibration |
| `run_t1(model, config)` | T1 relaxation calibration |
| `run_t2_echo(model, config)` | T2 echo coherence calibration |
| `run_drag_tuning(model, config)` | DRAG coefficient optimization |
| `CalibrationResult` | Common result type |

## Usage Guidance

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.calibration_targets import run_rabi, run_ramsey

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9, omega_q=2*np.pi*6e9,
    alpha=2*np.pi*(-200e6), chi=2*np.pi*(-2.84e6),
    kerr=2*np.pi*(-2e3), n_cav=8, n_tr=2,
)

rabi_result = run_rabi(model, config={"duration_s": 200e-9, "n_points": 50})
print("Rabi frequency:", rabi_result.fitted_params["rabi_frequency_hz"])

ramsey_result = run_ramsey(model, config={"max_delay_s": 10e-6, "n_points": 100})
print("T2*:", ramsey_result.fitted_params["t2_star_s"])
```

## Important Assumptions / Conventions

- All protocols use the `cqed_sim.sim` simulator internally; results reflect the model's dispersive Hamiltonian, not experimental hardware effects.
- Model parameters must be in `rad/s` (consistent with `cqed_sim.core`).
- Returned time scales (`T1`, `T2*`, `T2`) are in seconds.
- Fitted parameters in `CalibrationResult.fitted_params` use SI units unless otherwise noted.
- These routines simulate ideal qubit initialization (ground state) before each sweep point.

## Relationships to Other Modules

- **`cqed_sim.sim`**: all protocols call `simulate_sequence(...)` internally.
- **`cqed_sim.core`**: the model passed to each calibration routine is built using `DispersiveTransmonCavityModel` or equivalent.
- **`cqed_sim.pulses`**: calibration routines use `build_rotation_pulse(...)` internally for qubit drives.
- **`cqed_sim.calibration`**: provides SQR/multitone-specific calibration and is not the same as the standard single-qubit calibrations here.

## Limitations / Non-Goals

- The protocols currently support single-qubit (two-mode) systems. Three-mode or multi-qubit variants are not implemented.
- Results depend on the model accuracy; if the dispersive approximation breaks down for the chosen parameters, calibration curves may deviate from the ideal shapes.
- These are simulation-only routines. They do not interface with hardware control systems.
