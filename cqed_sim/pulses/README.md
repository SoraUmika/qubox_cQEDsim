# `cqed_sim.pulses`

The `pulses` module defines the `Pulse` data type, standard envelope shapes, calibrated pulse builders for common cQED operations, and hardware distortion utilities. It is the pulse-definition layer of the library — pulses constructed here are later compiled into a channel timeline by `cqed_sim.sequence` and fed to the simulator in `cqed_sim.sim`.

## Relevance in `cqed_sim`

Pulses sit between the physical model (in `cqed_sim.core`) and the simulation runner (in `cqed_sim.sim`). The module ensures that:

- drive waveforms are parameterized in physical units consistent with the rest of the library,
- common gate types (rotations, displacements, SQR multitone drives, sidebands) can be constructed from calibrated parameters without manual envelope arithmetic,
- and the carrier-sign convention required by the simulator is maintained throughout.

## Main Capabilities

- **`Pulse`**: The core data class. Holds a channel label, carrier frequency, envelope samples, amplitude, phase, start time, and duration. The carrier sign convention is `carrier = -omega_transition(frame)`.
- **Standard envelopes**: `gaussian_envelope`, `square_envelope`, `cosine_rise_envelope`, `multitone_gaussian_envelope`, `normalized_gaussian`. All return normalized waveform arrays; `Pulse.amplitude` controls the overall scale.
- **Calibrated builders**: `build_rotation_pulse`, `build_displacement_pulse`, `build_sideband_pulse`, `build_sqr_multitone_pulse`. These take a gate spec and a dictionary of calibration parameters and return `(pulses, drive_ops, meta)` tuples ready for `SequenceCompiler`.
- **Amplitude formulas**: `displacement_square_amplitude`, `rotation_gaussian_amplitude`, `sqr_lambda0_rad_s`, `sqr_rotation_coefficient`, `sqr_tone_amplitude_rad_s`. Closed-form physical amplitude calculations used internally by the builders.
- **SQR tone construction**: `build_sqr_tone_specs`, `pad_parameter_array`, `pad_sqr_angles` for assembling multitone SQR waveforms from per-Fock angle arrays.
- **Hardware distortion**: `HardwareConfig` wraps optional hardware-level distortion models (e.g. IIR/FIR filters) that can be applied to pulse envelopes before simulation.
- **Strong-readout seeds**: `square_readout_seed`, `gaussian_readout_seed`, `ramped_readout_seed`, and `clear_readout_seed` build sampled readout envelopes for high-power readout studies.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `Pulse` | Core pulse data class |
| `build_rotation_pulse(gate, cal_params)` | Gaussian qubit rotation pulse |
| `build_displacement_pulse(gate, cal_params)` | Cavity displacement pulse |
| `build_sideband_pulse(gate, cal_params)` | Sideband interaction pulse |
| `build_sqr_multitone_pulse(gate, cal_params)` | SQR multitone drive |
| `gaussian_envelope(...)` | Standard Gaussian envelope |
| `square_envelope(...)` | Top-hat envelope |
| `multitone_gaussian_envelope(...)` | Multi-frequency Gaussian |
| `MultitoneTone` | Single-tone spec for multitone waveforms |
| `HardwareConfig` | Optional hardware distortion model |
| `clear_readout_seed(...)` | CLEAR-like kick-up, plateau, and depletion seed |

## Usage Guidance

```python
import numpy as np
from cqed_sim.io import RotationGate
from cqed_sim.pulses import build_rotation_pulse

gate = RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0)
pulses, drive_ops, meta = build_rotation_pulse(
    gate,
    {
        "duration_rotation_s": 100.0e-9,
        "rotation_sigma_fraction": 0.18,
    },
)
```

For a custom drive, construct `Pulse` directly:

```python
from cqed_sim.pulses import Pulse, gaussian_envelope
import numpy as np

dt = 2.0e-9
t_end = 200.0e-9
n = int(t_end / dt)
env = gaussian_envelope(n, sigma=0.18)
p = Pulse(
    channel="qubit",
    carrier=-2.0 * np.pi * 6.0e9,   # negative of transition frequency
    envelope=env,
    amplitude=2.0 * np.pi * 1.0e6,
    phase=0.0,
    t0=0.0,
    dt=dt,
)
```

## Important Assumptions / Conventions

- **Carrier sign**: the raw low-level `Pulse.carrier` must equal `−omega_transition(frame)`. This follows from the `exp(+i*(omega*t+phase))` drive convention used throughout the library. For user-facing positive drive-tone frequencies, prefer `drive_frequency_for_transition_frequency(...)`, `internal_carrier_from_drive_frequency(...)`, and `drive_frequency_from_internal_carrier(...)` from `cqed_sim.core`.
- **Amplitude units**: amplitudes are in `rad/s` (matching the Hamiltonian units). Builders return physically calibrated amplitudes based on the supplied calibration dictionaries.
- **Envelope normalization**: envelope arrays from the standard helpers are normalized (peak 1 or area-normalized depending on context); `Pulse.amplitude` scales the overall drive strength.
- **Time units**: `t0` and `dt` are in seconds.
- **Channel labels**: the standard channel labels are `"qubit"`, `"cavity"` / `"storage"`, `"readout"`, `"sideband"`. Multilevel variants use `TransmonTransitionDriveSpec` and `SidebandDriveSpec` from `cqed_sim.core`.
- **Strong-readout pulse seeds**: `clear.py` returns sampled complex envelopes in angular-frequency units. Each builder accepts an optional frequency-domain AWG transfer function so the simulated pulse can include command-chain filtering before it reaches the Hamiltonian.

## Relationships to Other Modules

- **`cqed_sim.sequence`**: takes a list of `Pulse` objects and compiles them onto a global time grid per channel.
- **`cqed_sim.sim`**: uses the compiled channel timelines and the `drive_ops` mapping to assemble the time-dependent Hamiltonian.
- **`cqed_sim.io`**: defines `RotationGate`, `DisplacementGate`, `SQRGate` that are passed to the builders.
- **`cqed_sim.core`**: provides frequency helpers needed to compute physically correct carrier values.

## Limitations / Non-Goals

- Pulses are ideal waveforms; they do not model AWG quantization or finite-bandwidth effects unless `HardwareConfig` is used explicitly.
- The builders assume a Gaussian-based pulse family for qubit rotations and flat-top or Gaussian multitone for SQR. Custom waveform shapes require constructing `Pulse` directly.
- The calibration amplitude formulas are closed-form estimates; they are not derived from full Floquet or dressing theory.
