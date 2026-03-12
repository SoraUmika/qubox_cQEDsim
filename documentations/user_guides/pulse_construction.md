# Pulse Construction

Pulses describe the time-dependent drives applied to the system. Each `Pulse` is defined on a named channel and carries an envelope, carrier frequency, amplitude, and phase.

---

## The Pulse Dataclass

```python
from cqed_sim.pulses import Pulse

pulse = Pulse(
    channel="q",              # Drive channel name
    t0=0.0,                   # Start time (s)
    duration=100e-9,          # Duration (s)
    envelope=my_envelope_fn,  # Callable(t_rel) -> ndarray, or ndarray
    carrier=0.0,              # Carrier frequency (rad/s)
    phase=0.0,                # Phase offset (rad)
    amp=1.0,                  # Amplitude scaling
    drag=0.0,                 # DRAG coefficient
    label="my_pulse",         # Optional label
)
```

### Waveform Formula

$$\varepsilon(t) = \text{amp} \cdot \text{envelope}(t_{\text{rel}}) \cdot e^{i(\text{carrier} \cdot t + \text{phase})}$$

where $t_{\text{rel}} = (t - t_0) / \text{duration}$ ranges from 0 to 1.

If `drag ≠ 0`, a quadrature correction is added: the envelope derivative scaled by `drag` is added in the imaginary channel.

### Properties

- `t1` — end time: `t0 + duration`
- `sample(t)` — sample the pulse at arbitrary time points

---

## Built-in Envelopes

```python
from cqed_sim.pulses.envelopes import (
    square_envelope,
    gaussian_envelope,
    cosine_rise_envelope,
    normalized_gaussian,
    multitone_gaussian_envelope,
)
```

| Envelope | Signature | Description |
|---|---|---|
| `square_envelope` | `(t_rel) -> ndarray` | Constant 1.0 |
| `gaussian_envelope` | `(t_rel, sigma, center=0.5)` | Gaussian peak |
| `cosine_rise_envelope` | `(t_rel, rise_fraction=0.1)` | Flat-top with cosine edges |
| `normalized_gaussian` | `(t_rel, sigma_fraction)` | Gaussian normalized to unit area |
| `multitone_gaussian_envelope` | `(t_rel, duration_s, sigma_fraction, tone_specs)` | Multitone modulated Gaussian |

### Custom Envelopes

Any callable that maps `t_rel` (ndarray in [0, 1]) to a complex ndarray can be used:

```python
import numpy as np

def my_envelope(t_rel):
    return np.sin(np.pi * t_rel)

pulse = Pulse("q", 0.0, 100e-9, my_envelope, amp=0.5)
```

---

## Pulse Builders

Builders construct common pulse types from gate specifications and configuration parameters. All builders return a tuple: `(pulses, drive_ops, metadata)`.

### Qubit Rotation

```python
from cqed_sim.io import RotationGate
from cqed_sim.pulses import build_rotation_pulse

gate = RotationGate(index=0, name="x90", theta=np.pi / 2, phi=0.0)
pulses, drive_ops, meta = build_rotation_pulse(
    gate,
    {"duration_rotation_s": 100e-9, "rotation_sigma_fraction": 0.18},
)
```

Creates a normalized-Gaussian pulse on channel `"q"`. Drive mapping: `{"q": "qubit"}`.

Amplitude: $\Omega = \theta / (2T)$ under the RWA.

### Cavity Displacement

```python
from cqed_sim.io import DisplacementGate
from cqed_sim.pulses import build_displacement_pulse

gate = DisplacementGate(index=0, name="d1", re=1.0, im=0.5)
pulses, drive_ops, meta = build_displacement_pulse(
    gate,
    {"duration_displacement_s": 200e-9},
)
```

Creates a square-envelope pulse on channel `"storage"`. Drive mapping: `{"storage": "cavity"}`.

Amplitude: $\varepsilon = i\alpha / T$.

### Sideband Pulse

```python
from cqed_sim.core import SidebandDriveSpec
from cqed_sim.pulses import build_sideband_pulse

target = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red")
pulses, drive_ops, meta = build_sideband_pulse(
    target,
    duration_s=500e-9,
    amplitude_rad_s=2 * np.pi * 1e6,
    channel="sideband",
)
```

Builds an effective multilevel sideband pulse. Drive mapping: `{"sideband": SidebandDriveSpec(...)}`.

### SQR Multitone Pulse

```python
from cqed_sim.io import SQRGate
from cqed_sim.pulses import build_sqr_multitone_pulse

gate = SQRGate(index=0, name="sqr1", theta=(np.pi,) * 8, phi=(0.0,) * 8)
pulses, drive_ops, meta = build_sqr_multitone_pulse(
    gate, model,
    {"duration_sqr_s": 1e-6, "sqr_sigma_fraction": 0.18, "sqr_theta_cutoff": 1e-4},
    frame=frame,
)
```

Creates a multitone Gaussian pulse on channel `"q"` with per-manifold tone frequencies.

---

## Channel Names and Drive Operators

Pulse channels are mapped to physical operators via the `drive_ops` dictionary passed to the simulator:

```python
drive_ops = {
    "q": "qubit",          # Channel "q" drives the qubit
    "storage": "cavity",   # Channel "storage" drives the cavity
}
```

String targets use the model's `drive_coupling_operators()` to resolve the physical operator pair `(raising, lowering)`.

For multilevel ancilla or sideband drives, use structured targets:

```python
from cqed_sim.core import TransmonTransitionDriveSpec, SidebandDriveSpec

drive_ops = {
    "q": TransmonTransitionDriveSpec(lower_level=0, upper_level=2),  # |g⟩↔|f⟩
    "sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red"),
}
```

---

## Carrier Frequency Convention

Because the waveform convention is $e^{+i\omega t}$, a transition at rotating-frame frequency $\omega_{\text{tr}}$ requires:

$$\text{carrier} = -\omega_{\text{tr}}$$

Use the helpers:

```python
from cqed_sim.core import carrier_for_transition_frequency

omega_tr = model.manifold_transition_frequency(n=3, frame=frame)
carrier = carrier_for_transition_frequency(omega_tr)

pulse = Pulse("q", 0.0, 100e-9, gaussian_envelope, carrier=carrier, amp=0.5)
```
