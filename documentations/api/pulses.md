# API Reference — Pulse System (`cqed_sim.pulses`)

The pulse module provides the `Pulse` dataclass, analytic envelope functions, high-level pulse builders, calibration formulas, and hardware distortion models.

---

## Pulse

**Module path:** `cqed_sim.pulses.pulse.Pulse`

```python
@dataclass(frozen=True)
class Pulse:
    channel: str                          # Drive channel name
    t0: float                             # Start time (s)
    duration: float                       # Duration (s)
    envelope: Callable[[ndarray], ndarray] | ndarray  # Analytic or pre-sampled
    carrier: float = 0.0                  # Carrier frequency (rad/s)
    phase: float = 0.0                    # Phase offset (rad)
    amp: float = 1.0                      # Amplitude scaling
    drag: float = 0.0                     # DRAG coefficient
    sample_rate: float | None = None      # For discrete envelopes (Hz)
    label: str | None = None              # Optional identifier
```

| Property / Method | Returns | Description |
|---|---|---|
| `t1` | `float` | End time: `t0 + duration` |
| `sample(t)` | `ndarray` | Sample pulse at arbitrary time points |

### Waveform Formula

$$\varepsilon(t) = \text{amp} \cdot \text{envelope}(t_{\text{rel}}) \cdot e^{i(\text{carrier} \cdot t + \text{phase})}$$

where $t_{\text{rel}} = (t - t_0) / \text{duration}$.

**Sign convention:** exp(+iωt) throughout the repository. The resonant rotating-frame transition frequency is `-carrier`. Use `carrier_for_transition_frequency(...)` when you want the detuning axis to match the physical transition.

If `drag ≠ 0`, a quadrature correction is added: the envelope derivative scaled by `drag` is added in the imaginary channel.

---

## Envelopes

**Module path:** `cqed_sim.pulses.envelopes`

| Function | Signature | Description |
|---|---|---|
| `square_envelope(t_rel)` | `ndarray -> ndarray` | Constant 1.0 |
| `gaussian_envelope(t_rel, sigma, center=0.5)` | `-> ndarray` | $\exp(-(t-c)^2 / (2\sigma^2))$ |
| `cosine_rise_envelope(t_rel, rise_fraction=0.1)` | `-> ndarray` | Flat-top with cosine edges |
| `normalized_gaussian(t_rel, sigma_fraction)` | `-> ndarray` | Gaussian with unit area |
| `gaussian_area_fraction(sigma_fraction, n_pts=4097)` | `-> float` | Numerical area of Gaussian |
| `multitone_gaussian_envelope(t_rel, duration_s, sigma_fraction, tone_specs)` | `-> ndarray` | Multi-tone Gaussian modulated envelope |

### MultitoneTone

```python
@dataclass(frozen=True)
class MultitoneTone:
    manifold: int          # Fock level n
    omega_rad_s: float     # Tone frequency (rad/s)
    amp_rad_s: float       # Tone amplitude (rad/s)
    phase_rad: float       # Tone phase (rad)
```

Multitone envelope formula:

$$w(t) = \text{env}(t_{\text{rel}}) \cdot \sum_n a_n \, e^{i(\phi_n + \omega_n \cdot t)}$$

---

## Pulse Builders

**Module path:** `cqed_sim.pulses.builders`

All builders return `(pulse_list, drive_operator_mapping, metadata)`.

### `build_displacement_pulse`

```python
def build_displacement_pulse(
    gate: DisplacementGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]
```

**Config keys:** `"duration_displacement_s"` (float, seconds).

Creates a square-envelope pulse on channel `"storage"`. Drive mapping: `{"storage": "cavity"}`. Amplitude: $\varepsilon = i\alpha / T$.

### `build_rotation_pulse`

```python
def build_rotation_pulse(
    gate: RotationGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]
```

**Config keys:** `"duration_rotation_s"`, `"rotation_sigma_fraction"`.

Creates a normalized-Gaussian pulse on channel `"q"`. Drive mapping: `{"q": "qubit"}`. Amplitude: $\Omega = \theta / (2T)$.

### `build_sideband_pulse`

```python
def build_sideband_pulse(
    target: SidebandDriveSpec,
    *,
    duration_s: float,
    amplitude_rad_s: float,
    channel: str = "sideband",
    carrier: float = 0.0,
    phase: float = 0.0,
    sigma_fraction: float | None = None,
    label: str | None = None,
) -> tuple[list[Pulse], dict[str, SidebandDriveSpec], dict[str, Any]]
```

Builds an effective multilevel sideband pulse. The target specifies the ancilla transition, bosonic mode, and red/blue sideband.

### `build_sqr_multitone_pulse`

```python
def build_sqr_multitone_pulse(
    gate: SQRGate,
    model: DispersiveTransmonCavityModel,
    config: Mapping[str, Any],
    *,
    frame: FrameSpec | None = None,
    calibration: SQRCalibrationResult | None = None,
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]
```

**Config keys:** `"duration_sqr_s"`, `"sqr_sigma_fraction"`, `"sqr_theta_cutoff"`, optionally `"use_rotating_frame"`, `"fock_fqs_hz"`.

---

## Calibration Formulas

**Module path:** `cqed_sim.pulses.calibration`

| Function | Signature | Description |
|---|---|---|
| `displacement_square_amplitude(alpha, duration_s)` | `(complex, float) -> complex` | $\varepsilon = i\alpha / T$ |
| `rotation_gaussian_amplitude(theta, duration_s)` | `(float, float) -> float` | $\Omega = \theta / (2T)$ |
| `sqr_lambda0_rad_s(duration_s)` | `(float) -> float` | $\lambda_0 = \pi / (2T)$ |
| `sqr_rotation_coefficient(theta, d_lambda_norm)` | `-> float` | $s = \theta/\pi + d_\lambda$ |
| `sqr_tone_amplitude_rad_s(theta, duration_s, d_lambda_norm)` | `-> float` | $a = \lambda_0 \cdot s$ |
| `pad_parameter_array(values, n_cav)` | `-> ndarray` | Pad/truncate to n_cav |
| `build_sqr_tone_specs(model, frame, thetas, phis, duration_s, ...)` | `-> list[MultitoneTone]` | Build tone specs per active manifold |

**SQR frequency convention:** tone frequencies are the **negative** of the manifold transition frequency in the rotating frame, aligning with the exp(+iωt) waveform convention.

---

## HardwareConfig

**Module path:** `cqed_sim.pulses.hardware.HardwareConfig`

Models IQ distortion, quantization, and filtering for realistic waveform generation.

```python
@dataclass(frozen=True)
class HardwareConfig:
    lo_freq: float = 0.0             # LO frequency (rad/s)
    if_freq: float = 0.0             # IF frequency (rad/s)
    gain_i: float = 1.0              # I-channel gain
    gain_q: float = 1.0              # Q-channel gain
    quadrature_skew: float = 0.0     # IQ phase skew (rad)
    dc_i: float = 0.0                # DC offset on I
    dc_q: float = 0.0                # DC offset on Q
    image_leakage: float = 0.0       # Image sideband leakage
    channel_gain: float = 1.0        # Overall channel gain
    zoh_samples: int = 1             # Zero-order hold samples
    lowpass_bw: float | None = None  # Lowpass bandwidth (Hz)
    detuning: float = 0.0            # Extra frequency detuning (rad/s)
    timing_quantum: float | None = None  # Timing resolution (s)
    amplitude_bits: int | None = None    # DAC bit depth
```

### Hardware Processing Helpers

| Function | Description |
|---|---|
| `apply_timing_quantization(t0, quantum)` | Round time to nearest quantum |
| `apply_zoh(x, zoh_samples)` | Zero-order hold interpolation |
| `apply_first_order_lowpass(x, dt, bw)` | First-order IIR lowpass filter |
| `apply_amplitude_quantization(x, bits)` | DAC quantization of I/Q |
| `apply_iq_distortion(baseband, t, hw)` | Full IQ distortion chain → (distorted, RF) |
| `image_ratio_db(gain_i, gain_q)` | Image suppression in dB |
