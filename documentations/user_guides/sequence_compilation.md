# Sequence Compilation

The `SequenceCompiler` takes a list of `Pulse` objects and produces a `CompiledSequence` — a sampled, processed timeline of drive waveforms ready for the simulator.

---

## Basic Usage

```python
from cqed_sim.sequence import SequenceCompiler

compiler = SequenceCompiler(dt=2e-9)
compiled = compiler.compile(pulses, t_end=102e-9)
```

| Parameter | Type | Description |
|---|---|---|
| `dt` | `float` | Global sample time step (seconds) |
| `hardware` | `dict[str, HardwareConfig] \| None` | Per-channel hardware distortion configs |
| `crosstalk_matrix` | `dict[str, dict[str, float]] \| None` | Channel-to-channel crosstalk coefficients |
| `enable_cache` | `bool` | Memoize compiled sequences for repeated calls |

### Compile Method

```python
compiled = compiler.compile(pulses, t_end=None)
```

- `pulses` — list of `Pulse` objects
- `t_end` — explicit end time; defaults to `max(pulse.t1 for pulse in pulses)`

---

## Compilation Pipeline

1. **Time grid construction** — uniform grid from 0 to `t_end` with step `dt`
2. **Per-pulse sampling** — apply timing quantization, carrier IF offset, sample envelope and accumulate into per-channel baseband
3. **Crosstalk mixing** — if a crosstalk matrix is configured, mix signals between source and destination channels
4. **Hardware processing** — per-channel: zero-order hold → lowpass filter → amplitude quantization → IQ distortion

---

## CompiledSequence

```python
from cqed_sim.sequence import CompiledSequence, CompiledChannel
```

```python
@dataclass
class CompiledChannel:
    baseband: np.ndarray     # Complex baseband after signal processing
    distorted: np.ndarray    # Complex baseband after IQ distortion
    rf: np.ndarray           # Real RF waveform (upconverted)

@dataclass
class CompiledSequence:
    tlist: np.ndarray                      # Time grid (starts at 0.0)
    dt: float                              # Sample step (s)
    channels: dict[str, CompiledChannel]   # Per-channel compiled waveforms
```

Access compiled waveforms:

```python
t = compiled.tlist
qubit_waveform = compiled.channels["q"].baseband
```

---

## Hardware Distortion

Add realistic hardware imperfections via `HardwareConfig`:

```python
from cqed_sim.pulses import HardwareConfig

hw = HardwareConfig(
    gain_i=1.01,                # I-channel gain imbalance
    gain_q=0.99,                # Q-channel gain imbalance
    quadrature_skew=0.02,       # IQ phase skew (rad)
    dc_i=0.001,                 # DC offset on I
    dc_q=-0.001,               # DC offset on Q
    amplitude_bits=14,          # DAC bit depth
    lowpass_bw=500e6,           # Lowpass bandwidth (Hz)
    timing_quantum=1e-9,        # Timing resolution (s)
)

compiler = SequenceCompiler(dt=2e-9, hardware={"q": hw})
compiled = compiler.compile(pulses)
```

### Available Distortion Parameters

| Parameter | Description |
|---|---|
| `gain_i`, `gain_q` | IQ gain imbalance |
| `quadrature_skew` | IQ phase skew (rad) |
| `dc_i`, `dc_q` | DC offsets |
| `image_leakage` | Image sideband leakage |
| `channel_gain` | Overall gain |
| `zoh_samples` | Zero-order hold samples |
| `lowpass_bw` | Lowpass bandwidth (Hz) |
| `amplitude_bits` | DAC quantization depth |
| `timing_quantum` | Timing resolution (s) |
| `detuning` | Extra frequency detuning (rad/s) |

---

## Crosstalk

Model cross-channel leakage:

```python
crosstalk = {
    "q": {"storage": 0.01},   # 1% of qubit signal leaks into storage
    "storage": {"q": 0.005},  # 0.5% of storage signal leaks into qubit
}

compiler = SequenceCompiler(dt=2e-9, crosstalk_matrix=crosstalk)
```
