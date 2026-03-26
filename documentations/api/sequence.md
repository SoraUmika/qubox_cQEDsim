# API Reference — Sequence Compilation (`cqed_sim.sequence`)

The sequence module compiles high-level `Pulse` objects into time-discretized waveforms with optional hardware distortion.

---

## SequenceCompiler

**Module path:** `cqed_sim.sequence.scheduler.SequenceCompiler`

```python
class SequenceCompiler:
    def __init__(
        self,
        dt: float,
        hardware: dict[str, HardwareConfig] | None = None,
        crosstalk_matrix: dict[str, dict[str, float]] | None = None,
        enable_cache: bool = False,
    )
```

| Parameter | Type | Description |
|---|---|---|
| `dt` | `float` | Global sample time step (seconds) |
| `hardware` | `dict[str, HardwareConfig] \| None` | Per-channel hardware configs |
| `crosstalk_matrix` | `dict[str, dict[str, float]] \| None` | source → {dest: coefficient} |
| `enable_cache` | `bool` | Memoize compiled sequences |

### `compile`

```python
def compile(self, pulses: list[Pulse], t_end: float | None = None) -> CompiledSequence
```

**Processing pipeline:**

1. Build uniform time grid from 0 to max pulse end (or explicit `t_end`)
2. Per-pulse: apply timing quantization, carrier IF offset, sample and accumulate to baseband
3. Apply crosstalk mixing between channels if configured
4. Per-channel hardware processing: ZOH → lowpass → amplitude quantization → IQ distortion
5. Return `CompiledSequence` with per-channel baseband, distorted, and RF waveforms

---

## CompiledSequence / CompiledChannel

```python
@dataclass
class CompiledChannel:
    baseband: np.ndarray    # Complex baseband after signal processing
    distorted: np.ndarray   # After hardware distortion chain
    rf: np.ndarray          # RF waveform (baseband × carrier)

@dataclass
class CompiledSequence:
    tlist: np.ndarray                      # Time grid
    dt: float                              # Step size
    channels: dict[str, CompiledChannel]   # Per-channel waveforms
```

---

## HardwareContext Integration

The compiler supports an optional `HardwareContext` for higher-level transfer chain modeling (cable/filter/calibration effects via `ControlLine` objects). This is applied after the per-channel `HardwareConfig` step.

```python
compiler = SequenceCompiler(
    dt=0.2e-9,
    hardware={"q": hw_config},
    hardware_context=my_hardware_context,  # optional
)
```

See [Hardware Pipeline](hardware.md) for details on `ControlLine` and `HardwareContext`.

---

## Usage

```python
from cqed_sim.pulses import Pulse, GaussianEnvelope
from cqed_sim.sequence import SequenceCompiler

# Define a pulse
pulse = Pulse(
    channel="q",
    t0=0.0,
    duration=100e-9,
    envelope=GaussianEnvelope(sigma=25e-9),
    carrier=-6.15e9 * 2 * 3.14159,
    amp=0.1,
    label="pi_pulse",
)

# Compile
compiler = SequenceCompiler(dt=0.2e-9)
compiled = compiler.compile([pulse])

# Inspect waveforms
import matplotlib.pyplot as plt
ch = compiled.channels["q"]
plt.plot(compiled.tlist * 1e9, ch.baseband.real, label="I")
plt.plot(compiled.tlist * 1e9, ch.baseband.imag, label="Q")
plt.xlabel("Time (ns)")
plt.legend()
```
    distorted: np.ndarray   # Complex baseband after IQ distortion
    rf: np.ndarray          # Real RF waveform (upconverted)

@dataclass
class CompiledSequence:
    tlist: np.ndarray                    # Time grid (starts at 0.0)
    dt: float                            # Sample step (s)
    channels: dict[str, CompiledChannel] # Per-channel compiled waveforms
```
