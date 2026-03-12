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
    distorted: np.ndarray   # Complex baseband after IQ distortion
    rf: np.ndarray          # Real RF waveform (upconverted)

@dataclass
class CompiledSequence:
    tlist: np.ndarray                    # Time grid (starts at 0.0)
    dt: float                            # Sample step (s)
    channels: dict[str, CompiledChannel] # Per-channel compiled waveforms
```
