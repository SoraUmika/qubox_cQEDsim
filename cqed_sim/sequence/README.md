# `cqed_sim.sequence`

The `sequence` module provides `SequenceCompiler`, which takes a list of `Pulse` objects and assembles them into a per-channel piecewise-constant timeline ready for the simulation runner.

## Relevance in `cqed_sim`

`SequenceCompiler` is the glue between the pulse-definition layer (`cqed_sim.pulses`) and the simulation layer (`cqed_sim.sim`). Without it, the simulator would require the caller to manually discretize and align pulse waveforms onto a common time grid. The compiler handles overlapping pulses, heterogeneous channel sets, and the time-sampling details that the solver expects.

## Main Capabilities

- **`SequenceCompiler`**: constructed with a global `dt` (time step in seconds). Accepts a list of `Pulse` objects and a `t_end` and returns a `CompiledSequence`.
- **`CompiledSequence`**: holds the assembled per-channel timelines and metadata about the total duration, time grid, and channel list.
- **`CompiledChannel`**: per-channel piecewise-constant waveform array, covering the full `[0, t_end]` interval.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `SequenceCompiler(dt)` | Compiler instance for a given time step |
| `SequenceCompiler.compile(pulses, t_end)` | Compile a pulse list into a `CompiledSequence` |
| `CompiledSequence` | Result of compilation: per-channel arrays + metadata |
| `CompiledChannel` | Single-channel compiled waveform |

## Usage Guidance

```python
from cqed_sim.sequence import SequenceCompiler

dt = 2.0e-9  # seconds
t_end = max(p.t0 + p.dt * len(p.envelope) for p in pulses) + dt

compiled = SequenceCompiler(dt=dt).compile(pulses, t_end=t_end)
```

The resulting `compiled` object is passed directly to `simulate_sequence(...)` or `prepare_simulation(...)` in `cqed_sim.sim`.

For performance-sensitive workflows that reuse the same compiled sequence across many initial states:

```python
from cqed_sim.sim import SimulationConfig, prepare_simulation

session = prepare_simulation(model, compiled, drive_ops, config=SimulationConfig(frame=frame))
result = session.run(initial_state)
```

## Important Assumptions / Conventions

- All pulses passed to a single `compile(...)` call share the same time origin (`t=0`).
- `dt` is the global simulation time step; it should match the intended solver step to avoid interpolation errors.
- Pulses on the same channel and time interval are summed (additive waveform superposition).
- The `t_end` argument controls the total simulation window. Pulses that extend beyond `t_end` will be truncated at compilation.

## Relationships to Other Modules

- **`cqed_sim.pulses`**: provides the `Pulse` objects consumed by the compiler.
- **`cqed_sim.sim`**: consumes `CompiledSequence` via `simulate_sequence(...)` and `prepare_simulation(...)`.

## Limitations / Non-Goals

- Does not perform frequency-domain analysis or check for spectral crowding between channels.
- Does not validate physical calibration of pulse amplitudes — that responsibility lies with the builders in `cqed_sim.pulses`.
- The compiler supports only piecewise-constant (sample-and-hold) waveforms; continuous-time envelopes are sampled at `dt` during `Pulse` construction.
