# `cqed_sim.map_synthesis`

`cqed_sim.map_synthesis` is the preferred public namespace for the synthesis stack previously exposed as `cqed_sim.unitary_synthesis`.

The package still optimizes unitaries, but it also supports reduced-state, isometry, channel, observable, and trajectory targets. The new namespace reflects that broader scope more accurately.

## Preferred Entry Point

- `QuantumMapSynthesizer`

## Compatibility

- `cqed_sim.unitary_synthesis` remains available for backward compatibility during the transition period.
- Existing code using `UnitarySynthesizer` continues to work, but that namespace is scheduled for deprecation.

## Implementation Note

This package currently re-exports the existing synthesis implementation from `cqed_sim.unitary_synthesis` so the transition can happen without breaking current workflows.