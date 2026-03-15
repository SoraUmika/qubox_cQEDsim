## Summary

This report documents the architectural inconsistency addressed during the
`UnitarySynthesizer` future-proofing refactor on 2026-03-15.

## Confirmed Issue

- What the inconsistency is:
  The synthesis engine exposed flexible primitives, targets, and optimization
  backends, but the orchestration layer still depended directly on cQED model
  objects for Hilbert-space inference, waveform execution, and robust
  parameter-override evaluation.
- Where it appears:
  `cqed_sim/unitary_synthesis/optim.py`,
  `cqed_sim/unitary_synthesis/sequence.py`,
  and the legacy gateset-construction path used by
  `UnitarySynthesizer._build_initial_sequence(...)`.
- What components it affects:
  `UnitarySynthesizer`, waveform-backed `PrimitiveGate`, `GateSequence`
  dimension inference, and any workflow using `ParameterDistribution`.
- Why it is inconsistent:
  The public synthesis API had already been broadened beyond a single cQED use
  case, but the internal execution path still assumed direct access to a cQED
  model instead of an abstract system backend.
- Consequences:
  This made future non-cQED integrations harder, left the optimizer with hidden
  hardware assumptions, and prevented a clean backend boundary for future
  universal system adapters.

## Suspected / Open Questions

- No additional convention or physics inconsistency was identified in this
  refactor pass. The change is architectural and API-facing rather than a
  physics-model change.

## Fix Update

- Fixed on:
  2026-03-15
- What was fixed:
  Added a `QuantumSystem` abstraction and `CQEDSystemAdapter`, updated
  `UnitarySynthesizer` to resolve and use a system backend, routed sequence
  simulation through that interface, and moved cQED-specific default gateset
  construction behind the system layer.
- Files updated:
  `cqed_sim/unitary_synthesis/systems.py`,
  `cqed_sim/unitary_synthesis/optim.py`,
  `cqed_sim/unitary_synthesis/sequence.py`,
  `cqed_sim/unitary_synthesis/backends.py`,
  `cqed_sim/unitary_synthesis/__init__.py`
- Remaining concerns:
  Lower-level compatibility shims still allow direct `model` access in some
  sequence/backends helpers for backward compatibility, but the synthesizer
  itself now runs through the system interface.
