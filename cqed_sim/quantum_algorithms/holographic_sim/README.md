# Holographic Quantum Algorithms

This subpackage provides a generic, reusable foundation for the holographic
quantum-algorithm architecture described in
`paper_summary/holographic_quantum_algorithms.pdf`.

## Core vocabulary

- `physical register`: the small register prepared, coupled to the bond system,
  and optionally measured at each step
- `bond register`: the persistent Hilbert space that carries the effective
  many-body or MPS boundary information
- `holographic channel`: the CPTP map induced on the bond register by preparing
  the physical register in a reference state, applying a joint unitary, and
  tracing or conditioning on the physical subsystem
- `burn-in`: repeated channel application before measurements, used to approach
  bulk or steady-state behavior
- `observable schedule`: explicit insertion plan for where physical observables
  are measured during a holographic run

## Main public entry points

- `HolographicChannel`
- `ObservableSchedule`
- `HolographicSampler`
- `HolographicMPSAlgorithm`
- `HoloVQEObjective`
- `HoloQUADSProgram`

## Terminology mapping from the report

| Report concept | Implementation |
|---|---|
| bond-space transfer channel | `HolographicChannel` |
| purified / Stinespring embedding | `PurifiedChannelStep` |
| measurement insertion pattern | `ObservableSchedule` |
| holographic Monte Carlo estimator | `HolographicSampler.sample_correlator(...)` |
| exact small-system branch table | `HolographicSampler.enumerate_correlator(...)` |
| right-canonical MPS tensor / transfer matrix | `MatrixProductState`, `HolographicChannel.from_right_canonical_mps(...)` |
| holoVQE energy decomposition | `HoloVQEObjective` + `EnergyTerm` |
| holoQUADS time slicing | `HoloQUADSProgram` + `TimeSlice` |

## Scope implemented now

- dense-unitary, Kraus, and right-canonical MPS channel construction
- Monte Carlo and exact branch enumeration
- explicit burn-in and right-boundary postselection
- diagnostics for unitarity, completeness, trace preservation, and normalization
- example channel constructors and spin-inspired helpers
- backward-compatible wrappers in `holographicSim.py`

## Intentionally deferred

- full holographic variational optimization loops
- noisy hardware backends beyond the ideal purified channel abstraction
- general non-translation-invariant public channel sequences
- large-scale MPS tooling beyond the included right-canonical helpers

The current design keeps those extensions natural without hardcoding the package
to any one hardware model or paper benchmark.
