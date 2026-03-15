# API Reference: Holographic Quantum Algorithms (`cqed_sim.quantum_algorithms.holographic_sim`)

This package implements the generic holographic channel-estimation viewpoint from
`paper_summary/holographic_quantum_algorithms.pdf` in a reusable API-oriented form.

!!! note "Scope"
    The package is generic in physical and bond Hilbert-space dimensions. It is
    not hardcoded to bosonic cQED models, even though it lives inside the
    `cqed_sim` repository.

---

## Main Abstractions

```python
channel = HolographicChannel.from_unitary(U, physical_dim=2, bond_dim=4)
channel = HolographicChannel.from_kraus(kraus_ops)
channel = HolographicChannel.from_right_canonical_mps(tensor)
```

```python
schedule = ObservableSchedule(
    [
        {"step": 10, "operator": Z},
        {"step": 14, "operator": X},
    ],
    total_steps=20,
)
```

```python
sampler = HolographicSampler(channel, burn_in=BurnInConfig(steps=50))
result = sampler.sample_correlator(schedule, shots=5000)
exact = sampler.enumerate_correlator(schedule)
```

Key user-facing objects:

- `HolographicChannel`
- `PurifiedChannelStep`
- `ObservableSchedule`
- `BurnInConfig`
- `BoundaryCondition`
- `HolographicSampler`
- `HolographicMPSAlgorithm`

---

## Channels

`HolographicChannel` stores the bond-space transfer channel in standard Kraus form

```python
rho -> sum_k K_k rho K_k^dagger
```

while also optionally retaining a dense joint unitary or right-canonical MPS data.

Diagnostics:

- `kraus_completeness_error()`
- `right_canonical_error()`
- `channel_diagnostics(channel)`
- `validate_trace_preservation(channel)`

---

## Schedules and Observables

`ObservableSchedule` makes insertions explicit instead of relying on ad hoc tuples.

- `ObservableInsertion(step=..., observable=...)`
- `PhysicalObservable(matrix=...)`
- convenience observables: `pauli_x()`, `pauli_y()`, `pauli_z()`, `identity(dim)`

`total_steps` controls how many channel iterations occur, including identity /
no-measurement steps.

---

## Estimation Paths

Monte Carlo:

- `HolographicSampler.sample_correlator(...)`
- returns `CorrelatorEstimate(mean, variance, stderr, ...)`

Exact small-system branch enumeration:

- `HolographicSampler.enumerate_correlator(...)`
- returns `ExactCorrelatorResult(mean, variance, branches, normalization_error, ...)`

Burn-in:

- `HolographicSampler.summarize_burn_in(...)`
- returns `BurnInSummary`

---

## MPS Connection

`MatrixProductState` provides lightweight helpers for:

- right-canonical conversion
- left-canonical conversion
- expectation-value checks
- conversion of a selected site tensor into `HolographicChannel`

This is the main bridge between the report's channel/MPS language and the
sampling API implemented here.

---

## Future-Facing Scaffolding

```python
objective = HoloVQEObjective([...])
program = HoloQUADSProgram([...])
```

- `HoloVQEObjective` combines correlator schedules into a minimal energy objective.
- `HoloQUADSProgram` and `TimeSlice` provide time-sliced schedule composition for future dynamics workflows.

---

## Legacy Compatibility

The old prototype-style functions remain available from:

```python
from cqed_sim.quantum_algorithms.holographic_sim.holographicSim import (
    holographic_sim,
    holographic_sim_cached,
    holographic_sim_bfs,
)
```

These wrappers now delegate to the new internal abstractions and should be
treated as compatibility utilities rather than the primary public API.
