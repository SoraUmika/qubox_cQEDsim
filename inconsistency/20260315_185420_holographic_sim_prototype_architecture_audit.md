## Summary

This report records the inconsistencies found while refactoring the prototype
`cqed_sim/quantum_algorithms/holographic_sim` code into a reusable package on
2026-03-15.

## Confirmed Issues

- What the inconsistency is:
  The prototype lived inside the reusable library tree, but its actual
  structure was still notebook/script-style rather than package-style.
- Where it appeared:
  `cqed_sim/quantum_algorithms/holographic_sim/holographicSim.py` and
  `cqed_sim/quantum_algorithms/holographic_sim/mps.py`.
- What components it affected:
  public API shape, extensibility, testing, and documentation.
- Why it was inconsistent:
  The repository public modules are generally organized around clear
  dataclasses, result objects, and submodule boundaries, while the holographic
  prototype concentrated channel stepping, measurement logic, Monte Carlo
  execution, exact enumeration, and dataframe formatting in one file.
- Consequences:
  The module was difficult to document, validate, and extend toward holoVQE /
  holoQUADS without breaking users or duplicating logic.

- What the inconsistency is:
  The main holographic sampling path hardcoded qubit-system assumptions.
- Where it appeared:
  `SYS_DIM = 2` and fixed binary branch logic in
  `cqed_sim/quantum_algorithms/holographic_sim/holographicSim.py`.
- What components it affected:
  unitary stepping, exact enumeration, and observable handling.
- Why it was inconsistent:
  The scientific design in `paper_summary/holographic_quantum_algorithms.pdf`
  is framed in terms of generic physical dimension `Q`, bond dimension `chi`,
  and sliced geometries, not a permanently qubit-only physical register.
- Consequences:
  The prototype was harder to reuse for higher local dimension, multi-qubit
  physical registers, or model-inspired examples outside the specific ideal
  qubit case.

- What the inconsistency is:
  The prototype returned low-level tuples / arrays where the rest of the
  library now prefers named result/config objects.
- Where it appeared:
  Monte Carlo and exact sampling outputs in
  `cqed_sim/quantum_algorithms/holographic_sim/holographicSim.py`.
- What components it affected:
  interactive usage, downstream notebooks, diagnostics, and serialization.
- Why it was inconsistent:
  The repository newer API surfaces expose structured result containers with
  clear fields and validation metadata.
- Consequences:
  The interface was harder to read, harder to extend, and more fragile for
  future algorithm layers.

## Suspected / Open Questions

- The current `mps.py` uses a specific tensor orientation and "complete right
  isometry" construction that needed explicit documentation during the refactor.
  That mapping is now documented in the new channel and conventions docs.
- The public sampler API now supports finite explicit per-step channel
  sequences through `HolographicChannelSequence`. The remaining intentional
  limitation is narrower: `burn_in` retains its repeated-channel meaning and is
  therefore not defined for finite explicit sequences.

## Fix Update

- Status:
  Fixed on 2026-03-15 for the confirmed issues listed above.
- What was fixed:
  The monolithic prototype was replaced with a structured package centered on
  `HolographicChannel`, `PurifiedChannelStep`, `ObservableSchedule`,
  `HolographicSampler`, diagnostics modules, result objects, example channel
  constructors, and future-facing `holo_vqe.py` / `holoquads.py` scaffolding.
- Where it was fixed:
  `cqed_sim/quantum_algorithms/holographic_sim/`
  including `__init__.py`, `api.py`, `channel.py`, `channel_embedding.py`,
  `sampler.py`, `results.py`, `diagnostics.py`, `models/`, `holo_vqe.py`,
  `holoquads.py`, and the compatibility wrapper `holographicSim.py`.
- How the hardcoded qubit issue was addressed:
  The new core abstractions are parameterized by `physical_dim` and `bond_dim`.
  The legacy wrapper still defaults to `d=2` for backward compatibility, but
  the reusable package API no longer hardcodes a qubit-only physical register.
- How the tuple-heavy API issue was addressed:
  Public workflows now return named result/config objects such as
  `CorrelatorEstimate`, `ExactCorrelatorResult`, `BurnInSummary`, and
  `EnergyEstimate`, each with serialization helpers.
- Remaining concern:
  Fixed on 2026-03-27.
  A first-class public abstraction for arbitrary finite per-step channel
  sequences was added through
  `cqed_sim/quantum_algorithms/holographic_sim/step_unitary.py`,
  `cqed_sim/quantum_algorithms/holographic_sim/step_sequence.py`, and the
  generalized public sampler path in
  `cqed_sim/quantum_algorithms/holographic_sim/sampler.py`.
  The remaining narrower limitation is that `burn_in` retains its original
  repeated-channel meaning and is therefore intentionally defined only for
  single-channel translation-invariant workflows, not for finite explicit
  sequences.
