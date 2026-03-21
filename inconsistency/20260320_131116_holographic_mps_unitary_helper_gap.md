# Holographic MPS Unitary Helper Gap

## Summary

This report records the helper-level API gap found while validating the
holographic simulator against known MPS states on 2026-03-20.

## Confirmed Issues

- What the inconsistency is:
  The holographic package exposes right-canonical MPS-to-channel construction,
  but it does not expose a package-level helper that completes a right-canonical
  MPS tensor into a dense Stinespring unitary suitable for the finite-sequence
  `holographicSim.py` execution path.
- Where it appears:
  `cqed_sim/quantum_algorithms/holographic_sim/mps.py`,
  `cqed_sim/quantum_algorithms/holographic_sim/channel.py`, and
  `cqed_sim/quantum_algorithms/holographic_sim/api.py`.
- What components it affects:
  validation workflows, downstream finite-chain holographic studies, and any
  user code that needs to move from `MatrixProductState` / right-canonical
  tensors into the legacy `U_list`-based simulator.
- Why it is inconsistent:
  The paper summary and module documentation both describe a purified /
  Stinespring unitary embedding as a core holographic primitive, but the current
  public implementation stops at channel construction from MPS data and does not
  provide the dense-unitary bridge for users who need explicit unitary lists.
- Consequences:
  Validation or downstream usage must duplicate null-space completion logic
  outside the package, which risks orientation mistakes and makes the intended
  MPS-to-unitary path harder to discover.

## Suspected / Follow-up Questions

- Should the missing bridge live as a free helper such as a
  right-canonical-tensor-to-unitary function, or as a method on
  `HolographicChannel` / `MatrixProductState`?
- Should the helper expose the physical reference-state convention explicitly,
  or should it standardize on the current `|0>` input used by
  `HolographicChannel.from_unitary(...)`?

## Status

- Fixed on 2026-03-20.
- Numerical validation itself passed for product, GHZ, W, and seeded random
  four-qubit states once a local test-only Stinespring completion helper was
  used.

## Fix Record

- Package-level API fix applied in
  `cqed_sim/quantum_algorithms/holographic_sim/mps.py` by adding
  `right_canonical_tensor_to_stinespring_unitary(...)` and
  `MatrixProductState.site_stinespring_unitary(...)`.
- Convenience MPS entry points were also added through
  `HolographicChannel.from_mps_state(...)`,
  `HolographicSampler.from_mps_state(...)`, and
  `HolographicMPSAlgorithm.from_mps_state(...)`.
- Regression coverage now uses the public helper in
  `tests/quantum_algorithms/test_holographic_mps_validation.py` and verifies the
  direct MPS-convenience path in
  `tests/quantum_algorithms/test_holographic_sim.py`.
