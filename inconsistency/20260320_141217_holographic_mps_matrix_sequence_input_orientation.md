# Holographic MPS Matrix-Sequence Input Orientation

## Summary

This report records the matrix-sequence input ambiguity discovered while
refactoring the holographic MPS convenience layer on 2026-03-20.

## Confirmed Issues

- What the inconsistency is:
  `HolographicChannel.from_right_canonical_mps(...)` and the new public
  Stinespring helper can accept either a rank-3 right-canonical tensor or a
  sequence of square MPS matrices, but a raw NumPy coercion of a matrix sequence
  collapses it into a rank-3 array with a different axis order.
- Where it appears:
  `cqed_sim/quantum_algorithms/holographic_sim/channel.py` and
  `cqed_sim/quantum_algorithms/holographic_sim/mps.py`.
- What components it affects:
  direct matrix-sequence construction, public Stinespring completion, and any
  user workflow that stores `V_sigma` matrices explicitly rather than as a single
  rank-3 tensor.
- Why it is inconsistent:
  the public API promises both input forms, but unqualified array coercion can
  reinterpret a `(physical_dim, bond_dim, bond_dim)` sequence as a
  `(bond_left, physical_dim, bond_right)` tensor and scramble the intended
  physical/bond axis meaning.
- Consequences:
  sequence-based callers can obtain the wrong channel or dense unitary even when
  the individual MPS matrices are correct.

## Status

- Fixed on 2026-03-20.

## Fix Record

- Fixed by adding an internal resolver that treats explicit matrix sequences and
  rank-3 tensors as separate input forms before any NumPy coercion.
- Regression coverage was added in
  `tests/quantum_algorithms/test_holographic_mps_validation.py` for both
  `HolographicChannel.from_right_canonical_mps(...)` and
  `right_canonical_tensor_to_stinespring_unitary(...)` on matrix-sequence input.