Carlton M. Caves, "Quantum limits on noise in linear amplifiers," Physical Review D 26, 1817-1839 (1982). DOI: 10.1103/PhysRevD.26.1817

## Summary

- This paper is a foundational reference for quantum limits and added noise in linear bosonic amplifiers and passive/active linear transformations.
- It frames linear microwave components in terms of input and output bosonic modes plus additional noise modes required by quantum mechanics.
- Although focused on amplifiers, it is a standard anchor for understanding why passive lossy transformations must couple to environmental modes.

## Relevance to `cqed_sim`

- It supports the `PassiveSMatrixComponent` representation, where passive loss is encoded through the positive semidefinite defect `I - S S^dagger`.
- It motivates validating passivity before adding thermal bath covariance to multiport S-matrix propagation.

## Notes for This Feature Pass

- The implementation is restricted to passive S-matrix components; active amplifier noise is intentionally not modeled by `PassiveSMatrixComponent`.
