M. P. Silveri, J. A. Tuorila, E. V. Thuneberg, and G. S. Paraoanu, "Quantum systems under frequency modulation," Reports on Progress in Physics 80, 056002 (2017). DOI: 10.1088/1361-6633/aa5170

## Summary

- This review summarizes Floquet, sideband, multiphoton-resonance, and modulation physics across two-level systems, harmonic oscillators, and cQED-style coupled systems.
- Section 3.2 gives the most directly reusable analytical reference for sinusoidal longitudinal frequency modulation of a two-level system.
- The review makes explicit that the harmonic sideband weights for sinusoidal modulation are set by Bessel functions of the first kind, which is the key analytical target for the initial Floquet transition-strength validation pass.

## Relevance to `cqed_sim`

- It is the best existing cQED-focused review citation for the repository's new Floquet module because it connects canonical Floquet theory to superconducting-circuit modulation workflows.
- It supports validating `cqed_sim.floquet.compute_floquet_transition_strengths(...)` against a published analytical sideband formula instead of relying only on internal numerical consistency checks.
- It also provides a convenient reference surface for future cQED-specific Floquet validations such as multiphoton resonances, dynamic Stark shifts, and sideband activation.

## Notes for This Feature Pass

- The first validation task derived from this review focuses on Section 3.2, Eq. (45): the Bessel-weighted sideband amplitudes under sinusoidal modulation.
- The repository validation uses a closed two-level Floquet model with longitudinal modulation of the excited-state projector and a transverse probe operator, which is the simplest setting that matches the published analytical result.
- The review also cites the canonical foundational sources for Floquet theory, including Shirley 1965 and Sambe 1973, which should be used when future validation work needs original theory citations rather than review-level guidance.