Gengyan Zhang, Yanbing Liu, James J. Raftery, and Andrew A. Houck, "Suppression of photon shot noise dephasing in a tunable coupling superconducting qubit," npj Quantum Information 3, 1 (2017). DOI: 10.1038/s41534-016-0002-2

## Summary

- This paper studies photon-shot-noise dephasing in a tunable-coupling cQED device and demonstrates suppressing dephasing by reducing the dispersive coupling to the readout cavity.
- Eq. (3) gives an analytic photon-shot-noise dephasing expression that interpolates beyond the weak-dispersive limit.
- The work emphasizes the roles of cavity linewidth, thermal photon occupation, and dispersive shift in limiting qubit coherence.

## Relevance to `cqed_sim`

- It is the direct reference for `thermal_photon_dephasing(..., exact=True)`.
- It supports exposing both the exact expression and common weak/strong limiting approximations.

## Notes for This Feature Pass

- The implementation uses angular-rate inputs for `kappa` and `chi`, consistent with the rest of the simulator's rate convention.
