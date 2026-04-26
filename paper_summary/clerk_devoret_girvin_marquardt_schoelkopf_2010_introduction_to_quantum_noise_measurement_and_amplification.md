A. A. Clerk, M. H. Devoret, S. M. Girvin, F. Marquardt, and R. J. Schoelkopf, "Introduction to quantum noise, measurement, and amplification," Reviews of Modern Physics 82, 1155-1208 (2010). DOI: 10.1103/RevModPhys.82.1155

## Summary

- This review is the standard superconducting-circuit-facing reference for quantum noise spectra, input-output fields, and amplifier/noise-temperature conventions.
- It distinguishes normally ordered emission/absorption spectra from symmetrized noise quantities, which is important when translating thermal microwave noise into master-equation rates.
- It provides the broader context for treating microwave lines and amplifiers as quantum noise environments rather than purely classical Johnson-noise sources.

## Relevance to `cqed_sim`

- It anchors the warning that `sym_noise_temperature(...)` is reporting-only and must not be substituted for the Bose occupation used in Lindblad rates.
- It supports keeping normally ordered bath occupation separate from calibrated spectrum or amplifier reporting conventions.

## Notes for This Feature Pass

- `cqed_sim.microwave_noise` exposes both `bose_occupation(...)` and `sym_noise_temperature(...)`, but only the former is intended for `NoiseSpec.nth*`.
