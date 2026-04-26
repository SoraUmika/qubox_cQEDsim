S. Krinner, S. Storz, P. Kurpiers, P. Magnard, J. Heinsoo, R. Keller, J. Luetolf, C. Eichler, and A. Wallraff, "Engineering cryogenic setups for 100-qubit scale superconducting circuit systems," EPJ Quantum Technology 6, 2 (2019). DOI: 10.1140/epjqt/s40507-019-0072-0

## Summary

- This paper is the main cryogenic wiring reference for attenuation placement, thermal anchoring, heat load, and thermal photon occupations in superconducting-circuit fridge lines.
- Section 2.2.1 models attenuators as thermalizing beamsplitters and gives the cascade update for noise photon occupation through attenuator stages.
- It explicitly uses the Bose-Einstein occupation as the relevant dimensionless photon number at microwave frequencies.

## Relevance to `cqed_sim`

- It anchors the `PassiveLoss` and `NoiseCascade` scalar propagation convention in `cqed_sim.microwave_noise`.
- It supports the positive insertion-loss convention: a 20 dB attenuator transmits 1 percent of incident noise and emits 99 percent of its local blackbody occupation.
- It motivates returning normally ordered photon occupation as the bath quantity that should feed Lindblad `NoiseSpec.nth*` fields.

## Notes for This Feature Pass

- The implementation generalizes the paper's scalar attenuator cascade to frequency-vectorized calculations, sliced distributed lines, directional scalar loss, and passive S-matrix covariance propagation.
