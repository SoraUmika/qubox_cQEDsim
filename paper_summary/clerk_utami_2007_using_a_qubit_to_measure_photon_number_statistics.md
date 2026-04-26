A. A. Clerk and D. W. Utami, "Using a qubit to measure photon-number statistics of a driven thermal oscillator," Physical Review A 75, 042302 (2007). DOI: 10.1103/PhysRevA.75.042302

## Summary

- This theoretical paper analyzes how a dispersively coupled qubit senses photon-number statistics of a driven damped thermal oscillator.
- It extends qubit dephasing spectroscopy beyond simple zero-temperature coherent-drive limits and is a source for analytic photon-number-statistics dephasing expressions.
- Later cQED work cites this paper when using qubit dephasing to infer thermal photon populations.

## Relevance to `cqed_sim`

- It is a reference anchor for the exact thermal-photon dephasing expression exposed through `thermal_photon_dephasing(...)`.
- It supports treating cavity photon statistics as the intermediate quantity linking microwave bath occupation to qubit dephasing.

## Notes for This Feature Pass

- The implementation does not attempt general driven nonthermal photon statistics; it provides the thermal-bath expression and the common limiting approximations requested for cQED Lindblad usage.
