A. P. Sears, A. Petrenko, G. Catelani, L. Sun, H. Paik, G. Kirchmair, L. Frunzio, L. I. Glazman, S. M. Girvin, and R. J. Schoelkopf, "Photon shot noise dephasing in the strong-dispersive limit of circuit QED," Physical Review B 86, 180504(R) (2012). DOI: 10.1103/PhysRevB.86.180504

## Summary

- This experiment studies qubit dephasing caused by random photon arrival and departure events in the strong-dispersive limit of cQED.
- In the strong-dispersive, low-occupation regime, each photon event can effectively measure the qubit state, leading to the intuitive scaling `Gamma_phi ~= kappa*nbar`.
- The paper connects residual thermal photons and filtering/thermalization quality to observed coherence limits.

## Relevance to `cqed_sim`

- It anchors the `thermal_photon_dephasing(..., approximation="strong_low_occupation")` helper.
- It motivates using the propagated microwave bath occupation to estimate coherence impacts from residual photons.

## Notes for This Feature Pass

- The exact interpolation formula in the implementation is cited to Zhang et al. and Clerk/Utami; this paper supplies the strong-dispersive physical limit and experimental motivation.
