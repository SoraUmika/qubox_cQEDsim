Z. Wang, S. Shankar, Z. K. Minev, P. Campagne-Ibarcq, A. Narla, and M. H. Devoret, "Cavity Attenuators for Superconducting Qubits," Physical Review Applied 11, 014031 (2019). DOI: 10.1103/PhysRevApplied.11.014031

## Summary

- This paper introduces a band-pass microwave attenuator based on a dissipative cavity thermalized to the dilution-refrigerator mixing chamber.
- The motivation is residual thermal photon population in readout resonators, typically suspected to enter through weakly protected input and output ports.
- A cold dissipative cavity attenuator reduces the readout mode's effective thermal occupation by coupling it more strongly to a cold internal bath than to hot external line noise.
- In the reported brass implementation, the internal-to-external coupling ratio is approximately six, so the simple cold-bath model predicts about a sevenfold reduction in the readout-mode thermal occupation when the internal bath occupation is negligible.
- The paper reports Hahn-echo coherence approaching the relaxation limit and a noise-induced-dephasing upper bound near `2e-4` residual photons in the fundamental readout mode.

## Relevance to `cqed_sim`

- It anchors `EffectiveCavityAttenuator`, `TwoModeCavityAttenuatorModel`, and the design sweeps for cold-bath coupling ratio.
- It motivates the explicit distinction between lossless filters, which reject selected transmitted noise but do not thermalize the in-band mode, and cold dissipative attenuators, which add a real cold bath at the readout frequency.
- It motivates the synthetic noise-induced-dephasing extraction helpers that fit the offset in dephasing versus added photon number.

## Notes for This Feature Pass

- `cqed_sim` models the effective cold-bath limit as
  `nbar_eff = (kappa_internal*nbar_internal + kappa_external*nbar_external)/(kappa_internal + kappa_external)`.
- The two-mode model is a linear hybridization model for readout-like and attenuator-like normal modes; it reports participation, linewidth, and inherited dispersive-shift estimates, not a full electromagnetic simulation of the aperture geometry.
- Frequencies and rates in the cQED-facing helpers follow the repository convention: angular rates in rad/s, times in seconds, and Hamiltonian dispersive terms using the projector form `+ chi*n_mode*n_q`.
