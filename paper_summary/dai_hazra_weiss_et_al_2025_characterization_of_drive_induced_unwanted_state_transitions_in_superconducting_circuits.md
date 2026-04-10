W. Dai, S. Hazra, D. K. Weiss, P. D. Kurilovich, T. Connolly, H. K. Babla, S. Singh, V. R. Joshi, A. Z. Ding, P. D. Parakh, J. Venkatraman, X. Xiao, L. Frunzio, and M. H. Devoret, "Characterization of drive-induced unwanted state transitions in superconducting circuits," arXiv:2506.24070, 2025. DOI: 10.48550/arXiv.2506.24070

## Summary

- The paper classifies drive-induced unwanted state transitions in a fixed-frequency 3D transmon into three mechanisms: ac-Stark-shift-assisted resonant exchange with spurious modes, intrinsic multiphoton resonances of the transmon itself, and inelastic scattering into external modes.
- For the intrinsic mechanism-B features, the paper models the system with a transmon-only cosine Hamiltonian driven through the charge operator and analyzes the resulting Floquet branches as a function of drive frequency and drive strength.
- Appendix E also gives a coupled cosine-transmon and readout-mode Hamiltonian for a readout-assisted mechanism-C analysis, with fitted parameters EJ / h = 16.40 GHz, EC / h = 0.1695 GHz, g / h = 0.153 GHz, and omega_r / (2 pi) = 9.029 GHz.
- The paper's fixed-frequency branch-analysis slices most directly usable in this repository are the intrinsic avoided crossings from Figs. 4 and A8 and the readout-assisted K/L slices from Fig. A5 at wd / (2 pi) = 10.48 GHz and 10.73 GHz.

## Relevance to `cqed_sim`

- This is a strong literature target for the repository's Floquet tooling because it exercises exactly the kind of periodically driven closed-system analysis that `cqed_sim.floquet` is designed to handle.
- The cleanest reproducible subsets with the current repo are the intrinsic mechanism-B transmon-only analysis and the Appendix E readout-assisted mechanism-C branch slices. Neither path requires modeling TLSs, parasitic package modes, or dissipative steady states.
- The repository reproductions use `cqed_sim.floquet` directly with explicit static Hamiltonians and explicit charge-drive operators built from the paper's Eq. (1) and Eq. (E10), instead of the repo's standard Duffing-style runtime models.

## Notes for This Reproduction

- The reproduction script `test_against_papers/dai_et_al_2025_intrinsic_multiphoton_floquet_resonances.py` focuses on three branch-analysis targets: the low-power |1> <-> |5> avoided crossing near 8.02 GHz, the same resonance shifted to higher power near 7.825 GHz, and the |0> <-> |4> avoided crossing near 8.45 GHz.
- The reproduction script `test_against_papers/dai_et_al_2025_readout_assisted_floquet_resonances.py` focuses on the Appendix E readout-assisted K/L branch-analysis slices, tracking the dressed branches connected to |1_t,0_r> <-> |7_t,1_r> at 10.48 GHz and |1_t,0_r> <-> |4_t,1_r> at 10.73 GHz.
- Internally the script follows the repo's documented convention that Hamiltonian coefficients are represented in rad/s and reports quoted GHz values by dividing by 2 pi. This is consistent with the repo physics-conventions report and does not introduce a convention mismatch.
- The scripts keep ng = 0 and use minimum tracked branch gap plus bare-state mixing as the resonance marker. They do not attempt to reproduce the paper's full offset-charge averaging or ideal-displaced-state hybridization parameter Theta_j.
- Extrinsic mechanisms involving TLSs, readout-assisted inelastic scattering into a dissipative bath, and parasitic package modes remain out of scope for this reproduction pass.