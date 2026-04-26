# `cqed_sim.models`

`cqed_sim.models` contains the strong-readout model hierarchy added for
high-power superconducting-qubit readout studies.

The module is separate from the existing dispersive `cqed_sim.core` models so
that the legacy public APIs stay stable while the strong-readout path can use a
microscopic cosine transmon, explicit readout/filter modes, and dressed-state
diagnostics.

## Entry Points

- `transmon.py`: charge-basis cosine transmon
  `H = 4 EC (n-ng)^2 - EJ cos(phi)` plus a lighter Duffing approximation.
- `multilevel_cqed.py`: multilevel transmon and readout resonator Hamiltonian
  with complex IQ drive samples.
- `purcell_filter.py`: explicit filter resonator mode with output collapse
  operator `sqrt(kappa_f) f`.
- `dressed_basis.py`: undriven dressed-state diagonalization, bare-label
  assignment, dressed qubit projectors, transition matrices, and leakage.
- `mist_floquet.py`: semiclassical MIST penalty scans using charge-driven
  cosine-transmon spectra.

## Units and Ordering

Frequencies, rates, Hamiltonian coefficients, and drive amplitudes are angular
frequencies, normally rad/s.  Times are seconds.  Offset charge `ng` is
dimensionless.  Tensor ordering is transmon first:

- no filter: `|q, n_r>`
- with explicit filter: `|q, n_r, n_f>`

The cosine transmon charge basis is ordered from `-n_cut` to `+n_cut`.  The
exported charge matrix is the Cooper-pair number operator `n` in the transmon
eigenbasis.

## References

[1] J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer,
A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, "Charge-insensitive
qubit design derived from the Cooper pair box," Physical Review A 76, 042319
(2007). DOI: 10.1103/PhysRevA.76.042319

[2] A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum
electrodynamics," Reviews of Modern Physics 93, 025005 (2021). DOI:
10.1103/RevModPhys.93.025005

[3] W. Dai et al., "Characterization of drive-induced unwanted state transitions
in superconducting circuits," arXiv:2506.24070 (2025). DOI:
10.48550/arXiv.2506.24070
