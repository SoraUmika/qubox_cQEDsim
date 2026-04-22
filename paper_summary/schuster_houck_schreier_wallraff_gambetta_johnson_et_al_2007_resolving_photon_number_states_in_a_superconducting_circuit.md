D. I. Schuster, A. A. Houck, J. A. Schreier, A. Wallraff, J. M. Gambetta, B. R. Johnson, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, "Resolving photon number states in a superconducting circuit," Nature 445, 515-518 (2007). DOI: 10.1038/nature05461

## Summary

- This is the canonical experimental number-splitting paper in circuit QED.
- The hallmark result is that the qubit spectrum resolves into distinct lines for different cavity photon-number manifolds in the strong-dispersive regime.
- It directly connects spectroscopic peak spacing to the dispersive shift and peak amplitudes to the cavity photon-number distribution.

## Relevance to `cqed_sim`

- It is the most important external reference for the number-splitting and displacement-spectroscopy tutorial pages.
- It anchors the claim that resolved spectroscopy can act as a photon-number discriminator when the linewidth is narrower than the manifold spacing.
- It provides the literature basis for using peak spacing as a direct measurement of `chi`.

## Notes for This Feature Pass

- The public number-splitting tutorial uses a selective Gaussian probe and coherent-state Poisson weights rather than attempting a line-by-line reproduction of the original experiment.
- The validation target in the repo is therefore theory-consistent resolved peak spacing and peak-weight agreement, not a claim of full experimental reproduction.
