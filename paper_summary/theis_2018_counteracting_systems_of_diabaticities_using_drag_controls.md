L. S. Theis, F. Motzoi, S. Machnes, and F. K. Wilhelm, "Counteracting systems of diabaticities using DRAG controls: The status after 10 years," EPL (Europhysics Letters) 123, 60001 (2018). DOI: 10.1209/0295-5075/123/60001

## Summary

- This paper reviews the DRAG framework as a control strategy for suppressing unwanted transitions in weakly anharmonic or spectrally crowded quantum systems.
- The core idea is to cancel diabatic couplings to leakage states by adding structured control corrections that are compatible with available experimental control channels.
- The review emphasizes that the relevant quantity is often not only the final logical fidelity but also how the control suppresses excursions into unwanted nearby levels across the pulse.

## Relevance to `cqed_sim`

- The review provides canonical background for interpreting leakage-aware control as part of the optimization problem rather than as a purely downstream validation metric.
- It is especially relevant to the new path-leakage and edge-occupancy discussion because it frames leakage suppression as a control-design concern tied to unwanted intermediate excursions.
- The repository does not implement DRAG pulses in this feature pass; instead, it borrows the higher-level lesson that leakage-aware objectives and diagnostics should be explicit and inspectable.

## Notes for This Feature Pass

- The paper is best used here as background for why path leakage matters.
- It does not define a canonical edge-of-truncation projector penalty, so the new edge projector remains a repository-level diagnostic/regularization choice rather than a direct paper reproduction.
