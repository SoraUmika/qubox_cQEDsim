M. Werninghaus, D. J. Egger, F. Roy, S. Machnes, F. K. Wilhelm, and S. Filipp, "Leakage reduction in fast superconducting qubit gates via optimal control," npj Quantum Information 7, 14 (2021). DOI: 10.1038/s41534-020-00346-2

## Summary

- The paper studies single-qubit transmon gates in the fast-gate regime, where weak anharmonicity makes leakage outside the computational subspace a dominant error source.
- The reported workflow uses leakage-aware pulse optimization and closed-loop calibration to reduce leakage while preserving gate fidelity at very short pulse durations.
- The paper reports a 4.16 ns single-qubit pulse with 99.76% fidelity and 0.044% leakage on the studied hardware, emphasizing that leakage suppression can be treated as an explicit optimization target rather than only a post hoc diagnostic.

## Relevance to `cqed_sim`

- This is the most directly relevant citation for treating final leakage as a first-class optimization term in superconducting-qubit control.
- It supports the design choice to keep the relevant logical task objective separate from explicit leakage regularizers.
- The repository feature added in this pass is more general than the paper's closed-loop pulse setting: it applies the same idea to gate-sequence synthesis, reduced-state targets, isometries, and visualization/reporting infrastructure.

## Notes for This Feature Pass

- The paper motivates leakage-aware objective terms, not the specific edge-of-truncation projector used in the new tutorial example.
- In the repository, path leakage and edge occupancy are exposed as additional diagnostics and regularizers on top of the retained-subspace objective.
