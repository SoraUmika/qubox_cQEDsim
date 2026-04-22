Stefan Krastanov, Victor V. Albert, Chao Shen, Chang-Ling Zou, Reinier W. Heeres, Brian Vlastakis, Robert J. Schoelkopf, Lieven M. K. Vandersypen, and Liang Jiang, "Universal control of an oscillator with dispersive coupling to a qubit," Physical Review A 92, 040303(R) (2015). DOI: 10.1103/PhysRevA.92.040303

## Summary

- The paper gives the canonical cQED formulation of the SNAP gate: an independently programmable phase on each cavity Fock level enabled by dispersive number splitting of the ancilla transition.
- It proves that combining SNAP with displacements is sufficient for universal oscillator control.
- The paper also provides the core physical intuition behind displacement-SNAP-displacement state-preparation workflows.

## Relevance to `cqed_sim`

- It is the primary reference for the SNAP and Fock-state-preparation tutorial pages.
- It anchors the statement that a number-selective ancilla drive implements a diagonal unitary in the cavity Fock basis.
- It justifies using displacement-plus-SNAP synthesis as the physics model behind the public DSD workflow documentation.

## Notes for This Feature Pass

- The public tutorial does not reproduce a specific figure from the paper; it uses the paper as the canonical reference for the gate set and universality claim.
- The repo's unitary-synthesis and SNAP workflow pages should cite this paper whenever they explain why the D-SNAP-D primitive family is expressive.
