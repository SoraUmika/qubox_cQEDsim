P. Rebentrost and F. K. Wilhelm, "Optimal control of a leaking qubit," Physical Review B 79, 060507(R) (2009). DOI: 10.1103/PhysRevB.79.060507

## Summary

- The paper studies optimal control for a qubit embedded in a larger Hilbert space where leakage to noncomputational levels is unavoidable unless it is explicitly accounted for.
- It frames leakage as part of the control landscape and shows that optimized controls can significantly improve logical performance relative to leakage-oblivious strategies.
- The paper is an early canonical reference for the idea that the optimization target should distinguish the desired logical action from unwanted population transfer into leakage states.

## Relevance to `cqed_sim`

- This reference supports the repository design choice to preserve a logical/relevant-map objective while layering explicit leakage penalties and diagnostics on top.
- It is especially relevant for the new isometry- and reduced-state-aware leakage treatment because those workflows care about the logical map without requiring a full-space target unitary.
- The repository's new visualization helpers extend that idea by making the leakage block, projected logical density, and path profile directly inspectable after a synthesis run.

## Notes for This Feature Pass

- The paper motivates leakage-aware optimization at the level of objective design.
- The new `edge_projector` option in `cqed_sim` is a repository-specific extension for truncation-aware studies rather than a direct construct from this paper.