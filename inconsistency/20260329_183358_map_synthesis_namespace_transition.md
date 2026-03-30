# Map Synthesis Namespace Transition

Created: 2026-03-29 18:33:58 local time  
Status: fixed

## Confirmed Issues

### The legacy `unitary_synthesis` name no longer matched the public task surface

- What:
  - The synthesis stack now supports unitary targets, reduced-state targets, isometries, channels, observables, and trajectory checkpoints, but the primary public package and class names still centered only the unitary case.
- Where:
  - `cqed_sim/unitary_synthesis`
  - root `README.md`
  - `API_REFERENCE.md`
  - `documentations/api/unitary_synthesis.md`
  - `documentations/tutorials/unitary_synthesis.md`
- Affected components:
  - public API discoverability
  - user-facing examples and tutorials
  - naming consistency between implementation and supported task families
- Why this was inconsistent:
  - the package had evolved into a broader quantum-map synthesis layer, but the canonical namespace still implied only square-unitary matching.
- Consequences:
  - new users were steered toward an outdated conceptual model of the package surface
  - examples and documentation obscured the preferred abstraction for channel, isometry, and relevant-map workflows

### The new preferred namespace initially inherited the old namespace deprecation warning

- What:
  - the first `cqed_sim.map_synthesis` wrapper imported through `cqed_sim.unitary_synthesis`, which caused the legacy import warning to fire even when users adopted the new package.
- Where:
  - `cqed_sim/map_synthesis/__init__.py`
  - `cqed_sim/unitary_synthesis/__init__.py`
- Affected components:
  - new namespace import path
  - wrapper submodule imports such as `cqed_sim.map_synthesis.metrics`
- Why this was inconsistent:
  - the migration path should reward users for switching to the preferred namespace, not emit the same warning as the legacy one.
- Consequences:
  - the deprecation path looked broken from the user perspective
  - new examples would still surface legacy warnings despite using the new API

## Suspected / Follow-up Questions

- The current implementation modules still physically live under `cqed_sim/unitary_synthesis` and are re-exported through `cqed_sim.map_synthesis`. That is acceptable for this transition, but a future internal move may still be desirable if the repo wants the filesystem layout to match the public namespace exactly.
- Some downstream notebooks or external users may still intentionally reference `UnitarySynthesizer`. The compatibility alias remains in place, but the eventual removal schedule is still a release-management decision rather than an implementation decision made here.

## Status

- Fixed on 2026-03-29.
- `cqed_sim.map_synthesis` is now the preferred public namespace.
- `QuantumMapSynthesizer` is now the preferred public class name.
- `cqed_sim.unitary_synthesis` remains available as a backward-compatible compatibility facade and emits a deprecation warning only for direct legacy imports.
- The new namespace now imports cleanly without surfacing the legacy warning.
- Canonical docs and user-facing examples now prefer `cqed_sim.map_synthesis`.

## Fix Record

- Code changes:
  - `cqed_sim/map_synthesis/__init__.py`
  - `cqed_sim/unitary_synthesis/__init__.py`
- Tests:
  - `tests/unitary_synthesis/test_namespace_compatibility.py`
- Examples:
  - `examples/unitary_synthesis_demo.py`
  - `examples/unitary_synthesis_flexible_target_actions.py`
  - `examples/unitary_synthesis_leakage_aware_visualization.py`
  - `examples/unitary_synthesis_relevance_aware_optimizer.py`
  - `examples/synthesis_grape_validate_pipeline.py`
  - `examples/user_defined_gates_and_ansatz_demo.py`
  - `examples/grape_storage_subspace_gate_demo.py`
  - `examples/studies/sqr_speedlimit_multitone_gaussian.py`
- Documentation:
  - `README.md`
  - `cqed_sim/unitary_synthesis/README.md`
  - `documentations/api/gates.md`
  - `documentations/api/unitary_synthesis.md`
  - `documentations/tutorials/unitary_synthesis.md`
  - `documentations/tutorials/snap_fock_state_prep.md`
  - `API_REFERENCE.md`
  - `cqed_sim/gates/README.md`
  - `cqed_sim/io/README.md`
  - `cqed_sim/optimal_control/README.md`
  - `cqed_sim/sim/README.md`