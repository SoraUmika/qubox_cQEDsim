# Unitary Synthesis Leakage Metric Visibility and Visualization Gap

Created: 2026-03-29 15:19:12 local time  
Status: fixed

## Confirmed Issues

### Path leakage metrics existed in reports but were not consistently usable as selected task metrics

- What:
  - Intermediate checkpoint/path leakage was already computed for some synthesis runs, but selected-metric resolution happened before those checkpoint-derived aliases were merged into the metric dictionary.
- Where:
  - `cqed_sim/unitary_synthesis/optim.py`
  - `cqed_sim/unitary_synthesis/fast_eval.py`
- Affected components:
  - leakage-aware synthesis tasks
  - relevant-map optimization where users wanted to score path leakage directly
  - notebook/example workflows that needed consistent path-leakage reporting
- Why this was inconsistent:
  - The public metric layer exposed names such as `path_leakage_worst`, but the evaluator ordering could still hide them from actual metric selection.
- Consequences:
  - users could inspect path leakage after a run but could not rely on it as a first-class selected metric in the same way as the other public synthesis metrics

### Leakage-aware visualization and tutorial surfaces lagged the implemented optimizer behavior

- What:
  - The optimizer already had retained-subspace leakage and truncation diagnostics, but there was no dedicated public plotting layer for leakage blocks, projected logical densities, leakage-vs-step profiles, or edge-projector summaries.
- Where:
  - `cqed_sim/unitary_synthesis`
  - `cqed_sim/unitary_synthesis/README.md`
  - `documentations/api/unitary_synthesis.md`
  - `documentations/tutorials/unitary_synthesis.md`
  - `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`
- Affected components:
  - user-facing leakage diagnostics
  - example/tutorial discoverability
  - API/doc synchronization for the unitary-synthesis module
- Why this was inconsistent:
  - The public docs described leakage-aware synthesis at a high level, but the concrete workflow for comparing logical leakage, path leakage, and edge occupancy was missing.
- Consequences:
  - users had no clear supported path for post-fit leakage visualization or for reproducing the intended relevant-map leakage comparison workflow

## Suspected / Follow-up Questions

- Reduced-state and channel objectives still fall back off the accelerated ideal evaluator. This is an expected performance limitation of the current implementation, not a correctness issue fixed in this pass.
- The notebook execution tooling did not reliably report completion for the new Workflow 6 cell during this session, even after the comparison runtime was reduced. The example script itself runs quickly and the notebook cell now uses the reduced-iteration path, so the remaining concern appears to be notebook execution behavior rather than optimizer cost.

## Status

- Fixed on 2026-03-29.
- Path-leakage aliases are now visible to selected-metric resolution in both the legacy and accelerated evaluators.
- Path-leakage metrics remain reportable even when `LeakagePenalty.checkpoint_weight == 0.0`.
- Edge-projector leakage terms and serialized leakage diagnostics are now exposed through the public report surface.
- Dedicated visualization helpers, a runnable example, and a notebook workflow were added for leakage-aware relevant-map optimization.

## Fix Record

- Code changes:
  - `cqed_sim/unitary_synthesis/config.py`
  - `cqed_sim/unitary_synthesis/metrics.py`
  - `cqed_sim/unitary_synthesis/objectives.py`
  - `cqed_sim/unitary_synthesis/optim.py`
  - `cqed_sim/unitary_synthesis/fast_eval.py`
  - `cqed_sim/unitary_synthesis/visualization.py`
  - `cqed_sim/unitary_synthesis/__init__.py`
- Tests:
  - `tests/unitary_synthesis/test_relevance_objectives_and_execution.py`
  - `tests/unitary_synthesis/test_leakage_visualization.py`
- Examples and tutorials:
  - `examples/unitary_synthesis_leakage_aware_visualization.py`
  - `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`
- Documentation:
  - `cqed_sim/unitary_synthesis/README.md`
  - `documentations/api/unitary_synthesis.md`
  - `documentations/tutorials/unitary_synthesis.md`
  - `documentations/tutorials/index.md`
  - `documentations/examples.md`
  - `API_REFERENCE.md`
  - `physics_and_conventions/physics_conventions_report.tex`
