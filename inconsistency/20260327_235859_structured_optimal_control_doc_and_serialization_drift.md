# 2026-03-27 23:58:59 Structured Optimal-Control Documentation and Serialization Drift

## Confirmed Issues

### 1. Public optimal-control documentation contained concatenated duplicate content after the structured-control rewrite

- What:
  - the rewritten optimal-control tutorial and API page retained appended stale content from older GRAPE-only versions, producing malformed Markdown and outdated public guidance.
- Where:
  - `documentations/tutorials/optimal_control.md`
  - `documentations/api/optimal_control.md`
- Affected components:
  - public docs, tutorial discoverability, API accuracy, and the generated `site/` output.
- Why this is inconsistent:
  - the implementation now exposes both structured control and GRAPE, but the documentation files contained duplicated GRAPE-era sections, stale API names, and malformed splice points that no longer matched the current public surface.
- Consequences:
  - readers could see duplicated sections, invalid references such as `GrapeConfig(n_random_restarts=...)` or `LeakagePenaltyObjective`, and a misleading description of the supported workflow.

### 2. Structured-control artifact export exposed a JSON serialization gap for complex NumPy arrays

- What:
  - `ControlResult.save()` relied on `json_ready(...)`, but the helper did not recursively normalize complex-valued NumPy arrays once they appeared inside nested result payloads.
- Where:
  - `cqed_sim/optimal_control/utils.py`
  - surfaced through structured-control artifact export in `save_structured_control_artifacts(...)`
- Affected components:
  - `result.json` generation, study artifact export, and any future result payload carrying nested complex arrays.
- Why this is inconsistent:
  - `ControlResult` is intended to be a serializable result surface, but the serialization helper only handled simple container cases and not the richer nested payloads introduced by the structured-control workflow.
- Consequences:
  - structured study export could fail even when the optimization itself succeeded, breaking the requested reportable artifact path.

## Suspected / Follow-up Questions

- The public notebook companion for optimal control is still GRAPE-centric. If structured control becomes a primary tutorial workflow, the repo may eventually want a dedicated structured-control notebook rather than relying on the website tutorial page plus the example script.
- Because the malformed docs came from appended stale content rather than a semantic API mismatch alone, future large Markdown rewrites should get an explicit drift pass before site rebuilds.

## Status

- Current status: fixed on 2026-03-27
- Resolution summary:
  - removed duplicated stale content from the optimal-control tutorial and API page,
  - repaired malformed splice artifacts left by the doc rewrites,
  - added the structured optimal-control example to the public examples index,
  - fixed `json_ready(...)` so nested NumPy arrays are recursively normalized and complex values become JSON-safe.

## Fix Record

### Documentation drift

- Fixed by:
  - `documentations/tutorials/optimal_control.md`
  - `documentations/api/optimal_control.md`
  - `documentations/examples.md`
  - `README.md`
  - `API_REFERENCE.md`
  - `cqed_sim/optimal_control/README.md`
  - `mkdocs.yml`

### Serialization gap

- Fixed by:
  - `cqed_sim/optimal_control/utils.py`
  - validated through `tests/test_52_structured_optimal_control.py`