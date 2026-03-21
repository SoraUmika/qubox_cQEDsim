---
name: physics-convention-update
description: "Update cQED physics conventions consistently across code, tests, API docs, and the physics report. Use when changing chi sign, Kerr sign, Hamiltonians, rotating frames, basis ordering, units, measurement meaning, or parameter naming tied to physical interpretation."
argument-hint: "Describe the convention change, the old and new meaning, and the affected models or documents."
---

# Physics Convention Update

Use this skill when a change affects physical meaning, not just implementation detail.

## When to Use

- Changing `chi`, `Kerr`, frame, basis-ordering, sign, or unit conventions.
- Updating Hamiltonians, observables, measurement definitions, or parameter translations.
- Reconciling runtime conventions with unitary-synthesis or calibration conventions.
- Fixing convention drift between code, tests, tutorials, `API_REFERENCE.md`, and `physics_and_conventions/physics_conventions_report.tex`.

## Common Failure Modes to Prevent

- Updating runtime code but not `analysis/`, calibration helpers, or synthesis code.
- Fixing code and tests but leaving stale prose in the physics report or `documentations/`.
- Preserving old parameter names in tutorials or examples after the canonical meaning changed.
- Missing tensor-dimension or basis-ordering implications when the convention affects state labeling.

## Procedure

1. State the canonical meaning first.
   - Write down the old meaning and the new meaning in one or two precise sentences.
   - Identify whether the change affects runtime behavior, documentation only, or both.
2. Search all affected surfaces.
   - Runtime model layers in `cqed_sim/core/`, `cqed_sim/sim/`, and `cqed_sim/measurement/`.
   - Convention-sensitive helpers in `cqed_sim/analysis/`, `cqed_sim/calibration/`, `cqed_sim/calibration_targets/`, `cqed_sim/unitary_synthesis/`, and `cqed_sim/optimal_control/`.
   - Regression tests under `tests/` and literature checks under `test_against_papers/`.
   - Public docs in `API_REFERENCE.md`, `documentations/`, and `physics_and_conventions/physics_conventions_report.tex`.
3. Update code and tests together.
   - Do not change a physical convention in implementation without updating the tests that lock it in.
   - Prefer explicit regression coverage for sign, basis ordering, and frequency-translation behavior.
4. Keep the repo's canonical convention visible.
   - Make the project-level meaning explicit at wrapper boundaries.
   - Do not leave hidden sign or factor-of-two conversions undocumented.
5. Synchronize documentation.
   - Update `API_REFERENCE.md` if public naming, semantics, or examples change.
   - Update the relevant pages under `documentations/`.
   - Update `physics_and_conventions/physics_conventions_report.tex` whenever physical meaning changes.
6. Rebuild the physics report.
   - Run the existing physics-report build workflow after editing the `.tex` source.
7. Record the change.
   - Create or update an inconsistency report if the task fixes or reveals convention drift.

## Completion Checklist

- Canonical old vs. new meaning is stated clearly.
- Convention-sensitive code paths were searched, not just the first obvious model file.
- Tests were updated with explicit regression coverage.
- `API_REFERENCE.md` and `documentations/` match the new meaning.
- `physics_and_conventions/physics_conventions_report.tex` was updated and rebuilt.
- Any related inconsistency report was created or marked fixed.