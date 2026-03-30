---
description: "Audit the optimal-control I+ iQ convention migration, clean up stale public docs or artifacts, and make the final user-facing explanation clear and readable."
name: "Optimal-Control I/Q Doc Audit"
argument-hint: "Optional extra context, files to prioritize, or surfaces to exclude"
agent: "agent"
---
Review the recent optimal-control convention migration in this workspace and perform a focused follow-up audit of the documentation and user-facing artifacts.

Context:
- The intended public optimal-control baseband rule is now `c(t) = I(t) + i Q(t)`.
- Model-backed `Q` terms are intentionally built as `+i(raising - lowering)` so replay through the unchanged runtime pulse assembly remains Hamiltonian-consistent.
- The implementation and core docs were already updated, but there may still be stale wording in generated artifacts, tutorial outputs, or public-facing explanations.

Your task:
1. Audit the repo specifically for lingering user-facing or artifact-level references to the legacy optimal-control rule `I - iQ`.
2. Update only the surfaces that should now reflect the canonical `I + iQ` convention.
3. Rewrite any touched documentation in a clean, readable style that explains the convention clearly without repetitive migration-report wording.
4. Keep historical reports historical: if a file is intentionally documenting the old state as part of an inconsistency report or fix record, do not rewrite it as if the old state never existed.

Priorities:
- Check the public and developer-facing documentation first:
  - [API_REFERENCE.md](../../API_REFERENCE.md)
  - [documentations/api/optimal_control.md](../../documentations/api/optimal_control.md)
  - [documentations/physics_conventions.md](../../documentations/physics_conventions.md)
  - [documentations/tutorials/optimal_control.md](../../documentations/tutorials/optimal_control.md)
  - [cqed_sim/optimal_control/README.md](../../cqed_sim/optimal_control/README.md)
  - [physics_and_conventions/physics_conventions_report.tex](../../physics_and_conventions/physics_conventions_report.tex)
- Then inspect likely residual artifact surfaces:
  - generated `site/` output if source docs change
  - `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
  - plain-text outputs or benchmark artifacts that are meant to be user-consumable
- Treat migration reports under `inconsistency/` and implementation reports under `docs/` as historical records. Only update them if they now inaccurately describe what was fixed or what residue remains.

What to verify explicitly:
- No active public doc should still teach `c(t) = I(t) - i Q(t)` for optimal control.
- Any clean explanation of the new rule should also state why replay remains correct: model-backed `Q` is built as `+i(raising - lowering)` while the runtime assembly is unchanged.
- Structured optimal-control explanations should match the same sign convention as the main export path.
- Notebook code and stored outputs should be distinguished. If only stored output is stale, either refresh it or report it explicitly as remaining residue.
- Do not broaden the task into unrelated quadrature-system audits unless you find a direct contradiction tied to this migration.

Editing rules:
- Prefer minimal edits.
- Improve readability while staying technically precise.
- Avoid changelog-style prose in user-facing docs unless the page is actually a report.
- Do not rewrite unrelated sections just for style consistency.
- If you update public docs, also update generated `site/` output in the same task.
- If you update the physics report, build it. If the predefined VS Code task is still broken by workspace-path quoting, run the batch file directly and note that the task wiring remains broken.

Validation:
- Run targeted searches for stale `I - iQ` wording after edits.
- If public docs changed, run `python -m mkdocs build --strict`.
- If `physics_and_conventions/physics_conventions_report.tex` changed, run `physics_and_conventions/build_physics_conventions_report.bat` directly if needed.
- Run only the smallest relevant verification commands needed for the touched surfaces.

Final output requirements:
- Summarize exactly which docs or artifacts were updated.
- State whether any legacy wording remains and why.
- Call out separately whether any remaining matches are intentional historical records versus stale public-facing residue.

Additional context from the current migration state:
- Focus especially on the gap between implementation truth and rendered or generated artifacts.
- The main risk is not broken code anymore; it is stale or unclear explanation.
- Favor clarity for future users over preserving older phrasing.
