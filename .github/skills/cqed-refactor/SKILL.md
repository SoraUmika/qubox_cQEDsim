---
name: cqed-refactor
description: "Refactor cqed_sim modules or repo-side workflow tooling while preserving project conventions. Use when cleaning up modules, reorganizing APIs, renaming symbols, removing dead code, or doing behavior-preserving maintenance that may require inconsistency reports, tests, API_REFERENCE.md updates, website documentation sync, examples, or physics documentation updates."
argument-hint: "Describe the refactor target, affected modules, and whether public APIs or physics meaning change."
---

# cQED Refactor

Use this skill for repo-wide maintenance work where the main risk is incomplete synchronization rather than isolated code editing.

## When to Use

- Refactoring `cqed_sim` modules, `agent_workflow`, or repo-side tooling.
- Consolidating duplicate implementations or removing dead code.
- Renaming modules, classes, functions, parameters, or file layout.
- Reorganizing internals while preserving the intended feature set.
- Doing cleanup work that can affect tests, examples, physics conventions, or public docs.

## Do Not Use

- Greenfield feature work with no refactor component.
- Tiny local bug fixes that do not affect structure, public behavior, or conventions.

## Repo-Specific Rules

- Read `README.md` and `AGENTS.md` before touching code.
- Inspect relevant files under `inconsistency/` before changing behavior.
- Prefer minimal, reviewable edits over speculative rewrites.
- Reuse existing `cqed_sim` infrastructure instead of creating parallel paths.
- If simulation logic is involved, follow the QuTiP-native-first policy.

## Procedure

1. Scope the refactor.
   - List the modules, public entry points, tests, examples, and docs that may be touched.
   - Decide whether the change is internal-only, public-API-visible, or convention-changing.
2. Review prior inconsistencies.
   - Search `inconsistency/` for related audits or bug reports.
   - If the refactor addresses an existing report, plan to update that report with a fix record instead of leaving it stale.
3. Protect architecture boundaries.
   - Keep repo-side orchestration in `agent_workflow/` or `tools/` when it does not belong in `cqed_sim`.
   - Do not bypass existing `cqed_sim` flows with one-off standalone simulation code unless the task requires a justified exception.
4. Apply the refactor at the root cause.
   - Remove dead code only after confirming it is unreferenced.
   - Avoid adding duplicate abstractions or temporary compatibility layers unless the task explicitly calls for them.
5. Synchronize affected surfaces.
   - Public API change: update `API_REFERENCE.md` and the matching pages under `documentations/`.
   - Physics or convention change: update `physics_and_conventions/physics_conventions_report.tex` and rebuild the PDF.
   - Major reusable feature area: add or update the local module `README.md`.
6. Add validation.
   - Add or update automated coverage under `tests/`.
   - Add or update `examples/` only if the intended user workflow changed.
7. Close the loop.
   - Create or update an inconsistency report when required by `AGENTS.md`.
   - Record remaining risks, deferred follow-up work, and any unresolved assumptions.

## Completion Checklist

- The refactor is minimal and reviewable.
- No duplicate or parallel implementation path was introduced.
- Related inconsistency reports were created or updated.
- Tests were added or updated under `tests/`.
- `API_REFERENCE.md` was updated if public behavior changed.
- `documentations/` was updated if user-facing or developer-facing docs changed.
- The physics report was updated and rebuilt if conventions changed.
- Local module `README.md` files were updated for any major feature area that changed.