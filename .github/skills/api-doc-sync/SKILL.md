---
name: api-doc-sync
description: "Keep public API and documentation surfaces synchronized. Use when changing function signatures, parameter names, module organization, public workflows, file paths, or examples and when API_REFERENCE.md, documentations pages, physics docs, examples, or module READMEs may drift out of sync."
argument-hint: "Describe the public API or workflow change and list any renamed files, symbols, or parameters."
---

# API and Documentation Sync

Use this skill when the primary risk is stale documentation after code changes.

## When to Use

- Public function, class, or module signatures changed.
- Parameter names, return shapes, or workflow steps changed.
- Files or tutorials moved and old cross-references may now be stale.
- `API_REFERENCE.md` and `documentations/` need to stay aligned.
- A convention change also affects user-facing explanations.

## Sync Surfaces

Start from the bundled [sync checklist](./assets/sync-checklist.md).

Core surfaces in this repo:

- `API_REFERENCE.md`
- `documentations/api/`
- `documentations/user_guides/`
- `documentations/tutorials/`
- `documentations/index.md` and related navigation pages
- local module `README.md` files for major reusable features
- `physics_and_conventions/physics_conventions_report.tex` when semantics change

## Common Failure Modes to Prevent

- Updating `API_REFERENCE.md` but leaving `documentations/` stale.
- Renaming parameters in code but not in examples or tutorials.
- Leaving deleted file paths or notebook references in docs.
- Treating a physics-meaning change as if it were documentation-only.

## Procedure

1. Inventory the public surface change.
   - List renamed symbols, changed parameters, moved files, updated workflows, and any new limitations.
2. Map code changes to docs.
   - Find the matching sections in `API_REFERENCE.md`.
   - Find the related pages under `documentations/`.
   - Check module-level `README.md` files where the feature is explained locally.
3. Update examples and references.
   - Fix stale keys, paths, imports, and usage snippets.
   - If the intended user workflow changed, update `examples/` or tutorial assets accordingly.
4. Check whether physics meaning changed.
   - If yes, update `physics_and_conventions/physics_conventions_report.tex` as part of the same task.
5. Verify navigation and discoverability.
   - Make sure new or moved pages remain reachable from the existing documentation structure.
6. Finish with a drift pass.
   - Search for old names, old file paths, and outdated terminology after the main edit is done.

## Completion Checklist

- `API_REFERENCE.md` matches the implemented public API.
- Relevant `documentations/` pages were updated in the same task.
- Examples, tutorials, and module READMEs use the current names and paths.
- Physics documentation was updated if semantics changed.
- A final search confirmed that stale names or references were removed.