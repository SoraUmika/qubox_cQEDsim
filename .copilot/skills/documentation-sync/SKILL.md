---
name: documentation-sync
description: "Verify consistency across API_REFERENCE.md, documentations/ website, module-level READMEs, and mkdocs.yml navigation. Use when: refactoring public APIs, changing function signatures, adding modules, updating documentation, or before marking any refactor complete. Detects stale docs, missing symbols, signature mismatches."
---

# Skill: Cross-Document Synchronization Checker

## Identity

You are a documentation-audit agent for the `cqed_sim` project. Your job is to verify
that the public API surface, `API_REFERENCE.md`, website documentation under
`documentations/`, module-level READMEs, and `physics_and_conventions/physics_conventions_report.tex`
are mutually consistent and complete.

## Trigger

Invoke this skill when:
- After a refactor that changes public APIs, function signatures, class behavior,
  module organization, or configuration structure.
- After adding a new public module or sub-package to `cqed_sim/`.
- When asked to "check docs", "sync documentation", or "audit API reference".
- Before considering any refactor task complete (per AGENTS.md § Refactor Documentation
  Synchronization).
- After updating `API_REFERENCE.md` or any page under `documentations/`.

## Inputs

The user may provide:
- `scope`: specific modules or documentation files to audit (default: full scan).
- `fix`: whether to apply fixes automatically or only report gaps (default: report only).

## Workflow

Execute these steps **in order**. Do not skip steps.

### Step 1 — Snapshot Public API Surface

Read every `__init__.py` under `cqed_sim/` recursively.
For each module, extract:
- All names in `__all__` (if defined).
- All public symbols (classes, functions, constants not starting with `_`).
- The source file where each symbol is defined.

Build a structured inventory: `{module_path: [symbol_list]}`.

### Step 2 — Audit API_REFERENCE.md

Read `API_REFERENCE.md`. For each documented section:
1. Verify every documented symbol exists in the corresponding module.
2. Verify every documented function signature matches the current code.
3. Flag symbols that exist in code but are missing from `API_REFERENCE.md`.
4. Flag symbols documented in `API_REFERENCE.md` that no longer exist in code.

Produce a delta table:

| Symbol | Module | In Code | In API_REFERENCE | Status |
|--------|--------|---------|------------------|--------|
| ... | ... | ✓/✗ | ✓/✗ | Missing/Stale/OK |

### Step 3 — Audit Website Documentation

Read the `documentations/` folder structure and `mkdocs.yml` navigation.

For each API page under `documentations/api/`:
1. Verify it documents symbols that exist in `API_REFERENCE.md`.
2. Check for stale references to removed functions or classes.
3. Verify code examples use current import paths and signatures.

For each user guide and tutorial page under `documentations/`:
1. Check that import statements reference valid modules.
2. Flag any usage pattern that contradicts `API_REFERENCE.md`.

Produce a sync table:

| Page | Issue | Details |
|------|-------|---------|
| ... | Stale import / Missing symbol / Signature mismatch | ... |

### Step 4 — Audit Module-Level READMEs

For each sub-package under `cqed_sim/` that qualifies as a "major feature area"
(per AGENTS.md § Module-Level README Policy):
1. Check whether a `README.md` exists in the sub-package directory.
2. If it exists, verify it covers: purpose, entry points, important classes/functions,
   inputs/outputs, usage examples, and known limitations.
3. Flag missing or incomplete module-level READMEs.

Major feature areas that must have READMEs:
- `rl_control/`
- `optimal_control/`
- `unitary_synthesis/`
- `quantum_algorithms/`
- `calibration/`
- `tomo/`
- `backends/`
- `system_id/`

### Step 5 — Check mkdocs.yml Navigation Consistency

Read `mkdocs.yml`. Verify:
1. Every API page listed in `nav` corresponds to an existing `.md` file.
2. Every `.md` file under `documentations/` is reachable from `nav`.
3. Major feature areas with module-level READMEs are represented in the site.

### Step 6 — Generate Sync Report

Write a structured report summarizing all gaps:

```markdown
# Documentation Sync Audit — <date>

## API Surface Summary
- Total public symbols: N
- Documented in API_REFERENCE.md: M
- Documented on website: K

## API_REFERENCE.md Gaps
### Missing from API_REFERENCE.md (exist in code)
| Symbol | Module |
|--------|--------|

### Stale in API_REFERENCE.md (removed from code)
| Symbol | Section |
|--------|---------|

### Signature Mismatches
| Symbol | API_REFERENCE signature | Code signature |
|--------|------------------------|----------------|

## Website Documentation Gaps
| Page | Issue | Details |
|------|-------|---------|

## Module README Gaps
| Module | README exists | Complete | Issues |
|--------|--------------|----------|--------|

## mkdocs.yml Issues
| Issue | Details |
|-------|---------|

## Recommendations
1. ...
```

### Step 7 — Apply Fixes (if fix mode is enabled)

If the user requested automatic fixes:
1. Add missing symbols to `API_REFERENCE.md` with stub documentation.
2. Remove stale entries from `API_REFERENCE.md`.
3. Update signature mismatches to match current code.
4. Create skeleton `README.md` files for modules that lack them.
5. Update `mkdocs.yml` nav for missing pages.

Document all changes made in the report.

## Key References

- `API_REFERENCE.md` — canonical public API reference.
- `documentations/` — MkDocs website source.
- `mkdocs.yml` — site navigation and configuration.
- `cqed_sim/**/__init__.py` — re-export declarations.
- `README.md` — project-level API summary.
- `AGENTS.md` §§ Refactor Documentation Synchronization, API Reference and Website
  Documentation Synchronization, Module-Level README Policy.

## Quality Standards

- Every gap must cite the specific file, symbol, or section affected.
- Do not flag internal symbols (prefixed with `_`) as missing from documentation.
- Signature comparisons must use the actual function signature from source, not
  inferred from usage.
- For module-level READMEs, evaluate completeness against the AGENTS.md checklist,
  not against an arbitrary standard.
- The report must be actionable: each gap should have a clear next step.
