---
name: inconsistency-report
description: "Create or update timestamped inconsistency reports in the inconsistency folder. Use when a refactor, bug fix, audit, documentation mismatch, convention drift, or API inconsistency is found and the repo requires a structured confirmed-issues, follow-up, status, and fix-record report."
argument-hint: "Describe the inconsistency, where it appears, whether it is confirmed, and whether it is already fixed."
---

# Inconsistency Report

Use this skill whenever repo policy requires a tracked inconsistency record instead of a silent fix.

## When to Use

- Refactor tasks that uncover behavior, API, documentation, or physics mismatches.
- Convention fixes that need an audit trail.
- Documentation drift between `API_REFERENCE.md`, `documentations/`, and implementation.
- Cleanup work that resolves or partially resolves an existing report.

## File Naming

- Use the existing repo pattern: `YYYYMMDD_HHMMSS_short_slug.md`.
- Keep the slug descriptive and specific to the issue being tracked.

## Structure

Start from the bundled [report template](./assets/report-template.md).

Required sections:

- `## Confirmed Issues`
- `## Suspected / Follow-up Questions`
- `## Status`
- `## Fix Record`

Each confirmed issue should capture:

- What
- Where
- Affected components
- Why this is inconsistent
- Consequences

## Procedure

1. Check for an existing report first.
   - Search `inconsistency/` for related reports before creating a new one.
   - If the issue is already documented, update that file instead of duplicating it.
2. Separate confidence levels.
   - Put verified mismatches under `Confirmed Issues`.
   - Put uncertain theories or open design questions under `Suspected / Follow-up Questions`.
3. Describe the inconsistency concretely.
   - Name the files or modules involved.
   - Explain the project rule, convention, or expected behavior that is being violated.
   - State the practical consequence, not just the abstract mismatch.
4. Track resolution state.
   - If the issue is fixed in the same task, update `Status` and `Fix Record` immediately.
   - If the fix is partial, say exactly what remains unresolved.
5. Keep history intact.
   - Do not delete a report because it has been fixed.
   - Mark it as fixed, outdated, or partially resolved with a dated note.

## Completion Checklist

- Report filename matches the repo's timestamped pattern.
- Confirmed issues are separated from hypotheses.
- Status states whether the issue is open, fixed, partially fixed, or outdated.
- Fix Record names the files or changes that resolved the issue.
- Remaining concerns are explicit.