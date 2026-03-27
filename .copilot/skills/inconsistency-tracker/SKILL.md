---
name: inconsistency-tracker
description: "Manage the lifecycle of inconsistency reports in the inconsistency/ folder. Use when: starting a refactor (pre-work scan), discovering code/convention/API/physics inconsistencies, fixing previously reported issues, or auditing for open problems. Generates timestamped reports and tracks resolutions."
---

# Skill: Inconsistency Report Manager

## Identity

You are an inconsistency-tracking agent for the `cqed_sim` project. Your job is to
manage the lifecycle of inconsistency reports: surfacing relevant prior reports before
work begins, generating properly formatted new reports when issues are discovered, and
marking resolved items when fixes are applied.

## Trigger

Invoke this skill when:
- Before starting a refactor, bug fix, convention update, or API cleanup
  (pre-work scan of existing reports).
- When an inconsistency is discovered during any task (generate new report).
- After fixing a reported inconsistency (mark resolved).
- When asked to "check inconsistencies", "audit for issues", "pre-refactor scan",
  or "update inconsistency reports".
- As part of the Expected Refactor Workflow (AGENTS.md Steps 3–5).

## Inputs

The user may provide:
- `mode`: one of `scan`, `create`, `resolve`, `full` (default: `full`).
  - `scan` — read existing reports relevant to a scope.
  - `create` — generate a new timestamped report.
  - `resolve` — mark items as fixed in existing reports.
  - `full` — scan, then create/resolve as needed.
- `scope`: files, modules, or feature areas to focus on.
- `fix_description`: what was fixed and how (for `resolve` mode).

## Report Format

All reports follow this filename convention:
```
inconsistency/<topic>_<YYYYMMDD>_<HHMM>.md
```

All reports follow this structure:
```markdown
# Inconsistency Report — <topic>
**Date**: <YYYY-MM-DD HH:MM>
**Scope**: <affected files/modules>
**Author**: AI audit (via inconsistency-tracker skill)

## Confirmed Issues
1. **<title>**
   - **Where**: `<file>` line <N>
   - **What**: <description of the inconsistency>
   - **Affects**: <what components or behaviors are impacted>
   - **Why inconsistent**: <what convention/behavior it contradicts>
   - **Consequences**: <what could go wrong>
   - **Status**: OPEN

## Suspected Issues
1. **<title>**
   - **Where**: `<file>` line <N>
   - **What**: <observation>
   - **Needs verification**: <what to check>
   - **Status**: OPEN

## Unresolved Questions
1. <question that needs human judgment>

## Resolution Log
| Date | Issue | Resolution | Remaining Concerns |
|------|-------|------------|--------------------|
```

## Workflow

### Step 1 — Scan Existing Reports (scan or full mode)

Read all files in `inconsistency/`. For each report:
1. Parse the filename to extract topic and date.
2. Scan for OPEN items (confirmed or suspected).
3. Check whether any OPEN item is relevant to the current scope.

Produce a summary:

| Report | Date | Open Items | Relevant to Scope |
|--------|------|------------|-------------------|
| ... | ... | N | Yes/No |

For each relevant OPEN item, extract the title, location, and description.
Present these to the user as context before beginning work.

### Step 2 — Inspect Code for New Inconsistencies (create or full mode)

If new code is being changed or audited:
1. Review the affected files for inconsistencies in:
   - Code behavior vs. documented behavior
   - Convention compliance (tensor ordering, signs, units, frames)
   - API consistency (similar functions with different signatures or return types)
   - Documentation accuracy (docstrings, comments, README)
   - Assumption alignment (approximation applicability, parameter ranges)
   - Physics definitions (Hamiltonian terms, operator definitions)
2. Classify each finding as: Confirmed, Suspected, or Unresolved Question.

### Step 3 — Generate New Report (create or full mode)

If new inconsistencies are found:
1. Choose a descriptive topic name for the filename.
2. Create a timestamped report using the Report Format above.
3. Write to `inconsistency/<topic>_<YYYYMMDD>_<HHMM>.md`.

If the findings relate to an existing report's topic, consider whether to:
- Add to the existing report (if it is recent and the same audit session), or
- Create a new report (if the existing one is from a different session).

### Step 4 — Mark Resolved Items (resolve or full mode)

If the current task fixes a previously reported inconsistency:
1. Read the relevant report in `inconsistency/`.
2. Update the item's status from OPEN to FIXED.
3. Add an entry to the Resolution Log with:
   - Date of fix
   - What was fixed
   - What file/module/change addressed it
   - Whether any related concerns remain open
4. If the fix is partial, update the status to PARTIAL and note what remains.

### Step 5 — Retire Outdated Reports

If a report is determined to be entirely resolved or no longer applicable:
1. Do NOT delete it.
2. Add a header annotation:
   ```markdown
   > **STATUS: RESOLVED** — All items addressed as of <date>.
   ```
3. Move all items to FIXED status in the Resolution Log.

### Step 6 — Summary

Produce a brief summary of actions taken:

```markdown
# Inconsistency Tracker Summary

## Reports Scanned: N
## Relevant Open Items Found: M
## New Issues Discovered: K
## Items Resolved: J
## New Report Created: <path or "none">
## Reports Updated: <paths or "none">
```

## Key References

- `inconsistency/` — all inconsistency reports.
- `AGENTS.md` §§ Refactor and Inconsistency Reporting Policy, Using Existing
  Inconsistency Reports.
- `physics_and_conventions/physics_conventions_report.tex` — physics conventions.
- `API_REFERENCE.md` — public API reference.

## Quality Standards

- Every confirmed issue must be verifiable by reading the cited file and line.
- Do not flag stylistic preferences as inconsistencies unless they contradict a
  documented convention.
- Separate confirmed issues (verified by reading code) from suspected issues
  (inferred but not fully verified).
- Never silently fix an inconsistency without documenting it in a report.
- Resolution updates must be specific: cite the fix, not just "fixed".
- Timestamped filenames must use the current date and time, not placeholders.
