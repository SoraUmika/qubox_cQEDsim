# Physics Report Build Task Automation Gap

## Summary

This report records the physics-conventions PDF build automation mismatch found
while following up on the holographic refactor on 2026-03-20.

## Confirmed Issues

- What the inconsistency is:
  The repository provided a reusable batch build script for the physics report,
  but it did not provide checked-in VS Code task definitions that used that
  script, and the script itself paused for interactive input on both success and
  failure.
- Where it appears:
  `physics_and_conventions/build_physics_conventions_report.bat` and the absence
  of a repo-backed `.vscode/tasks.json` task definition for the report build.
- What components it affects:
  VS Code task execution, agent-driven validation flows, and any non-interactive
  developer workflow that expects the PDF build command to terminate cleanly.
- Why it is inconsistent:
  Project policy requires physics-report updates to be followed by a reproducible
  build step, but the checked-in build surface was interactive while the active
  workspace tasks were external, tool-generated, and not stable repo artifacts.
- Consequences:
  Task-based builds could hang or fail even when direct `pdflatex` compilation
  succeeded, making documentation validation less reproducible than the repo
  intends.

## Suspected / Follow-up Questions

- If additional report-build workflows are added later, should they all delegate
  to the same batch script so task behavior stays centralized?

## Status

- Fixed on 2026-03-20.

## Fix Record

- Removed interactive `pause` calls from
  `physics_and_conventions/build_physics_conventions_report.bat` so the script
  returns a normal process exit code in non-interactive contexts.
- Added checked-in workspace task definitions in `.vscode/tasks.json` that call
  the batch script directly, replacing the prior ad hoc task wiring.
