# .copilot/skills — Reusable AI Workflow Skills

This directory contains structured Skill definitions for AI-assisted development
on the `cqed_sim` codebase.

## Available Skills

| Skill | Directory | Purpose |
|-------|-----------|---------|
| **Refactor Reviewer** | `refactor-reviewer/` | Audit structural changes: imports, tests, notebooks, API surface |
| **Calibration Audit** | `calibration-audit/` | Validate SQR calibration results for physics consistency and convergence |
| **Artifact Builder** | `artifact-builder/` | Generate LaTeX tables, figure captions, README updates, Overleaf fragments |

## How to Invoke

Mention the skill by name in a Copilot prompt, e.g.:

```
@workspace Invoke the refactor-reviewer skill on the current working tree.
```

Or use short-form:

```
@workspace /refactor-reviewer
```

## Outputs

All Skill outputs are written to `outputs/report/`.
