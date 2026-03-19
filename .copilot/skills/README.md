# .copilot/skills — Reusable AI Workflow Skills

This directory contains structured Skill definitions for AI-assisted development
on the `cqed_sim` codebase.

## Available Skills

### Original Skills

| Skill | Directory | Purpose |
|-------|-----------|---------|
| **Refactor Reviewer** | `refactor-reviewer/` | Audit structural changes: imports, tests, notebooks, API surface |
| **Calibration Audit** | `calibration-audit/` | Validate SQR calibration results for physics consistency and convergence |
| **Artifact Builder** | `artifact-builder/` | Generate LaTeX tables, figure captions, README updates, Overleaf fragments |

### Physics & Convention Skills

| Skill | Directory | Purpose |
|-------|-----------|---------|
| **Physics Conventions** | `physics-conventions/` | Validate physics consistency: Hamiltonians, signs, frames, units, chi convention, drive phase; update `physics_conventions_report.tex` |
| **QuTiP Wrapper** | `qutip-wrapper/` | Check QuTiP 5.x native support before custom implementation; design convention-compliant wrappers; flag ad hoc usage |

### Documentation & Tracking Skills

| Skill | Directory | Purpose |
|-------|-----------|---------|
| **Documentation Sync** | `documentation-sync/` | Verify consistency across `API_REFERENCE.md`, `documentations/`, module READMEs, and `mkdocs.yml` |
| **Inconsistency Tracker** | `inconsistency-tracker/` | Manage lifecycle of `inconsistency/` reports: pre-work scan, new report generation, resolution tracking |

### Testing & Validation Skills

| Skill | Directory | Purpose |
|-------|-----------|---------|
| **Test Strategy** | `test-strategy/` | Recommend test placement (`tests/`, `test_against_papers/`, `examples/smoke_tests/`), design test cases, check coverage |
| **Notebook Curriculum** | `notebook-curriculum/` | Validate tutorial/example notebook imports, curriculum structure, manifest consistency, documentation site coverage |

## How to Invoke

Mention the skill by name in a Copilot prompt, e.g.:

```
@workspace Invoke the refactor-reviewer skill on the current working tree.
```

Or use short-form:

```
@workspace /refactor-reviewer
```

## Skill Combinations

Common multi-skill workflows:

| Task | Skills to Invoke |
|------|-----------------|
| Full refactor | `inconsistency-tracker` → `refactor-reviewer` → `physics-conventions` → `documentation-sync` → `test-strategy` |
| New simulation feature | `qutip-wrapper` → `physics-conventions` → `test-strategy` → `documentation-sync` |
| API change | `refactor-reviewer` → `documentation-sync` → `notebook-curriculum` |
| Post-calibration review | `calibration-audit` → `artifact-builder` |
| Pre-merge validation | `notebook-curriculum` → `test-strategy` → `documentation-sync` |

## Outputs

All Skill outputs are written to `outputs/report/` or `inconsistency/` as appropriate.
