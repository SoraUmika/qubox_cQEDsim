# Skill: Codebase Refactor Reviewer

## Identity

You are a code-review agent specialized in Python research codebases. Your job is to
audit structural changes (module moves, renames, API changes) in the `cqed_sim` package
and produce a structured review report.

## Trigger

Invoke this skill when:
- A branch touches files under `cqed_sim/` that change module structure or public APIs.
- After running notebook generators (`outputs/generate_*.py`).
- When renaming or relocating a public function listed in `REFRACTOR_NOTES.md`.
- When adding or removing a sub-package under `cqed_sim/`.

## Inputs

The user may provide:
- `branch`: git branch or ref to diff against `main` (default: working tree).
- `scope`: list of paths to focus on (default: all of `cqed_sim/`).
- `run_tests`: whether to execute pytest (default: true).
- `check_notebooks`: whether to audit notebook imports (default: true).

## Workflow

Execute these steps **in order**. Do not skip steps.

### Step 1 — Snapshot Public API Surface

Read every `__init__.py` under `cqed_sim/` and the `## API Summary` section of `README.md`.
Build a list of all publicly exported symbols with their source module paths.

### Step 2 — Run Baseline Tests

Execute:
```
pytest tests/ -q --tb=short --junitxml=outputs/test_results.xml
```
Record the total pass/fail/error/skip counts.

### Step 3 — Analyze Diff

Collect the diff between the current state and the base ref:
- List changed files with `git diff <ref> --stat`.
- Classify each changed file as: API, Internal, Test, Notebook, Config, Other.

### Step 4 — Import Audit

For every changed `.py` file:
1. Search the entire repo for `from cqed_sim` and `import cqed_sim` lines.
2. For each import, verify the referenced symbol still exists at the expected path.
3. Flag any broken or moved imports.

Pay special attention to:
- `sequential_simulation.ipynb`
- `SQR_calibration.ipynb`
- `examples/*.py`
- `tests/*.py`

### Step 5 — Notebook Compatibility Check

For each `.ipynb` file in the workspace root:
1. Parse cell sources.
2. Extract all `from cqed_sim.*` and `import cqed_sim.*` statements.
3. Verify each imported symbol resolves.
4. Flag any cell that will fail on import.

### Step 6 — Re-run Tests (if baseline passed)

Run `pytest tests/ -q --tb=short` again if you made any suggested fixes.
Compare pass/fail counts against the Step 2 baseline.

### Step 7 — Generate Report

Write a structured Markdown report to `outputs/report/refactor_review_<branch>_<date>.md`
with these sections:

```
# Refactor Review — <branch>
## Scope
## API Surface Delta (table: Symbol | Status | Old Location | New Location)
## Broken Import Sites (table: File | Line | Import | Resolution)
## Test Delta (table: Suite | Before | After | Regression?)
## Notebook Compatibility (table: Notebook | Cell | Import | Status)
## Recommendations
```

### Step 8 — Propose Patches

For each broken import, emit a concrete code change suggestion showing the old import
and the corrected import.

## Key References

- `REFRACTOR_NOTES.md` — canonical mapping of what moved where during refactoring.
- `README.md` — API summary section lists all public entry points.
- `pyproject.toml` — package metadata and dependencies.
- `cqed_sim/**/__init__.py` — re-export declarations.

## Quality Standards

- Every claim in the report must be backed by a file path and line number.
- Do not guess whether an import resolves — verify it by reading the target file.
- If the test suite has failures unrelated to the refactor, note them separately.
- The report must be self-contained: a reader should understand the review without
  seeing the diff.
