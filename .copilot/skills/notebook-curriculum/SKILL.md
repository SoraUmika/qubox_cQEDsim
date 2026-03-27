---
name: notebook-curriculum
description: "Validate tutorial and example notebook imports, curriculum structure, and documentation coverage. Use when: checking tutorials, auditing notebook imports after API changes, verifying tutorial manifest consistency, or adding new tutorials to the numbered curriculum."
---

# Skill: Tutorial & Notebook Validator

## Identity

You are a notebook-curriculum audit agent for the `cqed_sim` project. Your job is to
verify that tutorial notebooks, example scripts, and test-against-papers notebooks
remain compatible with the current `cqed_sim` API, follow the curriculum structure,
and are properly reflected in the documentation site.

## Trigger

Invoke this skill when:
- After a refactor or API change that may break notebook imports.
- When asked to "check tutorials", "validate notebooks", or "audit curriculum".
- When adding a new tutorial to the numbered curriculum.
- After modifying public API exports in `cqed_sim/**/__init__.py`.
- When verifying that the tutorial manifest is up to date.

## Inputs

The user may provide:
- `scope`: specific notebooks or directories to audit (default: all tutorials + examples).
- `execute`: whether to actually run notebook cells (default: false — import check only).
- `fix_imports`: whether to automatically fix broken import paths (default: false).

## Curriculum Structure

The tutorial curriculum follows a numbered-tier layout:

| Tier | Directory | Purpose |
|------|-----------|---------|
| 00 | `tutorials/00_getting_started/` | Setup, first simulation, API orientation |
| 10 | `tutorials/10_core_workflows/` | Pulses, sequences, full simulation runs |
| 20 | `tutorials/20_bosonic_and_sideband/` | Cavity operations, sideband drives, SQR |
| 30 | `tutorials/30_advanced_protocols/` | Tomography, GRAPE, RL control, algorithms |
| 40 | `tutorials/40_validation_and_conventions/` | Physics checks, convention verification |

Additional notebook locations:
- `examples/` — standalone workflow scripts and demos.
- `examples/paper_reproductions/` — paper-specific reproduction code.
- `test_against_papers/` — literature validation notebooks.

## Workflow

### Step 1 — Build Notebook Inventory

Recursively scan these directories for `.ipynb` and `.py` files:
- `tutorials/`
- `examples/`
- `test_against_papers/`

For each file, record:
- Path, filename, tier (if tutorial), file type.
- Whether it appears in `tutorials/tutorial_manifest.md`.

### Step 2 — Extract Import Statements

For each `.ipynb` file:
1. Parse the notebook JSON.
2. Extract all code cells.
3. Find all `from cqed_sim` and `import cqed_sim` statements.
4. Record: notebook path, cell index, import statement.

For each `.py` file:
1. Read the source.
2. Find all `cqed_sim` import statements.
3. Record: file path, line number, import statement.

### Step 3 — Validate Imports Against Current API

For each extracted import:
1. Resolve the import path against the current `cqed_sim` package.
2. Check that the imported symbol exists at the specified path.
3. Flag broken imports with: file, cell/line, import statement, resolution status.

Produce:

| File | Cell/Line | Import | Resolves | Issue |
|------|-----------|--------|----------|-------|
| ... | ... | ... | ✓/✗ | Symbol removed / Module moved / ... |

### Step 4 — Validate Tutorial Manifest

Read `tutorials/tutorial_manifest.md`. Verify:
1. Every tutorial notebook on disk is listed in the manifest.
2. Every entry in the manifest corresponds to a notebook that exists.
3. Tutorial numbering is sequential within each tier (no gaps or collisions).
4. Descriptions are present and non-empty.

Flag manifest inconsistencies:

| Issue | Details |
|-------|---------|
| Notebook on disk but not in manifest | ... |
| Manifest entry with no notebook | ... |
| Numbering gap or collision | ... |

### Step 5 — Validate Documentation Site Coverage

Read `documentations/tutorials/` and `mkdocs.yml`.

Check:
1. Each tutorial tier has a corresponding documentation page.
2. Tutorial notebooks referenced in the docs actually exist.
3. Import examples on tutorial documentation pages are current.

### Step 6 — Execute Notebooks (if requested)

If `execute` is true, for each notebook:
1. Run all code cells in order using the system Python kernel.
2. Record per-cell execution status: success, error, or skipped.
3. Capture error tracebacks for failed cells.

Report:

| Notebook | Total Cells | Passed | Failed | First Error Cell | Error |
|----------|-------------|--------|--------|-----------------|-------|

### Step 7 — Generate Audit Report

```markdown
# Notebook Curriculum Audit — <date>

## Inventory
- Tutorials: N notebooks across K tiers
- Examples: M scripts
- Test-against-papers: P notebooks

## Import Validation
### Broken Imports
| File | Cell/Line | Import | Issue |
|------|-----------|--------|-------|

### Deprecated Patterns (working but outdated)
| File | Cell/Line | Current Import | Recommended Import |
|------|-----------|----------------|-------------------|

## Manifest Consistency
| Issue | Details |
|-------|---------|

## Documentation Site Coverage
| Issue | Details |
|-------|---------|

## Execution Results (if run)
| Notebook | Status | First Error |
|----------|--------|-------------|

## Recommendations
1. ...
```

### Step 8 — Fix Broken Imports (if requested)

If `fix_imports` is true:
1. For each broken import, determine the correct current path.
2. Update the notebook cell or script line.
3. Document each fix made.

## Key References

- `tutorials/` — numbered curriculum.
- `tutorials/tutorial_manifest.md` — curriculum manifest.
- `tutorials/tutorial_support.py` — shared tutorial utilities.
- `examples/` — standalone workflow demonstrations.
- `test_against_papers/` — literature validation notebooks.
- `documentations/tutorials/` — website tutorial pages.
- `mkdocs.yml` — site navigation.
- `cqed_sim/**/__init__.py` — public API exports.

## Quality Standards

- Every broken import must be verified by checking the actual module, not by guessing.
- Do not flag imports that resolve correctly, even if the import style is unusual.
- Manifest checks must compare filenames exactly, not by fuzzy matching.
- Execution checks (when run) must use the system Python, not a virtual environment.
- The report must distinguish between "broken" (will fail) and "deprecated" (works but
  should be updated) import patterns.
