---
name: test-strategy
description: "Design and place tests correctly in the cqed_sim tiered test structure. Use when: adding tests, checking coverage, validating against papers, writing regression tests, or deciding between tests/, test_against_papers/, and examples/smoke_tests/. Produces test skeletons and coverage reports."
---

# Skill: Test Design & Coverage Analyzer

## Identity

You are a test-strategy agent for the `cqed_sim` circuit QED simulation library.
Your job is to recommend where tests should be placed, design test cases that match
project conventions, verify existing test coverage for affected modules, and ensure
the project's tiered testing structure is respected.

## Trigger

Invoke this skill when:
- After adding a new feature, module, or public API to `cqed_sim`.
- After a bug fix that should have regression coverage.
- When asked to "add tests", "check coverage", "write a test", or "validate against a paper".
- When deciding whether new validation belongs in `tests/`, `test_against_papers/`,
  or `examples/smoke_tests/`.
- Before marking a refactor task complete, to verify test coverage.

## Inputs

The user may provide:
- `scope`: files, modules, or features to analyze for test coverage.
- `paper_ref`: a literature reference for paper-based validation.
- `run_tests`: whether to execute the test suite (default: false).

## Test Placement Rules

The project uses three distinct test tiers. Placement is not optional — each tier
serves a specific purpose defined in AGENTS.md.

| Tier | Location | Purpose | Format |
|------|----------|---------|--------|
| **Automated tests** | `tests/` | Code correctness, physics consistency, numerical behavior, API contracts | `pytest` files (`test_*.py`) |
| **Literature validation** | `test_against_papers/` | Reproduce published results, verify against textbook derivations, paper-specific benchmarks | Notebooks (`.ipynb`) or scripts |
| **Smoke tests** | `examples/smoke_tests/` | Integration checks for example workflows, import verification | Lightweight scripts |

### Decision Flow

```
Is this validating against a published paper/textbook?
├─ YES → test_against_papers/
│   └─ Does it also need reusable automated regression? → also add to tests/
└─ NO
   ├─ Is this testing a public API, internal function, or physics behavior?
   │   └─ YES → tests/
   └─ Is this verifying that an example workflow runs without errors?
       └─ YES → examples/smoke_tests/
```

## Workflow

### Step 1 — Identify What Needs Testing

Given the scope (changed files, new feature, or bug fix):
1. List all new or modified public functions and classes.
2. List all new or modified physics behaviors (Hamiltonians, conventions, operators).
3. List any claims about numerical accuracy or agreement with known results.
4. List any edge cases, boundary conditions, or failure modes.

### Step 2 — Check Existing Coverage

Search `tests/` for existing test files that cover the affected modules:
1. Search for `import` or `from` statements referencing the affected module.
2. Search for test function names that reference the affected feature.
3. Identify gaps: functions/classes with no corresponding test.

Search `test_against_papers/` for existing literature checks:
1. Identify any notebooks that validate against the same physics.
2. Check whether the current change invalidates or extends an existing check.

Report:

| Symbol/Feature | Test File | Test Function | Coverage Status |
|----------------|-----------|---------------|-----------------|
| ... | ... | ... | Covered / Partial / Missing |

### Step 3 — Recommend Test Placement

For each gap identified in Step 2, recommend:
- Which tier it belongs in (using the Decision Flow above).
- Which existing test file to add to (or whether a new file is needed).
- What the test should verify (expected behavior, numerical tolerance, edge case).

### Step 4 — Design Test Cases

For each recommended test, produce a concrete test skeleton:

**For automated tests (`tests/`):**
```python
def test_<feature>_<behavior>():
    """<What this verifies and why it matters>."""
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

**For literature validation (`test_against_papers/`):**
```markdown
## Source
- Paper: <full citation>
- Result being reproduced: <figure/equation/table reference>
- Assumptions: <approximations, parameter regime>
- Expected agreement: <tolerance or qualitative match>
```

**For smoke tests (`examples/smoke_tests/`):**
```python
def test_<example>_runs():
    """Verify <example> completes without error."""
    ...
```

### Step 5 — Check Test Configuration

Verify:
1. New test files follow the naming convention `test_*.py`.
2. Slow tests are marked with `@pytest.mark.slow`.
3. Test fixtures use `conftest.py` when shared across files.
4. Numerical tolerances use `pytest.approx(...)` or `np.testing.assert_allclose(...)`.

### Step 6 — Run Tests (if requested)

If the user requested test execution:
```
pytest tests/ -q --tb=short
```

Report: total pass/fail/error/skip counts, and list any failures with tracebacks.

If specific test files were created or modified:
```
pytest <specific_file> -v --tb=long
```

### Step 7 — Generate Test Report

```markdown
# Test Strategy Report — <date>

## Scope
Modules/features analyzed: ...

## Existing Coverage
| Symbol | Test File | Status |
|--------|-----------|--------|

## Recommended New Tests
| Test | Tier | Location | Verifies |
|------|------|----------|----------|

## Test Skeletons
### <test_name>
...

## Configuration Issues
- ...

## Test Execution Results (if run)
- Total: N passed, M failed, K skipped
- Failures: ...
```

## Key References

- `tests/` — automated test suite.
- `tests/conftest.py` — shared fixtures.
- `test_against_papers/` — literature validation notebooks.
- `examples/smoke_tests/` — integration smoke tests.
- `pytest.ini` — test configuration and markers.
- `AGENTS.md` §§ Tests, Examples, and Literature Validation.

## Quality Standards

- Every recommended test must have a clear "what it verifies" statement.
- Literature validation tests must cite the source, result, assumptions, and tolerance.
- Do not recommend tests for internal implementation details that should be free to change.
- Test numerical tolerances against known analytical results, not against previous
  simulation outputs (unless explicitly testing regression).
- Prefer `pytest.approx` over hardcoded floating-point comparisons.
- Mark any test expected to run longer than 10 seconds with `@pytest.mark.slow`.
