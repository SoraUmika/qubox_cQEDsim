# Refactor Review Checklist

Use this checklist for every structural change to `cqed_sim/`.

## Pre-Review

- [ ] Identify the base ref (usually `main`).
- [ ] Confirm the branch builds: `pip install -e .` succeeds.
- [ ] Confirm baseline tests pass: `pytest tests/ -q`.

## API Surface

- [ ] All `__init__.py` re-exports are consistent with `REFRACTOR_NOTES.md`.
- [ ] No public symbol was removed without a deprecation note.
- [ ] `README.md` API Summary matches actual exports.
- [ ] New public functions have docstrings with parameter descriptions.

## Import Integrity

- [ ] All `from cqed_sim.*` imports in `cqed_sim/` resolve.
- [ ] All `from cqed_sim.*` imports in `tests/` resolve.
- [ ] All `from cqed_sim.*` imports in `examples/` resolve.
- [ ] All `from cqed_sim.*` imports in notebooks resolve.

## Test Suite

- [ ] No new test failures introduced (compare pre/post counts).
- [ ] Any moved test still runs under its new path.
- [ ] `conftest.py` fixtures still accessible to all test files.

## Notebooks

- [ ] `sequential_simulation.ipynb` imports all resolve.
- [ ] `SQR_calibration.ipynb` imports all resolve.
- [ ] Notebook generators (`outputs/generate_*.py`) run without error.

## Documentation

- [ ] `REFRACTOR_NOTES.md` updated if modules moved.
- [ ] `README.md` updated if API changed.
- [ ] `experiment_mapping.md` unchanged (or updates noted).

## Final

- [ ] Report generated at `outputs/report/refactor_review_*.md`.
- [ ] All broken imports have proposed patches.
- [ ] No open questions remain.
