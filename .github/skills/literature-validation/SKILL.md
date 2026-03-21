---
name: literature-validation
description: "Validate cqed_sim against papers, textbooks, or external reference results. Use when reproducing figures, spectra, dynamics, gate benchmarks, analytical limits, or paper-specific methods and when deciding what belongs in test_against_papers versus tests or examples."
argument-hint: "Describe the source being validated, the target result, and the expected level of agreement."
---

# Literature Validation

Use this skill when the task is about matching external reference material rather than only internal behavior.

## When to Use

- Reproducing a paper figure, equation, spectrum, dynamics trace, or benchmark.
- Checking the simulator against textbook or literature expectations.
- Organizing a validation workflow under `test_against_papers/`.
- Splitting literature reproduction work from reusable regression tests.

## Placement Rules

- Put literature-driven validation workflows under `test_against_papers/`.
- Put reusable automated regression coverage under `tests/`.
- Put typical user-facing demos under `examples/`.
- Do not turn notebooks into the default path unless the task explicitly asks for notebook work.

## Procedure

1. Define the validation target.
   - Name the source, the result being reproduced, and the physical quantity being checked.
   - State the expected level of agreement and any acceptable tolerance.
2. Capture the assumptions.
   - Note approximations, truncation, rotating frame, units, parameter mapping, and any deviations from the source.
   - Consult `paper_summary/` when the repo already contains a summary for the source.
3. Build the workflow in the right place.
   - Keep the literature-specific reproduction in `test_against_papers/`.
   - Extract stable regression pieces into `tests/` if they should guard future changes.
4. Use existing simulator infrastructure.
   - Prefer `cqed_sim` entry points over standalone simulation code.
   - If `cqed_sim` is insufficient, state the limitation explicitly before deviating.
5. Record the comparison clearly.
   - State what matched, what differed, and whether the disagreement is expected.
   - Include tolerances and interpretation instead of only raw plots or numbers.
6. Synchronize documentation when relevant.
   - Update module docs, examples, or physics documentation if the validation establishes or changes a repo convention.

## Completion Checklist

- The source and target result are clearly identified.
- Assumptions and approximations are explicit.
- Literature-specific work lives under `test_against_papers/`.
- Reusable regression coverage was moved into `tests/` when appropriate.
- The comparison states the expected agreement level and observed outcome.