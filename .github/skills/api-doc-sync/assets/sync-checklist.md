# API and Documentation Sync Checklist

Use this checklist after any public API or workflow change.

## Public API

- `API_REFERENCE.md`
- top-level package exports or import examples
- public function signatures, parameter names, defaults, and return descriptions

## Website Documentation

- `documentations/api/`
- `documentations/user_guides/`
- `documentations/tutorials/`
- `documentations/index.md`
- `mkdocs.yml` when navigation or page coverage changed
- run `python -m mkdocs build --strict` after website-source changes
- include the regenerated `site/` output in the same commit

## Local Feature Docs

- module `README.md` files for major feature areas
- `examples/` snippets or tutorial assets when usage changed

## Physics and Validation

- `physics_and_conventions/physics_conventions_report.tex` if semantics changed
- `tests/` coverage for public behavior
- `test_against_papers/` when the change affects literature-validation workflows

## Final Drift Pass

- search for old parameter names
- search for moved file paths
- search for stale imports or tutorial references
