Status: fixed on 2026-03-12.

## Confirmed issue

### `cqed_sim.experiment` mixed reusable primitives with workflow recipes

- What it is:
  - The package exposed reusable state-preparation and measurement helpers together with high-level protocol orchestration and canned workflow modules.
- Where it appeared:
  - `cqed_sim/experiment/*`
  - top-level `cqed_sim/__init__.py`
  - `README.md`, `API_REFERENCE.md`, and website docs under `documentations/`
  - tests and notebooks importing workflow APIs from the library surface
- Affected components:
  - package namespace design
  - public API documentation
  - workflow tests and example notebooks
- Why inconsistent:
  - The intended repository boundary is that `cqed_sim` remains a reusable simulation library, while typical experiment recipes and orchestration flows belong in `examples/`.
- Consequences:
  - the core package looked larger and more workflow-specific than intended
  - notebooks and tests were coupled to recipe-style APIs instead of stable low-level modules
  - documentation blurred the distinction between library primitives and example workflows

## Resolution status

- Fixed:
  - moved reusable state-preparation primitives to `cqed_sim/core/state_prep.py`
  - moved reusable measurement and readout-chain primitives to `cqed_sim/measurement/`
  - moved Kerr free-evolution and sequential sideband-reset workflows to `examples/workflows/` with top-level example scripts under `examples/`
  - removed the `cqed_sim.experiment` package namespace from the import package layout
  - rewired tests, notebooks, and docs to the new low-level module paths or example-side workflow modules
- Remaining open:
  - none identified in this refactor pass
