# Spectrum / Notebook Alignment Inconsistency Report

Created: 2026-03-12 23:59:11 local time
Status: fixed on 2026-03-13 for items 1-2.

## Confirmed issues

### 1. `examples/sideband_interactions.ipynb` assumes a repo-root import context

- What it is:
  - The notebook imports `examples.workflows.sequential_sideband_reset` and writes to `Path("examples") / "outputs" / ...` without first resolving the repository root.
- Where it appears:
  - `examples/sideband_interactions.ipynb`
- Affected components:
  - sideband example notebook execution
  - example output-path placement
- Why inconsistent:
  - the notebook lives under `examples/`, so a normal notebook kernel started from that directory does not have the repository root on `sys.path`.
  - the hard-coded `Path("examples") / "outputs"` path also nests outputs under `examples/examples/outputs/...` when the notebook runs from its own directory.
- Consequence:
  - the notebook fails to import the workflow helper under a normal `examples/` working directory and writes artifacts to an inconsistent location when it does run.

### 2. The library exposes static Hamiltonians but not a reusable public eigenspectrum API

- What it is:
  - models expose `static_hamiltonian(...)` and the codebase performs exact diagonalization inside `cqed_sim.analysis.parameter_translation`, but there is no public helper for users to compute vacuum-referenced eigenenergies/eigenstates from a model.
- Where it appears:
  - `cqed_sim/core/*`
  - `cqed_sim/analysis/parameter_translation.py`
  - `API_REFERENCE.md`
- Affected components:
  - model inspection workflows
  - notebook examples that need dressed levels
  - future spectrum-based analyses and plots
- Why inconsistent:
  - the repository already presents `cqed_sim` as a reusable simulator library with Hamiltonian access, but users still have to perform ad hoc diagonalization and energy shifting themselves to inspect energy levels.
- Consequence:
  - spectrum analysis is not available through a stable public API, and notebooks are pushed toward custom one-off code instead of reusing a documented library feature.

## Resolution summary

- Item 1 was resolved by making the notebook resolve the repository root explicitly before importing example workflow helpers and by writing outputs to the canonical `examples/outputs/...` tree.
- Item 2 was resolved by adding a reusable QuTiP-backed spectrum API with vacuum-referenced energies, eigenstates, and plotting support to the public library surface and using it from the notebook.
