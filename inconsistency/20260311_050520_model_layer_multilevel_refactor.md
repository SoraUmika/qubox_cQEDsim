# Model Layer Multilevel Refactor Inconsistency Report

Created: 2026-03-11 05:05:20 local time

Status: Fixed on 2026-03-11.

## Confirmed issues

### 1. The reusable model layer is duplicated across the two concrete dispersive models

- What it is:
  - `DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel` each manually implement operator embedding, operator caches, transmon projectors, transmon transition operators, sideband operators, and static Hamiltonian assembly.
- Where it appears:
  - `cqed_sim/core/model.py`
  - `cqed_sim/core/readout_model.py`
- Affected components:
  - model maintenance
  - multilevel transmon support
  - future extensions to more bosonic modes or other subsystem types
- Why inconsistent:
  - the repository already exposes multilevel-transmon behavior in both models, but the shared physics is not factored through a single reusable abstraction.
- Consequence:
  - changes to conventions, couplings, or operator naming risk diverging between the two concrete models.

### 2. Generic helper code infers model structure indirectly instead of using an explicit subsystem specification

- What it is:
  - helper paths such as `cqed_sim/core/frequencies.py` branch on `len(model.subsystem_dims)` to decide which arguments are valid.
- Where it appears:
  - `cqed_sim/core/frequencies.py`
  - related helper code that assumes only the current two concrete model shapes
- Affected components:
  - transition-frequency helpers
  - future model extensibility
- Why inconsistent:
  - the repository is moving toward multilevel and multimode support, but the generic helper layer still reasons from tuple length rather than declared subsystem metadata.
- Consequence:
  - adding new model shapes would require more ad hoc branching instead of using a canonical model specification.

### 3. The public API exposes multilevel transmon dimensions, but there is no general-purpose public model abstraction behind them

- What it is:
  - current public models accept `n_tr >= 2`, but users can only access that through the specialized two-mode and three-mode dispersive classes.
- Where it appears:
  - `cqed_sim/core/model.py`
  - `cqed_sim/core/readout_model.py`
- Affected components:
  - public model API
  - future extension toward more general cQED systems
- Why inconsistent:
  - the implementation already supports multilevel transmons numerically, but the API still lacks a reusable subsystem-based model layer that represents those systems directly.
- Consequence:
  - architecture pressure accumulates in the specialized wrapper classes instead of the shared core.

## Confirmed non-issues

### 4. The runtime transmon dimension itself is not the blocker

- What was checked:
  - core model dataclasses already accept `n_tr > 2`
  - sideband tests already exercise `n_tr = 3`
  - noise and extractor paths already include basic multilevel support
- Conclusion:
  - the main gap is architectural centralization, not an inability to simulate multilevel transmons at all.

## Suspected / unresolved items before the refactor

### A. SU(2)-specific workflow helpers should remain explicit

- Suspicion:
  - qubit-rotation and tomography helpers should stay clearly two-level rather than being silently generalized as part of the model refactor.
- Why unresolved:
  - that decision needs to be reflected in the final API/docs after the universal model is introduced.

## Resolution

The model-layer refactor was completed with the following changes:

- Added `cqed_sim/core/universal_model.py` with:
  - `TransmonModeSpec`
  - `BosonicModeSpec`
  - `DispersiveCouplingSpec`
  - `UniversalCQEDModel`
- Re-implemented:
  - `cqed_sim/core/model.py::DispersiveTransmonCavityModel`
  - `cqed_sim/core/readout_model.py::DispersiveReadoutTransmonStorageModel`
  as compatibility wrappers that delegate Hamiltonian assembly, operator generation, basis helpers, and transition-frequency logic to `UniversalCQEDModel`.
- Updated `cqed_sim/core/frequencies.py` so the generic transition helpers use explicit model capabilities instead of branching on `len(model.subsystem_dims)`.
- Added public exports for the new universal model layer in:
  - `cqed_sim/core/__init__.py`
  - `cqed_sim/__init__.py`
- Updated runtime helpers so cavity-only universal models behave consistently:
  - `cqed_sim/sim/runner.py`
  - `cqed_sim/sim/noise.py`
- Added regression coverage in `tests/test_34_universal_cqed_model.py`.

### Additional issue discovered and fixed during implementation

- What it was:
  - the first draft of the universal operator-alias policy exposed the same single storage mode under both the two-mode aliases (`a`, `adag`, `n_c`) and the three-mode aliases (`a_s`, `adag_s`, `n_s`).
- Consequence:
  - the noise builder could double-count storage decay if both alias families were present.
- Fix:
  - `cqed_sim/core/universal_model.py` now preserves the previous wrapper surfaces:
    - single-mode storage/cavity paths expose `a`, `adag`, `n_c`
    - storage/readout paths expose `a_s`, `adag_s`, `n_s` and `a_r`, `adag_r`, `n_r`
  - `cqed_sim/sim/noise.py` also now guards against duplicate bosonic collapse channels by identity.

### Final disposition

- Issue 1: fixed
- Issue 2: fixed
- Issue 3: fixed
- Item A: resolved by leaving SU(2)-specific workflow helpers explicit and documenting the boundary in the design note and API docs
