# 2026-03-27 16:00:00 Full Codebase Consistency Audit

Audit triggered by a project-wide consistency sweep covering .gitignore setup,
API documentation sync, module README coverage, test organization, mkdocs
navigation, and prior inconsistency report review.

Update on 2026-03-27 later in the day:

- The `ConditionalPhaseSQRGate` symbol referenced below was subsequently
  removed from the active Gate I/O public API.
- Mentions of that symbol in this report are historical notes about the API
  surface that existed when this audit was written.

---

## Confirmed Issues

### 1. API_REFERENCE.md missing documentation for several public symbols

- **What:**
  The following public API surfaces are exported and functional but have no
  corresponding section in `API_REFERENCE.md`:
  - `GateRegistry` class and `gate_registry` singleton (`cqed_sim.unitary_synthesis`)
  - `GateOrderOptimizer` and `GateOrderConfig` (`cqed_sim.unitary_synthesis`)
  - `make_gate_from_matrix`, `make_gate_from_callable`, `make_gate_from_waveform` factory helpers
  - `ConditionalPhaseSQRGate` in the Gate I/O type union (`cqed_sim.io`)
  - `run_sweep()` parallel sweep helper (`cqed_sim.sim`)
  - `ReadoutTrace` dataclass (`cqed_sim.measurement`)
- **Where:** `API_REFERENCE.md` §17 (Unitary Synthesis), §8 (Gate I/O), §6 (Simulation Engine), §7 (Measurement).
- **Affected components:** `API_REFERENCE.md`, corresponding `documentations/api/` pages.
- **Why this is inconsistent:** AGENTS.md requires `API_REFERENCE.md` to stay synchronized with the public API. The symbols listed above are in `__all__` of their respective modules but undocumented.
- **Consequences:** Users and developers cannot discover these features from the documentation alone.

### 2. `cqed_sim/control/` module missing README.md

- **What:** The `control` module contains `ControlLine`, `HardwareContext`, `postprocess_grape_waveforms`, calibration maps, and the `make_three_line_cqed_context` factory, but has no module-level `README.md`.
- **Where:** `cqed_sim/control/`
- **Affected components:** Module discoverability, developer onboarding.
- **Why this is inconsistent:** AGENTS.md requires every major module area to have a local `README.md`.
- **Consequences:** Developers must read the `__init__.py` docstring or scattered references to understand the module's role.

### 3. Test file inside module directory instead of central `tests/` folder

- **What:** `cqed_sim/unitary_synthesis/tests/test_native_primitives.py` contains validation tests that belong under the central `tests/` directory.
- **Where:** `cqed_sim/unitary_synthesis/tests/`
- **Affected components:** Test discovery, test organization conventions.
- **Why this is inconsistent:** AGENTS.md states all tests must be placed under the top-level `tests/` folder.
- **Consequences:** These tests may not be picked up by project-wide pytest runs unless the runner explicitly traverses module subdirectories; they break the numbered naming convention.

### 4. Test numbering gap: `test_30` missing

- **What:** The numbered test sequence jumps from `test_29_performance_paths.py` to `test_31_multilevel_sideband_extension.py` with no `test_30_*` file.
- **Where:** `tests/`
- **Affected components:** Test organization continuity.
- **Why this is inconsistent:** Numbering gap suggests a planned or deleted test was never created/replaced.
- **Consequences:** Minor; no functional impact, but may confuse developers tracking test coverage by number.

### 5. No `.gitignore` file existed prior to this audit

- **What:** The repository had no `.gitignore`, so build artifacts (`__pycache__/`, `*.pyc`, `build/`, `site/`, `cqed_sim.egg-info/`, `.pytest_cache/`, LaTeX build files, `outputs/`, `agent_runs/`) were all unfiltered.
- **Where:** Repository root.
- **Affected components:** Repository hygiene, commit noise, storage.
- **Why this is inconsistent:** Standard Python project practice and general git hygiene require a `.gitignore`.
- **Consequences:** Over 950 `.pyc` files, 100+ mkdocs site files, and various generated artifacts would be committed if git is initialized without one.

---

## Suspected / Follow-up Questions

### A. `cqed_sim/tomo/README.md` unit convention

Resolved on 2026-04-13. The tomo helper docs now describe `DeviceParameters` as a tomography-specific Hz-to-rad/ns convenience path while keeping the core `cqed_sim` model layer documented as unit-coherent.

### B. `documentations/getting_started.md` placement

`mkdocs.yml` lists `getting_started.md` under "User Guides" in the nav, but the file is at the root of `documentations/` rather than inside `documentations/user_guides/`. MkDocs resolves this correctly so there is no build error, but it is mildly inconsistent with the nav hierarchy.

### C. Six submodules not re-exported at top level

`unitary_synthesis`, `observables`, `plotting`, `operators`, `quantum_algorithms`, and `holographic` (inside `quantum_algorithms`) are not re-exported from `cqed_sim/__init__.py`. This may be intentional to keep the top-level namespace focused, but it is worth confirming whether all of these should remain import-only via submodule paths.

### D. Missing READMEs in `examples/` subdirectories and `tools/`

`examples/audits/`, `examples/workflows/`, `examples/smoke_tests/`, `examples/hardware_context/`, `examples/quantum_algorithms/`, and `tools/` lack descriptive `README.md` files. These are lower-priority but reduce discoverability.

### E. Benchmark result files at root of `benchmarks/`

`benchmarks/latest_results.json` and `benchmarks/latest_results_review_*.json` are generated outputs that probably should be in `outputs/` or `.gitignore`'d. Currently included in the new `.gitignore`.

---

## Status

- **Current status:** Fixed on 2026-04-13. Issues 1, 2, and 5 were already resolved at audit time. Issues 3-4 were resolved by moving the stray module-local validation test into the numbered top-level suite as `tests/test_30_native_primitives.py`. Suspected item A is also resolved.
- **Resolution summary:**
  - Issue 1 resolved at audit time: Added `run_sweep` to §6.4, `ConditionalPhaseSQRGate` to §8, and §17.9–17.11 (user-defined gate factories, gate registry, gate-order search) to `API_REFERENCE.md`. Matching updates applied to `documentations/api/unitary_synthesis.md`, `documentations/api/gate_io.md`, and `documentations/api/simulation.md`.
  - Later on 2026-03-27, the `ConditionalPhaseSQRGate` portion of that documentation update was superseded because the symbol was removed from the active public API.
  - Issue 2 resolved: Created `cqed_sim/control/README.md`.
  - Issue 5 resolved: `.gitignore` created at repository root.
  - Issues 3-4 resolved on 2026-04-13: `cqed_sim/unitary_synthesis/tests/test_native_primitives.py` was migrated into the top-level suite as `tests/test_30_native_primitives.py`, which also closes the missing `test_30_*` numbering gap.
  - Suspected item A resolved on 2026-04-13: the tomo helper docs now describe the rad/ns path as helper-specific rather than a global simulator requirement.

---

## Fix Record

- **Fixed by:**
  - `.gitignore` created at repository root covering Python artifacts, build outputs, IDE files, OS files, Jupyter checkpoints, LaTeX build artifacts, mkdocs site output, `outputs/`, `agent_runs/`, and benchmark caches.
  - `API_REFERENCE.md` updated with documentation for `run_sweep`, `ConditionalPhaseSQRGate`, `make_gate_from_matrix`, `make_gate_from_callable`, `make_gate_from_waveform`, `GateRegistry`, `gate_registry`, `GateOrderOptimizer`, `GateOrderConfig`, and `GateOrderSearchResult`. The `ConditionalPhaseSQRGate` portion was later superseded when that symbol was removed from the active API.
  - `documentations/api/gate_io.md` updated with `ConditionalPhaseSQRGate` and corrected Gate union type. The `ConditionalPhaseSQRGate` portion was later superseded when that symbol was removed from the active API.
  - `documentations/api/unitary_synthesis.md` updated with user-defined gate factories, gate registry, and gate-order search sections.
  - `cqed_sim/control/README.md` created with module purpose, key classes, usage examples, and integration notes.
  - `tests/test_30_native_primitives.py` now carries the previously module-local unitary-synthesis primitive validation coverage.
  - `cqed_sim/unitary_synthesis/tests/test_native_primitives.py` was removed from the package tree.
  - `cqed_sim/tomo/device.py`, `cqed_sim/tomo/README.md`, `API_REFERENCE.md`, `documentations/api/tomography.md`, and `documentations/api/analysis.md` were updated on 2026-04-13 so the tomography helper's rad/ns path is documented as helper-specific and consistent with the repo's unit-coherent global contract.
- **Remaining concerns:**
  - Suspected items B-E should be evaluated individually.
