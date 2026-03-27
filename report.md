# `cqed_sim` — Repository Sweep Report

**Date:** 2026-03-20  
**Scope:** Full `cqed_sim` package (141 Python files, 22 subpackages), tests, documentation, and repo-side tooling.

---

## Executive Summary

The `cqed_sim` codebase is in **good overall health**. The package imports cleanly
under Python 3.12.10, all 141 `.py` files pass AST syntax checking with zero errors,
no bare `except:` clauses exist, no `TODO`/`FIXME`/`HACK` markers remain, and all
`__init__.py` re-exports resolve without circular imports. The VS Code workspace
reports only 2 markdown lint warnings (missing language specifiers on fenced code blocks
in `documentations/architecture.md`), with zero Python errors.

This report catalogs the current state of **previously reported issues** (19 of 21
inconsistency reports fully resolved), **open items carried forward** from the prior
audit, and **newly identified issues** found during this sweep.

---

## 1. Syntax and Import Health

| Check | Result |
|-------|--------|
| AST syntax check (141 files) | **Pass** — 0 errors |
| Top-level `import cqed_sim` | **Pass** — 313 exported names |
| Circular imports | **None detected** |
| All `__init__.py` exports valid | **Yes** (22/22 subpackages) |
| Bare `except:` clauses | **None** |
| `TODO`/`FIXME`/`HACK` markers | **None** |
| VS Code compile/lint errors | 2 markdown lint warnings only (no Python errors) |

---

## 2. Deprecated API Usage

| Pattern | Occurrences | Status |
|---------|-------------|--------|
| `np.trapz` (removed in NumPy 2.x) | 3 | **Correctly handled** — all sites use `np.trapezoid` with `np.trapz` fallback |
| `np.random.seed()` (legacy RNG) | 0 | **Previously fixed** |
| Bare `except Exception: pass` | 0 | **Previously fixed** (narrowed in `universal_model.py`) |

---

## 3. Previously Reported Inconsistencies — Status

Of the 21 inconsistency reports in the `inconsistency/` folder:

| # | Report | Date | Status |
|---|--------|------|--------|
| 1 | Dispersive Convention Migration | 2026-03-11 | **Fixed** |
| 2 | Sideband / Multilevel / Kerr Audit | 2026-03-11 | **Fixed** |
| 3 | Usage Examples Spectroscopy Sign Mismatch | 2026-03-11 | **Fixed** |
| 4 | Notebook/Example Organization Audit | 2026-03-11 | **Fixed** |
| 5 | Model Layer Multilevel Refactor | 2026-03-11 | **Fixed** |
| 6 | Experiment Namespace Boundary Refactor | 2026-03-12 | **Fixed** |
| 7 | Spectrum Notebook Alignment | 2026-03-12 | **Fixed** |
| 8 | Sideband Notebook Core-Model Setup | 2026-03-13 | **Fixed** |
| 9 | Tutorial Curriculum Refactor Audit | 2026-03-13 | **Fixed** |
| 10 | Convention Cleanup Audit | 2026-03-13 | **Fixed** |
| 11 | Tutorial Execution dim=1 Bosonic Mode | 2026-03-13 | **Fixed** |
| 12 | Example Tutorial Migration Audit | 2026-03-15 | **Fixed** |
| 13 | Unitary Synthesis Pulse Backend Scope | 2026-03-15 | **Fixed** |
| 14 | Unitary Synthesis Phase 2 Runtime Audit | 2026-03-15 | **Fixed** |
| 15 | Unitary Synthesis System Coupling Refactor | 2026-03-15 | **Fixed** |
| 16 | Holographic Sim Architecture Audit | 2026-03-15 | **Fixed** (deferred: non-TI channels) |
| 17 | Optimal Control Follow-up Audit | 2026-03-17 | **Fixed** |
| 18 | Performance Optimization Refactor | 2026-03-17 | **Fixed** (deferred: GPU/JAX GRAPE) |
| 19 | Cleanup Pass 1 | 2026-03-17 | **Fixed** |
| 20 | Cleanup Pass 2 | 2026-03-17 | **Fixed** |
| 21 | OC Hardware Pipeline Refactor | 2026-03-18 | **Fixed** (deferred: hard constraints) |

**19 of 21 fully resolved.** The remaining 2 have documented deferred follow-up items
(non-TI holographic channels, GPU/JAX GRAPE gradients, hardware hard constraints) that
are design scope boundaries rather than bugs.

---

## 4. Open Items Carried Forward

These items were identified in the prior report (2026-03-18) and remain open:

### 4.1 ~~Missing `@abstractmethod` Decorators~~ (FIXED)

- **`cqed_sim/unitary_synthesis/systems.py`:** `QuantumSystem` now inherits `ABC` and
  `hilbert_dimension()` is decorated with `@abstractmethod`.
- **`cqed_sim/unitary_synthesis/sequence.py`:** `GateBase` now inherits `ABC` and
  `ideal_unitary()` is decorated with `@abstractmethod`.

**Status:** Fixed in this sweep.

### 4.2 `FrameSpec.omega_c_frame` Naming Ambiguity

`FrameSpec.omega_c_frame` is the underlying frozen field; `omega_s_frame` is a
read-only alias. In three-mode contexts, the "c" in `omega_c_frame` is ambiguous
("cavity" vs "storage"). No migration to clearer naming has been done.

**Severity:** Low  
**Impact:** Confusing for new developers working with three-mode models.

### 4.3 Waveform Bridge — Limited Gate Type Coverage (Documented)

`waveform_bridge` supports only `QubitRotation`, `Displacement`, and `SQR`. The
gate types `SNAP` and `FreeEvolveCondPhase` raise `TypeError`.
This is a fundamental design limitation (no pulse builders for these types).

**Severity:** Documented — No change needed unless pulse builders are added.

### 4.4 Dense Backend Limitations (Documented)

`NumPyBackend` and `JaxBackend` are piecewise-constant dense solvers, not drop-in
replacements for QuTiP's adaptive ODE solver on large Hilbert spaces. GPU/JAX GRAPE
integration is deferred.

**Severity:** Documented — Working as intended for small-system checks.

### 4.5 Synthetic I/Q Readout Model (Documented)

The fallback I/Q model in `measurement/qubit.py` is a simple uncalibrated Gaussian
cluster model. Documented in API_REFERENCE.md.

**Severity:** Documented — Expected behavior.

---

## 5. Newly Identified Issues

### ~~5.1 `API_REFERENCE.md` — Stale Gap Entries in Section 21~~ (FIXED)

Two entries in API_REFERENCE.md §21 have been marked as resolved:

1. **"Higher-Order Coefficients Lack Isolated Tests"** — Marked resolved (tests in `test_46`).
2. **"`targets.py` Contains User-Specific Hardcoded Paths"** — Marked resolved (cleaned 2026-03-17).

**Status:** Fixed in this sweep.

### ~~5.2 `API_REFERENCE.md` — `plot_energy_levels` Module Path~~ (FIXED)

`plot_energy_levels` has been moved to a dedicated `### Energy Levels (plotting.energy_levels)`
subsection in API_REFERENCE.md.

**Status:** Fixed in this sweep.

### ~~5.3 Test File Numbering Conflicts~~ (FIXED)

Conflicting test files have been renumbered:

| Original | New |
|----------|-----|
| `test_35_tutorial_api_conventions.py` | `test_55_tutorial_api_conventions.py` |
| `test_52_operators.py` | `test_56_operators.py` |
| `test_52_phase5_advanced.py` | `test_57_phase5_advanced.py` |

**Status:** Fixed in this sweep.

### ~~5.4 `architecture.md` Markdown Lint Warnings~~ (FIXED)

Both fenced code blocks in `documentations/architecture.md` now have `text` language
specifiers. The MD040 lint warnings are resolved.

**Status:** Fixed in this sweep.

### ~~5.5 `mkdocs.yml` — `architecture.md` Navigation Placement~~ (FIXED)

`architecture.md` has been moved under a "Developer Guide" section in the mkdocs nav.

**Status:** Fixed in this sweep.

### 5.6 Empty `tests/rl_control/` Subdirectory

`tests/rl_control/` exists but contains no test files. All RL tests are top-level
(`test_38`, `test_39`, `test_43`, `test_44`), inconsistent with the pattern used for
`tests/unitary_synthesis/`, `tests/sim/`, etc.

**Severity:** Low — Tests run fine; organizational inconsistency only.

---

## 6. Documentation Coverage

### 6.1 API_REFERENCE.md

| Area | Coverage |
|------|----------|
| Core modules (core, sim, pulses, gates, sequence) | **Complete** |
| Calibration (SQR, multitone, targets) | **Complete** |
| Optimal control (GRAPE, hardware, penalties) | **Complete** |
| Unitary synthesis (engine, primitives, targets) | **Complete** |
| RL control | **Partial** — main classes listed; action/observation/reward details in module README |
| System ID | **Sparse** — referenced but minimal detail |
| Holographic simulation | **Referenced** — defers to module README |

### 6.2 Module-Level READMEs

All modules required by AGENTS.md have READMEs:

| Module | README Present |
|--------|----------------|
| `optimal_control` | Yes |
| `rl_control` | Yes |
| `unitary_synthesis` | Yes |
| `quantum_algorithms/holographic_sim` | Yes |
| `sim` | Yes |
| `sequence` | Yes |
| `tomo` | Yes |
| `system_id` | Yes |
| `io` | Yes |
| `observables` | Yes |
| `operators` | Yes |
| `gates` | Yes |

Modules without READMEs (not strictly required but would improve discoverability):
`core`, `pulses`, `measurement`, `calibration`, `calibration_targets`, `analysis`,
`backends`, `plotting`.

### 6.3 Website Documentation (`documentations/`)

| Page | Status |
|------|--------|
| Getting Started, Installation | Present and current |
| Physics & Conventions | Present, links to LaTeX source |
| User Guides (8 pages) | Present and current |
| Tutorials (curriculum index) | Present |
| API Reference pages | Present for major modules |
| Architecture | Present (minor lint warnings) |

No missing pages detected.

---

## 7. Test Suite Structure

### 7.1 Coverage by Subpackage

| Module | Dedicated Test Coverage | Assessment |
|--------|------------------------|------------|
| core (models, frames, conventions) | 15+ test files | Excellent |
| sim (solver, noise, extractors) | 10+ test files | Excellent |
| pulses (envelopes, builders, hardware) | 5+ test files | Good |
| gates (all categories) | 3 test files | Good |
| calibration (SQR, multitone) | 4 test files | Good |
| calibration_targets (all protocols) | Subdirectory with per-protocol tests | Good |
| optimal_control (GRAPE, hardware) | 5 test files | Good |
| rl_control (env, physics, regression) | 4 test files | Good |
| unitary_synthesis | Subdirectory with dedicated tests | Good |
| measurement | 2 test files | Moderate |
| tomo | 1 test file | Light |
| observables | 1 test file (`test_53`) | Light |
| operators | 1 test file (`test_52_operators`) | Light |
| system_id | 1 test file (`test_48`) | Light |
| io | 1 test file (`test_47`) | Light |
| plotting | 1 smoke test (`test_49`) | Minimal |
| backends | Covered via sim tests | Indirect |
| quantum_algorithms | Subdirectory with tests | Moderate |

### 7.2 Test Infrastructure

- **pytest.ini:** Minimal and valid; defines `slow` marker and quiet output.
- **Total numbered test files:** ~54 (plus subdirectory tests).
- **Naming convention:** Sequential `test_NN_description.py` (with noted numbering gaps/collisions).

---

## 8. Package Dependencies

**Declared in `pyproject.toml`:**

| Dependency | Required By | Status |
|------------|-------------|--------|
| `numpy ≥1.24` | Everywhere | Active |
| `scipy ≥1.10` | Solvers, optimization | Active |
| `qutip ≥5.0` | Core simulation | Active |
| `matplotlib ≥3.8` | Plotting, observables | Active |
| `pandas ≥2.0` | `holographicSim.py`, `unitary_synthesis/progress.py` | Active (2 usage sites) |
| `jax` (optional) | `backends/jax_backend.py`, `propagators_jax.py` | Optional, gracefully handled |

No undeclared dependencies detected. `pandas` is used in 2 files and is correctly
declared as a required dependency.

---

## 9. Previously Fixed Issues (from 2026-03-18 audit)

The following issues from the prior audit have been confirmed fixed:

| Issue | Fix Location |
|-------|-------------|
| `NameError` in GRAPE leakage penalty path | `optimal_control/grape.py` |
| `np.random.seed()` in unitary synthesis | `unitary_synthesis/optim.py` |
| Dead code in `sim/runner.py` | `sim/runner.py` |
| Missing `KeyError` guard in `calibration/sqr.py` | `calibration/sqr.py` |
| Silent exception swallowing in `universal_model.py` | `core/universal_model.py` |
| `DeviceParameters` unit mismatch in tomo README | `tomo/README.md` |
| Dephasing rate convention undocumented | `sim/noise.py` docstrings + API_REFERENCE.md |
| Missing `observables` tests | `tests/test_53_observables.py` |
| Missing `operators` tests | `tests/test_52_operators.py` |

---

## 10. Recommendations Summary

### Medium Priority

| # | Item | Status |
|---|------|--------|
| 1 | ~~Update API_REFERENCE.md §21 to mark 2 stale gap entries as resolved~~ | **Fixed** |
| 2 | ~~Correct `plot_energy_levels` module path in API_REFERENCE.md~~ | **Fixed** |
| 3 | ~~Add `@abstractmethod` to `QuantumSystem.hilbert_dimension()` and `GateBase.ideal_unitary()`~~ | **Fixed** |

### Low Priority

| # | Item | Status |
|---|------|--------|
| 4 | ~~Renumber conflicting test files (`test_35`, `test_52`)~~ | **Fixed** |
| 5 | ~~Add language tags to fenced code blocks in `architecture.md`~~ | **Fixed** |
| 6 | Add READMEs for `core`, `pulses`, `measurement`, `calibration` subpackages | Open |
| 7 | Expand API_REFERENCE.md coverage for `rl_control` and `system_id` | Open |
| 8 | Clean up empty `tests/rl_control/` directory or move RL tests into it | Open |
| 9 | ~~Group `architecture.md` under a nav section in `mkdocs.yml`~~ | **Fixed** |

---

## 11. Conclusion

The `cqed_sim` package is well-organized, free of syntax errors, and imports cleanly.
All 21 previously tracked inconsistency reports are resolved or have documented
deferred scope items. The sweep identified 6 issues (§5.1–5.6), of which **5 have been
fixed** in this session:
- `@abstractmethod` decorators added to `QuantumSystem` and `GateBase`
- Two stale API_REFERENCE.md §21 entries marked resolved
- `plot_energy_levels` moved to correct module path in API_REFERENCE.md
- Test file numbering conflicts resolved (test_35 → test_55, test_52 → test_56/57)
- `architecture.md` code fences given language tags
- `architecture.md` moved under "Developer Guide" in mkdocs nav

Three low-priority items remain open: adding READMEs for additional subpackages,
expanding API_REFERENCE.md coverage for `rl_control`/`system_id`, and cleaning up the
empty `tests/rl_control/` directory.

---

*Generated from full repository sweep on 2026-03-20. Fixes applied same session.*
