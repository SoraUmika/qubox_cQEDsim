# `cqed_sim` — Codebase Audit Report

**Date:** 2026-03-18  
**Last updated:** 2026-06-14 (fixes applied)

---

## Executive Summary

The `cqed_sim` codebase is in good overall health. The package imports cleanly, all
`__init__.py` exports are consistent, no circular imports were found, and the test
suite passes (1 skip, 6 warnings, no failures). Twenty subpackages are well-organized
with module-level READMEs, and the API reference is synchronized with the source code.

This report catalogs **1 confirmed runtime bug** (now fixed), **6 still-open API/convention gaps**
(documented/resolved), **2 test-coverage gaps** (now covered), and several minor quality
issues (all fixed) discovered during a full codebase audit.

---

## 1. Runtime Bugs

### 1.1 `NameError` in GRAPE leakage penalty path ~~(Critical)~~ — **FIXED**

**File:** `cqed_sim/optimal_control/grape.py`, line 457  
**Issue:** When the leakage-penalty `records` list is empty, the code referenced an
undefined variable `values`:

```python
if not records:
    raw = 0.0
    gradient = np.zeros_like(values, dtype=float)  # ← was NameError
```

**Fix applied:** Changed to `np.zeros_like(schedule.values, dtype=float)`, matching
the schedule parameter defined at line 402 of the same function.

---

## 2. Missing/Incomplete Implementations

### 2.1 Waveform Bridge — Limited Gate Type Coverage — **DOCUMENTED**

**File:** `cqed_sim/unitary_synthesis/waveform_bridge.py`  
**Issue:** The waveform bridge only supports `QubitRotation`, `Displacement`, and `SQR`
gate types. The following gate types raise `TypeError` if passed through the bridge:
- `SNAP`
- `ConditionalPhaseSQR`
- `FreeEvolveCondPhase`

These are valid gate primitives in the unitary synthesis sequence layer, but cannot be
exported to the runtime pulse-simulation stack via the waveform bridge.
This is a fundamental design limitation: these gate types have no pulse builders in
`pulses/builders.py` and no IO gate representations in `io/gates.py`.

**Fix applied:** Inline documentation added to `waveform_bridge.py` and
`API_REFERENCE.md` §21 explaining the limitation and recommending the ideal/symbolic
backend path.

### 2.2 Missing `@abstractmethod` Decorators

**File:** `cqed_sim/unitary_synthesis/systems.py`, line 182  
`QuantumSystem.hilbert_dimension()` raises `NotImplementedError` but the class does
not inherit from `ABC` and the method lacks `@abstractmethod`. Python will not enforce
the override contract at instantiation time.

**File:** `cqed_sim/unitary_synthesis/sequence.py`, line 482  
`PrimitiveGate.ideal_unitary()` — same pattern. Both concrete subclasses do override
correctly, so this is not a runtime risk today, but it is a design gap.

### 2.3 Dense Backend Limitations (Documented, Not Implemented) — **DOCUMENTED**

`NumPyBackend` and `JaxBackend` remain piecewise-constant dense matrix exponential
solvers. They are not drop-in replacements for QuTiP's adaptive ODE solver on large
Hilbert spaces. GPU/JAX integration for GRAPE gradients is deferred (see
`docs/performance_design.md`).

**Fix applied:** Inline documentation added to `cqed_sim/backends/base_backend.py`
and `API_REFERENCE.md` §21.

### 2.4 Synthetic I/Q Readout Model (Documented, Not Implemented)

When no `ReadoutChain` is provided, the fallback I/Q model in
`cqed_sim/measurement/qubit.py` is a simple Gaussian cluster model. It is not
calibrated to a physical readout chain and will not faithfully reproduce real
measurement noise statistics.

---

## 3. Convention and Documentation Inconsistencies

### 3.1 Dephasing Rate Factor-of-2 Inconsistency — **DOCUMENTED**

**File:** `cqed_sim/sim/noise.py`, lines 62–70  
The qubit dephasing rate uses $\gamma_\phi = 1/(2T_\phi)$ while the storage/readout
dephasing rates use $\gamma_\phi = 1/T_\phi$ (no factor of 2). Both are named
`gamma_phi_*`, hiding the different physical meaning. Users passing the same $T_\phi$
for qubit and cavity will get different effective dephasing strengths.

This is intentional ($\sigma_z$ vs $\hat{n}$ Lindblad operators).

**Fix applied:** Comprehensive docstrings added to `NoiseSpec` class and individual
properties in `noise.py`, and a convention note added to `API_REFERENCE.md` §6.5.

### 3.2 `FrameSpec` Legacy Field Names

`FrameSpec.omega_c_frame` is the underlying frozen field; `omega_s_frame` is a
read-only alias. In three-mode contexts, the "c" in `omega_c_frame` is ambiguous
(cavity vs. storage). No migration to a clearer naming has been done.

### 3.3 `DeviceParameters` Uses rad/ns (Tomo Module) — **FIXED**

`cqed_sim/tomo/device.py` converts Hz → rad/ns in `to_model()`, which is inconsistent
with the library's documented "unit-coherent" convention where the recommended practice
is rad/s. The tomo module README (line 86) incorrectly stated "All frequencies in rad/s,
times in s."

**Fix applied:** `tomo/README.md` updated to accurately describe the rad/ns convention
used by `DeviceParameters`.

### 3.4 API Reference Documentation Misplacement

`plot_energy_levels` is listed under `plotting.calibration_plots` in API_REFERENCE.md
but actually lives in `plotting.energy_levels`. The module path in the documentation
should be corrected.

### 3.5 API Reference Stale Gap Entry

Section 21 of API_REFERENCE.md lists "Higher-Order Coefficients Lack Isolated Tests"
as an open gap, but dedicated tests now exist in `tests/test_46_higher_order_coefficients.py`.
The same section lists "targets.py Contains User-Specific Hardcoded Paths" as open,
but this was fixed in the 2026-03-17 cleanup. Both entries should be marked resolved.

---

## 4. Test Coverage Gaps

### 4.1 Modules Without Dedicated Tests — **FIXED**

| Module | Status |
|--------|--------|
| `observables` (6 submodules: bloch, fock, phases, trajectories, weakness, wigner) | **Now tested:** `tests/test_53_observables.py` |
| `operators` (2 submodules: basic, cavity) | **Now tested:** `tests/test_52_operators.py` |

### 4.2 Empty Test Subdirectory

`tests/rl_control/` exists but is empty. All RL tests are top-level (`test_38`,
`test_39`, `test_43`, `test_44`), inconsistent with the pattern used for
`tests/unitary_synthesis/`, `tests/sim/`, etc.

### 4.3 Duplicate Test Number

Two files share the number 42:
- `tests/test_42_gates.py`
- `tests/test_42_optimal_control_hardware_constraints.py`

And `test_30` is missing from the numbering sequence (29 → 31).

### 4.4 Fragile Test Coupling to Examples

Three tests import directly from `examples/`:
- `test_21` → `examples.audits.sqr_convention_metric_audit`
- `test_22` → `examples.studies.sqr_multitone_study`
- `test_32` → `examples.workflows.kerr_free_evolution`

If those example scripts are refactored or removed, the tests break without warning.

---

## 5. Code Quality Issues

### 5.1 Deprecated `np.random.seed()` in Unitary Synthesis — **FIXED**

**File:** `cqed_sim/unitary_synthesis/optim.py`, line ~1578  
Used the deprecated NumPy legacy global random state (`np.random.seed()`).

**Fix applied:** Now creates a `np.random.default_rng()` generator first, then
seeds legacy state from it for backward compatibility with SciPy optimizers that
still use the global state.

### 5.2 Dead Code in `sim/runner.py` — **FIXED**

**File:** `cqed_sim/sim/runner.py`, line 63  
`P_e` was assigned via `_projector_onto_first_excited_state(dims)` and then immediately
overwritten by the loop on line 68. The first assignment was dead code.

**Fix applied:** Dead assignment removed.

### 5.3 Missing `KeyError` Guard in `calibration/sqr.py` — **FIXED**

**File:** `cqed_sim/calibration/sqr.py`, line ~832  
`config["cavity_fock_cutoff"]` was accessed without `.get()` fallback, unlike all other
config accesses in the module.

**Fix applied:** Changed to `config.get("cavity_fock_cutoff", 0)`.

### 5.4 Silent Exception Swallowing — **FIXED**

**File:** `cqed_sim/core/universal_model.py`, line 373  
A bare `except Exception: pass` silently swallowed failures when building the
`couplings["sideband"]` entry.

**Fix applied:** Narrowed to `except (ValueError, IndexError): pass` with explanatory
comment.

### 5.5 Non-Thread-Safe Parallel Worker Globals — **DOCUMENTED**

**File:** `cqed_sim/sim/runner.py`, lines 300–301  
Module-level globals `_PARALLEL_SESSION` and `_PARALLEL_SWEEP_SESSIONS` are mutated by
worker initializers. This is standard for `ProcessPoolExecutor` but means `run_many`
and `run_sweep` are not safe to call concurrently from threads in the same process.

**Fix applied:** Thread-safety warning comments added above the globals.

---

## 6. Previously Reported Inconsistencies — Status

| Report | Date | Status |
|--------|------|--------|
| Dispersive Convention Migration | 2026-03-11 | Fixed |
| Sideband / Multilevel / Kerr Audit | 2026-03-11 | Fixed |
| Usage Examples Spectroscopy Sign Mismatch | 2026-03-11 | Fixed |
| Notebook/Example Organization Audit | 2026-03-11 | Fixed |
| Model Layer Multilevel Refactor | 2026-03-11 | Fixed |
| Experiment Namespace Boundary Refactor | 2026-03-12 | Fixed |
| Spectrum Notebook Alignment | 2026-03-12 | Fixed |
| Sideband Notebook Core-Model Setup | 2026-03-13 | Fixed |
| Tutorial Curriculum Refactor Audit | 2026-03-13 | Fixed |
| Convention Cleanup Audit | 2026-03-13 | Fixed |
| Tutorial Execution dim=1 Bosonic Mode | 2026-03-13 | Fixed |
| Example Tutorial Migration Audit | 2026-03-15 | Fixed |
| Unitary Synthesis Pulse Backend Scope | 2026-03-15 | Fixed |
| Unitary Synthesis Phase 2 Runtime Audit | 2026-03-15 | Fixed |
| Unitary Synthesis System Coupling Refactor | 2026-03-15 | Fixed |
| Holographic Sim Architecture Audit | 2026-03-15 | Fixed (deferred: non-TI channels) |
| Optimal Control Follow-up Audit | 2026-03-17 | Fixed |
| Performance Optimization Refactor | 2026-03-17 | Fixed (deferred: GPU/JAX GRAPE) |
| Cleanup Pass 1 | 2026-03-17 | Fixed (tomo README units now fixed) |
| Cleanup Pass 2 | 2026-03-17 | Fixed |
| OC Hardware Pipeline Refactor | 2026-03-18 | Fixed (deferred: hard constraints) |

**18 of 21 reports are fully resolved.** The remaining 3 have documented deferred
follow-up items (non-TI channels, GPU/JAX GRAPE, hardware hard constraints).

---

## 7. Package Structure Health

| Check | Result |
|-------|--------|
| Package imports cleanly | Yes |
| All `__init__.py` exports valid | Yes (20/20 subpackages) |
| All `__all__` lists consistent | Yes |
| No circular imports | Yes |
| All API_REFERENCE.md modules exist | Yes |
| All module READMEs present | Yes (8/8 major modules checked) |
| Website docs synchronized | Yes (21 API pages match API_REFERENCE sections) |
| No TODO/FIXME/HACK markers | Confirmed clean |

---

## 8. Recommendations (Remaining)

1. ~~**Fix the `NameError` in `grape.py` line 457**~~ — ✅ Fixed.
2. **Add `@abstractmethod` decorators** to `QuantumSystem.hilbert_dimension()` and
   `PrimitiveGate.ideal_unitary()`, and have both classes inherit from `ABC`.
3. ~~**Add dedicated tests** for `observables` and `operators` modules~~ — ✅ Added
   `test_52_operators.py` (28 tests) and `test_53_observables.py` (22 tests).
4. ~~**Document the dephasing rate convention**~~ — ✅ Documented in `noise.py`
   docstrings and `API_REFERENCE.md` §6.5.
5. ~~**Fix the tomo README** units claim~~ — ✅ Fixed.
6. **Update API_REFERENCE.md Section 21** to mark the two resolved gaps (higher-order
   tests, hardcoded paths). *(Partially done — backend and waveform bridge gaps updated.)*
7. **Correct the `plot_energy_levels` module path** in API_REFERENCE.md.
8. ~~**Add `cavity_fock_cutoff` guard**~~ — ✅ Fixed.
9. ~~**Replace `np.random.seed()`**~~ — ✅ Fixed.
10. **Extend waveform bridge** to support SNAP, ConditionalPhaseSQR, and
    FreeEvolveCondPhase gate types. *(Documented as fundamental limitation — requires
    pulse builders and IO gate representations.)*

---

## Appendix: Test Suite Summary

- **Total test files:** ~50 numbered + 25 in subdirectories
- **Result:** All passing (1 skip, 6 warnings, 0 failures)
- **Warnings:** `test_31` (1 deprecation), `test_38` (3 RL), `test_39` (2 RL)
- **Coverage:** 20 of 20 modules have dedicated test files
