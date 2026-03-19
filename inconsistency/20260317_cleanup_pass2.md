# Cleanup Pass 2 — Inconsistency and Findings Report

**Date:** 2026-03-17
**Scope:** Unit-convention documentation, repository hygiene, missing READMEs, API docs, test coverage

---

## 1. Unit Convention Inconsistency

### Finding

The library does not enforce physical units. Frequencies and times must be internally consistent within any single simulation (e.g., all rad/s + seconds, or all rad/ns + nanoseconds), but the library works correctly under any consistent unit system.

However, several files stated conflicting conventions:

| File | Previous claim | Status |
|---|---|---|
| `README.md` line 82 | "Internal Hamiltonian and frame frequencies are in rad/s; times are in s" | Too narrow — implies units are enforced |
| `API_REFERENCE.md` (Overview, Internal units) | "Hamiltonian coefficients ... in rad/s; times are in seconds" | Too narrow |
| `cqed_sim/core/frame.py` docstring | "All frequencies are in rad/s (angular frequency, i.e. 2*pi*f_Hz)" | Too narrow |
| `cqed_sim/sim/noise.py` `NoiseSpec` docstring | "repository convention H / hbar in rad/s and t in seconds" | Too narrow |
| `cqed_sim/pulses/pulse.py` `Pulse` docstring | "All time and frequency fields share the same unit: nanoseconds (t0, duration) and rad/ns (carrier)" | Contradicts all other files |

The `pulse.py` docstring was recently updated to say rad/ns + ns, while all other documentation says rad/s + s. Neither is wrong as an example, but having conflicting claims across files is confusing.

### Resolution

All five files were updated to the unit-coherent framing:
> "The library is unit-coherent: it does not enforce specific physical units for frequencies or times. Any internally consistent unit system is valid (for example, rad/s with times in seconds, or rad/ns with times in nanoseconds). The recommended convention used in the main examples and calibration function naming is rad/s and seconds."

The `pulse.py` docstring was updated to align with this framing while retaining the note that all fields within a single Pulse must use the same unit system.

---

## 2. Naming vs. Runtime Convention Tension

### Finding

The calibration module parameter names (e.g., `duration_s`, `omega_rad_s`, `amp_rad_s`) nominally imply rad/s + seconds, but the test suite (`conftest.py`) uses:
- `alpha = -2*np.pi*0.22` (anharmonicity in rad/GHz scale)
- `dt = 0.5` (numerically consistent with nanoseconds)

Both work because the library is unit-agnostic. The `_s` suffix in parameter names is a naming convention (rad/s + s is the reference convention for naming purposes), not a runtime enforcement.

### Status: Documented, no code change needed

The unit-coherent framing in the docstrings now makes this explicit.

---

## 3. Nested `.git` in holographic_sim

### Finding

`cqed_sim/quantum_algorithms/holographic_sim/.git/` was a nested git repository. This causes issues with git operations on the parent repo (submodule confusion, untracked status inconsistency).

### Resolution

- Removed the nested `.git` directory.
- Moved `notebook.ipynb` from the package directory to `examples/quantum_algorithms/holographic_sim_demo.ipynb` (notebooks belong in examples, not in the installable package).
- `IMPLEMENTATION_PLAN.md` had useful design notes not present in `README.md`; its content was appended as a "Development Notes" section in `holographic_sim/README.md`, then the file was removed.

---

## 4. Stale Root Artifacts

### Finding

The following files were present at the repository root but should not be tracked:

| File | Category | Action |
|---|---|---|
| `analytic_verification.py` | Useful verification script | Moved to `examples/` |
| `analytic_verification_results.json` | Output artifact | Removed |
| `av_results.txt`, `av_results2.txt` | Output artifacts | Removed |
| `sqr_calibration_result.json` | Output artifact | Removed |
| `test_output.txt`, `test_results.txt`, `test_results_full.txt`, `pytest_final.txt` | Test output artifacts | Removed |
| `PHYSICS_CORRECTNESS_EVALUATION.md` | Real evaluation report | Moved to `documentations/` |
| `texput.log` | LaTeX build artifact | Removed |

All tracked files were removed via `git rm`. `.gitignore` was updated to prevent recurrence of all patterns.

---

## 5. Missing Module READMEs

### Finding

The following modules lacked README files:
- `cqed_sim/io/` — gate sequence I/O module, read-only scope not documented
- `cqed_sim/system_id/` — calibration bridge, thin layer not documented
- `cqed_sim/operators/` — operator primitives, when to use vs. model-level not documented

### Resolution

README files created for all three modules.

---

## 6. Missing Test Coverage

### Finding

No tests existed for:
- `cqed_sim.io.gates` — all load/validate/format functions untested
- `cqed_sim.system_id` — CalibrationEvidence, randomizer_from_calibration, priors untested
- `cqed_sim.plotting` — plotting functions untested (headless smoke test absent)
- `cqed_sim.pulses.calibration` — closed-form formulas untested analytically

### Resolution

Added four test files:
- `tests/test_47_io_gates.py` — 20 tests covering load, validate, summarize, render, roundtrip
- `tests/test_48_system_id.py` — 13 tests covering CalibrationEvidence, randomizer, all prior types
- `tests/test_49_plotting_smoke.py` — 9 tests covering plot_energy_levels, plot_bloch_track, save_figure
- `tests/test_50_calibration_formulas.py` — 31 tests covering all 5 calibration formula functions

All 73 tests pass.

---

## 7. API Documentation Gaps

### Finding

No API documentation pages existed for:
- `cqed_sim.system_id` (documented in `17B` of `API_REFERENCE.md` but no dedicated doc page)
- `cqed_sim.gates` (referenced but no dedicated doc page)

### Resolution

Created:
- `documentations/api/system_id.md`
- `documentations/api/gates.md`

---

## 8. Windows Multiprocessing Note

### Finding

`SimulationSession.run_many` and `simulate_batch` use `ProcessPoolExecutor` with `spawn` context by default (required on Windows). No documentation warned users about per-worker spawn overhead.

### Resolution

Added docstring notes to both `run_many` and `simulate_batch` in `cqed_sim/sim/runner.py` explaining the Windows spawn overhead and recommending the serial path for most workloads.

---

## Summary of Files Changed

**Updated (existing files):**
- `README.md` — unit-coherent convention note
- `API_REFERENCE.md` — unit-coherent convention note
- `cqed_sim/core/frame.py` — unit-coherent docstring
- `cqed_sim/sim/noise.py` — unit-coherent docstring
- `cqed_sim/pulses/pulse.py` — unit-coherent docstring
- `cqed_sim/quantum_algorithms/holographic_sim/README.md` — Development Notes appended
- `.gitignore` — stale artifact patterns added
- `cqed_sim/sim/runner.py` — Windows spawn overhead notes

**Created (new files):**
- `cqed_sim/io/README.md`
- `cqed_sim/system_id/README.md`
- `cqed_sim/operators/README.md`
- `documentations/api/system_id.md`
- `documentations/api/gates.md`
- `tests/test_47_io_gates.py`
- `tests/test_48_system_id.py`
- `tests/test_49_plotting_smoke.py`
- `tests/test_50_calibration_formulas.py`
- `documentations/PHYSICS_CORRECTNESS_EVALUATION.md` (moved from root)
- `examples/analytic_verification.py` (moved from root)
- `examples/quantum_algorithms/holographic_sim_demo.ipynb` (moved from package dir)
- `inconsistency/20260317_cleanup_pass2.md` (this file)

**Removed (via git rm or rm):**
- `analytic_verification.py` (root)
- `analytic_verification_results.json`
- `av_results.txt`, `av_results2.txt`
- `sqr_calibration_result.json`
- `test_output.txt`, `test_results.txt`, `test_results_full.txt`, `pytest_final.txt`
- `PHYSICS_CORRECTNESS_EVALUATION.md` (root)
- `texput.log`
- `cqed_sim/quantum_algorithms/holographic_sim/.git/` (nested repo)
- `cqed_sim/quantum_algorithms/holographic_sim/IMPLEMENTATION_PLAN.md`
- `cqed_sim/quantum_algorithms/holographic_sim/notebook.ipynb` (moved)
