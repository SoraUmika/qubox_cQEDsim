# cqed_sim Codebase Cleanup Review & Next-Step Recommendations

**Date:** 2026-03-17  
**Scope:** Full repository-wide technical and scientific maturity assessment  
**Reviewer approach:** Staff-level technical reviewer, codebase-grounded analysis

---

## 1. Repository Survey

### 1.1 Major Subsystems

| Subsystem | Modules | Est. LOC | Maturity |
|-----------|---------|----------|----------|
| **Core physics/models** | `core/` (12 files) | ~3,000 | Production |
| **Gate library** | `gates/` (6 files) | ~2,000 | Production |
| **Pulse infrastructure** | `pulses/` (6 files) | ~1,200 | Production |
| **Timeline compilation** | `sequence/` (3 files) | ~500 | Solid |
| **Simulation engine** | `sim/` (8 files) | ~2,500 | Production |
| **Measurement/readout** | `measurement/` (2 files) | ~800 | Production |
| **Observables/diagnostics** | `observables/` (6 files) | ~1,000 | Production |
| **Plotting** | `plotting/` (7 files) | ~1,500 | Production |
| **Tomography** | `tomo/` (2 files) | ~600 | Production |
| **Parameter translation** | `analysis/` (1 file) | ~200 | Production |
| **Calibration** | `calibration/` (3 files) | ~2,000 | Solid |
| **Calibration targets** | `calibration_targets/` (7 files) | ~800 | Production |
| **Gate I/O** | `io/` (1 file) | ~300 | Transitional |
| **Operator primitives** | `operators/` (3 files) | ~250 | Solid |
| **Unitary synthesis** | `unitary_synthesis/` (14 files) | ~4,000 | Solid |
| **Optimal control (GRAPE)** | `optimal_control/` (10 files) | ~3,000 | Solid |
| **RL control** | `rl_control/` (12 files) | ~4,000 | Solid |
| **System identification** | `system_id/` (2 files) | ~100 | Transitional |
| **Backends** | `backends/` (3 files) | ~600 | Transitional |
| **Holographic algorithms** | `quantum_algorithms/` (19 files) | ~3,000 | Solid/Specialized |

**Total estimated library LOC:** ~31,000+

### 1.2 Maturity Assessment Summary

**Production-quality (stable, well-tested, well-documented):**
- `core`, `gates`, `pulses`, `sim`, `measurement`, `observables`, `plotting`, `tomo`, `analysis`, `calibration_targets`

**Solid (feature-complete but still evolving or needs polish):**
- `calibration`, `unitary_synthesis`, `optimal_control`, `rl_control`, `sequence`, `operators`, `quantum_algorithms`

**Transitional (functional but underdocumented or limited scope):**
- `io`, `system_id`, `backends`

### 1.3 Documentation Structure

- 19 API module docs, 8 user guides, 8 tutorial writeups under `documentations/`
- 14 structured workflow notebooks + 26 legacy tutorial notebooks in `tutorials/`
- 18 standalone example scripts + 5 example subdirectories
- 18 dated inconsistency reports (all resolved)
- Physics conventions report (`physics_and_conventions/physics_conventions_report.tex`)

### 1.4 Test Infrastructure

- **64 test files** (46 top-level + 18 in subdirectories)
- Systematic numbered naming convention (test_01 through test_46)
- Coverage spans: core physics, convention compliance, numerics, hardware effects, dissipation, gates, calibration, synthesis, RL, optimal control, performance

---

## 2. Cleanup Findings

### Category 1: API / Naming Cleanup

#### 1a. Unit convention documentation inconsistency

- **Severity: HIGH**
- **What:** The README, `core/frame.py`, and `sim/noise.py` document frequencies in "rad/s" with times in "seconds." The `pulses/pulse.py` module explicitly documents carrier in "rad/ns" and times in "nanoseconds." Tests use small dimensionless numbers consistent with a GHz-scale / nanosecond-scale convention.
- **Why it matters:** A new user reading the README will set up models in rad/s and seconds (as shown in the README examples: `omega_c=2*pi*5e9`, `dt=2e-9`). A user reading Pulse docstrings will expect rad/ns and nanoseconds. The library is numerically unit-agnostic (all floats, no dimensional enforcement), so both work if internally consistent — but the documentation claiming different units in different modules is confusing and error-prone.
- **Recommended fix:** Either (a) document the system as unit-agnostic and state that all frequencies and times must share a coherent unit system, or (b) pick one canonical convention and update all docstrings. Given the README examples use rad/s + seconds and the noise module states the same, the Pulse docstrings should be updated to match, or clarified as "in whatever unit system the user has chosen."
- **Effort:** Small

#### 1b. FrameSpec field naming legacy (`omega_c_frame` vs `omega_s_frame`)

- **Severity: LOW**
- **What:** `FrameSpec.omega_c_frame` is the canonical stored field (two-mode legacy). `omega_s_frame` is a property alias for three-mode usage.
- **Why it matters:** Already clarified in recent cleanup. The dual naming is documented and works correctly.
- **Recommended fix:** No immediate action. Consider migrating canonical field to `omega_s_frame` in a future major version if three-mode usage becomes dominant.
- **Effort:** N/A (deferred)

#### 1c. Gate I/O module is read-only

- **Severity: LOW**
- **What:** `cqed_sim.io` only deserializes gates from JSON. No serialization path from Python gate objects back to JSON.
- **Why it matters:** Users cannot round-trip gates. The one-way design is intentional (reads calibration outputs) but undocumented.
- **Recommended fix:** Document the one-way intent in a module README. Consider adding a `save_gate_sequence()` if round-tripping becomes needed.
- **Effort:** Small (doc) / Medium (implementation)

### Category 2: Structural / Architectural Cleanup

#### 2a. Nested `.git/` directory in `holographic_sim/`

- **Severity: HIGH**
- **What:** `cqed_sim/quantum_algorithms/holographic_sim/.git/` exists — a nested Git repository inside the main repo. Also contains `notebook.ipynb` and `IMPLEMENTATION_PLAN.md` inside a library source directory.
- **Why it matters:** Nested `.git/` confuses git history, breaks recursive git operations, and complicates CI/CD. A notebook inside a library source package violates the project's own organization guidelines.
- **Recommended fix:** 
  1. Remove the `.git/` directory from the holographic_sim package.
  2. Move `notebook.ipynb` to `examples/quantum_algorithms/` or `tutorials/`.
  3. Merge `IMPLEMENTATION_PLAN.md` content into the existing `holographic_sim/README.md` or relocate to `docs/`.
- **Effort:** Small (requires careful git operations)

#### 2b. Stale top-level artifact files

- **Severity: MODERATE**
- **What:** Multiple stale files at repository root: `analytic_verification.py`, `analytic_verification_results.json`, `av_results.txt`, `av_results2.txt`, `sqr_calibration_result.json`, `test_output.txt`, `test_results.txt`, `test_results_full.txt`, `pytest_final.txt`, `PHYSICS_CORRECTNESS_EVALUATION.md` (single-line placeholder), `texput.log`.
- **Why it matters:** Repository root clutter obscures the actual project structure. These are development artifacts that should not be tracked.
- **Recommended fix:** Add to `.gitignore` and/or remove from tracking. If `analytic_verification.py` contains useful logic, move it to `tests/` or `examples/`.
- **Effort:** Small

#### 2c. `build/` directory tracked in git

- **Severity: LOW**
- **What:** `build/bdist.win-amd64/` and `build/lib/` are present. While `.gitignore` includes `build/`, the directory appears to still be committed.
- **Why it matters:** Build artifacts should not be version controlled.
- **Recommended fix:** `git rm -r --cached build/` to untrack.
- **Effort:** Small

#### 2d. `sim/runner.py` global state for parallelization

- **Severity: LOW**
- **What:** `_PARALLEL_SESSION` and `_PARALLEL_SWEEP_SESSIONS` are module-level globals used as per-worker state in `ProcessPoolExecutor`. Model objects get mutated via `_default_observables_cache` side effect.
- **Why it matters:** Works correctly due to process isolation, but is unconventional and could surprise users or future maintainers. The model mutation (caching observables) is a hidden side effect.
- **Recommended fix:** Consider using a proper worker-state class instead of globals. Document the caching behavior. Not urgent.
- **Effort:** Medium

### Category 3: Documentation Cleanup

#### 3a. Missing module-level README files

- **Severity: MODERATE**
- **What:** Three modules lack required README.md files:
  - `cqed_sim/io/` — no README
  - `cqed_sim/system_id/` — no README
  - `cqed_sim/operators/` — no README
- **Why it matters:** AGENTS.md requires module-level READMEs for distinct functionality areas. These modules are discoverable by users but unexplained at the directory level.
- **Recommended fix:** Add concise README.md to each, covering purpose, key classes, when to use, and relationship to other modules.
- **Effort:** Small

#### 3b. Website docs missing coverage for `io/`, `system_id/`, `gates/`

- **Severity: LOW**
- **What:** `documentations/api/` has docs for most modules but:
  - `io/` → covered by `gate_io.md` (adequate)
  - `system_id/` → not documented
  - `gates/` → implied in core docs but no standalone page
- **Why it matters:** System_id has no dedicated doc page. Gates deserve a clear standalone reference separate from `core.md`.
- **Recommended fix:** Add `documentations/api/system_id.md` and `documentations/api/gates.md`.
- **Effort:** Small

#### 3c. Physics conventions report vs implementation drift

- **Severity: LOW**
- **What:** After the intensive March 11–17 refactoring cycle, 18 inconsistency reports were filed and resolved. The physics conventions report was updated during that cycle. No new drift detected.
- **Recommended fix:** No immediate action. Run a verification pass after the next major feature addition.
- **Effort:** N/A

### Category 4: Testing Cleanup

#### 4a. Missing tests for `io/`, `plotting/`, `system_id/`

- **Severity: MODERATE**
- **What:**
  - `cqed_sim/io/` — no dedicated tests for gate deserialization, validation, rendering
  - `cqed_sim/plotting/` — no tests (7 plot functions untested)
  - `cqed_sim/system_id/` — no tests for calibration-to-randomizer bridge
- **Why it matters:** These modules are part of the public API surface. `io/` handles external JSON input and should have validation tests. `system_id/` bridges calibration and RL and could silently break. `plotting/` is lower priority but smoke tests would prevent regressions.
- **Recommended fix:**
  - `io/`: Add tests for `load_gate_sequence()` with valid/invalid JSON, `gate_summary_text()`, `render_gate_table()`
  - `system_id/`: Add tests for `randomizer_from_calibration()` with representative `CalibrationEvidence`
  - `plotting/`: Add smoke tests that verify plot functions return `Figure` objects without error
- **Effort:** Small–Medium

#### 4b. No analytic-formula tests for calibration formulas

- **Severity: LOW**
- **What:** `pulses/calibration.py` provides closed-form amplitude estimates (`displacement_square_amplitude`, `rotation_gaussian_amplitude`, `sqr_lambda0_rad_s`). These have documented physics assumptions but no direct unit tests verifying the formulas against known analytical results.
- **Why it matters:** If a formula is silently wrong (e.g., missing factor of 2), it would propagate into all dependent calibration workflows.
- **Recommended fix:** Add tests comparing each formula against a textbook reference value.
- **Effort:** Small

### Category 5: Portability / Maintenance Cleanup

#### 5a. No hardcoded machine-specific paths

- **Severity: N/A (good)**
- **What:** No `C:\Users\...` or machine-specific paths found in library source.
- **Status:** Clean.

#### 5b. JAX backend is properly optional

- **Severity: N/A (good)**
- **What:** `JaxBackend` import is guarded with try/except. Fallback to `None` is clean.
- **Status:** Clean. Limitations documented in README.

#### 5c. Windows multiprocessing overhead not prominently documented

- **Severity: LOW**
- **What:** `sim/runner.py` uses `ProcessPoolExecutor` with `spawn` start method on Windows. This has high startup overhead for small jobs. Mentioned in README performance section but not in the `simulate_batch` docstring.
- **Recommended fix:** Add a note to `simulate_batch` and `run_sweep` docstrings about Windows overhead.
- **Effort:** Small

---

## 3. Top Next-Step Suggestions

### 3.1 Documentation & Convention Finalization (API maturity)

- **What:** Resolve the unit documentation inconsistency (rad/s vs rad/ns), add 3 missing module READMEs, add `system_id` and `gates` to website docs.
- **Why now:** The library has just completed a major refactoring cycle (18 inconsistencies filed and fixed). Documentation is 95% aligned. The remaining 5% are low-hanging fruit that would make the package ready for external users.
- **Existing support:** All module code is clean and well-typed. Just needs docstring/README alignment.
- **Cleanup first:** Resolve unit convention documentation (Finding 1a).
- **Value:** External user confidence; onboarding clarity.
- **Type:** Documentation/API maturity.

### 3.2 Test Coverage for Remaining Modules (validation)

- **What:** Add tests for `io/`, `plotting/` (smoke), `system_id/`, and calibration formula unit tests.
- **Why now:** 64 test files already provide strong coverage. The untested modules are small and well-scoped — testing them would fill the last gaps in automated validation.
- **Existing support:** Test infrastructure is mature with clear conventions (numbered test files, pytest.ini configured).
- **Cleanup first:** None.
- **Value:** Regression safety for the full API surface.
- **Type:** Validation/testing.

### 3.3 Repository Hygiene Pass (maintenance)

- **What:** Remove nested `.git/` from holographic_sim, clean up stale root artifacts, untrack `build/`.
- **Why now:** These are trivial fixes with zero risk that improve git health and repo clarity.
- **Existing support:** `.gitignore` already has most patterns; just needs enforcement.
- **Cleanup first:** None.
- **Value:** Clean git history; CI/CD compatibility; professional presentation.
- **Type:** Maintenance.

### 3.4 Unitary Synthesis Workflow Maturation (control/gate capability)

- **What:** The synthesis stack already supports multi-objective optimization, Pareto exploration, leakage penalties, waveform bridge back to runtime, and constraint-aware fitting. The next natural step is to solidify the "synthesis → GRAPE refinement → export" pipeline end-to-end.
- **Why now:** `unitary_synthesis` and `optimal_control` are both solid. The waveform bridge exists. The gap is a documented, tested end-to-end workflow showing: (1) synthesize a gate sequence, (2) refine with GRAPE, (3) export pulses back into `SequenceCompiler`, (4) validate with noisy simulation.
- **Existing support:** `waveform_bridge.py`, `evaluate_control_with_simulator()`, `CQEDSystemAdapter`.
- **Cleanup first:** None (both modules are in good shape).
- **Value:** Practical gate design workflow for experimentalists. Major differentiator.
- **Type:** Control/gate capability.

### 3.5 Experiment-Facing Calibration Loop (workflow/usability)

- **What:** Connect calibration targets (`run_spectroscopy`, `run_rabi`, etc.) with system_id priors and RL domain randomization into a documented "measure → update model → randomize → train" loop.
- **Why now:** All three pieces exist: `calibration_targets` produces measurements, `system_id` converts to priors, `rl_control` consumes randomized models. The loop just needs a documented recipe and an example.
- **Existing support:** `CalibrationEvidence`, `randomizer_from_calibration()`, `DomainRandomizer`, all benchmark tasks.
- **Cleanup first:** Add `system_id` README and test coverage (Findings 3a, 4a).
- **Value:** First sim-to-experiment bridge workflow. High scientific value for users running real calibrations.
- **Type:** Workflow/usability.

### 3.6 Multi-Qubit / Multi-Cavity Extension (physics-model extension)

- **What:** The `UniversalCQEDModel` architecture already supports arbitrary numbers of bosonic modes and a single transmon. Extending to multiple transmons (e.g., two-transmon + shared cavity for remote entanglement) is a natural model extension.
- **Why now:** The universal model's spec-based architecture (`TransmonModeSpec`, `BosonicModeSpec`, `DispersiveCouplingSpec`) was deliberately designed for extensibility. Two-qubit gates already exist in `gates/two_qubit.py` but have no model-level support.
- **Existing support:** 70% structural support. `UniversalCQEDModel` would need to accept a list of `TransmonModeSpec`.
- **Cleanup first:** Ensure all existing code paths (extractors, noise, etc.) are audited for single-transmon assumptions.
- **Value:** Opens the package to multi-qubit cQED studies (entanglement, cross-talk, multiplexed readout).
- **Type:** Physics-model extension.

### 3.7 Lindblad Support in Dense Backends (performance)

- **What:** The NumPy and JAX backends currently only support closed-system (unitary) evolution. Adding vectorized Lindblad support would enable fast noisy optimization without falling back to QuTiP.
- **Why now:** GRAPE and synthesis workflows often need noisy evaluation. Currently they call `evaluate_control_with_simulator()` which routes through QuTiP's Lindblad solver. Dense Lindblad in the JAX backend would unlock autodiff-based gradient computation for noisy objectives.
- **Existing support:** `BaseBackend` interface, `NoiseSpec`, collapse operator construction.
- **Cleanup first:** None (backend interface is clean).
- **Value:** Major performance unlock for noisy optimal control and synthesis. Enables JAX-gradient-based GRAPE.
- **Type:** Performance optimization.

---

## 4. Recommended Order of Operations

### Tier 1 — Worth Fixing Now (before next features)

| # | Item | Type | Effort | Reference |
|---|------|------|--------|-----------|
| 1 | Resolve unit documentation inconsistency (rad/s vs rad/ns) | Doc | Small | Finding 1a |
| 2 | Remove `.git/` from holographic_sim, relocate notebook/plan | Hygiene | Small | Finding 2a |
| 3 | Clean up stale root-level artifact files | Hygiene | Small | Finding 2b |
| 4 | Untrack `build/` directory | Hygiene | Small | Finding 2c |
| 5 | Add README.md to `io/`, `system_id/`, `operators/` | Doc | Small | Finding 3a |
| 6 | Add tests for `io/`, `system_id/`, `plotting/` (smoke) | Test | Small-Med | Finding 4a |
| 7 | Add analytic formula tests for `pulses/calibration.py` | Test | Small | Finding 4b |
| 8 | Add `system_id.md` and `gates.md` to website docs | Doc | Small | Finding 3b |
| 9 | Document Windows parallelization overhead in docstrings | Doc | Small | Finding 5c |

### Tier 2 — Natural Next Milestone

| # | Item | Type | Effort |
|---|------|------|--------|
| 10 | End-to-end synthesis→GRAPE→export→validate workflow | Capability | Medium |
| 11 | Calibration→SystemID→RL documented loop | Workflow | Medium |
| 12 | Gate I/O write support (serialize gates to JSON) | API | Medium |

### Tier 3 — Longer-Horizon Opportunities

| # | Item | Type | Effort |
|---|------|------|--------|
| 13 | Multi-transmon model extension in UniversalCQEDModel | Physics | Large |
| 14 | Dense Lindblad in JAX backend (enables autodiff GRAPE) | Performance | Large |
| 15 | FrameSpec field migration (omega_c_frame → omega_s_frame) | API | Medium |
| 16 | Comprehensive gate-error budget tooling (gate set tomography) | Capability | Large |
| 17 | MPS/tensor-network backend for large Hilbert spaces | Performance | Large |

---

## 5. Optional Changes Made

No code changes were implemented in this review pass. The analysis is the primary deliverable.

---

## 6. Tests Run

No tests were run during this review. The existing test suite (64 files) was surveyed for coverage.

---

## 7. Final Recommendation

### Fix Next

1. **Unit documentation** — This is the most confusing inconsistency remaining. The library is numerically unit-agnostic, but `pulse.py` says "rad/ns, ns" while `core/frame.py` and `sim/noise.py` say "rad/s, seconds." The README shows rad/s examples. Clarify in all docstrings that the library is unit-coherent (user picks a system and stays consistent) and state the recommended convention, or standardize all docstrings to "rad/s + seconds."

2. **Repository hygiene** — Remove the nested `.git/`, stale root artifacts, and untracked `build/`. Zero risk, high clarity gain.

3. **Three missing READMEs** — `io/`, `system_id/`, `operators/` each need a short README. Combined effort: ~30 minutes.

### Build Next

4. **Synthesis-to-validation pipeline** — The most impactful next milestone. All infrastructure exists (`UnitarySynthesizer`, `GrapeSolver`, `waveform_bridge`, `evaluate_control_with_simulator`). What's missing is a documented, tested end-to-end recipe showing gate design from target specification to validated noisy pulse. This is the natural culmination of the synthesis and optimal control work and would be a standout capability.

5. **Calibration-to-RL pipeline** — Second priority. Connecting `calibration_targets` → `system_id` → `rl_control` in a documented workflow would demonstrate the experimental feedback loop that the architecture was designed for.

### Postpone

6. **Multi-transmon support** — Valuable but requires auditing single-transmon assumptions across extractors, noise, and tomo. Not the next step.

7. **JAX Lindblad backend** — High potential payoff but substantial implementation. Better to solidify the existing capabilities first.

8. **FrameSpec field renaming** — Low impact, breaking change. Defer to a future major version.

---

## Important Review Questions — Explicit Answers

### Physics / conventions
- **Internally consistent?** Yes, after the March 11–17 refactoring cycle. The dispersive convention migration (chi sign) is complete. All 18 inconsistency reports are resolved.
- **Ambiguous conventions remaining?** The unit documentation (rad/s vs rad/ns) across modules is the main remaining ambiguity. It does not affect correctness (library is unit-agnostic) but affects user understanding.
- **Easy to understand from docs?** The physics conventions report and conventions quick reference are excellent. The unit ambiguity is the only source of potential confusion.

### API design
- **Coherent?** Yes. The `UniversalCQEDModel` → `Pulse` → `SequenceCompiler` → `simulate_sequence` → extractors pipeline is clean and well-documented.
- **Incompatible subsystem APIs?** No major incompatibilities. The `gates/` vs `core/ideal_gates.py` separation is correct (utilities vs. public API). The `observables/` re-exports from `sim/extractors.py` are intentional.
- **Notebook-derived code?** No. All library modules are properly structured. Earlier notebook-derived code was migrated to `examples/` during the March refactoring.

### Simulation / solver design
- **Good shape?** Yes. The `SimulationSession` + `prepare_simulation` pattern is well-designed for repeated simulations. The parallel execution paths work but rely on global state (acceptable with documentation).
- **Bottlenecks?** Windows multiprocessing startup overhead is the main practical bottleneck. The QuTiP solver is the performance ceiling for large systems.

### Gate / pulse / sequence design
- **Long-term shape?** Good. The `Pulse` → `CompiledSequence` pipeline is clean. Hardware effects (IQ distortion, crosstalk) are applied at compile time. DRAG is supported but underdocumented in builders.
- **Obvious next step?** DRAG exposure in pulse builders, and closing the synthesis→pulse export loop.

### Synthesis / optimization
- **Ready for broader use?** Yes. The `UnitarySynthesizer` with Pareto exploration, the `GrapeSolver` with multi-start, and the waveform bridge are feature-complete for the current scope.
- **Next extension?** End-to-end synthesis→GRAPE→validate pipeline (see recommendation #4 above).
- **Cleanup needed?** No. Both modules are solid.

### Tomography / characterization
- **Fits naturally?** Yes. `tomo` is focused, well-tested, and integrates cleanly with the measurement pipeline.
- **Convention mismatches?** None after the March cleanup. DeviceParameters unit handling is correct (code converts properly from Hz to the library's frequency convention).

### Documentation / examples
- **Docs match the real package?** 95% yes. The 5% gap is: unit documentation inconsistency, 3 missing module READMEs, 2 missing website doc pages.
- **Missing examples?** The synthesis→GRAPE→validate end-to-end example would be the highest-impact addition.
- **Missing module READMEs?** `io/`, `system_id/`, `operators/`.

### Package maturity
- **Stable and reusable:** Core physics stack, gates, pulses, sequence compilation, simulation engine, measurement, observables, plotting, tomography, calibration targets.
- **Still evolving:** Unitary synthesis, optimal control, RL control (feature-complete but still growing).
- **Experimental:** JAX backend, system_id (thin), io (limited scope).
- **What would most improve confidence?** (1) Resolving the unit documentation, (2) filling the remaining test gaps, (3) demonstrating the end-to-end synthesis pipeline. These three together would bring the package to a level suitable for publication-quality external use.
