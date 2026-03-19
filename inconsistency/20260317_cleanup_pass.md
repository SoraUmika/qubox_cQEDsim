# cqed_sim Cleanup Pass — 2026-03-17

## Overview

This document records all findings and changes made during the 2026-03-17 audit
and cleanup pass of the `cqed_sim` repository.

---

## Phase 1 — Remove hardcoded user paths (`targets.py`)

**File:** `cqed_sim/unitary_synthesis/targets.py`

**Finding:** `_default_reference_root()` contained two absolute Windows paths
(`C:\Users\dazzl\...` and `C:\Users\jl82323\...`) that are machine-specific and
would silently fail on any other developer's machine.

**Change:** Replaced the hardcoded path list with:
1. An `CQED_REFERENCE_ROOT` environment-variable override (with a `UserWarning`
   if the env var is set but the path does not exist).
2. The existing relative sibling-repo lookup (`<repo_root>/../noah_repo/...`).

Callers are unaffected when neither path exists — `None` is returned as before
and reference-matrix targets raise an informative error at use time.

---

## Phase 2 — FrameSpec naming clarification (`frame.py`)

**File:** `cqed_sim/core/frame.py`

**Finding:** `omega_c_frame` is the canonical stored field but the property
`omega_s_frame` was documented as "aliasing the legacy cavity-frame name to
storage", which is backwards — `omega_c_frame` is actually the legacy name and
`omega_s_frame` is the preferred three-mode name. The docstring was confusing.

**Change:** Updated class docstring to clearly state:
- `omega_c_frame` is the canonical stored field (backward-compatible).
- `omega_s_frame` is the preferred alias for three-mode contexts.
- Added field-level docstrings with units (rad/s).
- Updated the `omega_s_frame` property docstring to correctly explain the
  relationship.

No behavior changes. All existing code using `omega_c_frame` continues to work.

---

## Phase 3 — Analytic tests for `chi_higher` and `kerr_higher`

**New file:** `tests/test_46_higher_order_coefficients.py`

**Finding:** The existing `test_06_leakage_drag_and_higher_order.py` had smoke
tests for higher-order terms but did not isolate individual coefficient orders
or check the ground-state / low-Fock zero conditions.

**New tests added:**

- `TestKerrHigher`: verifies `kerr`, `kerr_higher[0]`, and `kerr_higher[0:2]`
  energy contributions against the closed-form falling-factorial formula
  `K_j * n^(j+2)_falling / (j+2)!` for all Fock levels 0–7; checks that
  `n=0` and `n=1` give zero Kerr energy; verifies `basis_energy` agrees with
  Hamiltonian expectation.
- `TestChiHigher`: verifies dispersive chi and `chi_higher[0]` (falling factorial
  order 2 = `n*(n-1)`) for the qubit-excited energy shift; checks that
  `n=0,1` give only linear chi contribution; verifies `|g,n>` energies are
  unaffected by chi/chi_higher; verifies `basis_energy` agrees with Hamiltonian.
- `TestUniversalModelHigherOrder`: verifies that `UniversalCQEDModel` and
  `DispersiveTransmonCavityModel` produce identical Hamiltonians for the same
  kerr/kerr_higher values.

**Implementation correctness note:** From the code in `universal_model.py`:
```
kerr_higher[0]  → order_index=2 → order=3 → coeff * n*(n-1)*(n-2) / 6
chi_higher[0]   → order=2       → coeff * n*(n-1) * n_q  (no factorial division)
```
Note that `chi_higher` does NOT divide by a factorial (unlike `kerr_higher`).
This asymmetry is present in the implementation and confirmed by `test_10_chi_convention.py`
line 183: `expected = ... + chi2 * (n * (n - 1)) + chi3 * (n * (n - 1) * (n - 2))`.

---

## Phase 4 — DeviceParameters unit explicitness (`device.py`)

**File:** `cqed_sim/tomo/device.py`

**Finding:** The `hz_to_rad_per_ns` method multiplies by `2*pi*1e-9`, producing
rad/ns, and feeds the result directly to `DispersiveTransmonCavityModel`. The
prompt flagged this as a potential bug "since the model expects rad/s". However,
after auditing the tomo workflow (`protocol.py`) and the simulation runner, it is
confirmed that the entire `cqed_sim` simulation uses **nanoseconds** as its time
unit (pulse durations, `dt`, `t_end` are all in ns). Therefore Hamiltonian
frequencies must be in **rad/ns** — and the conversion is **correct**.

The tomo README line 83 says "All frequencies in `rad/s`, times in `s`" — this
is **incorrect documentation**. The actual convention used throughout the code
is rad/ns and ns.

**Changes:**
- Added an extensive class docstring to `DeviceParameters` explaining the
  rad/ns convention and why it is correct.
- Added a docstring to `hz_to_rad_per_ns` explaining the conversion and its
  rationale.
- Added field-level docstrings with units (Hz for stored values, rad/ns after
  conversion).
- Added a docstring to `to_model()` explaining the unit convention.

**No behavior changes.** The tomo README contains a documentation error ("rad/s,
times in s") but correcting it is out of scope for this pass.

**Remaining issue:** `cqed_sim/tomo/README.md` line 83 says "All frequencies in
`rad/s`, times in `s`" — this should be updated to "rad/ns, times in ns" in a
future documentation pass.

---

## Phase 5 — Backend and synthetic I/Q documentation

**Files modified:**
- `cqed_sim/backends/base_backend.py` — added class docstring with memory scaling
  warnings and limitations.
- `cqed_sim/backends/numpy_backend.py` — added class docstring.
- `cqed_sim/backends/jax_backend.py` — added class docstring with GPU and JIT notes.
- `cqed_sim/measurement/qubit.py` — added `QubitMeasurementSpec` class docstring
  explaining the two I/Q sampling modes and documenting the limitations of the
  synthetic path.

---

## Phase 6 — Waveform bridge gate coverage (`waveform_bridge.py`)

**File:** `cqed_sim/unitary_synthesis/waveform_bridge.py`

**Finding:** The `TypeError` raised for unsupported gates contained only a brief
list of supported types with no guidance on how to simulate SNAP,
`ConditionalPhaseSQR`, or `FreeEvolveCondPhase` gates.

**Change:** Improved the error message to:
1. Name the unsupported gate type explicitly (using `type(gate).__name__`).
2. List all three supported types.
3. Tell the user what to use instead for SNAP/ConditionalPhaseSQR/FreeEvolveCondPhase
   (the model-backed path: `simulate_sequence` / `hamiltonian_time_slices`).

No new gate support was added — SNAP pulse building would require a dedicated
`build_snap_pulse` function that does not currently exist.

---

## Phase 7 — Pulse carrier sign convention documentation

**File:** `cqed_sim/pulses/pulse.py`

**Finding:** The carrier sign convention (`carrier = -omega_transition(frame)`)
was documented only in `cqed_sim/sim/README.md` line 141, not on the `Pulse`
dataclass itself.

**Change:** Added a comprehensive class docstring to `Pulse` that:
1. Shows the full waveform formula (`amp * envelope(t_rel) * exp(i*(carrier*t+phase))`).
2. Documents the carrier sign convention with explanation of why the sign is negative.
3. Documents all fields with units (ns for time, rad/ns for carrier).

---

## Phase 8 — Convention alignment verification

**Files checked:**
- `tests/test_10_chi_convention.py`
- `tests/test_32_kerr_sign_notebook_regression.py`
- `tests/conventions/test_conventions.py`
- `cqed_sim/core/universal_model.py`

**Findings:**

1. **|g> = |0>, |e> = |1>**: Confirmed. `basis_state(0, n)` gives the ground state,
   `basis_state(1, n)` gives the first excited state throughout. This is consistent
   with the QuTiP `qt.basis(2, 0)` = |g> = |0> convention.

2. **Dispersive chi sign**: The Hamiltonian is `H_chi = chi * n_c * n_q`. A positive
   `chi` increases the energy of `|e, n>` relative to `|g, n>`, making the qubit
   transition frequency increase with photon number. This is the standard "positive
   chi pulls transition up" convention. Tests confirm this in `test_10_chi_convention.py`.

3. **Kerr sign**: `H_kerr = kerr * n*(n-1) / 2`. A negative `kerr` decreases the
   energy of higher Fock states (cavity photons are less energetic at higher occupation).
   Tests confirm this in `test_32_kerr_sign_notebook_regression.py`.

4. **No inconsistencies found** between docs, tests, and implementation for these
   three conventions.

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `cqed_sim/unitary_synthesis/targets.py` | Phase 1: Remove hardcoded user paths, add env var |
| `cqed_sim/core/frame.py` | Phase 2: Clarify docstrings, fix naming description |
| `tests/test_46_higher_order_coefficients.py` | Phase 3: New analytic test file |
| `cqed_sim/tomo/device.py` | Phase 4: Add unit documentation |
| `cqed_sim/backends/base_backend.py` | Phase 5: Add limitation docstrings |
| `cqed_sim/backends/numpy_backend.py` | Phase 5: Add limitation docstrings |
| `cqed_sim/backends/jax_backend.py` | Phase 5: Add limitation docstrings |
| `cqed_sim/measurement/qubit.py` | Phase 5: Document synthetic I/Q limitations |
| `cqed_sim/unitary_synthesis/waveform_bridge.py` | Phase 6: Improve error message |
| `cqed_sim/pulses/pulse.py` | Phase 7: Document carrier sign convention |

## Remaining Issues / Known Documentation Debt

1. `cqed_sim/tomo/README.md` line 83: "All frequencies in `rad/s`, times in `s`"
   should be corrected to "rad/ns, times in ns".
2. No SNAP waveform bridge support exists; the error message now clearly directs
   users to the model-backed path.
3. The `FrameSpec.omega_c_frame` name ambiguity is documented but the field name
   is not changed (backward compatibility preserved).
