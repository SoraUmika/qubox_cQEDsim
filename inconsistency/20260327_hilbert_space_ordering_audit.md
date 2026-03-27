# Hilbert-Space Ordering Audit — 2026-03-27

## Summary

Full repository audit of tensor-product subsystem ordering convention.
The intended convention is **qubit ⊗ cavity** (qubit first, cavity second).

## Confirmed Issues

### 1. `sqr()` and `multi_sqr()` in `cqed_sim/gates/coupled.py` used cavity ⊗ qubit

**What:** The `sqr()` and `multi_sqr()` functions constructed operators in
**cavity-first, qubit-second** ordering (`qt.tensor(|n><n|, R)`) instead of
the repository-wide **qubit-first, cavity-second** convention.

**Where:** `cqed_sim/gates/coupled.py`, lines ~430–525.

**Affected components:**
- `cqed_sim.gates.coupled.sqr()`
- `cqed_sim.gates.coupled.multi_sqr()`
- `tests/test_42_gates.py` (SQR test states constructed as cavity ⊗ qubit)
- `API_REFERENCE.md` (documented the cavity-first exception)
- `documentations/physics_conventions.md` (documented cavity-first for SQR)
- `physics_and_conventions/physics_conventions_report.tex` (same)

**Why inconsistent:** Every other function in `gates/coupled.py` (dispersive_phase,
conditional_rotation, conditional_displacement, controlled_parity, controlled_snap,
jaynes_cummings, blue_sideband) uses qubit ⊗ cavity. The SQR functions used the
opposite ordering, creating a composability hazard: multiplying an SQR unitary with
any other coupled gate would silently produce a wrong result because the subsystems
are transposed.

**Consequences:**
- Gate composition `U_other * U_sqr` would have acted on the wrong subsystem
  slots without any error.
- The deviation was documented, but documentation does not prevent silent numerical
  errors when composing operators.
- `sqr_op()` in `cqed_sim/core/ideal_gates.py` already used qubit-first ordering,
  so the two SQR implementations were mutually inconsistent.

## Status

**Fixed** — 2026-03-27.

## Fix Record

### Code changes

| File | Change |
|------|--------|
| `cqed_sim/gates/coupled.py` | Swapped tensor-product arguments in `sqr()` and `multi_sqr()` to place qubit first, cavity second. Updated return-size docstrings and Notes sections. |
| `tests/test_42_gates.py` | Updated SQR test states from `qt.tensor(cav, qubit)` to `qt.tensor(qubit, cav)`. Updated identity comparison. |
| `API_REFERENCE.md` | Replaced "cavity first, qubit second" SQR note with "qubit first, cavity second". |
| `documentations/physics_conventions.md` | Updated SQR tensor ordering section. |
| `physics_and_conventions/physics_conventions_report.tex` | Updated SQR paragraph. |

### Regression tests added

| File | Tests added |
|------|-------------|
| `tests/test_25_tensor_product_convention.py` | `test_sqr_uses_qubit_first_cavity_second`, `test_multi_sqr_uses_qubit_first_cavity_second`, `test_sqr_composable_with_other_qubit_cavity_gates`, `test_sqr_op_and_sqr_produce_same_unitary`, `test_all_coupled_gates_have_qubit_times_cavity_dimension`, plus convention helpers for flat indexing, partial traces, and embeddings. |

## Suspected / Follow-up Questions

None. All other tensor-product sites audited across `core/`, `sim/`, `gates/`,
`operators/`, `tomo/`, `measurement/`, `analysis/`, `calibration/`,
`unitary_synthesis/`, `optimal_control/`, `floquet/`, `rl_control/`, and
`plotting/` are consistent with qubit ⊗ cavity.
