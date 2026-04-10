# 2026-04-09 17:19:35 Floquet Transition-Strength Scalar Extraction

## Confirmed Issues

### 1. `compute_floquet_transition_strengths(...)` assumed a 1x1 indexable object where QuTiP now returns a scalar

- What:
  - `cqed_sim.floquet.analysis.compute_floquet_transition_strengths(...)` computed matrix elements with
    `(bra * operator * ket)[0, 0]`.
  - Under the current QuTiP behavior in this repository environment, `bra * operator * ket` returns a Python complex scalar rather than a 1x1 `Qobj`.
- Where:
  - `cqed_sim/floquet/analysis.py`
- Affected components:
  - `cqed_sim.floquet.compute_floquet_transition_strengths(...)`
  - any example, study, or paper-validation workflow that tries to compute Floquet transition strengths
- Why this is inconsistent:
  - The public API advertises harmonic-resolved Floquet transition strengths as a supported analysis feature, but the implementation depended on an older return-type assumption and crashed in the current supported environment.
- Consequences:
  - transition-strength analysis raised `TypeError: 'complex' object is not subscriptable` instead of returning physical results
  - literature-backed validation of modulation sidebands could not run
  - users had no reliable path to validate probe-side Floquet observables even though the helper was publicly exposed

## Suspected / Follow-up Questions

- The helper now computes matrix elements robustly through `Qobj.overlap(...)`, but broader transition-strength validation is still needed for strongly hybridized multilevel and multimode problems.

## Status

Fixed on 2026-04-09.

## Fix Record

- Updated `cqed_sim/floquet/analysis.py` to compute Floquet transition matrix elements with `mode.overlap(operator * other_mode)` instead of indexing into a presumed 1x1 object.
- Added regression and theory-backed coverage in `tests/test_58_floquet.py`, including a Bessel-sideband validation for sinusoidal longitudinal modulation.
- Added a literature validation workflow in `test_against_papers/silveri_2017_frequency_modulation_sidebands.py`.
- Added a new reference summary in `paper_summary/silveri_tuorila_thuneberg_paraoanu_2017_quantum_systems_under_frequency_modulation.md`.