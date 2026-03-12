# Sideband / Multilevel / Kerr Audit Inconsistency Report

Created: 2026-03-11 02:44:49 local time
Status: fixed on 2026-03-11 for items 1-3; item 4 remained a confirmed non-issue.

## Confirmed issues

### 1. Reusable sideband API is too narrow for the requested physics

- What it is: the current reusable sideband path is only the fixed string target `"sideband"`, implemented as the operator pair `a^\dagger b` and `a b^\dagger`.
- Where it appears:
  - `cqed_sim/core/model.py`
  - `cqed_sim/sim/runner.py`
  - `examples/sideband_swap.py`
  - `tests/test_18_noise_and_sideband.py`
- Affected components:
  - pulse-to-Hamiltonian drive mapping
  - sideband examples
  - sideband tests
- Why inconsistent:
  - the repository already supports `n_tr >= 3`, but the reusable drive API does not let callers select a transmon manifold such as `|g> <-> |f>` or select which bosonic mode participates in an effective sideband interaction.
  - this prevents a clean implementation of the requested effective `|f><g| b` sideband without ad hoc notebook logic.
- Consequence:
  - the simulator can demonstrate a two-level exchange-like sideband, but it cannot yet express the requested fast multilevel bosonic-control physics through a stable public API.

### 2. Transmon dissipation is not resolved by ladder transition

- What it is: `NoiseSpec` exposes only aggregate `t1` and `tphi`, and `collapse_operators(...)` applies relaxation through the full lowering operator `b`.
- Where it appears:
  - `cqed_sim/sim/noise.py`
- Affected components:
  - open-system multilevel transmon simulations
  - shelving and sideband benchmarks
- Why inconsistent:
  - the requested physics explicitly needs separate `|f> -> |e>` and `|e> -> |g>` decay support, but the current API cannot assign independent rates to those channels.
- Consequence:
  - multilevel ancilla dynamics can be simulated only with a coarse aggregate decay model, which is insufficient for the requested shelving and sideband-degradation benchmarks.

### 3. Several helpers still assume a two-level qubit even though the core model allows `n_tr >= 3`

- What it is: ideal-gate, Bloch-vector, tomography, and some convenience helpers still hardcode Pauli or dimension-2 behavior.
- Where it appears:
  - `cqed_sim/core/ideal_gates.py`
  - `cqed_sim/sim/extractors.py`
  - `cqed_sim/tomo/protocol.py`
  - related examples/tests that build Pauli-based diagnostics
- Affected components:
  - multilevel analysis
  - shelving diagnostics
  - ancilla-state inspection
- Why inconsistent:
  - the repository advertises `n_tr >= 3` in the public model dataclasses, but several analysis paths still implicitly treat subsystem 0 as a strict qubit.
- Consequence:
  - multilevel simulations run, but the reporting and validation layer is incomplete for `|f>`-involving protocols.

## Confirmed non-issue from the audit

### 4. The core self-Kerr sign does not appear to be flipped

- What was checked:
  - documented convention in `README.md`, `API_REFERENCE.md`, and `physics_and_conventions/physics_conventions_report.tex`
  - implemented convention in `cqed_sim/core/model.py` and `cqed_sim/sim/couplings.py`
  - notebook logic in `test_against_papers/prl115_137002_bosonic_controls.ipynb`
- Conclusion:
  - the notebook uses the documented negative Kerr parameter in the rotating frame, and its short-time coherent-state phase direction matches the documented Hamiltonian sign.
- Why this still matters:
  - the notebook can look counterintuitive if one expects the coherent-state mean to rotate with the opposite sign under `H = +(K/2) n(n-1)`.
- Consequence:
  - the likely bug is not a flipped self-Kerr sign in the simulator core; the real gap is the lack of an explicit regression diagnostic that explains the rotating-frame interpretation.

## Suspected or unresolved follow-up items

### A. The sideband extension should stay inside the existing model/runner architecture

- Suspicion:
  - the cleanest path is to extend `drive_ops` so it can resolve structured multilevel sideband targets, instead of introducing a separate simulator stack.
- Why unresolved:
  - the concrete API shape still needs to be chosen during implementation.

### B. The notebook should probably be migrated into a reusable example or regression helper

- Suspicion:
  - the Kerr-sign diagnosis should live in a normal Python example and test, with the notebook reduced to presentation-only content.
- Why unresolved:
  - this depends on how much of the notebook is kept after the new regression benchmark is added.

## Resolution summary

- Item 1 was resolved by adding structured multilevel drive targets and sideband helpers in the core model/runner path.
- Item 2 was resolved by extending multilevel noise support with per-transition ancilla relaxation handling.
- Item 3 was resolved by adding multilevel extractors, shelving helpers, and explicit sideband validation tests/examples.
- The literature notebook was retained under `test_against_papers/` and renamed to `prl115_137002_bosonic_controls.ipynb` so its role is explicit.
