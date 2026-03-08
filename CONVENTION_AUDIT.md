# Tensor Convention Audit: `qubit ⊗ cavity`

## Phase 0: Audit / Inventory

### Pre-refactor summary

- The core simulator and operator-lifting code were primarily `cavity ⊗ qubit`.
- The unitary-synthesis stack had already started moving toward `qubit ⊗ cavity`, so the repository was mixed.
- Many tests, examples, and notebooks were ambiguous because they used helpers like `basis_state(...)`, while those helpers still encoded the old order.
- Several notebooks and saved notebook outputs still spelled out the old ordering in text.

### Files inspected and changed

| File | Before | Evidence used for classification | Change made |
| --- | --- | --- | --- |
| `README.md` | `cavity ⊗ qubit` docs | Explicit prose said `|n>_cavity tensor |q>_qubit` | Updated docs to state full-space `|q> ⊗ |n>` and clarified subspace ordering separately |
| `cqed_sim/core/model.py` | `cavity ⊗ qubit` | `a` and `b` were lifted in cavity-first order; `basis_state()` returned the old order | Rebuilt operators and basis states as qubit-first |
| `cqed_sim/core/ideal_gates.py` | `cavity ⊗ qubit` | Qubit/cavity embeddings and SQR block construction were lifted in the old order | Re-embedded all ideal operators as `O_q ⊗ I_c` or `I_q ⊗ O_c` |
| `cqed_sim/core/__init__.py` | ambiguous | Export surface did not expose convention helpers | Exported new convention helpers |
| `cqed_sim/core/conventions.py` | new | New explicit helper module | Added canonical dims/index helpers for `qubit ⊗ cavity` |
| `cqed_sim/operators/basic.py` | `cavity ⊗ qubit` | Tensor helper and joint basis-state builder were cavity-first | Replaced with `tensor_qubit_cavity(...)` and qubit-first basis states |
| `cqed_sim/operators/__init__.py` | ambiguous | Exported the old tensor helper name | Exported the qubit-first helper |
| `cqed_sim/sim/extractors.py` | `cavity ⊗ qubit` | Partial traces and conditioned projectors assumed cavity-first subsystem indices | Switched reductions/projectors to qubit-first subsystem indices |
| `cqed_sim/sim/noise.py` | `cavity ⊗ qubit` | Total identity for dephasing was built in old order | Rebuilt as qubit-first |
| `cqed_sim/sim/runner.py` | `cavity ⊗ qubit` | Default excited-state projector used the old embedding | Embedded `P_e` as `|e><e| ⊗ I_c` |
| `cqed_sim/simulators/common.py` | `cavity ⊗ qubit` | Initial product states were built cavity-first | Rebuilt all initial states as qubit-first |
| `cqed_sim/observables/fock.py` | mixed | Helper basis construction inherited old ordering; debug text was ambiguous | Fixed helper calls and made basis-order text explicit |
| `cqed_sim/tomo/protocol.py` | mixed | Qubit reductions, Fock projectors, and one `n_cav` dimension read still used old subsystem positions | Reworked tomography helpers and true-vector extraction for qubit-first |
| `cqed_sim/analysis/sqr_speedlimit_multitone_gaussian.py` | `cavity ⊗ qubit` | Validation extracted contiguous cavity-first 2x2 blocks | Reindexed full-space blocks as qubit-first |
| `cqed_sim/snap_opt/experiments.py` | `cavity ⊗ qubit` | Fast-reset `X` operator was cavity-first | Embedded qubit `X` as `X ⊗ I_c` |
| `cqed_sim/unitary_synthesis/sequence.py` | mixed | Full-space drift, primitive embeddings, and identity construction mixed conventions | Rebuilt full-space synthesis operators and drift indices as qubit-first |
| `cqed_sim/unitary_synthesis/subspace.py` | mixed | Full-space indexing assumed cavity-first while restricted subspace order was interleaved-by-Fock | Reindexed full-space access via qubit-first helpers and kept restricted block ordering explicit |
| `cqed_sim/unitary_synthesis/targets.py` | mixed | Target constructors permuted from cavity-first reference layouts | Built targets directly in qubit-first order and removed conversion helpers |
| `cqed_sim/tests/test_sanity.py` | mixed | Reduced cavity state used old trace index; two explicit product states were reversed | Fixed partial traces and product-state construction |
| `cqed_sim/tests/test_sqr_calibration.py` | mixed | One Bloch sanity test embedded the final state in the wrong slot | Re-embedded the qubit state as the first subsystem |
| `tests/test_03_dispersive_and_ramsey.py` | mixed | Qubit `ptrace` still used cavity-first index | Switched to qubit-first trace index |
| `tests/test_04_xy_phase_and_overlap.py` | mixed | Qubit `ptrace` still used cavity-first index | Switched to qubit-first trace index |
| `tests/test_06_leakage_drag_and_higher_order.py` | mixed | Leakage projector and qubit reductions assumed old order | Re-embedded projector and corrected qubit reduction index |
| `tests/test_10_chi_convention.py` | mixed | Qubit reductions still used cavity-first index | Switched to qubit-first trace index |
| `tests/test_11_model_invariants.py` | mixed | Product-state reconstruction and subsystem reductions needed qubit-first semantics | Corrected subsystem reductions and reconstruction order |
| `tests/test_14_dissipation.py` | mixed | Qubit coherence trace used old index | Switched to qubit-first trace index |
| `tests/test_16_ideal_primitives_and_extractors.py` | mostly `qubit ⊗ cavity` | One cavity/qubit tensor was still reversed | Fixed the remaining joint-state construction |
| `tests/test_17_gate_library_integration.py` | mostly `qubit ⊗ cavity` | Explicit ideal product state needed verification after the core flip | Kept qubit-first state construction consistent with refactor |
| `tests/test_18_noise_and_sideband.py` | mixed | Basis-state labels and one qubit coherence trace still reflected the old helper semantics | Rewrote explicit states/overlaps to the qubit-first basis and fixed the qubit trace index |
| `tests/test_19_fock_tomo.py` | mixed | Excited-state readout used old qubit trace index in one path | Switched to qubit-first trace index |
| `tests/test_20_gaussian_iq_convention.py` | mixed | Full-space `n=0` block extraction was hard-coded to old index positions | Reindexed by `n_cav` from qubit-first dims |
| `tests/test_25_tensor_product_convention.py` | new | Dedicated validation file added for this refactor | Added explicit convention regression tests |
| `cqed_sim/unitary_synthesis/tests/test_subspace.py` | mixed | Expected subspace indices reflected the old full-space mapping | Updated expected indices for qubit-first full-space mapping |
| `cqed_sim/unitary_synthesis/tests/test_free_evolve_condphase.py` | mixed | Full-space phase ratios extracted old 2x2 blocks | Reindexed full-space blocks |
| `cqed_sim/unitary_synthesis/tests/test_primitives_and_backends.py` | mixed | One full-operator validation path used old block indices | Reindexed qubit/cavity blocks |
| `cqed_sim/unitary_synthesis/tests/test_synthesis_and_targets.py` | mixed | Cluster reference helper permuted from cavity-first ordering | Matched the new direct qubit-first target constructors |
| `cqed_sim/unitary_synthesis/tests/test_time_policy_and_condphase.py` | mixed | Full-space phase-ratio extraction used old block indices | Reindexed qubit/cavity blocks |
| `examples/displacement_qubit_spectroscopy.py` | mixed | Reduced cavity state used old trace index | Switched to cavity trace on subsystem 1 |
| `examples/fock_tomo_workflow.py` | `cavity ⊗ qubit` | Correlated density-matrix construction and Fock projectors were embedded cavity-first | Rebuilt example states and projectors as qubit-first |
| `examples/ringdown_noise.py` | `cavity ⊗ qubit` | Initial coherent-state preparation was cavity-first | Rebuilt the initial state as qubit-first |
| `examples/simulate_fock_tomo_and_sqr_calibration.py` | mixed | Qubit traces, one cavity trace, and one `n_cav` dimension read still assumed the old order | Corrected trace indices and dimension lookup |
| `examples/sqr_multitone_study.py` | `cavity ⊗ qubit` | Reference-state builders, active-support helpers, phase analysis, and a runtime Bloch probe were cavity-first | Rebuilt all explicit states, traces, and dimension lookups as qubit-first |
| `landgraf_reproduction.ipynb` | `cavity ⊗ qubit` | Conditional Hamiltonian operators were built as `proj_n ⊗ sigma` | Rebuilt notebook operators as `sigma ⊗ proj_n` |
| `chi_evolution_copy.ipynb` | `cavity ⊗ qubit` | Product states and qubit reductions were cavity-first | Swapped state ordering and qubit trace indices |
| `one_tone_sqr_xyz_demo.ipynb` | `cavity ⊗ qubit` | Initial-state bank used cavity-first tensors and labels like `|n,g>` | Swapped tensor order and renamed labels to qubit-first form |
| `sequential_simulation.ipynb` | `cavity ⊗ qubit` docs | Saved notebook output text still printed the old basis ordering | Rewrote saved output strings to qubit-first wording |
| `examples/unitary_synthesis_cluster_optimization.ipynb` | mixed | Helper and markdown still described a cavity-first conversion step | Replaced it with qubit-first validation-only wording and helper |

### Files inspected and retained without code changes

The following files were scanned for convention-dependent tensor products, partial traces, or ordering text and did not require semantic changes after the core refactor:

- `examples/sanity_run.py`
- `examples/sideband_swap.py`
- `examples/unitary_synthesis_demo.py`
- `tests/test_01_sanity_and_free.py`
- `tests/test_02_cavity_drive_and_kerr.py`
- `tests/test_05_detuning_and_frames.py`
- `tests/test_07_convergence_regression.py`
- `tests/test_09_runtime_policy.py`
- `tests/test_12_pulse_semantics.py`
- `tests/test_13_hardware_extended.py`
- `tests/test_15_numerics_performance.py`
- `tests/test_24_displacement_selective_spectroscopy.py`
- `tests/test_prl133_unitarity_trace.py`
- `SQR_calibration.ipynb`
- `SQR_optimization_demo.ipynb`
- `SQR_speedlimit_multitone_gaussian.ipynb`
- `SQR_three_gate_optimization.ipynb`

Notes:

- Several of these files were previously ambiguous rather than explicitly correct, because they routed through `model.basis_state(...)` or extraction helpers. Once the core helpers were refactored, their effective convention became qubit-first without requiring local code changes.
- Files outside the package/tests/examples/notebook surface that did not contain tensor-product, basis-index, or partial-trace logic were keyword-scanned and not treated as convention-sensitive.

## Phase 1: Refactor Plan

### Current conventions found during the audit

- Core simulation internals: mostly `cavity ⊗ qubit`
- Unitary synthesis: mixed; restricted subspace APIs were qubit-aware, but full-space embeddings still assumed cavity-first in several places
- Tests/examples/notebooks: mixed and often ambiguous through helpers
- Docs/comments/notebook outputs: several explicit cavity-first statements remained

### Exact changes planned and then applied

1. Make the package-level convention explicit with reusable indexing helpers.
2. Flip all full-space state/operator embeddings to `qubit ⊗ cavity`.
3. Correct subsystem reductions:
   - qubit reduction is `ptrace(..., 0)`
   - cavity reduction is `ptrace(..., 1)`
4. Reindex all full-space 2x2 per-Fock blocks as `[n, n_cav + n]`.
5. Update tests, examples, notebooks, and docs to match the same convention.
6. Add explicit regression tests so the convention is checked directly rather than inferred.

### API-visible changes

- `DispersiveTransmonCavityModel.basis_state(q, n)` now explicitly returns `|q> ⊗ |n>`.
- Public embedding helpers now mean:
  - qubit-only operators -> `O_q ⊗ I_c`
  - cavity-only operators -> `I_q ⊗ O_c`
- `cqed_sim.operators.basic` now exposes `tensor_qubit_cavity(...)` instead of the old cavity-first helper semantics.
- Any user code that implicitly relied on the old flat-index order or old `ptrace` indices is intentionally broken and must be updated.

### Downstream risks identified before implementation

- Existing notebooks/examples could still execute numerically while silently labeling or interpreting states incorrectly.
- Tests that only used helper abstractions could remain green while masking a convention mismatch.
- Synthesis code had the highest risk of hidden mismatch because it mixed restricted-subspace order with full-space order.

## Phase 2: Implementation Summary

### Core simulator / helper changes

- Added explicit qubit-first indexing helpers in `cqed_sim/core/conventions.py`.
- Refactored the model, ideal-gate embeddings, operator helpers, tomography helpers, simulation extractors, runner defaults, and common initial-state builders to qubit-first.
- Reindexed full-space synthesis, target construction, and per-Fock block extraction to qubit-first.

### Tests / examples / notebooks updated

- Updated existing tests to use qubit-first `ptrace` indices, operator embeddings, and full-space block extraction.
- Added `tests/test_25_tensor_product_convention.py` as a dedicated convention regression file.
- Updated examples and notebooks to rebuild states/operators as `qt.tensor(qubit, cavity)` and to label states in qubit-first order.
- Rewrote stale notebook output strings and markdown that still documented the old basis ordering.

## Validation

### New or strengthened validation categories

- Basis-state ordering:
  - `tests/test_25_tensor_product_convention.py::test_basis_state_flat_indices_follow_qubit_then_cavity`
- Operator embedding:
  - `tests/test_25_tensor_product_convention.py::test_operator_embeddings_act_on_qubit_and_cavity_slots`
- Hamiltonian structure:
  - `tests/test_25_tensor_product_convention.py::test_static_hamiltonian_diagonal_uses_qubit_cavity_order`
- Expectation-value sanity:
  - `tests/test_25_tensor_product_convention.py::test_conditioned_bloch_uses_fock_projector_on_second_subsystem`
- Round-trip consistency:
  - `tests/test_25_tensor_product_convention.py::test_state_prep_evolution_and_measurement_keep_qubit_cavity_consistent`
- Existing regression coverage updated:
  - `tests/test_03_dispersive_and_ramsey.py`
  - `tests/test_10_chi_convention.py`
  - `tests/test_11_model_invariants.py`
  - `tests/test_16_ideal_primitives_and_extractors.py`
  - `tests/test_20_gaussian_iq_convention.py`
  - `cqed_sim/unitary_synthesis/tests/test_subspace.py`
  - `cqed_sim/unitary_synthesis/tests/test_synthesis_and_targets.py`
  - `cqed_sim/tests/test_sanity.py`
  - `cqed_sim/tests/test_sqr_calibration.py`

### Verification actually run

The following commands were run successfully:

```bash
python -m pytest tests/test_25_tensor_product_convention.py -q
python -m pytest tests/test_16_ideal_primitives_and_extractors.py -q
python -m pytest tests/test_20_gaussian_iq_convention.py -q
python -m pytest tests/test_03_dispersive_and_ramsey.py tests/test_10_chi_convention.py tests/test_11_model_invariants.py tests/test_19_fock_tomo.py -q
python -m pytest cqed_sim/tests/test_sanity.py -k "test_baseline_vs_refactor_case_a or test_conditioned_bloch_matches_known_state or test_relative_phase_definition" -q
python -m pytest cqed_sim/tests/test_sqr_calibration.py -k "test_identity_target_gives_high_fidelity" -q
python -m pytest tests/test_25_tensor_product_convention.py tests/test_16_ideal_primitives_and_extractors.py cqed_sim/tests/test_sanity.py -k "test_baseline_vs_refactor_case_a or test_conditioned_bloch_matches_known_state or test_relative_phase_definition or test_basis_state_flat_indices_follow_qubit_then_cavity or test_operator_embeddings_act_on_qubit_and_cavity_slots or test_static_hamiltonian_diagonal_uses_qubit_cavity_order or test_conditioned_bloch_uses_fock_projector_on_second_subsystem or test_state_prep_evolution_and_measurement_keep_qubit_cavity_consistent" -q
python -m pytest tests/test_20_gaussian_iq_convention.py cqed_sim/unitary_synthesis/tests/test_subspace.py cqed_sim/unitary_synthesis/tests/test_synthesis_and_targets.py -k "test_cluster_mps_target_matches_noah_u1_convention or test_a1_deterministic_basis_ordering or test_a2_projector_properties or test_a3_embed_extract_roundtrip or test_gaussian_i_quadrature_is_x90_with_g_to_minus_y or test_gaussian_q_quadrature_is_y90_with_g_to_plus_x or test_gaussian_phase_sweep_matches_cos_phi_sin_phi_axis or test_detuning_sign_flips_z_error_axis or test_single_tone_multitone_and_gaussian_match_iq_convention" -q
```

Additional notebook verification:

- Parsed the modified notebooks as JSON after editing to confirm they remained valid notebook files.
- Ran a repo scan excluding generated output directories and found no remaining old-order strings such as `|n>_cavity tensor |q>_qubit`, `qt.tensor(cavity, qubit)`, or `cavity ⊗ qubit`.

## Remaining ambiguities / caveats

- The repository contains many generated outputs and figures under `outputs/`; these were not treated as source-of-truth for subsystem semantics unless they were saved notebook outputs or developer-facing text.
- The restricted subspace returned by `Subspace.qubit_cavity_block(...)` intentionally remains interleaved by Fock sector (`|g,n>, |e,n>` per manifold). That is not a contradiction: it is a restricted-basis ordering layered on top of a full-space `qubit ⊗ cavity` convention.
- One large combined pytest invocation timed out; verification was therefore completed with smaller targeted runs that exercised the refactored convention paths directly.

## Final statement

The full codebase now uses the explicit full-space tensor-product convention:

`qubit ⊗ cavity`

That means:

- subsystem ordering is qubit first, cavity second
- qubit-only operators embed as `O_q ⊗ I_c`
- cavity-only operators embed as `I_q ⊗ O_c`
- state preparation, evolution, tomography, synthesis, examples, tests, and inspected notebooks now agree on that convention
