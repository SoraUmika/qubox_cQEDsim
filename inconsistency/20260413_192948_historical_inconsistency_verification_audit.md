# 2026-04-13 19:29:48 Historical Inconsistency Verification Audit

## Confirmed Issues

### 1. A unitary-synthesis validation test still lives outside the top-level `tests/` suite

- What:
  - `cqed_sim/unitary_synthesis/tests/test_native_primitives.py` is still present under the package tree rather than under the repository-level `tests/` folder.
- Where:
  - `cqed_sim/unitary_synthesis/tests/test_native_primitives.py`
  - previously tracked in `inconsistency/20260327_160000_full_codebase_consistency_audit.md`
- Affected components:
  - test organization
  - repo-wide pytest discoverability expectations
  - consistency with the repository's numbered top-level test layout
- Why this is inconsistent:
  - `AGENTS.md` requires validation tests to live under the top-level `tests/` folder.
  - The current tree still splits validation coverage across two locations.
- Consequences:
  - developers can miss coverage when auditing only the numbered top-level suite
  - the repo's test-placement convention remains partially unenforced

### 2. The numbered top-level test sequence still has a `test_30_*` gap

- What:
  - No file matching `tests/test_30*` exists in the current repository.
- Where:
  - `tests/`
  - previously tracked in `inconsistency/20260327_160000_full_codebase_consistency_audit.md`
- Affected components:
  - test numbering continuity
  - developer discoverability when tracking coverage by file number
- Why this is inconsistent:
  - The repository otherwise uses a sequential numbered naming scheme for the top-level tests.
- Consequences:
  - this is mostly cosmetic, but it still creates avoidable ambiguity about whether a numbered test was removed, renamed, or never migrated

### 3. Tomography helper documentation still overstates the global unit semantics of the core simulator

- What:
  - `cqed_sim/tomo/device.py` states that the simulation core uses nanoseconds as its internal time unit and instructs readers not to construct `DispersiveTransmonCavityModel` from rad/s values through that path.
  - The repository-wide public contract now says the library is unit-coherent and accepts any internally consistent unit system, with rad/s and seconds as the recommended convention.
  - The `DeviceParameters` helper itself is intentionally a Hz-to-rad/ns convenience layer, but that helper-specific convention is broader in the docstring than the actual library-wide guarantee.
- Where:
  - `cqed_sim/tomo/device.py`
  - `cqed_sim/tomo/README.md`
  - `README.md`
  - `API_REFERENCE.md`
- Affected components:
  - tomography helper documentation
  - global unit/convention guidance
  - user expectations about whether the core model is intrinsically tied to nanoseconds
- Why this is inconsistent:
  - The helper-specific rad/ns workflow is valid, but the current wording can be read as if the core simulator itself requires nanosecond time units.
  - That conflicts with the repo's documented unit-coherent model-layer contract.
- Consequences:
  - users may incorrectly infer that `DispersiveTransmonCavityModel` or the simulation core rejects rad/s plus seconds workflows
  - future documentation updates can drift further if the helper-specific wording is treated as the global convention

## Suspected / Follow-up Questions

- `documentations/getting_started.md` is still stored at the documentation root even though the MkDocs nav places it under the User Guides section. This remains a mild structure inconsistency, but not a build failure.
- `tools/README.md` is still missing, and only `examples/floquet/README.md` exists among the `examples/*` subdirectories that were previously flagged for discoverability drift. These remain low-priority documentation gaps rather than confirmed API inconsistencies.
- Benchmark cache JSON files still exist at the root of `benchmarks/`, but the active `.gitignore` now covers `benchmarks/latest_results.json` and `benchmarks/latest_results_review_*.json`. That mitigates commit-noise risk, though the files are still present in the tree.
- No active `ConditionalPhaseSQR` or `ConditionalPhaseSQRGate` references were found in `cqed_sim/`, `API_REFERENCE.md`, or `documentations/`. The older March 27 physics/API reports appear superseded rather than still live on that surface.

## Status

- Current status:
  - Fixed on 2026-04-13.
  - Reviewed all reports currently present under `inconsistency/`.
  - Historical reports tied to the positive-drive-frequency migration, optimal-control `I + iQ` migration, targeted-subspace Bloch-vector convention fix, and removed `ConditionalPhaseSQR` API surface all remain fixed or superseded in the active codebase.
  - The previously live test-organization and tomo-unit wording issues identified by this audit were resolved in the same task.
- Resolution summary:
  - Verified the positive-frequency helper migration by confirming the live presence of:
    - `drive_frequency_for_transition_frequency(...)`
    - `transition_frequency_from_drive_frequency(...)`
    - `internal_carrier_from_drive_frequency(...)`
    - `drive_frequency_from_internal_carrier(...)`
  - Verified the optimal-control baseband migration in live code by confirming:
    - `cqed_sim/optimal_control/utils.py` builds `Q` as `+i(raising - lowering)`
    - `cqed_sim/optimal_control/parameterizations.py` documents export as `c(t) = I(t) + i Q(t)`
  - Verified the targeted-subspace Bloch-convention fix by confirming `cqed_sim/calibration/targeted_subspace_multitone.py` now reuses `bloch_vector_from_angles(...)`.
  - Verified that the removed conditional-phase API surface is no longer present in active package/docs search targets.
  - Closed the remaining organization/documentation drift in the same task by moving the stray unitary-synthesis primitive tests into the numbered top-level suite and clarifying the tomography helper's helper-specific rad/ns wording across code and docs.

## Fix Record

- Fixed by:
  - `tests/test_30_native_primitives.py` added and `cqed_sim/unitary_synthesis/tests/test_native_primitives.py` removed, closing both the module-local test placement issue and the missing `test_30_*` numbering gap.
  - `cqed_sim/tomo/device.py` and `cqed_sim/tomo/README.md` updated so `DeviceParameters` is described as a tomography-specific Hz-to-rad/ns convenience path rather than a global simulator requirement.
  - `API_REFERENCE.md`, `documentations/api/tomography.md`, and `documentations/api/analysis.md` updated to match that clarified unit-coherent contract.
  - Focused regression verification passed with:
    - `python -m pytest tests/conventions/test_conventions.py tests/test_19_fock_tomo.py tests/test_36_targeted_subspace_multitone.py tests/test_40_optimal_control_grape.py tests/test_52_structured_optimal_control.py -q`
- Remaining concerns:
  - Lower-priority follow-up items outside the scope of this task remain the only open questions from the broader audit set.