# 2026-03-30 12:17:48 Positive Drive Frequency / Carrier API Migration

## Confirmed Issues

### 1. Public convention guidance centered the raw negative carrier instead of positive physical tone frequencies

- What:
  - The repo's low-level runtime is internally consistent with `Pulse.carrier = -omega_transition(frame)`.
  - Public-facing guidance in the README, API reference, website docs, and local module READMEs still presented that raw negative carrier as the main user convention.
- Where:
  - `README.md`
  - `API_REFERENCE.md`
  - `documentations/index.md`
  - `documentations/physics_conventions.md`
  - `cqed_sim/core/README.md`
  - `cqed_sim/pulses/README.md`
  - `physics_and_conventions/physics_conventions_report.tex`
- Affected components:
  - pulse construction
  - spectroscopy plotting and scan setup
  - external hardware integration and lab-side frequency translation
  - public documentation
- Why this is inconsistent:
  - The chosen canonical direction for the repo is a hybrid standard: textbook-style frame and Hamiltonian semantics, but positive physical tone frequencies at public API boundaries.
  - Presenting the raw negative carrier as the primary user convention forces users to reason about an internal rotating-frame sign rather than the physical tone frequency they actually set on hardware.
- Consequences:
  - user code can silently mix physical tone frequencies with raw internal carriers
  - documentation encourages a sign convention that is operationally easy to misuse outside the simulator internals

### 2. The public frequency-helper layer lacked an explicit positive-frequency translation path

- What:
  - `cqed_sim.core.frequencies` exposed only `carrier_for_transition_frequency(...)` and `transition_frequency_from_carrier(...)`.
  - There was no public helper family for converting between positive physical drive-tone frequencies, rotating-frame transition frequencies, and the raw low-level carrier representation.
- Where:
  - `cqed_sim/core/frequencies.py`
  - `cqed_sim/core/__init__.py`
  - `cqed_sim/__init__.py`
- Affected components:
  - public helper API
  - downstream calibration code
  - examples and user code that need explicit frame-aware frequency translation
- Why this is inconsistent:
  - The intended repo standard should make frame and carrier semantics explicit at public wrapper boundaries rather than requiring users to reconstruct them ad hoc.
- Consequences:
  - downstream callers had to open-code `frame - drive`, `drive - frame`, or `-transition` conversions
  - external code could remain correct numerically but opaque in meaning, which is exactly the kind of convention drift the repo is trying to eliminate

## Suspected / Follow-up Questions

- The raw `Pulse.carrier` field and low-level builder outputs still use the legacy negative rotating-frame representation. This report does not treat that as a bug for the current phase, but it remains a follow-up migration surface if the repo later decides to make positive physical tone frequencies the direct pulse-construction API.
- Follow-up completed on 2026-03-30: `cqed_sim.optimal_control` now documents and exports `c(t) = I(t) + i Q(t)`, with model-backed `Q` terms built as `+i(raising - lowering)` so runtime replay stays Hamiltonian-consistent.
- The existing VS Code task used to build `physics_and_conventions/physics_conventions_report.tex` is currently misquoted for this workspace path and fails before invoking the batch file. The report source itself may still be valid; the task wiring needs separate cleanup.
- Exact alignment to a single so-called `QuTiP standard` remains a separate design question rather than a residual inconsistency in the current repo. QuTiP does not expose the same complex-envelope carrier abstraction directly, so changing the runtime to an `exp(-i omega t)` pulse convention would be a deeper breaking migration, not part of the current consistency fix.

## Status

Fixed for the current hybrid public/internal standard on 2026-03-30.

The new helper layer now lets user-facing code stay in positive physical drive-tone frequencies while converting explicitly to the raw low-level carrier only at the pulse boundary. The main API docs, tutorial docs, user guides, generated tutorial notebooks, and checked-in site output have been synchronized to that public-facing convention. The underlying runtime representation has not yet been migrated away from the raw negative carrier field.

Additional wrapper-layer migration completed on 2026-03-30:

- SQR multitone helpers now preserve both the raw internal waveform carrier and explicit positive drive-frequency metadata.
- Tomography now exposes `selective_qubit_drive_frequency(...)` as the positive-frequency helper, while keeping `selective_qubit_freq(...)` as the backward-compatible raw carrier helper.
- RL qubit and storage primitives now translate via positive drive frequencies internally and record both `drive_frequency` and `internal_carrier` in segment metadata.
- Sideband RL metadata remains labeled as `modulation_frequency` rather than a positive physical drive frequency because that boundary is still an effective sideband-modulation surface rather than a simple single-mode carrier convention.

Final repo-wide audit update on 2026-03-30:

- A broad convention-sensitive search across code, module READMEs, API docs, tutorials, examples, and the physics documentation did not find an additional code-path regression that still exposed raw-carrier-first semantics at public boundaries.
- The remaining work was documentation explicitness: the canonical physics sources and a few residual public/module documentation surfaces were tightened so they now state the hybrid rule more directly.
- The canonical statement is now explicit: public boundaries use positive physical drive frequencies, the low-level runtime keeps `Pulse.carrier = -omega_transition(frame)` because of the `exp(+i(omega t + phase))` waveform convention, and sideband APIs may intentionally remain modulation-frequency surfaces.
- A final follow-up audit driven specifically by drive-sign, carrier, and frame-convention consistency found only one remaining maintainability issue: an internal tutorial-plot generator and one sideband tutorial page still hand-coded a red-sideband carrier instead of using the documented effective-sideband helper path. Those surfaces are now aligned as well.

## Fix Record

- Added explicit positive-frequency translation helpers in `cqed_sim/core/frequencies.py`:
  - `drive_frequency_for_transition_frequency(...)`
  - `transition_frequency_from_drive_frequency(...)`
  - `internal_carrier_from_drive_frequency(...)`
  - `drive_frequency_from_internal_carrier(...)`
- Re-exported the new helpers through:
  - `cqed_sim/core/__init__.py`
  - `cqed_sim/__init__.py`
- Added regression coverage in:
  - `tests/conventions/test_conventions.py`
  - `tests/test_10_chi_convention.py`
- Updated the main public documentation surfaces to describe the raw carrier as an internal compatibility representation and the positive drive-frequency helper layer as the preferred user-facing translation path:
  - `README.md`
  - `API_REFERENCE.md`
  - `documentations/index.md`
  - `documentations/physics_conventions.md`
  - `documentations/api/pulses.md`
  - `cqed_sim/core/README.md`
  - `cqed_sim/pulses/README.md`
  - `physics_and_conventions/physics_conventions_report.tex`
- Updated the tutorial-facing and workflow-facing documentation surfaces so spectroscopy examples now present positive physical drive frequencies first and translate to the raw low-level carrier only at pulse construction:
  - `tutorials/_generate_tutorials.py`
  - `tutorials/_generate_workflow_tutorials.py`
  - `tutorials/00_tutorial_index.ipynb`
  - `tutorials/02_units_frames_and_conventions.ipynb`
  - `tutorials/06_qubit_spectroscopy.ipynb`
  - `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`
  - `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`
  - `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb`
  - `tutorials/README.md`
  - `tutorials/conventions_quick_reference.md`
  - `documentations/tutorials/units_frames_conventions.md`
  - `documentations/tutorials/displacement_spectroscopy.md`
  - `documentations/tutorials/number_splitting.md`
  - `documentations/tutorials/frame_sanity_checks.md`
  - `documentations/tutorials/sideband_swap.md`
  - `documentations/user_guides/frames.md`
  - `documentations/user_guides/pulse_construction.md`
  - `documentations/api/core.md`
  - `documentations/api/pulses.md`
  - regenerated `site/` tutorial, API, and user-guide pages
- Extended regression coverage for the tutorial layer in:
  - `tests/test_33_usage_examples_spectroscopy_sign.py`
- Updated wrapper-layer and module-local convention surfaces so additional public helpers now expose positive drive frequencies explicitly while preserving the raw low-level runtime representation:
  - `cqed_sim/pulses/envelopes.py`
  - `cqed_sim/pulses/calibration.py`
  - `cqed_sim/pulses/builders.py`
  - `cqed_sim/calibration/conditioned_multitone.py`
  - `cqed_sim/calibration/sqr.py`
  - `cqed_sim/tomo/protocol.py`
  - `cqed_sim/tomo/__init__.py`
  - `cqed_sim/rl_control/runtime.py`
  - `cqed_sim/optimal_control/parameterizations.py`
  - `cqed_sim/sim/README.md`
  - `cqed_sim/unitary_synthesis/README.md`
  - `cqed_sim/rl_control/README.md`
  - `cqed_sim/optimal_control/README.md`
  - `cqed_sim/tomo/README.md`
  - `documentations/api/pulses.md`
  - `documentations/api/tomography.md`
  - `documentations/api/optimal_control.md`
  - `API_REFERENCE.md`
  - regenerated `site/` API pages and search index
- Tightened the canonical physics/convention wording and residual public/module documentation surfaces after the final repo-wide audit:
  - `documentations/physics_conventions.md`
  - `physics_and_conventions/physics_conventions_report.tex`
  - `README.md`
  - `documentations/api/unitary_synthesis.md`
  - `API_REFERENCE.md`
  - `cqed_sim/unitary_synthesis/README.md`
  - `cqed_sim/calibration/README.md`
- Replaced the last remaining hand-coded sideband-carrier example surfaces with the canonical effective-sideband helper path:
  - `documentations/tutorials/sideband_interactions.md`
  - `tools/generate_remaining_tutorial_plots.py`
- Added focused regression coverage for the wrapper-layer migration in:
  - `tests/test_19_fock_tomo.py`
  - `tests/test_23_sqr_additive_amplitude_correction.py`
  - `tests/test_39_rl_control_extensions.py`
- Verified the migration with:
  - `python -m pytest tests/conventions/test_conventions.py tests/test_19_fock_tomo.py tests/test_23_sqr_additive_amplitude_correction.py tests/test_39_rl_control_extensions.py -q`
  - `python -m mkdocs build --strict`

Remaining follow-up work:

- decide whether a later phase should change the direct `Pulse` construction API itself
- decide whether the sideband wrapper surface should gain an explicit positive-frequency lab convention or remain an effective modulation-frequency API
- repair or replace the broken physics-report VS Code task configuration