# Public Tutorial Physics Audit

Created: 2026-04-21 14:09:36 local time

## Confirmed Issues

### 1. `cross_kerr.md` public figure contradicted the documented theory claim

- What:
  - The public page claimed that the simulated phase accumulated linearly on top of the analytic cross-Kerr line.
  - The checked-in public figure instead showed the simulated phase remaining near zero.
- Where:
  - `documentations/tutorials/cross_kerr.md`
  - `documentations/assets/images/tutorials/cross_kerr_phase.png`
  - `tools/generate_tutorial_plots.py`
- Affected components:
  - public tutorial physics interpretation
  - cross-Kerr sign convention trust
  - conditional-phase gate explanation
- Why inconsistent:
  - The generator was not isolating the conditional storage-readout phase in a way consistent with the documented observable.
  - The page text also described the phase with the wrong sign relative to the repo's runtime convention.
- Consequences:
  - users could infer that the simulation disagreed with the Hamiltonian or that the sign convention was undefined.

### 2. `floquet_driven_systems.md` public figure flattened the resonance diagnostic

- What:
  - The public page promised a visible avoided crossing and a resonance-localized gap diagnostic.
  - The checked-in figure showed nearly flat quasienergy branches and a zero gap across the full sweep.
- Where:
  - `documentations/tutorials/floquet_driven_systems.md`
  - `documentations/assets/images/tutorials/floquet_quasienergy_scan.png`
  - `tools/generate_tutorial_plots.py`
- Affected components:
  - Floquet tutorial credibility
  - branch-tracking interpretation
  - sideband resonance teaching
- Why inconsistent:
  - The public plot was using a global minimum-adjacent-gap diagnostic that could be dominated by irrelevant or duplicate branches rather than the resonant pair of interest.
- Consequences:
  - users could conclude that the example failed to enter a resonant regime at all.

### 3. Several physics tutorial pages made theory claims without explicit references

- What:
  - High-value physics pages described number splitting, Kerr revival, SNAP control, readout physics, dispersive shifts, and cross-Kerr interactions without a reusable citation path.
- Where:
  - `documentations/tutorials/cross_kerr.md`
  - `documentations/tutorials/floquet_driven_systems.md`
  - `documentations/tutorials/number_splitting.md`
  - `documentations/tutorials/readout_resonator.md`
  - `documentations/tutorials/kerr_free_evolution.md`
  - `documentations/tutorials/dispersive_shift_dressed.md`
  - `documentations/tutorials/open_system_dynamics.md`
  - `documentations/tutorials/snap_fock_state_prep.md`
- Affected components:
  - public documentation provenance
  - tutorial maintainability
  - alignment with the repo citation policy
- Why inconsistent:
  - The repo already requires literature-grounded tutorial claims, but these pages had not been brought into a common citation standard.
- Consequences:
  - future maintenance would have to rediscover the source papers from scratch, and users could not easily tell which claims were review-level, paper-level, or internal.

### 4. The public tutorial layer did not have a tutorial-by-tutorial audit artifact

- What:
  - There was no single public matrix recording plot path, source path, validation lane, and status for each tutorial guide page.
- Where:
  - `documentations/tutorials/`
- Affected components:
  - docs maintenance
  - tutorial regression triage
  - public evidence provenance
- Why inconsistent:
  - Prior inconsistency reports and tests existed, but there was no synced public artifact tying each checked-in tutorial page to a declared validation target.
- Consequences:
  - it was too easy for a page, image, and generator to drift apart unnoticed.

## Suspected Issues

- Several public tutorial figures generated outside `tools/generate_tutorial_plots.py` still rely on script-level or workflow-level validation rather than plot-specific JSON summaries. That is acceptable for this pass, but future work could standardize summary payloads across `tools/generate_foundational_tutorial_plots.py` and `tools/generate_remaining_tutorial_plots.py`.
- Some older markdown files still contain mojibake punctuation from previous exports. This is mostly cosmetic now, but it increases patch fragility and should be normalized in a follow-up docs cleanup.

## Open Questions

- The current audit matrix declares a validation target for every public tutorial page, but not every page yet emits a machine-readable per-figure summary next to the PNG. That remains a worthwhile future standardization target.
- The current pass focuses on the public docs/tutorial surface. Notebook outputs beyond the public guide pages were not re-audited exhaustively unless they directly fed a checked-in website figure.

## Status

- Fixed in this pass.

## Fix Record

- `tools/generate_tutorial_plots.py`
  - repaired the cross-Kerr public observable extraction so the plotted quantity is the signed conditional storage-readout phase.
  - reworked the Floquet public figure to highlight the resonant hybridized pair and its finite avoided-crossing gap.
  - added validation JSON outputs for the repaired public figures.
- `tests/test_61_public_tutorial_plot_validation.py`
  - added public-plot regressions for the repaired cross-Kerr and Floquet figures.
- `tests/test_56_tutorial_physics_validation.py`
  - extended analytic-law regression coverage for power Rabi and storage decay.
- `documentations/tutorials/*.md`
  - updated cross-Kerr and Floquet tutorial text to match the repaired figures.
  - added missing reference sections to the highest-value physics pages touched in this pass.
  - corrected the readout-resonator notebook path drift.
- `paper_summary/`
  - added reusable summaries for the canonical SNAP, number-splitting, Kerr-revival, and circuit-QED review references.
- `documentations/tutorials/tutorial_physics_audit_matrix.md`
  - added the public tutorial audit matrix covering each tutorial guide page.
