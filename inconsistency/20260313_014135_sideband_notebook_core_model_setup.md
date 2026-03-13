# Sideband Notebook Core-Model Setup Inconsistency Report

Created: 2026-03-13 01:41:35 local time
Status: fixed on 2026-03-13.

## Confirmed issue

### `examples/sideband_interactions.ipynb` was presenting workflow-builder setup instead of direct `cqed_sim` model construction

- What it is:
  - The notebook imported `SequentialSidebandResetDevice`, `build_sideband_reset_model`, `build_sideband_reset_frame`, and `build_sideband_reset_noise` from `examples.workflows.sequential_sideband_reset` to create the simulation setup.
- Where it appears:
  - `examples/sideband_interactions.ipynb`
- Affected components:
  - notebook onboarding path
  - example clarity around the intended `cqed_sim` model layer
  - reuse of the notebook as a direct library demonstration
- Why inconsistent:
  - the notebook is framed as a sideband-interaction and spectrum example for `cqed_sim`, but its initial setup path went through a workflow-specific helper module instead of constructing a core library model directly.
  - that obscures the stable entry points the project expects users to learn first, namely `DispersiveReadoutTransmonStorageModel`, `FrameSpec`, and `NoiseSpec`.
- Consequence:
  - readers can come away thinking the workflow helper is the canonical way to create the modeled system, even though the architecture intends the helper to be an example-side orchestration layer on top of the reusable package.

## Resolution summary

- The notebook now constructs the three-mode system directly with `cqed_sim.DispersiveReadoutTransmonStorageModel`.
- The rotating frame is now created directly with `cqed_sim.FrameSpec`.
- The notebook noise model is now created directly with `cqed_sim.NoiseSpec` and `pure_dephasing_time_from_t1_t2(...)`.
- The final sequential-reset section remains example-side orchestration, but it now runs on top of the directly constructed core model rather than using workflow-side builder functions for setup.
