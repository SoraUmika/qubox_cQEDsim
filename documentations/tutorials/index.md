# Tutorial Curriculum

The primary guided-learning material for `cqed_sim` lives in the repository's top-level `tutorials/` directory. There are now two complementary tutorial tracks:

- a categorized workflow suite that rewrites the representative example scripts as notebook tutorials
- the earlier flat numbered curriculum, which remains useful as a broader API and conventions primer

Start with:

- `tutorials/README.md` for the current workflow-first reading order
- `tutorials/00_getting_started/01_protocol_style_simulation.ipynb`
- `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`
- `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb`
- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`

## Workflow Tutorial Taxonomy

- `tutorials/00_getting_started/`
  - first end-to-end protocol notebook
- `tutorials/10_core_workflows/`
  - displacement spectroscopy and Kerr free evolution
- `tutorials/20_bosonic_and_sideband/`
  - sideband swap, detuned synchronization, sequential reset, and shelving
- `tutorials/30_advanced_protocols/`
  - cross-Kerr, open-system degradation, unitary synthesis, GRAPE optimal control, RL-ready hybrid control, and SNAP optimization
- `tutorials/40_validation_and_conventions/`
  - convention-validation notebooks such as the Kerr sign check

## Foundational Curriculum Still Available

- `tutorials/00_tutorial_index.ipynb`
- `tutorials/01_getting_started_minimal_dispersive_model.ipynb`
- `tutorials/02_units_frames_and_conventions.ipynb`
- `tutorials/06_qubit_spectroscopy.ipynb`
- `tutorials/24_sideband_like_interactions.ipynb`
- `tutorials/26_frame_sanity_checks_and_common_failure_modes.ipynb`

## Conventions

The tutorials follow the runtime conventions documented in:

- `physics_and_conventions/physics_conventions_report.tex`
- `tutorials/conventions_quick_reference.md`

In particular, frame choice, carrier sign, dispersive `chi`, Kerr terms, and truncation meaning are treated consistently with the current public API.

## Tutorials vs Examples

`tutorials/` is for structured, user-facing, step-by-step learning material.

`examples/` is for standalone scripts, audits, studies, paper reproductions, and specialized workflow helpers that are useful in the repository but are not the primary onboarding curriculum.
