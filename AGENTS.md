# Project Guidelines

## Startup Policy

- Before taking any action, first read `README.md` to gather project context.
- Do not create, activate, or rely on a virtual environment unless the user explicitly asks for one.
- For Python execution, use the existing system Python at `E:\Program Files\Python311\python.exe` or `E:\Programs\python.exe` for existing python version 3.12.10.
- Do not run dependency installation or environment-management commands unless the user explicitly requests them.

## Python Environment

- Treat this repository as using the existing machine Python environment rather than a repo-local virtual environment.
- Avoid commands such as `python -m venv`, `virtualenv`, `conda create`, `poetry env use`, or other environment bootstrap steps unless the user explicitly asks for them.

## Working Style

- Inspect the current codebase first and prefer minimal changes that match existing conventions.
- Before changing project setup, execution workflow, or dependency state, confirm that the change is necessary for the task.
- Prefer using the repository's existing scripts, tests, and notebooks over introducing new setup layers.
- Do not introduce parallel implementations, duplicate abstractions, or ad hoc workarounds when an existing project path already supports the task.
- Keep new code, documentation, and refactors aligned with the repository’s current architecture unless the task explicitly requires an architectural change.

## QuTiP Native-First Policy

- When adding new functionality, first check whether QuTiP already supports the required operation, solver, object, transformation, visualization, or analysis natively.
- If QuTiP provides the needed functionality in a form that is compatible with the project, prefer using that native QuTiP capability rather than re-implementing the same behavior from scratch.
- In such cases, expose the functionality through a project-level wrapper function or wrapper abstraction so that repository conventions, validation, typing, defaults, units, documentation, and future extensibility remain under project control.
- Do not call QuTiP primitives in an ad hoc or scattered way across the codebase when a stable wrapped interface is more appropriate.
- New wrappers around QuTiP functionality should follow the repository’s existing API style and should make any project-specific conventions explicit, including truncation, basis ordering, rotating-frame assumptions, units, parameter naming, and return types.
- If QuTiP provides only part of the needed functionality, reuse the native QuTiP portion where appropriate and implement only the missing project-specific layer on top.
- If QuTiP does not support the needed functionality natively, document that gap clearly in the implementation or task notes before introducing a custom implementation.
- Any wrapper introduced around QuTiP functionality should also be reflected in `API_REFERENCE.md` if it is part of the intended public or developer-facing API.
- If the wrapped QuTiP functionality carries physics assumptions or convention-sensitive meaning, ensure those assumptions are also documented in `physics_and_conventions/physics_conventions_report.tex` when relevant.
- If changes are made to the `physics_and_conventions/physics_conventions_report.tex` file , you must also run  `physics_and_conventions/build_physics_conventions_report.bat` to ensure it compiles
## cQED Simulation Usage Policy

- For tasks that require simulation, numerical modeling, or reproducing a particular cQED experiment, prefer using `cqed_sim` whenever it is applicable.
- When asked to simulate a particular experiment, first check whether the task can and should be implemented through the existing `cqed_sim` framework rather than building a separate ad hoc simulation path.
- When working with `cqed_sim`, refer to `API_REFERENCE.md` to align usage with the intended public API, existing abstractions, and project conventions.
- Do not bypass `cqed_sim` with custom standalone simulation code unless there is a clear technical reason, and in such cases explain the limitation or gap in `cqed_sim` that motivates the deviation.

## Physics and Convention Maintenance

- Any new feature added to `cqed_sim` that fundamentally introduces new physics, modifies an existing physical model, changes assumptions, or alters conventions must also update the documentation in the `physics_and_conventions` folder.
- In particular, such changes must be reflected in `physics_and_conventions/physics_conventions_report.tex` so that the simulator’s documented physics, conventions, assumptions, and implementation remain synchronized.
- Do not treat physics-documentation updates as optional when a code change affects physical meaning, Hamiltonians, frames, sign conventions, units, approximations, measurement definitions, gate conventions, or parameter mappings.

## Tests and Examples

- Any new tests or test cases added to validate code correctness, physics consistency, numerical behavior, or API behavior must be placed under the `tests` folder.
- Any user-facing example scripts, demonstration workflows, or reference usage patterns showing how `cqed_sim` is intended to be used in practice should be placed under the `examples` folder.
- Do not place new validation tests inside ad hoc scripts or notebooks when they belong in the formal `tests` suite, and do not place typical usage demos outside `examples` unless there is a strong project-specific reason.

- If a task or prompt asks for verification against a paper, textbook, or other literature source—especially meaning reproduction of published or reference results—that work should be treated under `test_against_papers`.
- Requests to reproduce figures, equations, numerical benchmarks, analytical limits, spectra, dynamics, or other results from papers or textbooks should not be placed as ad hoc scripts or informal notes when they are intended as validation.
- Use `test_against_papers` for literature-based validation tasks whose purpose is to confirm that the implementation matches external reference results, standard derivations, or published cQED behavior.
- Such tests should make clear what source is being checked, what result is being reproduced, what assumptions or approximations are being used, and what level of agreement is expected.
- If reproducing a paper or textbook result also requires new reusable regression coverage, add the formal automated portion under `tests` as appropriate, while keeping the literature-reproduction workflow organized under `test_against_papers`.

## Refactor and Inconsistency Reporting Policy

- Anytime you are asked to refactor, all project guidelines above must still be followed strictly.
- A refactor task must include an inspection step for inconsistencies in code behavior, conventions, APIs, documentation, assumptions, and physics definitions, rather than assuming the existing implementation is internally consistent.
- If any inconsistency is discovered, write an inconsistency report in the `inconsistency` folder before or alongside the refactor changes.
- The report filename must include the date and time of creation.
- Each report must briefly specify:
  - what the inconsistency is,
  - where it appears,
  - what components it affects,
  - why it is inconsistent with the project’s intended conventions or behavior,
  - and what consequences it may cause.
- Reports should clearly separate confirmed issues from suspected issues or unresolved questions.
- Important inconsistencies must not be silently corrected without documentation.
- When multiple inconsistencies are discovered in the same task, either create one consolidated timestamped report for that refactor session or multiple clearly named timestamped reports if that is more readable.

- Before starting a refactor, bug fix, convention update, API cleanup, or related maintenance task, inspect the existing files in the `inconsistency` folder for previously reported issues relevant to the affected code paths.
- Do not treat prior inconsistency reports as archival only; use them as active task context when evaluating what should be fixed, preserved, or re-verified.
- If a reported inconsistency is addressed by the current task, update the corresponding inconsistency report to mark it as fixed rather than leaving the report in an unresolved state.
- A fix update should clearly indicate:
  - what issue was fixed,
  - when it was fixed,
  - what commit, file, module, or change addressed it,
  - and whether any related concerns remain open.
- If a task partially fixes a reported inconsistency, mark the resolved portion clearly and leave the remaining unresolved portion explicitly identified.
- Do not silently resolve previously reported inconsistencies without updating their status in the `inconsistency` folder.
- If a prior inconsistency report is determined to be outdated, invalid, or no longer applicable, annotate it accordingly rather than deleting its history without explanation.

## Refactor Documentation Synchronization

- If something is being refactored, update `API_REFERENCE.md` if the refactor changes public APIs, function signatures, class behavior, module organization, expected usage patterns, configuration structure, or any other user-facing developer interface.
- If something is being refactored and the refactor affects physical meaning, conventions, modeling assumptions, Hamiltonians, rotating frames, sign conventions, units, approximations, observables, tomography definitions, calibration meaning, or experiment-to-simulation interpretation, update `physics_and_conventions/physics_conventions_report.tex` as needed.
- Do not treat `API_REFERENCE.md` or `physics_and_conventions/physics_conventions_report.tex` as optional follow-up work when a refactor materially changes the documented behavior or meaning of the code.
- If a refactor does not require changes to these documents, verify that explicitly before leaving them unchanged.
- Refactors should preserve consistency between implementation, API documentation, and physics/conventions documentation at the end of the task.

## API Reference and Website Documentation Synchronization

- `API_REFERENCE.md` must remain consistent with the website documentation located under the `documentations` folder.
- When public APIs, module organization, function signatures, class behavior, configuration structures, usage patterns, or developer-facing workflows change, update both `API_REFERENCE.md` and the relevant website documentation pages under `documentations`.
- Do not update one documentation surface while leaving the other stale when they are intended to describe the same public behavior.
- If `API_REFERENCE.md` is treated as the canonical reference for the public API, ensure that the website documentation reflects that reference accurately in wording, signatures, examples, and module/class/function coverage.
- If the website documentation contains higher-level guides, tutorials, walkthroughs, or reorganized API material, those pages must still remain semantically consistent with `API_REFERENCE.md`.
- Any inconsistency discovered between `API_REFERENCE.md` and the website documentation should be treated as a documentation inconsistency and should be corrected as part of the task when relevant.
- If a refactor or feature addition changes the intended public or developer-facing interface, verify explicitly that `API_REFERENCE.md` and the `documentations` folder are synchronized before considering the task complete.

## Expected Refactor Workflow

- When performing a refactor, follow this general sequence unless the task explicitly requires a different order:
  1. Read `README.md` and inspect the relevant code paths.
  2. Determine whether the task should use existing `cqed_sim` infrastructure.
  3. Identify any inconsistencies in implementation, API usage, conventions, assumptions, or physics meaning.
  4. Write an inconsistency report in the `inconsistency` folder if issues are found.
  5. Apply the refactor with minimal necessary changes that match project conventions.
  6. Update `API_REFERENCE.md` if the refactor changes public-facing code behavior or usage.
  7. Update `physics_and_conventions/physics_conventions_report.tex` if the refactor changes physical meaning, conventions, or modeling assumptions. Generally this should be treated as a source of truth and should not need explicit change unless otherwise told to do so
  8. Add or update tests under `tests` as needed.
  9. Add or update examples under `examples` if the intended user workflow or recommended usage has changed.

## General Quality Bar

- Prefer correctness, consistency, and maintainability over introducing unnecessary abstractions.
- Keep simulation, experiment-facing usage, documentation, and conventions synchronized.
- Any change that affects behavior should be considered for its impact on implementation, tests, examples, API documentation, and physics/conventions documentation rather than modifying code in isolation.
