# AGENTS.md

# Project Guidelines

## Startup Policy

- Before taking any action, first read `README.md` to gather project context, repository structure, and expected workflows.
- Do not create, activate, or rely on a virtual environment unless the user explicitly requests it.
- For Python execution, use the existing system Python at:
  - `E:\Program Files\Python311\python.exe`, or
  - `E:\Programs\python.exe`
  (currently Python 3.12.10).
- Do not run dependency-installation or environment-management commands unless the user explicitly requests them.
- Only install missing packages when they are genuinely required for analysis, simple calculations, optimization, testing, or other common development tasks needed for the requested work.
- Any package installed should be broadly used, general-purpose, and low-risk.
- Do not install niche, experimental, or project-specific dependencies unless the user explicitly requests them.
- Do not use VS Code Jupyter extensions or other interactive notebook environments unless the user explicitly requests them.
- Do not treat notebooks as the default development environment unless the task specifically calls for notebook-based work.
- Prefer the built-in Python Jupyter notebook for Python 3.12.10 when notebook usage is explicitly requested, especially the one referenced by the `PYTHON_JUPYTER_NOTEBOOK` environment variable if it is set.
- Do not install a notebook environment unless the user explicitly requests it.

## Python Environment

- Treat this repository as using the existing machine Python environment rather than a repository-local virtual environment.
- Avoid commands such as:
  - `python -m venv`
  - `virtualenv`
  - `conda create`
  - `poetry env use`
  - or any other environment bootstrap/setup command
  unless the user explicitly asks for them.
- Prefer working with the already available interpreter and installed packages whenever practical.

## Working Style

- Inspect the current codebase first and prefer minimal changes that match existing conventions.
- Before changing project setup, execution workflow, dependency state, or file organization, confirm that the change is truly necessary for the task.
- Prefer using the repository’s existing scripts, tests, examples, and utilities over introducing new setup layers or parallel workflows.
- Do not introduce parallel implementations, duplicate abstractions, or ad hoc workarounds when an existing project path already supports the task.
- Keep new code, documentation, and refactors aligned with the repository’s current architecture unless the task explicitly requires an architectural change.
- Prefer explicit, reviewable changes over broad speculative rewrites.

## QuTiP Native-First Policy

- When adding new functionality, first check whether QuTiP already supports the required operation, solver, object, transformation, visualization, or analysis natively.
- If QuTiP provides the needed functionality in a form compatible with the project, prefer using that native QuTiP capability rather than re-implementing the same behavior from scratch.
- When native QuTiP functionality is used, expose it through a project-level wrapper or structured abstraction so that repository conventions, validation, typing, defaults, units, documentation, and future extensibility remain under project control.
- Do not call QuTiP primitives in an ad hoc or scattered manner across the codebase when a stable wrapped interface is more appropriate.
- New wrappers around QuTiP functionality should follow the repository’s existing API style and should make project-specific conventions explicit, including:
  - truncation,
  - basis ordering,
  - rotating-frame assumptions,
  - units,
  - parameter naming,
  - return types.
- If QuTiP provides only part of the needed functionality, reuse the native QuTiP portion where appropriate and implement only the missing project-specific layer on top.
- If QuTiP does not support the needed functionality natively, document that gap clearly in implementation notes, task notes, or code comments before introducing a custom implementation.
- Any wrapper introduced around QuTiP functionality should also be reflected in `API_REFERENCE.md` if it is part of the intended public or developer-facing API.
- If the wrapped QuTiP functionality carries physics assumptions or convention-sensitive meaning, ensure those assumptions are also documented in `physics_and_conventions/physics_conventions_report.tex` when relevant.
- If changes are made to `physics_and_conventions/physics_conventions_report.tex`, also run:
  - `physics_and_conventions/build_physics_conventions_report.bat`
  to ensure the document still compiles.

## cQED Simulation Usage Policy

- For tasks that require simulation, numerical modeling, or reproduction of a particular cQED experiment, prefer using `cqed_sim` whenever it is applicable.
- When asked to simulate a particular experiment, first check whether the task can and should be implemented through the existing `cqed_sim` framework rather than by building a separate ad hoc simulation path.
- When working with `cqed_sim`, refer to `API_REFERENCE.md` so that usage remains aligned with the intended public API, existing abstractions, and project conventions.
- Do not bypass `cqed_sim` with standalone simulation code unless there is a clear technical reason to do so.
- If `cqed_sim` is not sufficient for the requested task, explain the limitation or gap that motivates the deviation.

## Module-Level README Policy

- Any relatively separate functionality or feature area built into `cqed_sim` must include its own local `README.md`.
- This applies especially to self-contained or conceptually distinct modules, subsystems, or feature families, including but not limited to:
  - reinforcement learning,
  - GRAPE,
  - holographic simulation,
  - synthesis/optimization frameworks,
  - paper-inspired simulation modules,
  - or other major reusable capabilities.
- Each such `README.md` should explain, at minimum:
  - what the module or feature is,
  - why it is relevant,
  - what problem it is meant to solve,
  - what physics, control, simulation, or algorithmic role it serves,
  - how it fits into the broader `cqed_sim` ecosystem,
  - and how a user or developer is expected to use it.
- Each local module `README.md` should also include, where appropriate:
  - entry points,
  - important classes/functions,
  - expected inputs and outputs,
  - configuration or assumptions,
  - usage examples,
  - links or references to related papers or internal documentation,
  - and known limitations or non-goals.
- Do not treat module-level documentation as optional for major features.
- If a feature is sufficiently distinct that a user or developer would benefit from a dedicated conceptual and usage guide, it should have its own `README.md`.
- When adding a new major module or substantially expanding an existing one, create or update that module’s local `README.md` as part of the same task.

## Literature Search and Citation Policy

### When to Search

- Before implementing any new physics feature, algorithm, gate, model, or simulation method, **search online** for the canonical reference paper(s) that define or first describe that feature.
- Before reproducing a cQED experiment, spectroscopy measurement, control protocol, or numerical result, **search for the original publication** and retrieve the key equations, parameter values, and claimed results.
- When adding a tutorial, example, or test that demonstrates a particular physical effect or algorithm, **identify the authoritative source** for that effect before writing any code.
- When comparing simulator output to expected behavior, **search for published benchmarks or textbook derivations** to give the comparison a concrete external anchor.

### What to Search For

- The original paper introducing the method, gate, or protocol (e.g., Krastanov et al. for SNAP gates, GRAPE references for optimal control).
- Review articles or textbook chapters that present the standard derivation or convention for the relevant physics.
- Papers that provide the specific numerical values or qualitative behavior being reproduced.
- arXiv preprints, Physical Review, Nature, Science, and cQED-focused journals (PRL, PRA, npj Quantum Information, Applied Physics Letters, etc.) are all valid sources.

### How to Record the Reference

- Every tutorial page under `documentations/tutorials/` that reproduces, demonstrates, or is inspired by a published result **must include a References section** citing the relevant papers in the following format:

  ```
  [1] Author(s), "Title," Journal/arXiv, Year. DOI or URL.
  ```

- Every entry under `test_against_papers/` must state in a header comment which paper is being checked, which figure or equation is being reproduced, and a full citation.
- Code implementing a specific algorithm, decomposition, or formula from a paper must include a comment citing the source equation number and paper. Example:

  ```python
  # Implements Eq. (3) from Krastanov et al., Optica 2015
  # DOI: 10.1364/OPTICA.2.000880
  ```

- When a paper summary is added to the `paper_summary/` folder, that file must itself begin with the full citation so it is unambiguous which paper it covers.

### Integration with Existing Policies

- After finding a reference, check whether the `paper_summary/` folder already has a summary for that paper. If not, and if the paper is sufficiently important to the task, add a summary file.
- Physics conventions drawn from a found reference must be reconciled with `physics_and_conventions/physics_conventions_report.tex`. If they differ, document the discrepancy before choosing which convention to follow.
- Do not proceed with implementing a physics feature or reproducing a result without at least one concrete external reference. If no reference can be found after a reasonable search, document that gap explicitly in implementation notes or the relevant inconsistency report.

## Physics and Convention Maintenance

- Any new feature added to `cqed_sim` that introduces new physics, modifies an existing physical model, changes assumptions, or alters established conventions must also update the documentation in the `physics_and_conventions` folder.
- In particular, all such changes must be reflected in `physics_and_conventions/physics_conventions_report.tex` so that the simulator’s documented physics, conventions, assumptions, and implementation remain synchronized.
- Physics documentation updates are mandatory whenever a code change affects:
  - physical meaning,
  - Hamiltonians,
  - reference frames,
  - sign conventions,
  - units,
  - approximations,
  - measurement definitions,
  - gate conventions,
  - parameter interpretations,
  - parameter mappings.
- Do not treat physics-documentation updates as optional or defer them to a later cleanup pass when the underlying code change has physical significance.
- When a task involves reproducing results from a paper, implementing a published method, or aligning simulator behavior with a literature reference, also consult the `paper_summary` folder, which contains reports and summaries for specific papers.
- If a request depends on methods, assumptions, derivations, conventions, or target results drawn from a summarized paper, use the corresponding material in `paper_summary` during both implementation and documentation updates.

## Tests, Examples, and Literature Validation

- Any new tests or test cases added to validate code correctness, physics consistency, numerical behavior, or API behavior must be placed under the `tests` folder.
- Any user-facing example scripts, demonstration workflows, or reference usage patterns showing how `cqed_sim` is intended to be used in practice should be placed under the `examples` folder.
- Do not place new validation tests inside ad hoc scripts or notebooks when they belong in the formal `tests` suite.
- Do not place typical usage demos outside `examples` unless there is a strong project-specific reason.

### Tutorial Evidence and Plot Policy

- Any new or materially updated tutorial under `documentations/tutorials` must point to the concrete code that produces the demonstrated result.
- That code should live in `examples` unless there is a strong existing project convention requiring a different location.
- Tutorials covering simulation behavior must include generated evidence from an actual run, not only descriptive prose. At minimum, include:
  - the script, function, or workflow path that was run,
  - the resulting plot, table, or other artifact that shows the simulation outcome,
  - and enough reported values or narrative interpretation that a reader can tell what the simulation demonstrated.
- When a tutorial introduces or updates a plot, regenerate the underlying simulation or example so the checked-in asset matches the current code and text.
- If a tutorial page changes and it is part of the public website, rebuild the generated `site` output in the same task.
- Holographic simulation tutorials are not exempt from this rule. They must include the exact example code path and concrete generated outputs such as observable plots, benchmark tables, sampled-versus-exact comparisons, or other simulation results that demonstrate the claimed behavior.

### Literature-Based Validation

- If a task or prompt asks for verification against a paper, textbook, or other literature source—especially reproduction of published or reference results—that work should be organized under `test_against_papers`.
- Requests to reproduce figures, equations, numerical benchmarks, analytical limits, spectra, dynamics, or other literature-based results should not be treated as ad hoc scripts or informal notes when they are intended as validation.
- Use `test_against_papers` for literature-based validation tasks whose purpose is to confirm that the implementation matches:
  - external reference results,
  - standard derivations,
  - published cQED behavior,
  - or paper-specific methods.
- Such tests should clearly state:
  - what source is being checked,
  - what result is being reproduced,
  - what assumptions or approximations are being used,
  - and what level of agreement is expected.
- If reproducing a paper or textbook result also requires reusable regression coverage, add the formal automated portion under `tests` as appropriate, while keeping the literature-reproduction workflow organized under `test_against_papers`.
- Before writing any literature-based validation, follow the **Literature Search and Citation Policy** above: search online for the paper, extract the key result, and include the full citation in the test file header. Use `/literature-review` to perform this step if the reference is not already known.

## Refactor and Inconsistency Reporting Policy

- Any time you are asked to refactor, all project guidelines above must still be followed strictly.
- A refactor task must include an inspection step for inconsistencies in:
  - code behavior,
  - conventions,
  - APIs,
  - documentation,
  - assumptions,
  - and physics definitions.
- Do not assume the existing implementation is internally consistent merely because it already exists.
- If any inconsistency is discovered, write an inconsistency report in the `inconsistency` folder before or alongside the refactor changes.
- The report filename must include the date and time of creation.
- Each report must briefly specify:
  - what the inconsistency is,
  - where it appears,
  - what components it affects,
  - why it is inconsistent with the project’s intended conventions or behavior,
  - and what consequences it may cause.
- Reports should clearly separate:
  - confirmed issues,
  - suspected issues,
  - unresolved questions.
- Important inconsistencies must not be silently corrected without documentation.
- When multiple inconsistencies are discovered in the same task, either:
  - create one consolidated timestamped report for that refactor session, or
  - create multiple clearly named timestamped reports if that is more readable.

### Using Existing Inconsistency Reports

- Before starting a refactor, bug fix, convention update, API cleanup, or related maintenance task, inspect the existing files in the `inconsistency` folder for previously reported issues relevant to the affected code paths.
- Do not treat prior inconsistency reports as archival only; use them as active task context when evaluating what should be fixed, preserved, or re-verified.
- If a reported inconsistency is addressed by the current task, update the corresponding inconsistency report to mark it as fixed rather than leaving the report unresolved.
- A fix update should clearly indicate:
  - what issue was fixed,
  - when it was fixed,
  - what commit, file, module, or change addressed it,
  - and whether any related concerns remain open.
- If a task only partially fixes a reported inconsistency, mark the resolved portion clearly and leave the remaining unresolved portion explicitly identified.
- Do not silently resolve previously reported inconsistencies without updating their status in the `inconsistency` folder.
- If a prior inconsistency report is determined to be outdated, invalid, or no longer applicable, annotate it accordingly rather than deleting its history without explanation.

## Refactor Documentation Synchronization

- If something is being refactored, update `API_REFERENCE.md` if the refactor changes:
  - public APIs,
  - function signatures,
  - class behavior,
  - module organization,
  - expected usage patterns,
  - configuration structure,
  - or any other user-facing or developer-facing interface.
- If something is being refactored and the refactor affects:
  - physical meaning,
  - conventions,
  - modeling assumptions,
  - Hamiltonians,
  - rotating frames,
  - sign conventions,
  - units,
  - approximations,
  - observables,
  - tomography definitions,
  - calibration meaning,
  - experiment-to-simulation interpretation,
  then update `physics_and_conventions/physics_conventions_report.tex` as needed.
- Do not treat `API_REFERENCE.md` or `physics_and_conventions/physics_conventions_report.tex` as optional follow-up work when a refactor materially changes documented behavior or meaning.
- If a refactor does not require changes to these documents, verify that explicitly before leaving them unchanged.
- Refactors should preserve consistency between implementation, API documentation, and physics/conventions documentation by the end of the task.

## API Reference and Website Documentation Synchronization

- `API_REFERENCE.md` must remain consistent with the website documentation located under the `documentations` folder.
- The generated `site` folder is the checked-in official website output and must remain consistent with both `API_REFERENCE.md` and the MkDocs source under `documentations`.
- When public APIs, module organization, function signatures, class behavior, configuration structures, usage patterns, or developer-facing workflows change, update both:
  - `API_REFERENCE.md`, and
  - the relevant website documentation pages under `documentations`.
- When those website-source pages change, also rebuild and update the generated `site` output so the published HTML reflects the same changes.
- Do not update one documentation surface while leaving the other stale when they are intended to describe the same public behavior.
- If `API_REFERENCE.md` is treated as the canonical public API reference, ensure that the website documentation reflects it accurately in wording, signatures, examples, and coverage.
- If the website documentation contains higher-level guides, tutorials, walkthroughs, or reorganized API material, those pages must still remain semantically consistent with `API_REFERENCE.md`.
- Any inconsistency discovered between `API_REFERENCE.md` and the website documentation should be treated as a documentation inconsistency and corrected as part of the task when relevant.
- If a refactor or feature addition changes the intended public or developer-facing interface, verify explicitly that `API_REFERENCE.md`, the `documentations` folder, and the generated `site` output are synchronized before considering the task complete.

## Expected Refactor Workflow

- When performing a refactor, follow this general sequence unless the task explicitly requires a different order:
  1. Read `README.md` and inspect the relevant code paths.
  2. Determine whether the task should use existing `cqed_sim` infrastructure.
  3. Inspect the `inconsistency` folder for prior reports relevant to the affected code.
  4. Identify any inconsistencies in implementation, API usage, conventions, assumptions, or physics meaning.
  5. Write or update an inconsistency report in the `inconsistency` folder if issues are found.
  6. Apply the refactor with minimal necessary changes that match project conventions.
  7. Update `API_REFERENCE.md` if the refactor changes public-facing code behavior or usage.
  8. Update relevant pages under `documentations` if developer-facing or public-facing documentation is affected.
  9. Rebuild and update the generated `site` output when public documentation, tutorials, or API-reference pages changed.
  10. Update `physics_and_conventions/physics_conventions_report.tex` if the refactor changes physical meaning, conventions, or modeling assumptions.
  11. Run `physics_and_conventions/build_physics_conventions_report.bat` if the physics conventions document was changed.
  12. Add or update tests under `tests` as needed.
  13. Add or update examples under `examples` if the intended user workflow or recommended usage has changed.
  14. Add or update module-level `README.md` files for any new or substantially changed major feature areas.

## Auto-Commit and Push Policy

- After every Claude session that produces file changes, the project is automatically committed and pushed to `origin` via a `Stop` hook configured in `.claude/settings.local.json`.
- The hook fires when Claude finishes responding. It runs in the background (`async: true`) so it does not block session teardown.
- The commit message is derived from `git diff --cached --stat` and takes the form:
  `auto: <N files changed, X insertions(+), Y deletions(-)>`
- The hook only commits and pushes if `git status --porcelain` reports at least one change. It is a no-op if there are no uncommitted changes.
- The hook runs `git add -A` before committing, so all tracked and untracked changes in the working tree are included.
- This is a local-only setting (`.claude/settings.local.json`, gitignored) and does not affect other contributors.
- If a push fails (no remote, auth issue, branch protection), the hook silently swallows the error and the session ends normally.
- Do not rely on this hook for commits that require meaningful messages, branch management, or PR creation — use explicit git commands for those.

## General Quality Bar

- Prefer correctness, consistency, and maintainability over unnecessary abstraction.
- Keep simulation behavior, experiment-facing usage, documentation, and conventions synchronized.
- Any change that affects behavior should be evaluated for its impact on:
  - implementation,
  - tests,
  - examples,
  - API documentation,
  - website documentation,
  - generated website output under `site`,
  - module-level READMEs,
  - and physics/conventions documentation,
  rather than modifying code in isolation.
- Major reusable features should be discoverable, documented, and understandable without requiring a reader to infer intent solely from source code.
- Documentation should explain not only how a feature works, but also why it exists and when it should be used.