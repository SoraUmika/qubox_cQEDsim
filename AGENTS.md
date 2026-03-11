# Project Guidelines

## Startup Policy

- Before taking any action, first read `README.md` to gather project context.
- If the task is notebook- or simulation-specific, also read the most relevant local file before acting, such as `pyproject.toml`, `experiment_mapping.md`, or the active notebook.
- Do not create, activate, or rely on a virtual environment unless the user explicitly asks for one.
- For Python execution, use the existing system Python at `E:\Program Files\Python311\python.exe`. Or `E:\Programs\python.exe`
- Do not run dependency installation or environment-management commands unless the user explicitly requests them.

## Python Environment

- Treat this repository as using the existing machine Python environment rather than a repo-local venv.
- Avoid commands such as `python -m venv`, `virtualenv`, `conda create`, `poetry env use`, or other environment bootstrap steps unless the user explicitly asks for them.

## Working Style

- Inspect the current codebase first and prefer minimal changes that match existing conventions.
- Before changing project setup, execution workflow, or dependency state, confirm that the change is necessary for the task.
- Prefer using the repository's existing scripts, tests, and notebooks over introducing new setup layers.

## Physics and Convention Maintenance

- Any new feature added to `cqed_sim` that fundamentally introduces new physics, modifies an existing physical model, changes assumptions, or alters conventions must also update the documentation in the `physics_and_conventions` folder.
- In particular, such changes must be reflected in `physics_and_conventions/physics_conventions_report.tex` so that the simulator’s documented physics, conventions, assumptions, and implementation remain synchronized.
- Do not treat physics-documentation updates as optional when a code change affects physical meaning, Hamiltonians, frames, sign conventions, units, approximations, measurement definitions, gate conventions, or parameter mappings.

## Tests and Examples

- Any new tests or test cases added to validate code correctness, physics consistency, numerical behavior, or API behavior must be placed under the `tests` folder.
- Any user-facing example scripts, demonstration workflows, or reference usage patterns showing how `cqed_sim` is intended to be used in practice should be placed under the `examples` folder.
- Do not place new validation tests inside ad hoc scripts or notebooks when they belong in the formal `tests` suite, and do not place typical usage demos outside `examples` unless there is a strong project-specific reason.