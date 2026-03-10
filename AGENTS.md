# Project Guidelines

## Startup Policy

- Before taking any action, first read `README.md` to gather project context.
- If the task is notebook- or simulation-specific, also read the most relevant local file before acting, such as `pyproject.toml`, `experiment_mapping.md`, or the active notebook.
- Do not create, activate, or rely on a virtual environment unless the user explicitly asks for one.
- For Python execution, use the existing system Python at `E:\Program Files\Python311\python.exe`.
- Do not run dependency installation or environment-management commands unless the user explicitly requests them.

## Python Environment

- Treat this repository as using the existing machine Python environment rather than a repo-local venv.
- Avoid commands such as `python -m venv`, `virtualenv`, `conda create`, `poetry env use`, or other environment bootstrap steps unless the user explicitly asks for them.

## Working Style

- Inspect the current codebase first and prefer minimal changes that match existing conventions.
- Before changing project setup, execution workflow, or dependency state, confirm that the change is necessary for the task.
- Prefer using the repository's existing scripts, tests, and notebooks over introducing new setup layers.