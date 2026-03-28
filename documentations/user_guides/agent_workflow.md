# Autonomous Agent Workflow

The repository includes a repo-side two-agent workflow in `agent_workflow/` for autonomous implementation, review, documentation, and resume handling.

## Roles

- Human: provides the task goal and optional plan.
- Copilot `general-purpose`: executes code and test work.
- Copilot `code-review`: reviews the execution result.
- Copilot `general-purpose`: writes workflow documentation artifacts and the final summary.

## Launch Paths

- `python -m agent_workflow.run --task agent_workflow/tasks/example_task.yaml`
- `python tools/run_agent_workflow.py --task agent_workflow/tasks/example_task.yaml`
- `run_agent_workflow.ps1`
- `run_agent_workflow.bat`
- VS Code tasks:
  - `Agent Workflow: Run Task`
  - `Agent Workflow: Resume Last Run`

## Key Features

- machine-readable task specs
- persistent `RUN_STATE.json`
- timestamped artifacts under `agent_runs/`
- bounded execute/test/review repair loop
- resume support for incomplete runs
- generic command backend abstraction
- shipped Copilot CLI programmatic backend profile
- deterministic validation backend for local testing

## Important Note About Placement

This workflow is repo-side rather than part of the `cqed_sim` package surface. The repository already keeps high-level orchestration outside the reusable simulation library, and this workflow follows that boundary.

## Validation Demo

Use `agent_workflow/tasks/validation_demo_task.yaml` to validate the orchestration loop without modifying the main repository. It copies a small seed project into the run directory and uses the scripted backend profile to exercise incomplete-run resume behavior.

## Configuration

Backend profiles live in `agent_workflow/config.json`.

The default `copilot_cli_autonomous` profile invokes Copilot CLI in programmatic mode with `--agent`, `--prompt`, `--model gpt-5.4`, and `--allow-all-tools`, so the workflow can advance through its internal phases without waiting for the user to manually trigger each next step.

For fully headless runs, make sure:

- `copilot` is installed
- you are already signed in
- the repository has already been trusted by Copilot CLI

If trust has not been granted yet, launch `copilot` once from the repo root and choose the option to remember the folder, or add the repo with `/add-dir`.

## Related Docs

- `README_AGENT_WORKFLOW.md`
- `MULTI_AGENT_WORKFLOW_SURVEY.md`
- `MULTI_AGENT_WORKFLOW_DESIGN.md`
