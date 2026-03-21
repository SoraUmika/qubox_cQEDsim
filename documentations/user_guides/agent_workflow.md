# Semi-Autonomous Agent Workflow

The repository includes a repo-side two-agent workflow in `agent_workflow/` for semi-autonomous implementation, review, documentation, and resume handling.

## Roles

- Human: provides the task plan and acceptance criteria.
- Codex: executes code and test work.
- Opus 4.6: reviews, audits completion, writes documentation artifacts, and writes the final summary.

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
- generic command backend abstraction for later provider wiring
- deterministic validation backend for local testing

## Important Note About Placement

This workflow is repo-side rather than part of the `cqed_sim` package surface. The repository already keeps high-level orchestration outside the reusable simulation library, and this workflow follows that boundary.

## Validation Demo

Use `agent_workflow/tasks/validation_demo_task.yaml` to validate the orchestration loop without modifying the main repository. It copies a small seed project into the run directory and uses the scripted backend profile to exercise incomplete-run resume behavior.

## Configuration

Backend profiles live in `agent_workflow/config.json`.

The shipped `validation_demo` profile is fully runnable. Real Codex and Opus command integration should be configured explicitly through the generic command backend instead of relying on hardcoded assumptions about a local CLI contract.

## Related Docs

- `README_AGENT_WORKFLOW.md`
- `MULTI_AGENT_WORKFLOW_SURVEY.md`
- `MULTI_AGENT_WORKFLOW_DESIGN.md`
