# README: Agent Workflow

## What This Is

This repository now includes a repo-side semi-autonomous two-agent workflow under `agent_workflow/`.

The workflow is intended for tasks where:

- a human usually provides the plan
- Codex is the main implementation and test-running agent
- Opus 4.6 acts as reviewer, documentation writer, completion auditor, and final summarizer
- the run should be resumable and auditable instead of depending on continuous manual supervision

## Why It Lives Repo-Side

The repository already treats high-level orchestration as repo-side tooling rather than part of the reusable `cqed_sim` simulation package.

That means this workflow is implemented in `agent_workflow/` plus wrappers in `tools/` and the repository root, instead of expanding the `cqed_sim` import surface again.

## Main Entry Points

### Python module

```bash
python -m agent_workflow.run --task agent_workflow/tasks/example_task.yaml
```

### Python wrapper script

```bash
python tools/run_agent_workflow.py --task agent_workflow/tasks/example_task.yaml
```

### PowerShell wrapper

```powershell
./run_agent_workflow.ps1 -Task agent_workflow/tasks/example_task.yaml
```

### Batch wrapper

```bat
run_agent_workflow.bat --task agent_workflow/tasks/example_task.yaml
```

## CLI Options

Supported options include:

- `--task`
- `--resume`
- `--resume-last`
- `--force-restart`
- `--max-iterations`
- `--dry-run`
- `--verbose`

## Task Spec Fields

The workflow task files support:

- `title`
- `goal`
- `human_plan`
- `context_files`
- `constraints`
- `deliverables`
- `tests_to_run`
- `docs_to_update`
- `acceptance_criteria`
- `max_iterations`
- `strict_context`
- `allow_repo_edits`
- `review_required`
- `auto_resume`
- `stop_on_blocking_error`
- `backend_profile`
- `bootstrap_workspace_from`
- `working_directory`

The important design point is that `human_plan` is first-class. If a human already knows the plan, the workflow does not force an extra planner by default.

## Backend Profiles

Backend profiles are defined in `agent_workflow/config.json`.

Shipped profiles:

- `unconfigured`
  - explicit placeholder profile for real provider configuration
- `validation_demo`
  - deterministic scripted backend used for validation and tests

### Real provider integration

The workflow ships a generic `CommandTemplateBackend` for shell-based integration, but it does not hardcode a guessed Copilot CLI contract.

Reason:

- a local `copilot.ps1` exists on this machine
- direct invocation is currently blocked by PowerShell execution policy
- the exact command-line contract is environment-specific and should be configured explicitly

## Run Artifacts

Each run writes a timestamped folder under `agent_runs/` with:

- resolved task input
- persistent run state
- prompt and context files
- backend outputs
- test logs
- changed-file snapshots
- documentation artifact
- final summary

The run state file is `RUN_STATE.json` and is the source of truth for resume behavior.

## Repair Loop

The orchestration loop is:

1. execute with Codex
2. run tests
3. review with Opus
4. if review requests repair, continue with the next execution pass
5. stop only on acceptance, explicit blocking status, or the iteration cap

This is the bounded autonomy mechanism. It does not loop forever.

## Validation Demo

A deterministic validation task is included at `agent_workflow/tasks/validation_demo_task.yaml`.

It uses a sandbox workspace copied into the run directory so the validation does not modify the main repository.

### Demonstrate incomplete then resume

First run with a one-iteration cap:

```bash
python -m agent_workflow.run --task agent_workflow/tasks/validation_demo_task.yaml --force-restart --max-iterations 1
```

Expected result:

- the implementation fix is applied in the sandbox
- tests run
- review requests a documentation follow-up
- the workflow exits incomplete with status code `2`

Then resume with a higher cap:

```bash
python -m agent_workflow.run --task agent_workflow/tasks/validation_demo_task.yaml --resume --max-iterations 3
```

Expected result:

- the next execution pass updates the sandbox documentation
- review accepts the run
- documentation and final summary artifacts are written
- the workflow exits with status code `0`

## Exit Codes

- `0`: complete
- `1`: blocked
- `2`: incomplete

## Files Worth Reading

- `MULTI_AGENT_WORKFLOW_SURVEY.md`
- `MULTI_AGENT_WORKFLOW_DESIGN.md`
- `agent_workflow/README.md`
- `agent_workflow/tasks/example_task.yaml`
- `agent_workflow/tasks/validation_demo_task.yaml`
