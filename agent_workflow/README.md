# Agent Workflow Package

This folder contains the repo-side autonomous Copilot-driven workflow runtime.

## What It Contains

- task-spec loading and minimal YAML fallback support
- persistent run state and resumable phase tracking
- prompt rendering for execute, review, docs, and summary roles
- backend abstraction for command and scripted providers
- Copilot CLI programmatic invocation wrapper for built-in Copilot agents
- artifact writing under `agent_runs/`
- deterministic validation fixtures and demo backends

## What It Does Not Contain

- reusable cQED simulation primitives
- reusable package-level public APIs for `cqed_sim`

## Entry Points

- `python -m agent_workflow.run --task ...`
- `python tools/run_agent_workflow.py --task ...`
- `run_agent_workflow.ps1`
- `run_agent_workflow.bat`

## Important Files

- `config.json`
- `copilot_programmatic.py`
- `task_spec.py`
- `state.py`
- `backends.py`
- `orchestrator.py`
- `tasks/validation_demo_task.yaml`
- `demo/validation_script.json`
