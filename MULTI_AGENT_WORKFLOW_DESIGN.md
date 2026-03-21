# Multi-Agent Workflow Design

## Objective

Implement a repo-contained, semi-autonomous two-agent workflow that can be launched from one command, uses a human-authored plan as the default source of truth, persists run state, resumes safely, and separates execution from review and documentation.

## High-Level Design

### Role model

- Human: provides the plan, constraints, deliverables, and acceptance criteria.
- Codex: main execution agent for implementation, edits, tests, and repair passes.
- Opus 4.6: reviewer, completion auditor, documentation writer, and final summarizer.

### Repository placement

The orchestration lives repo-side in `agent_workflow/` rather than inside `cqed_sim/`.

Reason:

- the repository already treats high-level orchestration as outside the reusable simulation package
- this avoids reintroducing workflow-specific APIs into the public simulation namespace
- the tooling still remains fully contained in the workspace and accessible from one command

## Implemented File Layout

```text
agent_workflow/
  __init__.py
  artifacts.py
  backends.py
  cli.py
  orchestrator.py
  prompts.py
  run.py
  simple_yaml.py
  state.py
  task_spec.py
  README.md
  config.json
  demo/
    validation_script.json
  prompts/
    codex_executor.md
    opus_reviewer.md
    opus_docs.md
    opus_summary.md
    fallback_planner.md
  tasks/
    example_task.yaml
    validation_demo_task.yaml
  validation_fixture/
    seed_project/
      README.md
      AGENTS.md
      demo_math.py
      demo_docs.md
      check_demo_math.py

agent_runs/
  README.md

tools/
  run_agent_workflow.py

run_agent_workflow.ps1
run_agent_workflow.bat
.vscode/tasks.json
```

## Task Specification Model

### Format

The workflow accepts YAML or JSON task files. YAML is preferred.

### YAML support strategy

To avoid introducing a required YAML dependency, the implementation uses:

- PyYAML if it is already installed
- otherwise a standard-library fallback parser that supports the task-spec subset used here:
  - top-level key/value pairs
  - top-level scalar lists
  - top-level block strings with `|`

Advanced YAML features intentionally require PyYAML instead of an ad hoc partial reimplementation.

### Supported task fields

The core task spec supports the requested fields:

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

Additional implementation-oriented fields were added to support safe validation and backend selection:

- `backend_profile`
- `bootstrap_workspace_from`
- `working_directory`

### Human plan priority

`human_plan` is first-class. It is carried directly into the execution and review prompts and is treated as the default plan source. The fallback planner prompt exists but is not invoked unless a task omits the human plan and the operator chooses to use that path.

## Run Phases

The implemented run phases are:

1. initialize
2. execute
3. test
4. review
5. docs
6. summary
7. complete

### Phase behavior

- `initialize`: create the run directory, copy the optional sandbox workspace, write the resolved task, and initialize persistent state.
- `execute`: send the execution prompt to the Codex backend and record output plus workspace snapshot.
- `test`: run the configured commands, save a test log, and skip reruns when the workspace snapshot has not changed since the last test run.
- `review`: send artifacts to the Opus reviewer and decide between `accepted`, `needs_repair`, or `blocked`.
- `docs`: create a documentation artifact after acceptance.
- `summary`: create the final human-readable summary and enforce completion checks.

## State Persistence

Persistent state is stored in `RUN_STATE.json` and includes:

- task name and task file path
- run directory
- current phase
- run status
- iteration count
- max iteration limit
- human-plan presence flag
- review requirement flag
- acceptance-criteria satisfaction flag
- blocking reason or last error
- active working directory
- workspace snapshot tracking
- completion flags
- repair instructions
- per-iteration execution, test, and review records
- timestamps

This state is updated after every major phase.

## Artifact Model

Each run writes human-readable artifacts into a timestamped `agent_runs/<run_id>/` directory, including:

- `task_resolved.yaml`
- `RUN_STATE.json`
- `IMPLEMENTATION_LOG.md`
- `EXECUTION_PROMPT_iterNN.txt`
- `EXECUTION_CONTEXT_iterNN.json`
- `EXECUTION_OUTPUT_iterNN.txt`
- `TEST_LOG_iterNN.txt`
- `CHANGED_FILES_iterNN.txt`
- `REVIEW_PROMPT_iterNN.txt`
- `REVIEW_CONTEXT_iterNN.json`
- `REVIEW_OUTPUT_iterNN.txt`
- `DOC_PROMPT.txt`
- `DOC_CONTEXT.json`
- `DOC_OUTPUT.md`
- `SUMMARY_PROMPT.txt`
- `SUMMARY_CONTEXT.json`
- `FINAL_SUMMARY.md`

## Resume Strategy

### Normal behavior

- if `auto_resume` is enabled and there is an incomplete run for the task, a normal invocation resumes it automatically
- `--resume` explicitly resumes the latest incomplete run for the given task
- `--resume-last` targets the latest incomplete run regardless of task
- `--force-restart` always starts a fresh run directory

### Resume semantics

The workflow resumes from `current_phase` recorded in `RUN_STATE.json`:

- if execution finished but review did not, the next run resumes at review
- if review requested repair and iterations remain, the next run resumes at the next execution pass
- if the maximum iteration cap was hit, the run remains incomplete until relaunched with a higher limit or a different task decision

## Backend Abstraction

### Interface

The workflow uses a provider abstraction around `AgentRequest` and `AgentResponse` with a common `run()` method.

### Implemented backend types

- `UnavailableBackend`
  - explicit, honest failure mode when no provider is configured
- `CommandTemplateBackend`
  - generic subprocess wrapper with placeholder substitution for prompt file, context file, working directory, run directory, and other fields
- `ScriptedBackend`
  - deterministic backend for validation and tests

### Role wrappers

The orchestration exposes:

- `CodexBackend`
- `OpusBackend`

These are role-specific wrappers around the underlying provider backend instance.

### Why this design was chosen

The environment check showed that a local `copilot.ps1` command exists but is not directly runnable under the current default execution policy. Because the exact CLI contract is not guaranteed here, the workflow does not pretend to ship a fully working hardcoded Copilot integration.

Instead, it ships:

- a real command backend abstraction for later provider wiring
- a deterministic demo profile that proves the orchestration logic now
- explicit configuration in `agent_workflow/config.json`

## Bounded Autonomy

The main autonomous loop is the repair loop itself:

1. Codex executes.
2. Tests run.
3. Opus reviews.
4. If review says `needs_repair`, the workflow either starts another execution pass or stops incomplete when the iteration cap is reached.

This is bounded by `max_iterations` and therefore cannot loop forever.

## Test Execution Strategy

Task files carry an explicit list of commands under `tests_to_run`.

Behavior:

- no commands: tests are waived with a recorded explanation
- unchanged workspace snapshot since the last test run: tests are skipped
- otherwise: commands are executed and written to `TEST_LOG_iterNN.txt`

## Prompting Strategy

Reusable prompt templates exist for:

- Codex execution
- Opus review
- Opus documentation
- Opus final summary
- fallback planning

All templates explicitly require:

- reading `README.md` first
- reading `AGENTS.md` if present
- inspecting relevant files before editing
- minimal and local changes
- test updates when behavior changes
- documentation updates when usage or behavior changes
- explicit assumptions
- clear blocker reporting

## Validation Design

A deterministic validation path was added because real provider integration is environment-specific.

### Validation mechanism

- a seed project is copied into `agent_runs/<run_id>/workspace`
- the `validation_demo` backend profile uses `ScriptedBackend`
- the scripted Codex responses apply deterministic sandbox edits
- the scripted Opus responses force one repair loop before acceptance

### What this validates

- human-authored plan ingestion
- Codex execution phase
- test execution phase
- Opus review phase
- Opus documentation and summary phases
- run artifact generation
- incomplete status when the iteration cap is hit
- successful resume to completion

## CLI and VS Code Integration

### Python entrypoints

- `python -m agent_workflow.run --task ...`
- `python tools/run_agent_workflow.py --task ...`

### Shell wrappers

- `run_agent_workflow.ps1`
- `run_agent_workflow.bat`

### VS Code tasks

- `Agent Workflow: Run Task`
- `Agent Workflow: Resume Last Run`

## Honest Integration Boundary

### Fully implemented now

- task loading
- persistent run state
- artifact writing
- execute/test/review/docs/summary orchestration
- bounded repair loop
- automatic resume behavior
- deterministic validation backend
- CLI wrappers and VS Code tasks

### Intentionally scaffolded, not faked

- real Codex / Opus provider invocation details
- exact Copilot CLI command-line contract for this environment

Those are left to explicit `CommandTemplateBackend` configuration rather than guessed or hardcoded assumptions.
