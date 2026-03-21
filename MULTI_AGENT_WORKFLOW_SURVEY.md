# Multi-Agent Workflow Survey

## Scope

This survey records the repository context used to place the semi-autonomous two-agent workflow and identifies what can be reused versus what needed to be added.

## Existing Repository Structure

### Reusable Python package

The repository already has a large reusable library surface under `cqed_sim/` with subpackages for:

- models and conventions in `cqed_sim/core`
- pulse construction in `cqed_sim/pulses`
- sequence compilation in `cqed_sim/sequence`
- simulation and extractors in `cqed_sim/sim`
- analysis, calibration, measurement, tomography, plotting, optimal control, RL, and related domains

The package is installed from the repository root via `pyproject.toml`.

### Repo-side workflows and examples

The root README and architecture docs already establish a boundary:

- reusable simulation primitives stay inside `cqed_sim`
- guided notebooks live under `tutorials/`
- standalone workflows, studies, and orchestration helpers live repo-side under `examples/` or other top-level folders

This boundary is reinforced by:

- `API_REFERENCE.md` under the "Workflow boundary" note
- `documentations/architecture.md`
- `physics_and_conventions/experimental_protocol_alignment_note.md`
- `inconsistency/20260312_232943_experiment_namespace_boundary_refactor.md`

### Existing CLI patterns

The repository already uses standard-library `argparse` entrypoints in repo-side scripts such as:

- `benchmarks/run_optimal_control_benchmarks.py`
- `benchmarks/run_performance_benchmarks.py`
- many example scripts with `main()` plus `if __name__ == "__main__":`

There was no pre-existing repo-level multi-agent workflow runner.

### Existing documentation surfaces

The repository has several active documentation surfaces:

- root `README.md`
- `API_REFERENCE.md`
- the MkDocs site under `documentations/` with navigation in `mkdocs.yml`
- module-level `README.md` files across many reusable areas
- design and implementation reports under `docs/`

### Existing tests and test execution

The primary automated test harness is `pytest`, configured by `pytest.ini` with `-q` as the default addopt.

There is also an existing capture helper at `outputs/pytest_capture_runner.py` that shows the repo already stores human-readable execution artifacts when useful.

## Existing Constraints That Matter

### Architecture boundary

The most important pre-existing constraint is that high-level orchestration intentionally does not live inside the `cqed_sim` import package anymore. The repository has already been cleaned up once to remove workflow-style orchestration from the reusable library surface.

### Dirty worktree

The repository currently has unrelated local modifications in multiple files. That means the new workflow must be added additively and must not rely on cleaning or rewriting unrelated repo state.

### Python environment policy

The repo policy is to use the machine Python environment instead of a repository-local virtual environment. The implementation therefore avoids any environment bootstrapping and uses standard-library code for the workflow core.

## Provider / CLI Reality Check

A local `copilot.ps1` command exists on this machine, but direct invocation currently fails under the default PowerShell execution policy. Because of that, fully wired Copilot CLI integration could not be claimed as complete or reliable inside the repository itself.

What this means in practice:

- a real backend abstraction is still useful and was implemented
- deterministic scripted backends are needed for validation
- real provider commands should be configured explicitly by the user through a command backend profile instead of hardcoding an assumed CLI contract

## Reuse Opportunities

The following repo patterns were directly reusable:

- `argparse`-style CLI entrypoints
- standard-library JSON I/O for run state and backend profiles
- repo-side documentation pattern for workflows and examples
- the existing `agent_runs`-like artifact idea already seen in other generated outputs
- the repo's preference for minimal, additive changes and explicit documentation

## What Had To Be Added

The repo did not already contain:

- a machine-readable task spec format for agent runs
- a persistent run-state model for execute/test/review/docs/summary phases
- a backend abstraction for Codex and Opus roles
- timestamped workflow artifact folders with resumable state
- VS Code tasks or shell wrappers for agent orchestration
- a deterministic validation path for a multi-agent repair loop

## Placement Decision

### Best location for the new workflow

The best location is a new top-level `agent_workflow/` package plus repo-side wrappers in `tools/` and root shell scripts.

### Why this location was chosen

This preserves the existing architectural contract:

- the reusable `cqed_sim` package remains focused on cQED simulation primitives
- the new orchestration subsystem remains repo-contained and easy to run
- developer-facing workflow tooling is added without re-expanding the public simulation API surface

## Assumptions

- Users will author task specs from the repository root or provide paths relative to it.
- Real Codex / Opus provider commands may differ across environments, so command invocation must remain configurable.
- The workflow needs a deterministic validation path that does not mutate the main repository during tests.
- Human-authored plans are the default and should be treated as first-class task input.

## Outcome

The survey supports a repo-side implementation with:

- `agent_workflow/` for orchestration code, prompts, tasks, config, demo backend, and validation fixture
- `tools/run_agent_workflow.py` plus shell wrappers for one-command launch
- `agent_runs/` for resumable artifacts
- documentation updates in the root README, MkDocs site, and workflow-specific READMEs
