# Agent workflow Copilot autonomy gap

## Confirmed issues

- The workflow state machine already loops automatically across `execute -> test -> review -> docs -> summary`, but the shipped default backend profile was `unconfigured`, so real runs still required a user to manually drive Copilot outside the orchestrator.
- `README_AGENT_WORKFLOW.md` and `documentations/user_guides/agent_workflow.md` described a resumable semi-autonomous workflow, but the repo did not ship a concrete Copilot CLI programmatic profile even though the intended operator experience was hands-off phase progression.
- The task template pinned `backend_profile: unconfigured`, which overrode the config default and prevented the example path from using a real autonomous backend.

## Affected components

- `agent_workflow/config.json`
- `agent_workflow/tasks/example_task.yaml`
- `README_AGENT_WORKFLOW.md`
- `documentations/user_guides/agent_workflow.md`
- Copilot backend integration for repo-side workflow execution

## Why this is inconsistent

The repo positioned the workflow as a bounded autonomous loop, but the out-of-the-box configuration stopped short of wiring that loop to a real Copilot execution path. That forced users to act as the transition layer between phases, which contradicted the intended workflow behavior.

## Consequences

- Users must manually trigger the next Copilot step instead of letting the orchestrator own the full loop.
- The default user experience does not match the workflow documentation.
- The example task path advertises a runnable workflow but defaults to a non-runnable backend.

## Fix record

- Fixed by adding a shipped `copilot_cli_autonomous` backend profile that uses Copilot CLI programmatic mode with built-in Copilot agents and `--allow-all-tools`.
- Fixed by adding `agent_workflow/copilot_programmatic.py` and `tools/run_copilot_programmatic.py` to keep prompt handoff short and Windows-safe by referencing prompt/context artifacts instead of injecting huge prompt bodies directly into the command line.
- Fixed by updating the example task and workflow documentation to point at the Copilot-backed autonomous path, while preserving `validation_demo` as the deterministic test backend.

## Remaining open questions

- Fully headless runs still depend on one-time Copilot CLI login and trusted-directory setup outside the repo.
- If future workflows need different Copilot built-in agents per phase beyond the current execute/review/docs/summary mapping, the wrapper may need additional configuration knobs.
