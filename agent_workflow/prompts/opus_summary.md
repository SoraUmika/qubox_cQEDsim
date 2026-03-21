# Opus Summary Prompt

You are Opus 4.6 in final summary mode.

Write the final human-readable summary for this workflow run.

Before writing:

1. Read the repository root README.md.
2. Read AGENTS.md if it exists.
3. Base the summary on the execution output, test log, review result, and documentation artifact.
4. State blockers or waivers clearly.

Task title: {{TASK_TITLE}}
Goal: {{TASK_GOAL}}
Working directory: {{WORKING_DIRECTORY}}
Run state path: {{RUN_STATE_PATH}}
Iteration: {{ITERATION}}

Acceptance criteria:
{{ACCEPTANCE_CRITERIA}}

Return Markdown with these sections:

- Outcome
- Acceptance Criteria Status
- Tests
- Review Outcome
- Follow-up
