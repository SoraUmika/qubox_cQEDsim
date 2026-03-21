from __future__ import annotations

from pathlib import Path
from typing import Any


PROMPT_FILES = {
    "execute": "codex_executor.md",
    "review": "opus_reviewer.md",
    "docs": "opus_docs.md",
    "summary": "opus_summary.md",
    "planner": "fallback_planner.md",
}


class PromptTemplateError(ValueError):
    pass


def load_prompt_template(repo_root: Path, prompt_name: str) -> str:
    try:
        filename = PROMPT_FILES[prompt_name]
    except KeyError as exc:
        raise PromptTemplateError(f"Unknown prompt template '{prompt_name}'.") from exc
    path = repo_root / "agent_workflow" / "prompts" / filename
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def format_list(items: list[str] | tuple[str, ...], *, empty_text: str = "- none") -> str:
    if not items:
        return empty_text
    return "\n".join(f"- {item}" for item in items)


def format_repair_instructions(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def default_prompt_context(*, task: dict[str, Any], state: dict[str, Any], working_directory: Path) -> dict[str, Any]:
    return {
        "TASK_TITLE": task.get("title", ""),
        "TASK_GOAL": task.get("goal", ""),
        "HUMAN_PLAN": task.get("human_plan", "") or "No human plan was supplied. Use a small planning pass before editing.",
        "CONTEXT_FILES": format_list(task.get("context_files", []), empty_text="- none specified"),
        "CONSTRAINTS": format_list(task.get("constraints", []), empty_text="- none specified"),
        "DELIVERABLES": format_list(task.get("deliverables", []), empty_text="- none specified"),
        "TESTS_TO_RUN": format_list(task.get("tests_to_run", []), empty_text="- no explicit test commands"),
        "DOCS_TO_UPDATE": format_list(task.get("docs_to_update", []), empty_text="- no explicit docs targets"),
        "ACCEPTANCE_CRITERIA": format_list(task.get("acceptance_criteria", []), empty_text="- no explicit acceptance criteria"),
        "STRICT_CONTEXT": task.get("strict_context", False),
        "ALLOW_REPO_EDITS": task.get("allow_repo_edits", True),
        "REVIEW_REQUIRED": task.get("review_required", True),
        "STOP_ON_BLOCKING_ERROR": task.get("stop_on_blocking_error", True),
        "ITERATION": state.get("iteration_count", 0),
        "CURRENT_PHASE": state.get("current_phase", "initialize"),
        "RUN_STATUS": state.get("status", "initialized"),
        "REPAIR_INSTRUCTIONS": format_repair_instructions(state.get("repair_instructions", [])),
        "WORKING_DIRECTORY": str(working_directory),
        "RUN_STATE_PATH": state.get("run_state_path", "RUN_STATE.json"),
        "IMPLEMENTATION_LOG_PATH": state.get("implementation_log_path", "IMPLEMENTATION_LOG.md"),
        "TASK_PATH": task.get("task_file", ""),
    }
