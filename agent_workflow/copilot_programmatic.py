from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_AGENT = "general-purpose"
PHASE_AGENT_DEFAULTS = {
    "execute": "general-purpose",
    "review": "code-review",
    "docs": "general-purpose",
    "summary": "general-purpose",
    "planner": "general-purpose",
}


def resolve_agent_name(phase: str, *, default_agent: str, overrides: dict[str, str]) -> str:
    normalized = phase.strip().lower()
    if normalized in overrides and overrides[normalized].strip():
        return overrides[normalized].strip()
    if normalized in PHASE_AGENT_DEFAULTS:
        return PHASE_AGENT_DEFAULTS[normalized]
    return default_agent.strip() or DEFAULT_AGENT


def build_proxy_prompt(*, prompt_path: Path, context_path: Path, working_directory: Path, cwd: Path) -> str:
    prompt_ref = _mentionable_path(prompt_path, cwd=cwd)
    context_ref = _mentionable_path(context_path, cwd=cwd)
    return (
        f"Follow the instructions in @{prompt_ref}.\n"
        f"Use @{context_ref} as structured workflow context.\n"
        f"Work only inside `{working_directory}` unless the prompt explicitly says otherwise.\n"
        "Return only the final output format requested by the prompt instructions."
    )


def build_copilot_command(
    *,
    agent: str,
    model: str | None,
    prompt: str,
    allow_all_tools: bool,
) -> list[str]:
    command = ["copilot", "--agent", agent, "--prompt", prompt]
    effective_model = (model or "").strip()
    if effective_model:
        command.extend(["--model", effective_model])
    if allow_all_tools:
        command.append("--allow-all-tools")
    return command


def run_copilot_programmatic(
    *,
    agent: str,
    model: str | None,
    prompt: str,
    cwd: Path,
    allow_all_tools: bool,
) -> subprocess.CompletedProcess[str]:
    command = build_copilot_command(
        agent=agent,
        model=model,
        prompt=prompt,
        allow_all_tools=allow_all_tools,
    )
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Invoke GitHub Copilot CLI in programmatic mode for agent_workflow.")
    parser.add_argument("--prompt-file", type=Path, required=True, help="Rendered workflow prompt file.")
    parser.add_argument("--context-file", type=Path, required=True, help="Structured workflow context JSON file.")
    parser.add_argument("--working-directory", type=Path, required=True, help="Directory the agent should modify.")
    parser.add_argument("--phase", required=True, help="Workflow phase name.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Copilot CLI model ID.")
    parser.add_argument("--default-agent", default=DEFAULT_AGENT, help="Fallback Copilot agent name.")
    parser.add_argument("--execute-agent", default="", help="Copilot agent override for execute.")
    parser.add_argument("--review-agent", default="", help="Copilot agent override for review.")
    parser.add_argument("--docs-agent", default="", help="Copilot agent override for docs.")
    parser.add_argument("--summary-agent", default="", help="Copilot agent override for summary.")
    parser.add_argument("--planner-agent", default="", help="Copilot agent override for planner.")
    parser.add_argument("--allow-all-tools", action=argparse.BooleanOptionalAction, default=True, help="Pass --allow-all-tools to Copilot CLI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cwd = Path.cwd().resolve()
    prompt_path = args.prompt_file.resolve()
    context_path = args.context_file.resolve()
    working_directory = args.working_directory.resolve()
    overrides = {
        "execute": str(args.execute_agent),
        "review": str(args.review_agent),
        "docs": str(args.docs_agent),
        "summary": str(args.summary_agent),
        "planner": str(args.planner_agent),
    }
    agent = resolve_agent_name(
        str(args.phase),
        default_agent=str(args.default_agent),
        overrides=overrides,
    )
    prompt = build_proxy_prompt(
        prompt_path=prompt_path,
        context_path=context_path,
        working_directory=working_directory,
        cwd=cwd,
    )
    completed = run_copilot_programmatic(
        agent=agent,
        model=str(args.model),
        prompt=prompt,
        cwd=cwd,
        allow_all_tools=bool(args.allow_all_tools),
    )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return int(completed.returncode)


def _mentionable_path(path: Path, *, cwd: Path) -> str:
    try:
        relative = path.resolve().relative_to(cwd.resolve())
    except ValueError:
        relative = path.resolve()
    return relative.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
